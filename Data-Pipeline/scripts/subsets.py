"""Build the Civil Comments evaluation and drift subsets.

Inputs (CONTRACTS.md section 2, written by the preprocess stage):
    data/cache/civil/train_clean.parquet   (the 1.8M-row train split)
    data/processed/civil/{validation,test}_clean.parquet
    data/processed/civil/identities_test_clean.parquet
    config/civil_teacher_shift_ids.txt (paths.civil_exclude_ids; optional)

Outputs:
    data/processed/civil/h2_eval.parquet       10,000 from test, stratified on toxicity
    data/processed/civil/canary_civil.parquet  4,000 from validation, stratified on toxicity
    data/processed/civil/bias_eval.parquet     the full Identities test split
    data/processed/civil/drift/batch_00..19.parquet
                                               40,000 from train, stratified on toxicity,
                                               sorted by created_date, cut into 20 batches
    data/splits/{h2_eval,canary_civil,bias_eval}_ids.txt
    data/splits/civil_drift_ids.csv (id, batch, created_date)

Each subset comes from a different Civil split, so they never share a comment,
except that the Identities test split is an annotated subset of Civil test: some
bias_eval rows can also appear in h2_eval. That overlap is expected and is
reported in the summary rather than removed.

Comments the teacher lane used to choose its prompt (its shift check on Civil
validation) are listed in paths.civil_exclude_ids. They are removed before
sampling, so they never reach the canary or the drift stream. If the file does
not exist yet, nothing is excluded and the stage logs a warning.

    python -m scripts.subsets --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.preprocess import civil_paths
from scripts.utils import atomic_path, atomic_write, get_logger, load_config, timed, write_summary

logger = get_logger(__name__)
TASK_ID = "subsets"
STRATIFY_ON = "toxicity"
IDENTITIES_FILE = "identities_test_clean.parquet"


def _by_id(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by id so results depend only on the data and the seed, not on row order."""
    return df.sort_values("id", kind="stable").reset_index(drop=True)


def _read(path: Path, labels: list[str]) -> pd.DataFrame:
    """Load one input file and check the columns this stage relies on."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found in {path.parent.as_posix()}; run the upstream stage first"
        )
    df = pd.read_parquet(path)
    missing = [c for c in ("id", "text", "created_date", *labels) if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    duplicates = int(df["id"].duplicated().sum())
    if duplicates:
        raise ValueError(f"{path.name} has {duplicates} duplicate ids")
    return _by_id(df)


def _load_excluded(path: Path) -> set[str]:
    """Ids to keep out of every subset; empty (with a warning) if the file is missing."""
    if not path.exists():
        logger.warning("Exclusion list %s not found; no rows excluded", path.as_posix())
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def _sample(df: pd.DataFrame, n: int, seed: int, name: str) -> pd.DataFrame:
    """Draw exactly ``n`` rows, stratified on toxicity; the whole frame if it is smaller."""
    if n >= len(df):
        logger.warning("%s: wanted %d rows but only %d exist; using all of them",
                       name, n, len(df))
        return df
    _, sample = train_test_split(df, test_size=n, stratify=df[STRATIFY_ON],
                                 random_state=seed)
    return _by_id(sample)


def _drift_batches(df: pd.DataFrame, total: int, batches: int, seed: int) -> list[pd.DataFrame]:
    """Stratified sample, sorted by time, cut into ``batches`` equal batches."""
    per_batch = min(total, len(df)) // batches
    if per_batch < 1:
        raise ValueError(f"Civil train has {len(df)} rows; too few for {batches} drift batches")
    stream = _sample(df, per_batch * batches, seed, "drift")
    # Time order is the point: batch_00 is the oldest traffic, batch_19 the newest.
    stream = stream.sort_values(["created_date", "id"], kind="stable").reset_index(drop=True)
    return [stream.iloc[b * per_batch:(b + 1) * per_batch].reset_index(drop=True)
            for b in range(batches)]


def _write_parquet(df: pd.DataFrame, target: Path) -> None:
    with atomic_path(target) as tmp:
        df.to_parquet(tmp, index=False)


def _write_ids(ids: pd.Series, target: Path) -> None:
    atomic_write(target, "\n".join(ids) + "\n")


def _toxicity_pts(df: pd.DataFrame) -> float:
    return round(float(df[STRATIFY_ON].mean()) * 100, 3)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = cfg["seed"]
    labels = cfg["labels"]["names"]
    sizes = cfg["splits"]["civil"]
    civil_out = Path(cfg["paths"]["processed_dir"]) / "civil"
    splits_dir = Path(cfg["paths"]["splits_dir"])
    summary: dict[str, Any] = {"task": TASK_ID, "mode": "real", "dropped": {}}

    with timed(summary):
        # Not under processed_dir: preprocess keeps the 1.8M-row Civil train in the
        # untracked cache. civil_paths owns that rule, so ask it rather than
        # rebuilding the path here.
        train = _read(civil_paths(cfg, "train")[1], labels)
        validation = _read(civil_out / "validation_clean.parquet", labels)
        test = _read(civil_out / "test_clean.parquet", labels)
        identities = _read(civil_out / IDENTITIES_FILE, labels)
        summary["rows_in"] = {"train": len(train), "validation": len(validation),
                              "test": len(test), "identities_test": len(identities)}

        # Teacher prompt-selection rows stay out of the canary and the drift
        # stream. They come from Civil validation; train is filtered too for safety.
        excluded = _load_excluded(Path(cfg["paths"]["civil_exclude_ids"]))
        kept_validation = validation[~validation["id"].isin(excluded)].reset_index(drop=True)
        kept_train = train[~train["id"].isin(excluded)].reset_index(drop=True)
        removed = {"validation": len(validation) - len(kept_validation),
                   "train": len(train) - len(kept_train)}
        summary["excluded"] = {"ids_listed": len(excluded), "removed": removed}
        if sum(removed.values()):
            summary["dropped"]["teacher_shift_rows"] = sum(removed.values())
        validation, train = kept_validation, kept_train

        subsets = {
            "h2_eval": _sample(test, sizes["h2_size"], seed, "h2_eval"),
            "canary_civil": _sample(validation, sizes["canary_size"], seed, "canary_civil"),
            "bias_eval": identities,
        }
        drift = _drift_batches(train, sizes["drift_total"], sizes["drift_batches"], seed)

        for name, df in subsets.items():
            _write_parquet(df, civil_out / f"{name}.parquet")
            _write_ids(df["id"], splits_dir / f"{name}_ids.txt")
        for b, df in enumerate(drift):
            _write_parquet(df, civil_out / "drift" / f"batch_{b:02d}.parquet")
        manifest = pd.concat([
            pd.DataFrame({"id": df["id"], "batch": b,
                          "created_date": df["created_date"].map(lambda t: t.isoformat())})
            for b, df in enumerate(drift)
        ])
        atomic_write(splits_dir / "civil_drift_ids.csv", manifest.to_csv(index=False))

        summary["files"] = {name: len(df) for name, df in subsets.items()}
        summary["files"].update({f"drift/batch_{b:02d}": len(df) for b, df in enumerate(drift)})
        summary["rows_out"] = sum(summary["files"].values())
        summary["drift_date_range"] = [str(drift[0]["created_date"].iloc[0]),
                                       str(drift[-1]["created_date"].iloc[-1])]
        summary["toxicity_pts"] = {
            "h2_eval": _toxicity_pts(subsets["h2_eval"]), "civil_test": _toxicity_pts(test),
            "canary_civil": _toxicity_pts(subsets["canary_civil"]),
            "civil_validation": _toxicity_pts(validation),
            "drift": _toxicity_pts(pd.concat(drift)), "civil_train": _toxicity_pts(train),
        }
        # Expected, not an error: Identities test is an annotated subset of Civil test.
        summary["bias_eval_ids_also_in_h2_eval"] = int(
            subsets["bias_eval"]["id"].isin(subsets["h2_eval"]["id"]).sum()
        )

    logger.info("subsets: h2=%d canary=%d bias=%d drift=%d x %d",
                len(subsets["h2_eval"]), len(subsets["canary_civil"]),
                len(subsets["bias_eval"]), len(drift), len(drift[0]))
    write_summary(TASK_ID, summary, cfg)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config)), indent=2, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
