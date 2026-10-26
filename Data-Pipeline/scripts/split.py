"""Split deduplicated Wikipedia train into train / val / canary, and fix the gate set.

Inputs (CONTRACTS.md section 2, written by the preprocess and dedupe stages):
    data/processed/wikipedia/train_dedup.parquet
    data/processed/wikipedia/test_clean.parquet

Outputs:
    data/processed/wikipedia/{train,val,gate,canary_wikipedia}.parquet
    data/splits/{train,val,gate,canary_wikipedia}_ids.txt

Train and validation come from a seeded multilabel iterative stratification, so
every label, including rare ones such as threat, keeps the same prevalence in
both. The canary set is carved out of validation the same way. The test
split becomes the gate set unchanged.

    python -m scripts.split --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from scripts.utils import atomic_path, atomic_write, get_logger, load_config, timed, write_summary

logger = get_logger(__name__)
TASK_ID = "split"
OUTPUTS = ("train", "val", "gate", "canary_wikipedia")


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
    missing = [c for c in ("id", "text", *labels) if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    duplicates = int(df["id"].duplicated().sum())
    if duplicates:
        raise ValueError(f"{path.name} has {duplicates} duplicate ids")
    return _by_id(df)


def _stratified(
    df: pd.DataFrame, labels: list[str], fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (rest, taken) with every label's prevalence matched."""
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=fraction, random_state=seed
    )
    rest_idx, taken_idx = next(splitter.split(np.zeros((len(df), 1)), df[labels].to_numpy()))
    return df.iloc[np.sort(rest_idx)], df.iloc[np.sort(taken_idx)]


def _carve(
    df: pd.DataFrame, labels: list[str], n: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Take exactly ``n`` stratified rows out of ``df``; return (rest, taken)."""
    rest, taken = _stratified(df, labels, n / len(df), seed)
    diff = len(taken) - n
    if diff:
        # Stratification can land a row or two off the target. Move the
        # difference at random (seeded) so the canary size is exact.
        rng = np.random.default_rng(seed)
        src, dst = (taken, rest) if diff > 0 else (rest, taken)
        move = src.index[np.sort(rng.choice(len(src), abs(diff), replace=False))]
        dst = pd.concat([dst, src.loc[move]])
        src = src.drop(move)
        rest, taken = (dst, src) if diff > 0 else (src, dst)
    return _by_id(rest), _by_id(taken)


def _prevalence(df: pd.DataFrame, labels: list[str]) -> dict[str, float]:
    """Positive rate per label, in percentage points."""
    return {label: round(float(df[label].mean()) * 100, 3) for label in labels}


def _write(df: pd.DataFrame, name: str, wiki_out: Path, splits_dir: Path) -> None:
    with atomic_path(wiki_out / f"{name}.parquet") as tmp:
        df.to_parquet(tmp, index=False)
    atomic_write(splits_dir / f"{name}_ids.txt", "\n".join(df["id"]) + "\n")


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = cfg["seed"]
    labels = cfg["labels"]["names"]
    val_fraction = cfg["splits"]["val_fraction"]
    canary_target = cfg["splits"]["canary_wikipedia_size"]
    wiki_out = Path(cfg["paths"]["processed_dir"]) / "wikipedia"
    splits_dir = Path(cfg["paths"]["splits_dir"])
    summary: dict[str, Any] = {"task": TASK_ID, "mode": "real", "dropped": {}}

    with timed(summary):
        train_dedup = _read(wiki_out / "train_dedup.parquet", labels)
        gate = _read(wiki_out / "test_clean.parquet", labels)
        summary["rows_in"] = {"train_dedup": len(train_dedup), "test_clean": len(gate)}

        train, val = _stratified(train_dedup, labels, val_fraction, seed)
        train, val = _by_id(train), _by_id(val)

        # Small inputs (stubs, tests) cannot hold a full-size canary; cap it at
        # half of validation, the same rule the stub uses.
        canary_n = min(canary_target, len(val) // 2)
        if canary_n < 1:
            raise ValueError(f"Validation has {len(val)} rows; too few to carve a canary set")
        if canary_n < canary_target:
            logger.warning("Validation has %d rows; canary capped at %d instead of %d",
                           len(val), canary_n, canary_target)
        val, canary = _carve(val, labels, canary_n, seed)

        train["sample_weight"] = np.ones(len(train), dtype="float32")  # bias step overwrites
        # sample_weight belongs to train only (CONTRACTS.md section 2).
        val, gate, canary = (df.drop(columns="sample_weight", errors="ignore")
                             for df in (val, gate, canary))
        parts = {"train": train, "val": val, "gate": gate, "canary_wikipedia": canary}
        for name, df in parts.items():
            _write(df, name, wiki_out, splits_dir)

        n_in = len(train_dedup)
        summary["files"] = {name: len(df) for name, df in parts.items()}
        summary["rows_out"] = sum(summary["files"].values())
        summary["fraction_of_train_dedup"] = {
            name: round(len(parts[name]) / n_in, 4) for name in ("train", "val", "canary_wikipedia")
        }
        summary["prevalence_pts"] = {name: _prevalence(df, labels) for name, df in parts.items()}

    logger.info("split: train=%d val=%d canary=%d gate=%d",
                len(train), len(val), len(canary), len(gate))
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
