"""Merge the teacher lane's outputs into one training table (Phase 2).

Writes data/processed/teacher_labels.parquet: one row per raw Wikipedia training id (159,571),
joined by id from four sources in the shard store:

- p_<label>: the stack-v1 soft labels, the student's targets;
- p_<label>_raw: the raw Qwen probabilities (never a training target);
- tfidf_oof_<label>: the out-of-fold TF-IDF scores;
- p_<label>_calqwen: the calibrated-Qwen soft labels, for the ablation's attribution arm;

plus prompt_version and model_version (the soft labels') and teacher_version. There are no gate
or drift rows. Join it onto the processed train and validation sets by id; rows dedupe removed
simply drop out. The columns are in CONTRACTS.md.

It runs only after validate_teacher_shards: it reads that stage's report and stops unless every
shard run it reads passed. If validation was skipped (store unreachable), the merge skips too.
The TF-IDF file is checked against its committed manifest (SHA-256, row count, id hash, version).

Run on its own: python -m scripts.merge_teacher_labels
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from scripts.alerts import send_alert
from scripts.errors import TeacherShardError
from scripts.parquet_io import write_frame
from scripts.utils import get_logger, load_config, timed
from scripts.validate_teacher_shards import resolve, teacher_modules

logger = get_logger(__name__)


def require_validated(report_path: Path, prefixes: list[str]) -> bool:
    """True if every prefix passed validation, False if validation was skipped."""
    if not report_path.exists():
        raise TeacherShardError(f"{report_path} not found: run validate_teacher_shards first")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("skipped"):
        return False
    for prefix in prefixes:
        result = report.get("runs", {}).get(prefix)
        if result is None:
            raise TeacherShardError(f"{prefix} is not in {report_path}: validate it first")
        if result["errors"]:
            raise TeacherShardError(f"{prefix} failed validation ({result['errors']} errors)")
    return True


def load_shards(validate: ModuleType, storage, prefix: str, columns: list[str]) -> pd.DataFrame:
    """All shards under ``prefix`` as one frame, sorted by id."""
    with tempfile.TemporaryDirectory() as tmp:
        frames, _manifests, _schemas, errors = validate.load_run(storage, prefix, Path(tmp))
    if errors:
        raise TeacherShardError(f"{prefix}: could not load shards: {errors[:3]}")
    if not frames:
        raise TeacherShardError(f"{prefix}: no shards found")
    frame = pd.concat([frames[name] for name in sorted(frames)], ignore_index=True)
    return frame[columns].sort_values("id", ignore_index=True)


def load_tfidf(
    storage, path: str, manifest: dict[str, Any], shards: ModuleType, columns: list[str]
) -> pd.DataFrame:
    """The out-of-fold TF-IDF scores, checked against their committed manifest."""
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(storage.download(path, Path(tmp)))
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        frame = pd.read_parquet(local)
    ids = sorted(frame["id"].astype(str))
    problems = []
    if digest != manifest["sha256"]:
        problems.append(f"sha256 {digest}, manifest says {manifest['sha256']}")
    if len(frame) != manifest["row_count"]:
        problems.append(f"{len(frame)} rows, manifest says {manifest['row_count']}")
    if shards.ids_sha256(ids) != manifest["ids_sha256"]:
        problems.append("ids_sha256 differs from the manifest")
    versions = set(frame["model_version"].astype(str))
    if versions != {manifest["baseline_version"]}:
        problems.append(f"model_version {sorted(versions)}, manifest says "
                        f"{manifest['baseline_version']}")
    if problems:
        raise TeacherShardError(f"{path}: " + "; ".join(problems))
    return frame[columns].sort_values("id", ignore_index=True)


def check_ids(name: str, frame: pd.DataFrame, expected: set[str]) -> None:
    """Every source must hold exactly the expected ids, once each."""
    ids = frame["id"].astype(str)
    if ids.duplicated().any():
        raise TeacherShardError(f"{name}: {int(ids.duplicated().sum())} duplicate ids")
    found = set(ids)
    if found != expected:
        raise TeacherShardError(
            f"{name}: {len(expected - found)} ids missing and {len(found - expected)} extra, "
            "compared with the soft labels"
        )


def combine(
    soft: pd.DataFrame,
    others: list[tuple[str, pd.DataFrame, dict[str, str]]],
    teacher_version: str,
) -> pd.DataFrame:
    """Join every source onto the soft labels by id, with the contract's column names."""
    expected = set(soft["id"].astype(str))
    check_ids("soft labels", soft, expected)
    table = soft.copy()
    for name, frame, rename in others:
        check_ids(name, frame, expected)
        table = table.merge(frame.rename(columns=rename), on="id", how="inner",
                            validate="one_to_one")
    table["teacher_version"] = teacher_version
    scores = [c for c in table.columns if c.startswith(("p_", "tfidf_oof_"))]
    table[scores] = table[scores].astype("float32")
    for column in ("id", "prompt_version", "model_version", "teacher_version"):
        table[column] = table[column].astype(str)
    versions = ["prompt_version", "model_version", "teacher_version"]
    ordered = ["id"] + [c for c in table.columns if c not in ("id", *versions)] + versions
    return table[ordered].sort_values("id", ignore_index=True)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build data/processed/teacher_labels.parquet. Returns the stage summary."""
    summary: dict[str, Any] = {"task": "merge_teacher_labels", "dropped": {}}
    with timed(summary):
        tcfg = cfg["teacher"]
        merge = tcfg["merge"]
        prefixes = [merge["soft"], merge["raw"]] + ([merge["calqwen"]] if merge.get("calqwen")
                                                    else [])
        if not require_validated(resolve(tcfg["report"]), prefixes):
            logger.warning("Skipping the teacher-label merge: validation was skipped")
            send_alert("warn", "merge_teacher_labels skipped: validation was skipped", {})
            summary["skipped"] = True
            return summary

        mods = teacher_modules(resolve(tcfg["model_pipeline_dir"]))
        mcfg = mods["config"].load_config(None)
        labels = list(mods["config"].LABELS)
        p = [f"p_{label}" for label in labels]
        storage = mods["storage"].get_storage(
            mcfg, resolve(tcfg.get("local_root")), tcfg.get("backend")
        )
        validate = mods["validate"]

        soft = load_shards(validate, storage, merge["soft"],
                           ["id", *p, "prompt_version", "model_version"])
        others = [
            ("raw teacher", load_shards(validate, storage, merge["raw"], ["id", *p]),
             {c: f"{c}_raw" for c in p}),
        ]
        manifest = json.loads(resolve(merge["tfidf_manifest"]).read_text(encoding="utf-8"))
        tfidf = load_tfidf(storage, merge["tfidf"], manifest, mods["shards"], ["id", *p])
        others.append(("tfidf", tfidf, {f"p_{label}": f"tfidf_oof_{label}" for label in labels}))
        if merge.get("calqwen"):
            calqwen = load_shards(validate, storage, merge["calqwen"], ["id", *p])
            others.append(("calqwen", calqwen, {c: f"{c}_calqwen" for c in p}))

        table = combine(soft, others, merge["teacher_version"])
        output = resolve(merge["output"])
        rows = write_frame(table, output)

        summary.update({
            "skipped": False,
            "rows_out": rows,
            "columns": list(table.columns),
            "missing_targets": int(table[p].isna().any(axis=1).sum()),
            "output": str(output),
        })
        logger.info("Wrote %s: %d rows, %d columns", output, rows, len(table.columns))
    return summary


def main() -> None:
    print(json.dumps(run(load_config()), indent=2, default=str))


if __name__ == "__main__":
    main()
