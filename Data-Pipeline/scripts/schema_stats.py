"""TFDV schema inference and validation over every raw and processed file.

Two modes:

    python -m scripts.schema_stats --infer     build the schemas (run once, review, commit)
    python -m scripts.schema_stats             validate every file (what the DAG runs)

Runs in the TensorFlow environment at /opt/tfenv (Python 3.11, pandas 1.5,
pyarrow 10), so it imports nothing from the main-environment stages.

Two schemas, because raw and processed files give the same columns different
types (raw labels are float rater fractions or 0.0/1.0, processed labels int8):

    data/schema/raw_schema.pbtxt   raw files: inferred from raw Wikipedia train
                                   plus raw Civil identities test
    data/schema/schema.pbtxt       processed files: inferred from processed
                                   Wikipedia train plus processed bias_eval

Files with different column sets share one schema through TFDV environments.
Which columns belong to which environment is read from the files themselves at
--infer time and then fixed in the committed schema:

    RAW_WIKI     raw Wikipedia train and test
    RAW_CIVIL    raw Civil train, validation, test and identities test
    WIKI         processed Wikipedia files, the obfuscation set
    WIKI_TRAIN   processed Wikipedia train (the only file with sample_weight)
    CIVIL        processed Civil files, H2, canary, drift
    CIVIL_BIAS   processed Civil files with identity columns (bias_eval)

Curation applied on top of inference: label and flag domains {0, 1}, *_frac and
identity scores in [0, 1], char_len >= 1 (no empty text), id and text required
everywhere, created_date required in Civil, inferred string vocabularies
removed (they would flag every new publication or article id), and raw identity
scores allowed to be missing (most raw Civil rows have none). The work plan's
"language == en" cannot be checked from the data, since neither Kaggle source
has a language field; both datasets are English-only by construction.

Any anomaly fails the task after the full report is written to
reports/schema/validation.json. The two Civil train files (1.8M rows each) are
validated on a seeded sample of validation.raw_civil_stats_sample rows.

The TFDV statistics for every validated file are saved as text protos in
reports/schema/stats/, mirroring the target names (for example
reports/schema/stats/processed/civil/drift.pbtxt), so two runs can be compared
and the monitoring phase's drift comparator can load them as a baseline.
Before saving, the most-frequent-values lists of the text column are removed:
they hold real comments, and the Kaggle rules (work plan v3, section 2.2) keep
comment text out of the repository. Length, uniqueness and every other
column's statistics are kept.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.utils import atomic_path, atomic_write, get_logger, load_config, timed, write_summary

logger = get_logger(__name__)
TASK_ID = "schema_stats"
RAW_SCHEMA = "raw_schema.pbtxt"
PROCESSED_SCHEMA = "schema.pbtxt"

RAW_ENVS = ("RAW_WIKI", "RAW_CIVIL")
PROCESSED_ENVS = ("WIKI", "WIKI_TRAIN", "CIVIL", "CIVIL_BIAS")
FLAG_COLUMNS = ("has_identity_term", "has_obfuscation")
# Free-text columns whose example values must never be written to reports/.
TEXT_COLUMNS = ("text",)


class SchemaAnomalyError(RuntimeError):
    """Raised after the report is written when any file breaks its schema."""


def _tfdv():
    """Import TFDV lazily, so a clear message appears outside /opt/tfenv."""
    try:
        import tensorflow_data_validation as tfdv
    except ImportError as exc:
        raise ImportError(
            "schema_stats needs TFDV, which only exists in the TensorFlow environment. "
            "Run it with /opt/tfenv/bin/python inside the Docker image."
        ) from exc
    return tfdv


# --- Which file is validated against which schema and environment -------------------

def targets(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Every file to validate: (name, paths, schema, environment, sampled?)."""
    raw, processed, cache = (Path(cfg["paths"][k]) for k in ("raw_dir", "processed_dir",
                                                              "cache_dir"))
    wiki, civil = processed / "wikipedia", processed / "civil"
    out = [
        ("raw/wikipedia/train", [raw / "wikipedia/train.parquet"], "raw", "RAW_WIKI", False),
        ("raw/wikipedia/test", [raw / "wikipedia/test.parquet"], "raw", "RAW_WIKI", False),
        ("cache/civil/train", [cache / "civil/train.parquet"], "raw", "RAW_CIVIL", True),
        ("raw/civil/validation", [raw / "civil/validation.parquet"], "raw", "RAW_CIVIL", False),
        ("raw/civil/test", [raw / "civil/test.parquet"], "raw", "RAW_CIVIL", False),
        ("raw/civil/identities_test", [raw / "civil/identities_test.parquet"], "raw",
         "RAW_CIVIL", False),
    ]
    for name in ("train_clean", "test_clean", "train_dedup", "val", "gate", "canary_wikipedia"):
        out.append((f"processed/wikipedia/{name}", [wiki / f"{name}.parquet"], "processed",
                    "WIKI", False))
    out += [
        ("processed/wikipedia/train", [wiki / "train.parquet"], "processed", "WIKI_TRAIN", False),
        ("processed/obfuscation", [processed / "obfuscation.parquet"], "processed", "WIKI", False),
        ("cache/civil/train_clean", [cache / "civil/train_clean.parquet"], "processed", "CIVIL",
         True),
    ]
    for name in ("validation_clean", "test_clean", "h2_eval", "canary_civil"):
        out.append((f"processed/civil/{name}", [civil / f"{name}.parquet"], "processed", "CIVIL",
                    False))
    n_batches = cfg["splits"]["civil"]["drift_batches"]
    out.append(("processed/civil/drift", [civil / "drift" / f"batch_{b:02d}.parquet"
                                          for b in range(n_batches)], "processed", "CIVIL", False))
    for name in ("identities_test_clean", "bias_eval"):
        out.append((f"processed/civil/{name}", [civil / f"{name}.parquet"], "processed",
                    "CIVIL_BIAS", False))
    keys = ("name", "paths", "schema", "environment", "sampled")
    return [dict(zip(keys, row, strict=True)) for row in out]


# --- Reading files the way TFDV needs them --------------------------------------------

def read_for_tfdv(paths: list[Path], sample: int | None, seed: int) -> pd.DataFrame:
    """Load Parquet into a frame TFDV accepts.

    TFDV reads strings, ints and floats, but not timestamps or booleans, so
    timestamps become ISO strings and booleans become 0/1. A sample, when asked
    for, is drawn from the Arrow table before converting to pandas, so the
    1.8M-row Civil train never has to fit in memory as Python strings.
    """
    frames = []
    for path in paths:
        table = pq.read_table(path)
        if sample is not None and table.num_rows > sample:
            rng = np.random.default_rng(seed)
            table = table.take(np.sort(rng.choice(table.num_rows, sample, replace=False)))
        frames.append(table.to_pandas())
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        elif pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype("int64")
    return df


def statistics(df: pd.DataFrame):
    return _tfdv().generate_statistics_from_dataframe(df)


def columns(path: Path) -> set[str]:
    return set(pq.read_schema(path).names)


# --- Inference and curation -----------------------------------------------------------

def _merge_inferred(sources: list[pd.DataFrame]):
    """Infer one schema from each source and merge their features by name."""
    tfdv = _tfdv()
    merged = None
    for df in sources:
        schema = tfdv.infer_schema(statistics(df), infer_feature_shape=True)
        if merged is None:
            merged = schema
            continue
        known = {f.name for f in merged.feature}
        for feature in schema.feature:
            if feature.name not in known:
                merged.feature.add().CopyFrom(feature)
    return merged


def _assign_environments(schema, env_columns: dict[str, set[str]]) -> None:
    """Every feature lives in exactly the environments whose files contain it."""
    schema.ClearField("default_environment")
    schema.default_environment.extend(env_columns)
    for feature in schema.feature:
        del feature.not_in_environment[:]
        feature.not_in_environment.extend(
            env for env, cols in env_columns.items() if feature.name not in cols
        )


def _require(feature) -> None:
    feature.presence.min_fraction = 1.0
    feature.presence.min_count = 1


def _domain(feature, low: float | None = None, high: float | None = None) -> None:
    """Set an int or float range, matching the feature's type."""
    from tensorflow_metadata.proto.v0 import schema_pb2

    is_int = feature.type == schema_pb2.INT
    domain = feature.int_domain if is_int else feature.float_domain
    if low is not None:
        domain.min = int(low) if is_int else float(low)
    if high is not None:
        domain.max = int(high) if is_int else float(high)


def curate(schema, labels: list[str], identity: set[str], raw: bool) -> None:
    """Tighten the inferred schema to what the contracts actually promise."""
    schema.ClearField("string_domain")
    for feature in schema.feature:
        name = feature.name
        if feature.WhichOneof("domain_info") == "domain":
            # An inferred vocabulary would flag every unseen publication or article id.
            feature.ClearField("domain")
        if name in ("id", "text", "created_date"):
            _require(feature)
        if name in labels:
            _domain(feature, 0, 1)
        elif name.endswith("_frac") or name == "sexual_explicit":
            _domain(feature, 0, 1)
        elif name in FLAG_COLUMNS:
            _domain(feature, 0, 1)
        elif name == "char_len":
            _domain(feature, 1)  # empty text is dropped in preprocess
        elif name == "sample_weight":
            _domain(feature, 0)
        elif name == "identity_annotator_count":
            _domain(feature, 0)
        elif name in identity:
            if raw:
                # Most raw Civil rows were never shown to identity annotators. A fixed
                # shape means "always present", so swap it for "one value when present".
                feature.presence.min_fraction = 0.0
                feature.presence.min_count = 0
                feature.ClearField("shape")
                feature.value_count.min = 1
                feature.value_count.max = 1
            else:
                _domain(feature, 0, 1)


def infer(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build both schemas from the current pipeline outputs and write them."""
    tfdv = _tfdv()
    labels = cfg["labels"]["names"]
    raw, processed = Path(cfg["paths"]["raw_dir"]), Path(cfg["paths"]["processed_dir"])
    schema_dir = Path(cfg["paths"]["schema_dir"])
    wiki, civil = processed / "wikipedia", processed / "civil"

    raw_envs = {"RAW_WIKI": columns(raw / "wikipedia/train.parquet"),
                "RAW_CIVIL": columns(raw / "civil/identities_test.parquet")}
    env_cols = {"WIKI": columns(wiki / "val.parquet"),
                "WIKI_TRAIN": columns(wiki / "train.parquet"),
                "CIVIL": columns(civil / "test_clean.parquet"),
                "CIVIL_BIAS": columns(civil / "bias_eval.parquet")}
    identity = env_cols["CIVIL_BIAS"] - env_cols["CIVIL"] - {"identity_annotator_count"}

    raw_schema = _merge_inferred([
        read_for_tfdv([raw / "wikipedia/train.parquet"], None, cfg["seed"]),
        read_for_tfdv([raw / "civil/identities_test.parquet"], None, cfg["seed"]),
    ])
    curate(raw_schema, labels, identity, raw=True)
    _assign_environments(raw_schema, raw_envs)

    schema = _merge_inferred([
        read_for_tfdv([wiki / "train.parquet"], None, cfg["seed"]),
        read_for_tfdv([civil / "bias_eval.parquet"], None, cfg["seed"]),
    ])
    curate(schema, labels, identity, raw=False)
    _assign_environments(schema, env_cols)

    for name, proto in ((RAW_SCHEMA, raw_schema), (PROCESSED_SCHEMA, schema)):
        with atomic_path(schema_dir / name) as tmp:
            tfdv.write_schema_text(proto, str(tmp))
    logger.info("Wrote %s and %s to %s; review them, then commit both",
                RAW_SCHEMA, PROCESSED_SCHEMA, schema_dir.as_posix())
    return {"raw_features": len(raw_schema.feature), "processed_features": len(schema.feature),
            "identity_columns": len(identity)}


# --- Validation (what the DAG runs) ---------------------------------------------------

def load_schemas(cfg: dict[str, Any]) -> dict[str, Any]:
    tfdv = _tfdv()
    schema_dir = Path(cfg["paths"]["schema_dir"])
    schemas = {}
    for key, name in (("raw", RAW_SCHEMA), ("processed", PROCESSED_SCHEMA)):
        path = schema_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"No schema at {path.as_posix()}. Build it first with "
                "`python -m scripts.schema_stats --infer`, review it, and commit it."
            )
        schemas[key] = tfdv.load_schema_text(str(path))
    return schemas


def describe(anomalies) -> dict[str, str]:
    """Anomaly proto to {feature: description}, including dataset-level ones."""
    found = {name: info.description for name, info in anomalies.anomaly_info.items()}
    if anomalies.HasField("dataset_anomaly_info"):
        found["(dataset)"] = anomalies.dataset_anomaly_info.description
    return found


def stats_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["reports_dir"]) / "schema" / "stats"


def without_comment_text(stats):
    """A copy of the statistics with the text column's example values removed.

    TFDV records each string column's most frequent values, which for `text`
    are real comments. Those may not be committed (Kaggle rules), so they are
    cleared; the text column's length and uniqueness statistics remain.
    """
    clean = type(stats)()
    clean.CopyFrom(stats)
    for dataset in clean.datasets:
        for feature in dataset.features:
            name = feature.path.step[0] if feature.path.step else feature.name
            if name in TEXT_COLUMNS and feature.HasField("string_stats"):
                feature.string_stats.ClearField("top_values")
                feature.string_stats.ClearField("rank_histogram")
    return clean


def save_statistics(stats, cfg: dict[str, Any], target_name: str) -> str:
    """Write one target's TFDV statistics as a text proto; return its path under reports/."""
    path = stats_dir(cfg) / f"{target_name}.pbtxt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as tmp:
        _tfdv().write_stats_text(without_comment_text(stats), str(tmp))
    return path.relative_to(Path(cfg["paths"]["reports_dir"])).as_posix()


def validate(cfg: dict[str, Any], only: list[str] | None = None) -> dict[str, Any]:
    """Validate every target (or just the named ones) against its schema."""
    tfdv = _tfdv()
    schemas = load_schemas(cfg)
    sample = cfg["validation"]["raw_civil_stats_sample"]
    results = []
    for target in targets(cfg):
        if only is not None and target["name"] not in only:
            continue
        missing = [p.as_posix() for p in target["paths"] if not p.exists()]
        if missing:
            results.append({"target": target["name"], "rows": 0,
                            "anomalies": {"(file)": f"missing: {missing}"}})
            continue
        df = read_for_tfdv(target["paths"], sample if target["sampled"] else None, cfg["seed"])
        stats = statistics(df)
        anomalies = tfdv.validate_statistics(stats, schemas[target["schema"]],
                                             environment=target["environment"])
        results.append({"target": target["name"], "rows": len(df),
                        "environment": target["environment"],
                        "statistics": save_statistics(stats, cfg, target["name"]),
                        "anomalies": describe(anomalies)})
        logger.info("schema_stats: %s (%d rows) -> %d anomalies", target["name"], len(df),
                    len(results[-1]["anomalies"]))
    return {"results": results,
            "failed": [r["target"] for r in results if r["anomalies"]]}


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"task": TASK_ID, "mode": "real", "dropped": {}, "rows_out": 0}
    with timed(summary):
        report = validate(cfg)
        summary["files_checked"] = len(report["results"])
        summary["rows_checked"] = {r["target"]: r["rows"] for r in report["results"]}
        summary["files_with_anomalies"] = report["failed"]
        summary["statistics_dir"] = stats_dir(cfg).relative_to(
            Path(cfg["paths"]["reports_dir"]).parent).as_posix()
    atomic_write(Path(cfg["paths"]["reports_dir"]) / "schema" / "validation.json",
                 json.dumps(report, indent=2, sort_keys=True))
    write_summary(TASK_ID, summary, cfg)
    if report["failed"]:
        details = "; ".join(f"{r['target']}: {r['anomalies']}" for r in report["results"]
                            if r["anomalies"])
        raise SchemaAnomalyError(f"{len(report['failed'])} file(s) break the schema: {details}")
    logger.info("schema_stats: %d files match their schema", summary["files_checked"])
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--infer", action="store_true",
                        help="build the schemas from current outputs instead of validating")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    result = infer(cfg) if args.infer else run(cfg)
    print(json.dumps(result, indent=2, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
