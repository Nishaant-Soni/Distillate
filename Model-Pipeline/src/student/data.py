"""Training rows, the train/validation split, and the soft labels joined by id.

Two sources (``data.source``):

* ``temp``: until the data pipeline's splits merge, a seeded multilabel iterative
  stratification of raw train (``wiki_train_raw.parquet``, 00_prepare_data's output) into
  train and validation, the same method as ``Data-Pipeline/scripts/split.py``. The split ids
  are saved with the run. Phase 2 reruns on the official splits for reported numbers.
* ``official``: the pipeline's ``train.parquet`` and ``val.parquet`` (CONTRACTS.md §2.2),
  including the bias step's ``sample_weight``.

The gate set is never read here; ``check_no_gate`` fails the run if a gate id shows up.

Soft labels come from the shard store (``soft_labels.prefix``) and are checked before use:
versions, ``stack_sha256`` against the committed stack file, failure rate, and, for
``temp`` (which has every raw train id), the teacher validator's full §6 checks.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.teacher.config import LABELS, get_logger

logger = get_logger(__name__)

LOGIT_COLUMNS = [f"logit_{label}" for label in LABELS]


def read_rows(path: str | Path):
    """``id``, ``text``, six int8 labels, and ``sample_weight`` (1.0 if the file has none).

    Rows with empty text are dropped: the data pipeline drops them too, and there is
    nothing for the student to read.
    """
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; see docs/student.md for how to get it")
    df = pd.read_parquet(path)
    missing = [c for c in ("id", "text", *LABELS) if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}")
    if df["id"].duplicated().any():
        raise ValueError(f"{path.name} has {int(df['id'].duplicated().sum())} duplicate ids")
    for label in LABELS:
        if not df[label].isin((0, 1)).all():
            raise ValueError(f"{path.name}: {label} has values other than 0 and 1")
    keep = ["id", "text", *LABELS] + (["sample_weight"] if "sample_weight" in df else [])
    df = df[keep].copy()
    df["id"] = df["id"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df[list(LABELS)] = df[list(LABELS)].astype("int8")
    if "sample_weight" not in df:
        df["sample_weight"] = 1.0
    df["sample_weight"] = df["sample_weight"].astype("float32")
    empty = df["text"].str.strip().str.len() == 0
    if empty.any():
        logger.info("%s: dropped %d rows with empty text", path.name, int(empty.sum()))
    return df[~empty].sort_values("id", kind="stable").reset_index(drop=True)


def temp_split(df, val_fraction: float, seed: int):
    """``(train, val)``: seeded multilabel iterative stratification on rows sorted by id."""
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    df = df.sort_values("id", kind="stable").reset_index(drop=True)
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_fraction,
                                                random_state=seed)
    train_idx, val_idx = next(splitter.split(df[["id"]], df[list(LABELS)].to_numpy()))
    return (df.iloc[sorted(train_idx)].reset_index(drop=True),
            df.iloc[sorted(val_idx)].reset_index(drop=True))


def load_splits(cfg: Mapping[str, Any]):
    """``(train, val, info)`` for the configured source, with ``data.limit`` applied to train."""
    data, paths = cfg["data"], cfg["paths"]
    if data["source"] == "temp":
        rows = read_rows(paths["raw_train"])
        train, val = temp_split(rows, float(data["val_fraction"]), int(cfg["seed"]))
        info = {"source": "temp", "source_file": paths["raw_train"],
                "source_sha256": file_sha256(paths["raw_train"]), "source_rows": len(rows)}
    else:
        base = Path(paths["official_dir"])
        train, val = read_rows(base / "train.parquet"), read_rows(base / "val.parquet")
        info = {"source": "official", "source_file": str(base),
                "source_sha256": file_sha256(base / "train.parquet")}
    if set(train["id"]) & set(val["id"]):
        raise ValueError("train and validation share ids")
    if data.get("limit"):
        train = train.sample(n=min(int(data["limit"]), len(train)), random_state=cfg["seed"])
        train = train.sort_values("id", kind="stable").reset_index(drop=True)
    check_no_gate((train, val), paths.get("gate_ids"))
    info.update({"train_rows": len(train), "val_rows": len(val),
                 "train_prevalence": {k: round(float(train[k].mean()), 5) for k in LABELS},
                 "val_prevalence": {k: round(float(val[k].mean()), 5) for k in LABELS}})
    return train, val, info


def check_no_gate(frames: Sequence[Any], gate_ids_path: str | Path | None) -> None:
    """Fail if any row is in the gate subset: the gate set is never trained or tuned on."""
    import pandas as pd

    if not gate_ids_path or not Path(gate_ids_path).exists():
        return
    gate = set(pd.read_parquet(gate_ids_path, columns=["id"])["id"].astype(str))
    for frame in frames:
        leaked = gate & set(frame["id"])
        if leaked:
            raise ValueError(f"{len(leaked)} gate ids in training data, e.g. {sorted(leaked)[0]}")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_soft_labels(cfg: Mapping[str, Any], source_ids: Sequence[str] | None = None):
    """``(soft, info)``: ``id``, ``logit_<label>``, ``status`` for every shard under the prefix.

    ``source_ids`` (every raw train id) enables the teacher validator's full checks; without
    it, only versions, stack hash, and failure rate are checked.
    """
    import pandas as pd

    from src.teacher.storage import get_storage
    from src.teacher.validate_local import check_run, load_run

    sl = cfg["soft_labels"]
    storage = get_storage(cfg, sl.get("local_root"))
    cache = Path(cfg["paths"]["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    frames, manifests, schemas, errors = load_run(storage, sl["prefix"], cache)
    if not frames:
        raise FileNotFoundError(f"no soft-label shards under {sl['prefix']}")

    stack_sha = file_sha256(cfg["paths"]["stack_file"])
    for name, manifest in sorted(manifests.items()):
        if manifest.get("stack_sha256") != stack_sha:
            errors.append(f"{name}: stack_sha256 {manifest.get('stack_sha256')} differs from "
                          f"the committed stack file ({stack_sha})")
    if source_ids is not None:
        errors += check_run(frames, manifests, schemas, sorted(source_ids),
                            int(sl["shard_size"]), sl["prompt_version"], sl["model_version"],
                            max_failure=float(sl["max_failure"]))
    soft = pd.concat([frames[k] for k in sorted(frames)], ignore_index=True)
    if source_ids is None:
        for key in ("prompt_version", "model_version"):
            values = set(soft[key].astype(str))
            if values != {sl[key]}:
                errors.append(f"run: {key} {sorted(values)}, expected {sl[key]}")
        failed = float((soft["status"] != "ok").mean())
        if failed > float(sl["max_failure"]):
            errors.append(f"run: failure rate {failed:.2%} above {float(sl['max_failure']):.0%}")
    if errors:
        raise ValueError("soft labels failed validation:\n  " + "\n  ".join(errors))

    soft["id"] = soft["id"].astype(str)
    info = {"prefix": sl["prefix"], "shards": len(frames), "rows": len(soft),
            "failed_rows": int((soft["status"] != "ok").sum()), "stack_sha256": stack_sha,
            "model_version": sl["model_version"], "prompt_version": sl["prompt_version"],
            "full_validation": source_ids is not None}
    logger.info("soft labels: %s", json.dumps(info))
    return soft[["id", *LOGIT_COLUMNS, "status"]], info


def attach_soft_labels(rows, soft):
    """``rows`` plus ``logit_<label>`` and ``soft_ok``; every row must have a soft label."""
    merged = rows.merge(soft, on="id", how="left", validate="one_to_one")
    missing = merged["status"].isna()
    if missing.any():
        raise ValueError(f"{int(missing.sum())} training rows have no soft label, e.g. "
                         f"{merged.loc[missing, 'id'].iloc[0]}")
    merged["soft_ok"] = merged.pop("status") == "ok"
    return merged
