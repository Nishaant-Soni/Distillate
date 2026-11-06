"""Training rows, the train/validation split, and the soft labels joined by id.

Three sources (``data.source``):

* ``temp``: a seeded multilabel iterative stratification of raw train
  (``wiki_train_raw.parquet``, 00_prepare_data's output) into train and validation, the same
  method as ``Data-Pipeline/scripts/split.py`` (Phase 1, before the official splits existed).
* ``ids``: raw train restricted to the pipeline's official split ids
  (``paths.split_ids_dir``/``{train,val}_ids.txt``): the official split without DVC, for
  sweeps and ablations on Kaggle. No ``sample_weight`` (1.0) and the teacher lane's cleaning.
* ``official``: the pipeline's ``train.parquet`` and ``val.parquet`` (CONTRACTS.md §2.2),
  including the bias step's ``sample_weight``.

The split ids are saved with the run. The gate set is never read here; ``check_no_gate``
fails the run if a gate id shows up.

Soft labels come from the store and are checked before use (``soft_labels.arm``):

* ``format: shards`` (the stack, calibrated Qwen): versions, each manifest's
  ``check_field`` against the SHA-256 of the committed ``check_file`` (the stack, or the
  Platt calibration), failure rate, and, when every raw train id is known (``temp`` and
  ``ids``), the teacher validator's full §6 checks.
* ``format: oof`` (TF-IDF out-of-fold scores, one file): the file's SHA-256 against the
  committed manifest, model version, and coverage of the training ids. ``p_<label>`` becomes
  ``logit_<label>``.
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


def read_id_list(path: str | Path) -> list[str]:
    """One id per line (blank lines ignored); duplicates are an error."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: the data pipeline's split ids "
                                "(Data-Pipeline/data/splits/)")
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path.name} has {len(ids) - len(set(ids))} duplicate ids")
    return ids


def id_split(rows, train_ids: Sequence[str], val_ids: Sequence[str],
             max_missing: float = 0.001):
    """``(train, val, missing)``: ``rows`` restricted to each id list, sorted by id.

    Official ids missing from ``rows`` (the two lanes clean text slightly differently, so an
    id can be empty on one side) are dropped and counted; more than ``max_missing`` of
    either list is an error.
    """
    have = set(rows["id"])
    missing = {}
    out = []
    for name, ids in (("train", train_ids), ("val", val_ids)):
        lost = [i for i in ids if i not in have]
        if len(lost) > max_missing * len(ids):
            raise ValueError(f"{len(lost)} of {len(ids)} official {name} ids are not in raw "
                             f"train, e.g. {lost[:3]}")
        missing[name] = len(lost)
        keep = set(ids)
        out.append(rows[rows["id"].isin(keep)].sort_values("id", kind="stable")
                   .reset_index(drop=True))
    return out[0], out[1], missing


def load_splits(cfg: Mapping[str, Any]):
    """``(train, val, info)`` for the configured source, with ``data.limit`` applied to train."""
    data, paths = cfg["data"], cfg["paths"]
    if data["source"] == "temp":
        rows = read_rows(paths["raw_train"])
        train, val = temp_split(rows, float(data["val_fraction"]), int(cfg["seed"]))
        info = {"source": "temp", "source_file": paths["raw_train"],
                "source_sha256": file_sha256(paths["raw_train"]), "source_rows": len(rows)}
    elif data["source"] == "ids":
        rows = read_rows(paths["raw_train"])
        ids_dir = Path(paths["split_ids_dir"])
        train, val, missing = id_split(rows, read_id_list(ids_dir / "train_ids.txt"),
                                       read_id_list(ids_dir / "val_ids.txt"))
        info = {"source": "ids", "source_file": paths["raw_train"],
                "source_sha256": file_sha256(paths["raw_train"]), "source_rows": len(rows),
                "split_ids_sha256": {n: file_sha256(ids_dir / f"{n}_ids.txt")
                                     for n in ("train", "val")},
                "missing_split_ids": missing}
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
    """``(soft, info)``: ``id``, ``logit_<label>``, ``status`` for the configured arm.

    ``source_ids`` (every raw train id) enables the full checks (the teacher validator's for
    shards, full coverage for the out-of-fold file); without it, only versions, hashes, and
    failure rate are checked.
    """
    from src.teacher.storage import get_storage

    sl = cfg["soft_labels"]
    storage = get_storage(cfg, sl.get("local_root"))
    cache = Path(cfg["paths"]["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    if sl.get("format", "shards") == "oof":
        return _load_oof(sl, storage, cache, source_ids)
    return _load_shards(sl, storage, cache, source_ids)


def _info(sl: Mapping[str, Any], check_sha: str, **extra: Any) -> dict[str, Any]:
    arm = sl.get("arm", "stack")
    return {"arm": arm, "prefix": sl["prefix"], **extra,
            "check_field": sl["check_field"], "check_sha256": check_sha,
            "stack_sha256": check_sha if arm == "stack" else None,
            "model_version": sl["model_version"], "prompt_version": sl["prompt_version"],
            "teacher_version": sl["teacher_version"]}


def _load_shards(sl: Mapping[str, Any], storage, cache: Path,
                 source_ids: Sequence[str] | None):
    import pandas as pd

    from src.teacher.validate_local import check_run, load_run

    frames, manifests, schemas, errors = load_run(storage, sl["prefix"], cache)
    if not frames:
        raise FileNotFoundError(f"no soft-label shards under {sl['prefix']}")

    field = sl["check_field"]
    check_sha = file_sha256(sl["check_file"])
    for name, manifest in sorted(manifests.items()):
        if manifest.get(field) != check_sha:
            errors.append(f"{name}: {field} {manifest.get(field)} differs from the committed "
                          f"{Path(sl['check_file']).name} ({check_sha})")
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
    info = _info(sl, check_sha, shards=len(frames), rows=len(soft),
                 failed_rows=int((soft["status"] != "ok").sum()),
                 full_validation=source_ids is not None)
    logger.info("soft labels: %s", json.dumps(info))
    return soft[["id", *LOGIT_COLUMNS, "status"]], info


OOF_CLIP = 1e-6


def _load_oof(sl: Mapping[str, Any], storage, cache: Path, source_ids: Sequence[str] | None):
    """The TF-IDF out-of-fold file as soft labels: ``logit_<label> = log(p / (1 - p))``."""
    import numpy as np
    import pandas as pd

    if not sl.get("check_file"):
        raise ValueError("soft_labels.check_file must name the out-of-fold file's manifest")
    manifest = json.loads(Path(sl["check_file"]).read_text())
    local = storage.download(sl["prefix"], cache)
    check_sha = file_sha256(local)
    errors = []
    if manifest.get(sl["check_field"]) != check_sha:
        errors.append(f"{Path(sl['prefix']).name}: sha256 {check_sha} differs from "
                      f"{Path(sl['check_file']).name} ({manifest.get(sl['check_field'])})")
    frame = pd.read_parquet(local)
    p_columns = [f"p_{label}" for label in LABELS]
    missing = [c for c in ("id", *p_columns) if c not in frame.columns]
    if missing:
        raise ValueError(f"{Path(sl['prefix']).name} is missing columns {missing}")
    frame["id"] = frame["id"].astype(str)
    if frame["id"].duplicated().any():
        errors.append(f"{int(frame['id'].duplicated().sum())} duplicate ids")
    if "model_version" in frame and set(frame["model_version"].astype(str)) != {
            sl["model_version"]}:
        errors.append(f"model_version {sorted(set(frame['model_version'].astype(str)))}, "
                      f"expected {sl['model_version']}")
    if "row_count" in manifest and int(manifest["row_count"]) != len(frame):
        errors.append(f"{len(frame)} rows, manifest says {manifest['row_count']}")
    if source_ids is not None:
        lost = set(source_ids) - set(frame["id"])
        if lost:
            errors.append(f"{len(lost)} raw train ids have no label, e.g. {sorted(lost)[0]}")
    probs = frame[p_columns].to_numpy(dtype="float64")
    ok = np.isfinite(probs).all(axis=1)
    failed = float((~ok).mean())
    if failed > float(sl["max_failure"]):
        errors.append(f"failure rate {failed:.2%} above {float(sl['max_failure']):.0%}")
    if errors:
        raise ValueError("soft labels failed validation:\n  " + "\n  ".join(errors))

    clipped = np.clip(probs, OOF_CLIP, 1 - OOF_CLIP)
    soft = pd.DataFrame({"id": frame["id"]})
    for k, column in enumerate(LOGIT_COLUMNS):
        soft[column] = np.log(clipped[:, k] / (1 - clipped[:, k])).astype("float32")
    soft["status"] = np.where(ok, "ok", "failed")
    info = _info(sl, check_sha, shards=1, rows=len(soft), failed_rows=int((~ok).sum()),
                 full_validation=source_ids is not None)
    logger.info("soft labels: %s", json.dumps(info))
    return soft, info


def attach_soft_labels(rows, soft):
    """``rows`` plus ``logit_<label>`` and ``soft_ok``; every row must have a soft label."""
    merged = rows.merge(soft, on="id", how="left", validate="one_to_one")
    missing = merged["status"].isna()
    if missing.any():
        raise ValueError(f"{int(missing.sum())} training rows have no soft label, e.g. "
                         f"{merged.loc[missing, 'id'].iloc[0]}")
    merged["soft_ok"] = merged.pop("status") == "ok"
    return merged
