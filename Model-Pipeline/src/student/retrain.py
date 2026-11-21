"""Retraining on the drift stream (docs/retraining.md): the extra training rows and the batch
the candidate is judged on.

Off unless the config's ``retrain`` section asks for it, so CI runs and sweeps train exactly as
before. A retraining run (``retrain_dag``, after drift batch n-1 is released) sets:

* ``drift_batches``: the Civil drift batches 0 .. n-3, added to training. Their gold labels are
  the batch files' own (Civil, positive at a rater fraction of 0.5 or more); when the run uses
  soft labels, each batch's stack shard (``<soft_labels.prefix>drift/shard_<k>``) is checked as
  the training shards are. Their ``sample_weight`` is ``drift_weight``.
* ``obfuscation_augment.rows``: that many toxic Wikipedia train rows, rewritten with the data
  lane's seeded tricks (``Data-Pipeline/scripts/obfuscate.py``: look-alike digits, spaced-out
  letters, Cyrillic homoglyphs), keeping their labels and soft labels; ids ``<id>_aug``. The
  obfuscation evaluation set is built from gate comments, so these never overlap it.
* ``holdout_batch``: batch n-2, never trained on. ``src.pipeline.run`` scores it into
  ``predictions/drift_holdout.parquet``, which ``src.pipeline.promote_check`` compares with
  @production's scores.

Validation, early stopping and thresholds stay on Wikipedia validation, as in CI.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.teacher.config import LABELS, PIPELINE_ROOT, get_logger

logger = get_logger(__name__)

HOLDOUT_STEM = "drift_holdout"
AUG_SUFFIX = "_aug"
DATA_PIPELINE = PIPELINE_ROOT.parent / "Data-Pipeline"


def settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The ``retrain`` section with defaults, checked."""
    r = dict(cfg.get("retrain") or {})
    batches = [int(k) for k in (r.get("drift_batches") or [])]
    holdout = r.get("holdout_batch")
    holdout = None if holdout is None else int(holdout)
    aug = dict(r.get("obfuscation_augment") or {})
    out = {"drift_batches": batches, "holdout_batch": holdout,
           "drift_weight": float(r.get("drift_weight", 1.0)), "drift_dir": r.get("drift_dir"),
           "augment_rows": int(aug.get("rows", 0) or 0), "augment_seed": int(aug.get("seed", 42))}
    if batches and batches != list(range(len(batches))):
        raise ValueError(f"retrain.drift_batches must be 0, 1, ..., m-1 (docs/retraining.md, "
                         f"section 1), got {batches}")
    if holdout is not None and holdout in batches:
        raise ValueError(f"retrain.holdout_batch {holdout} is also a training batch")
    if holdout is not None and holdout != len(batches):
        # Training batches 0..n-3 and the held-out batch n-2: it comes right after them.
        raise ValueError(f"retrain.holdout_batch must be the batch right after the training "
                         f"batches ({len(batches)}, docs/retraining.md section 1), got {holdout}")
    if out["drift_weight"] <= 0:
        raise ValueError("retrain.drift_weight must be positive")
    if out["augment_rows"] < 0:
        raise ValueError("retrain.obfuscation_augment.rows can't be negative")
    return out


def enabled(cfg: Mapping[str, Any]) -> bool:
    s = settings(cfg)
    return bool(s["drift_batches"] or s["augment_rows"] or s["holdout_batch"] is not None)


def drift_dir(cfg: Mapping[str, Any]) -> Path:
    """Where the batch files are: ``retrain.drift_dir``, else ``<pipeline.data_dir>/civil/drift``
    (the data lane's processed folder when ``pipeline.data_dir`` is unset)."""
    from src.data.load_data import DEFAULT_DATA_DIR

    s = settings(cfg)
    if s["drift_dir"]:
        path = Path(s["drift_dir"])
    else:
        base = (cfg.get("pipeline") or {}).get("data_dir")
        base = Path(base) if base else DEFAULT_DATA_DIR
        path = base / "civil" / "drift"
    return path if path.is_absolute() else PIPELINE_ROOT / path


def read_batch(cfg: Mapping[str, Any], k: int):
    """Drift batch ``k``: ``id``, ``text``, the six labels, ``sample_weight``."""
    from src.student.data import read_rows

    path = drift_dir(cfg) / f"batch_{k:02d}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: the retraining job fetches the drift "
                                "batches it trains on and holds out")
    return read_rows(path)


def drift_soft_labels(cfg: Mapping[str, Any], batches: Mapping[str, int]):
    """``id``, ``logit_<label>``, ``status`` for the drift rows, from the stack's drift shards
    (one per batch), with the same checks as the training shards: the manifests' check field
    against the committed stack, versions, ids per batch, failure rate."""
    import pandas as pd

    from src.student.data import LOGIT_COLUMNS, file_sha256
    from src.teacher.storage import get_storage
    from src.teacher.validate_local import check_run, load_run

    sl = cfg["soft_labels"]
    prefix = sl["prefix"].rstrip("/") + "/drift/"
    cache = Path(cfg["paths"]["cache_dir"]) / "drift"
    cache.mkdir(parents=True, exist_ok=True)
    frames, manifests, schemas, errors = load_run(get_storage(cfg, sl.get("local_root")),
                                                  prefix, cache)
    wanted = {f"shard_{k:04d}" for k in set(batches.values())}
    missing = sorted(wanted - set(frames))
    if missing:
        raise FileNotFoundError(f"no drift soft-label shards {missing} under {prefix}")
    frames = {k: v for k, v in frames.items() if k in wanted}
    manifests = {k: v for k, v in manifests.items() if k in wanted}
    schemas = {k: v for k, v in schemas.items() if k in wanted}
    errors = [e for e in errors if e.split(":")[0] in wanted]

    field, check_sha = sl["check_field"], file_sha256(sl["check_file"])
    for name, manifest in sorted(manifests.items()):
        if manifest.get(field) != check_sha:
            errors.append(f"{name}: {field} {manifest.get(field)} differs from the committed "
                          f"{Path(sl['check_file']).name} ({check_sha})")
    errors += check_run(frames, manifests, schemas, sorted(batches), int(sl["shard_size"]),
                        sl["prompt_version"], sl["model_version"],
                        max_failure=float(sl["max_failure"]), batches=batches)
    if errors:
        raise ValueError("drift soft labels failed validation:\n  " + "\n  ".join(errors))
    soft = pd.concat([frames[k] for k in sorted(frames)], ignore_index=True)
    soft["id"] = soft["id"].astype(str)
    return soft[["id", *LOGIT_COLUMNS, "status"]]


def _obfuscate():
    """The data lane's ``obfuscate_text`` (the same tricks and seeding as the evaluation set)."""
    if str(DATA_PIPELINE) not in sys.path:
        sys.path.insert(0, str(DATA_PIPELINE))
    from scripts.obfuscate import obfuscate_text

    return obfuscate_text


def augment(train, rows: int, seed: int):
    """``(new_rows, techniques)``: up to ``rows`` toxic train rows rewritten, labels and soft
    labels kept. Rows that no trick changes are skipped. Seeded, and independent of row order."""
    from collections import Counter

    obfuscate_text = _obfuscate()
    toxic = train[train["toxicity"] == 1].sort_values("id", kind="stable")
    picked = toxic.sample(n=min(rows, len(toxic)), random_state=seed).sort_values("id",
                                                                                    kind="stable")
    texts, techniques = [], []
    for row in picked.itertuples(index=False):
        text, technique = obfuscate_text(row.text, row.id, seed)
        texts.append(text)
        techniques.append(technique)
    out = picked.assign(text=texts, id=picked["id"] + AUG_SUFFIX)
    kept = [t != "none" for t in techniques]
    return out[kept].reset_index(drop=True), dict(Counter(t for t in techniques if t != "none"))


def extend_training(cfg: Mapping[str, Any], train, soft: bool):
    """``(train plus the drift and augmented rows, info)``. ``train`` already has its soft
    labels when ``soft``; the new rows get theirs the same way."""
    import pandas as pd

    from src.student.data import attach_soft_labels

    s = settings(cfg)
    parts = [train]
    info: dict[str, Any] = {k: s[k] for k in ("drift_batches", "holdout_batch", "drift_weight")}
    if s["drift_batches"]:
        batches = []
        for k in s["drift_batches"]:
            batch = read_batch(cfg, k)
            batches.append(batch.assign(batch=k))
        drift = pd.concat(batches, ignore_index=True)
        if drift["id"].duplicated().any():
            raise ValueError("the drift batches share ids")
        drift["sample_weight"] = s["drift_weight"]
        drift["sample_weight"] = drift["sample_weight"].astype("float32")
        if soft:
            drift = attach_soft_labels(drift, drift_soft_labels(
                cfg, dict(zip(drift["id"], drift["batch"], strict=True))))
        parts.append(drift.drop(columns="batch"))
        info["drift_rows"] = len(drift)
        info["drift_prevalence"] = {k: round(float(drift[k].mean()), 5) for k in LABELS}
    if s["augment_rows"]:
        added, techniques = augment(train, s["augment_rows"], s["augment_seed"])
        parts.append(added)
        info["obfuscation_augment"] = {"requested": s["augment_rows"], "rows": len(added),
                                       "seed": s["augment_seed"], "techniques": techniques}
    out = pd.concat(parts, ignore_index=True)[list(train.columns)]
    if out["id"].duplicated().any():
        dup = out.loc[out["id"].duplicated(), "id"].iloc[0]
        raise ValueError(f"training rows share an id after adding the drift and augmented "
                         f"rows, e.g. {dup}")
    out = out.sort_values("id", kind="stable").reset_index(drop=True)
    info["train_rows"] = len(out)
    logger.info("retrain: %s", info)
    return out, info


def holdout_table(cfg: Mapping[str, Any]):
    """The held-out batch, for scoring and its gold file; None when no batch is held out."""
    s = settings(cfg)
    if s["holdout_batch"] is None:
        return None
    return read_batch(cfg, s["holdout_batch"])
