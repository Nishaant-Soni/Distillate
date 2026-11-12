"""Drift ingestion: release the Civil drift stream one batch per run (Phase 2 skeleton).

Each run of drift_ingest_dag handles the next batch of data/processed/civil/drift/:

1. release: read batch_k, check its schema, its size, and that its ids are exactly batch k of
   data/splits/civil_drift_ids.csv;
2. drift: drift statistics for the batch (a stub until Sai's src/monitoring/drift.py is merged);
3. score: send every comment through the serving API (v4 section 2.4) and store the responses
   without the text: id, text_sha256, char_len and scores only;
4. evaluate: attach the gold labels to the previous batch's predictions (the "delayed" labels)
   and export its AUC and F1 per label;
5. overrides: merge the moderation queue's overrides (a stub until the GCS feedback log exists).

Phase 2 runs against MockService (deterministic scores in the section 2.4 format). Phase 3 sets
drift_ingest.service_url to the live endpoint, which HttpService calls.

Run one batch on its own: python -m scripts.drift_ingest
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import f1_score, roc_auc_score

from scripts.columns import LABELS
from scripts.errors import DriftBatchError
from scripts.utils import PIPELINE_ROOT, atomic_write, get_logger, load_config, timed

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["id", "text", "created_date", *LABELS]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PIPELINE_ROOT / p).resolve()


# ---------------------------------------------------------------- state


def load_state(dcfg: dict[str, Any]) -> dict[str, Any]:
    path = resolve(dcfg["state_file"])
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"next_batch": 0, "batches": {}}


def save_state(dcfg: dict[str, Any], state: dict[str, Any]) -> None:
    atomic_write(resolve(dcfg["state_file"]), json.dumps(state, indent=2, default=str))


# ---------------------------------------------------------------- the service (v4 section 2.4)


def build_response(
    scores: dict[str, float], threshold: float, margin: float, model_version: str,
    latency_ms: float,
) -> dict[str, Any]:
    """A /predict response in the section 2.4 format."""
    flagged = {label: scores[label] >= threshold for label in LABELS}
    return {
        "scores": scores,
        "flagged": flagged,
        "flag": any(flagged.values()),
        "low_confidence": any(abs(scores[label] - threshold) < margin for label in LABELS),
        "model_version": model_version,
        "latency_ms": latency_ms,
    }


class MockService:
    """Stands in for the live service until Nov 22: same response format, deterministic scores.

    Scores come from the text's SHA-256, so they carry no signal (AUC near 0.5): the mock
    exercises the plumbing, not the model.
    """

    def __init__(self, threshold: float, margin: float):
        self.threshold, self.margin = threshold, margin

    def predict(self, text: str) -> dict[str, Any]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        scores = {label: round(digest[i] / 255, 4) for i, label in enumerate(LABELS)}
        return build_response(scores, self.threshold, self.margin, "mock-0", 0.0)


class HttpService:
    """The live endpoint (Phase 3): POST {url}/predict with {"text": ...}."""

    def __init__(self, url: str, timeout_s: float):
        self.url, self.timeout_s = url.rstrip("/"), timeout_s

    def predict(self, text: str) -> dict[str, Any]:
        response = requests.post(f"{self.url}/predict", json={"text": text}, timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()


def get_service(dcfg: dict[str, Any]):
    if dcfg.get("service_url"):
        return HttpService(dcfg["service_url"], dcfg.get("request_timeout_s", 10))
    return MockService(dcfg.get("mock_threshold", 0.5), dcfg.get("low_confidence_margin", 0.1))


# ---------------------------------------------------------------- the five steps


def batch_path(dcfg: dict[str, Any], k: int) -> Path:
    return resolve(dcfg["batches_dir"]) / f"batch_{k:02d}.parquet"


def release_batch(dcfg: dict[str, Any], k: int) -> pd.DataFrame:
    """Batch k, checked against the schema, its size and the drift manifest."""
    path = batch_path(dcfg, k)
    if not path.exists():
        raise DriftBatchError(f"{path} not found: run data_prep_dag (subsets) first")
    batch = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in batch.columns]
    if missing:
        raise DriftBatchError(f"batch {k}: missing columns {missing}")
    if len(batch) != dcfg["batch_size"]:
        raise DriftBatchError(f"batch {k}: {len(batch)} rows, expected {dcfg['batch_size']}")
    manifest = pd.read_csv(resolve(dcfg["manifest"]), dtype={"id": str})
    expected = set(manifest.loc[manifest["batch"] == k, "id"])
    if set(batch["id"].astype(str)) != expected:
        raise DriftBatchError(f"batch {k}: ids differ from {dcfg['manifest']}")
    return batch


def compute_drift(batch: pd.DataFrame) -> dict[str, Any]:
    """Basic batch statistics. TODO: call Sai's src/monitoring/drift.py (PSI) once merged."""
    return {
        "status": "stub: src/monitoring/drift.py not merged yet",
        "mean_char_len": float(batch["char_len"].mean()) if "char_len" in batch else None,
        "obfuscation_share": (float(batch["has_obfuscation"].mean())
                              if "has_obfuscation" in batch else None),
        "date_min": str(batch["created_date"].min()),
        "date_max": str(batch["created_date"].max()),
    }


def score_batch(batch: pd.DataFrame, service) -> pd.DataFrame:
    """Every comment through the service. The result holds no text, only its hash."""
    rows = []
    for record in batch[["id", "text"]].itertuples(index=False):
        started = time.perf_counter()
        response = service.predict(record.text)
        row = {
            "id": str(record.id),
            "text_sha256": hashlib.sha256(record.text.encode("utf-8")).hexdigest(),
            "char_len": len(record.text),
            "model_version": response["model_version"],
            "flag": bool(response["flag"]),
            "low_confidence": bool(response["low_confidence"]),
            "latency_ms": response.get("latency_ms") or (time.perf_counter() - started) * 1000,
        }
        for label in LABELS:
            row[f"score_{label}"] = float(response["scores"][label])
            row[f"flagged_{label}"] = bool(response["flagged"][label])
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(predictions: pd.DataFrame, gold: pd.DataFrame, labels: list[str]) -> dict[str, Any]:
    """The delayed gold labels against the stored predictions: AUC and F1 per label."""
    joined = predictions.merge(gold[["id", *labels]].astype({"id": str}), on="id",
                               validate="one_to_one")
    metrics: dict[str, Any] = {"n": len(joined)}
    for label in labels:
        y = joined[label].to_numpy()
        has_both = 0 < y.sum() < len(y)
        metrics[label] = {
            "positives": int(y.sum()),
            "auc": float(roc_auc_score(y, joined[f"score_{label}"])) if has_both else None,
            "f1": float(f1_score(y, joined[f"flagged_{label}"], zero_division=0)),
        }
    return metrics


def merge_overrides() -> dict[str, Any]:
    """TODO (Phase 3): merge the moderation queue's overrides from the GCS feedback log."""
    return {"status": "stub: feedback log not available yet", "overrides": 0}


# ---------------------------------------------------------------- the DAG's steps
# Each step reads and writes the state file, so drift_ingest_dag can run them as separate
# tasks (and retry or inspect one on its own); run() calls them in order for a local run.


def _reports(dcfg: dict[str, Any]) -> Path:
    return resolve(dcfg["reports_dir"])


def _pred_path(dcfg: dict[str, Any], k: int) -> Path:
    return _reports(dcfg) / "predictions" / f"batch_{k:02d}.parquet"


def _work_path(dcfg: dict[str, Any], k: int) -> Path:
    return _reports(dcfg) / "work" / f"batch_{k:02d}.json"


def _load_work(dcfg: dict[str, Any], k: int) -> dict[str, Any]:
    path = _work_path(dcfg, k)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"batch": k}


def _save_work(dcfg: dict[str, Any], k: int, work: dict[str, Any]) -> None:
    atomic_write(_work_path(dcfg, k), json.dumps(work, indent=2, default=str))


def _current(dcfg: dict[str, Any]) -> tuple[dict[str, Any], int]:
    state = load_state(dcfg)
    if state.get("current") is None:
        raise DriftBatchError("no drift batch in progress: run the release step first")
    return state, int(state["current"])


def step_release(cfg: dict[str, Any]) -> bool:
    """Release the next batch after checking it. False when the stream is finished."""
    dcfg = cfg["drift_ingest"]
    state = load_state(dcfg)
    k = state["next_batch"]
    if k >= dcfg["n_batches"]:
        logger.info("All %d drift batches already released; nothing to do", dcfg["n_batches"])
        return False
    batch = release_batch(dcfg, k)
    state["current"] = k
    save_state(dcfg, state)
    _save_work(dcfg, k, {"batch": k, "rows": len(batch)})
    logger.info("Released drift batch %d (%d rows)", k, len(batch))
    return True


def step_drift(cfg: dict[str, Any]) -> dict[str, Any]:
    dcfg = cfg["drift_ingest"]
    _, k = _current(dcfg)
    work = _load_work(dcfg, k)
    work["drift"] = compute_drift(release_batch(dcfg, k))
    _save_work(dcfg, k, work)
    return work["drift"]


def step_score(cfg: dict[str, Any]) -> dict[str, Any]:
    dcfg = cfg["drift_ingest"]
    _, k = _current(dcfg)
    predictions = score_batch(release_batch(dcfg, k), get_service(dcfg))
    path = _pred_path(dcfg, k)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(path, index=False)
    work = _load_work(dcfg, k)
    work.update({
        "model_version": predictions["model_version"].iloc[0],
        "flag_rate": float(predictions["flag"].mean()),
        "mean_latency_ms": float(np.mean(predictions["latency_ms"])),
    })
    _save_work(dcfg, k, work)
    return {"batch": k, "flag_rate": work["flag_rate"]}


def step_evaluate(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """The previous batch's predictions against its (now delivered) gold labels."""
    dcfg = cfg["drift_ingest"]
    _, k = _current(dcfg)
    previous = None
    if k > 0 and _pred_path(dcfg, k - 1).exists():
        previous = evaluate(pd.read_parquet(_pred_path(dcfg, k - 1)),
                            release_batch(dcfg, k - 1), list(dcfg["eval_labels"]))
    work = _load_work(dcfg, k)
    work["previous_batch_metrics"] = previous
    _save_work(dcfg, k, work)
    return previous


def step_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    dcfg = cfg["drift_ingest"]
    _, k = _current(dcfg)
    work = _load_work(dcfg, k)
    work["overrides"] = merge_overrides()
    _save_work(dcfg, k, work)
    return work["overrides"]


def step_finish(cfg: dict[str, Any]) -> dict[str, Any]:
    """Write the batch record and move the stream on to the next batch."""
    dcfg = cfg["drift_ingest"]
    state, k = _current(dcfg)
    work = _load_work(dcfg, k)
    atomic_write(_reports(dcfg) / f"batch_{k:02d}.json", json.dumps(work, indent=2, default=str))
    state["batches"][str(k)] = work
    state["next_batch"] = k + 1
    state.pop("current", None)
    save_state(dcfg, state)
    logger.info("Drift batch %d done: flag rate %.3f", k, work.get("flag_rate", float("nan")))
    return {"batch": k, "flag_rate": work.get("flag_rate")}


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Handle the next batch, all steps in order. Returns the run summary."""
    summary: dict[str, Any] = {"task": "drift_ingest", "dropped": {}}
    with timed(summary):
        if not step_release(cfg):
            summary["skipped"] = True
            return summary
        step_drift(cfg)
        step_score(cfg)
        step_evaluate(cfg)
        step_overrides(cfg)
        summary.update({"skipped": False, **step_finish(cfg)})
    return summary



def main() -> None:
    print(json.dumps(run(load_config()), indent=2, default=str))


if __name__ == "__main__":
    main()


