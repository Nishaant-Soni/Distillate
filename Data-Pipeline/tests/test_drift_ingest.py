"""Tests for scripts/drift_ingest.py: tiny made-up batches, the mock service, no network."""

import json

import pandas as pd
import pytest

from scripts import drift
from scripts import drift_ingest as di
from scripts.columns import LABELS
from scripts.errors import DriftBatchError


def make_batch(k, n=4):
    frame = pd.DataFrame({
        "id": [f"b{k}_{i}" for i in range(n)],
        "text": [f"made-up comment {k}-{i}" for i in range(n)],
        "created_date": pd.date_range("2016-01-01", periods=n, tz="UTC"),
        "char_len": [20] * n,
        "has_obfuscation": [False] * n,
    })
    for label in LABELS:
        frame[label] = [i % 2 for i in range(n)]  # both classes in every batch
    return frame


def setup(tmp_path, n_batches=2, n=4):
    train = make_batch(99, n=20)  # a tiny "training set" for the drift baseline
    train.to_parquet(tmp_path / "train.parquet", index=False)
    batches = tmp_path / "drift"
    batches.mkdir()
    manifest_rows = []
    for k in range(n_batches):
        batch = make_batch(k, n)
        batch.to_parquet(batches / f"batch_{k:02d}.parquet", index=False)
        manifest_rows += [{"id": i, "batch": k} for i in batch["id"]]
    pd.DataFrame(manifest_rows).to_csv(tmp_path / "civil_drift_ids.csv", index=False)
    return {
        "paths": {"processed_dir": str(tmp_path)},
        "drift": {
            "baseline_source": str(tmp_path / "train.parquet"),
            "baseline_file": str(tmp_path / "monitoring" / "drift_baseline.json"),
            "psi_bins": 10, "vocab_min_count": 1, "psi_warn": 0.1,
            "psi_alert": 0.25, "psi_epsilon": 0.0001,
        },
        "drift_ingest": {
            "batches_dir": str(batches), "manifest": str(tmp_path / "civil_drift_ids.csv"),
            "n_batches": n_batches, "batch_size": n,
            "state_file": str(tmp_path / "reports" / "state.json"),
            "reports_dir": str(tmp_path / "reports"), "service_url": None,
            "mock_threshold": 0.5, "low_confidence_margin": 0.1,
            "eval_labels": ["toxicity", "obscene", "threat", "insult", "identity_attack"],
            "reference_flag_rate": None, "build_baseline_if_missing": True,
        },
    }


def test_runs_one_batch_per_run_and_evaluates_the_previous_one(tmp_path):
    cfg = setup(tmp_path)
    first = di.run(cfg)
    assert first["batch"] == 0
    state = json.loads((tmp_path / "reports" / "state.json").read_text())
    assert state["next_batch"] == 1 and state["batches"]["0"]["previous_batch_metrics"] is None

    second = di.run(cfg)
    assert second["batch"] == 1
    record = json.loads((tmp_path / "reports" / "batch_01.json").read_text())
    metrics = record["previous_batch_metrics"]
    assert metrics["n"] == 4 and metrics["toxicity"]["positives"] == 2

    assert di.run(cfg)["skipped"] is True  # all batches released


def test_predictions_never_store_text(tmp_path):
    cfg = setup(tmp_path)
    di.run(cfg)
    predictions = pd.read_parquet(tmp_path / "reports" / "predictions" / "batch_00.parquet")
    assert "text" not in predictions.columns
    assert predictions["text_sha256"].str.len().eq(64).all()


def test_mock_service_follows_the_api_format_and_is_deterministic():
    service = di.MockService(threshold=0.5, margin=0.1)
    response = service.predict("hello")
    assert set(response) == {"scores", "flagged", "flag", "low_confidence", "model_version",
                             "latency_ms"}
    assert set(response["scores"]) == set(LABELS)
    assert response["flag"] == any(response["flagged"].values())
    assert service.predict("hello") == response


def test_build_response_flags_and_low_confidence():
    scores = {label: 0.1 for label in LABELS} | {"threat": 0.55}
    response = di.build_response(scores, threshold=0.5, margin=0.1, model_version="m",
                                 latency_ms=1.0)
    assert response["flagged"]["threat"] and response["flag"]
    assert response["low_confidence"]  # 0.55 is within 0.1 of 0.5


def test_batch_with_wrong_ids_is_rejected(tmp_path):
    cfg = setup(tmp_path)
    manifest = pd.read_csv(tmp_path / "civil_drift_ids.csv")
    manifest.loc[0, "id"] = "someone_else"
    manifest.to_csv(tmp_path / "civil_drift_ids.csv", index=False)
    with pytest.raises(DriftBatchError, match="ids differ"):
        di.run(cfg)


def test_batch_with_missing_columns_is_rejected(tmp_path):
    cfg = setup(tmp_path)
    path = tmp_path / "drift" / "batch_00.parquet"
    pd.read_parquet(path).drop(columns=["created_date"]).to_parquet(path, index=False)
    with pytest.raises(DriftBatchError, match="missing columns"):
        di.run(cfg)


def test_missing_batch_file_names_it(tmp_path):
    cfg = setup(tmp_path)
    (tmp_path / "drift" / "batch_00.parquet").unlink()
    with pytest.raises(DriftBatchError, match="batch_00.parquet"):
        di.run(cfg)



def test_finished_stream_evaluates_the_last_batch_once(tmp_path):
    cfg = setup(tmp_path)
    di.run(cfg)
    di.run(cfg)
    assert di.run(cfg)["skipped"] is True  # the stream is finished...
    state = json.loads((tmp_path / "reports" / "state.json").read_text())
    final = state["final_batch_metrics"]
    assert final["batch"] == 1 and final["toxicity"]["positives"] == 2  # but batch 1 is evaluated
    assert (tmp_path / "reports" / "batch_01_final_metrics.json").exists()
    assert di.run(cfg)["skipped"] is True  # and later runs don't redo it


def test_drift_step_records_input_psi(tmp_path):
    cfg = setup(tmp_path)
    di.run(cfg)
    record = json.loads((tmp_path / "reports" / "batch_00.json").read_text())
    assert set(record["drift"]["features"]) == set(drift.INPUT_FEATURES)
    assert record["drift"]["status"] in {"ok", "warn", "alert"}


def test_flag_rate_drift_is_unknown_without_a_reference(tmp_path):
    cfg = setup(tmp_path)
    di.run(cfg)
    record = json.loads((tmp_path / "reports" / "batch_00.json").read_text())
    assert record["flag_rate_drift"]["status"] == "unknown"


def test_missing_baseline_fails_clearly(tmp_path):
    cfg = setup(tmp_path)
    cfg["drift_ingest"]["build_baseline_if_missing"] = False
    with pytest.raises(FileNotFoundError, match="drift baseline"):
        di.run(cfg)


def test_zero_latency_is_kept():
    predictions = di.score_batch(make_batch(0), di.MockService(threshold=0.5, margin=0.1))
    assert (predictions["latency_ms"] == 0.0).all()  # the mock reports 0.0
