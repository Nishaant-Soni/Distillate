"""The student runs' MLflow logger, on the committed Phase 2 records and a local store."""

import pytest

from src.student.config import load_config
from src.student.mlflow_log import RECORDS_DIR, build_records, log_records

pytest.importorskip("mlflow")


@pytest.fixture(scope="module")
def records():
    return build_records()


def local_cfg(root):
    return load_config(None, [f"paths.mlflow_db={root}/mlruns/mlflow.db",
                              f"paths.mlflow_artifacts={root}/mlruns/artifacts"])


def test_one_record_per_arm_and_the_selection(records):
    arms = sorted(p.name for p in (RECORDS_DIR / "runs").iterdir())
    assert len(records) == len(arms) + 1 and records[-1].name == "selection"
    by_arm = {r.name.rsplit("-", 1)[0]: r for r in records[:-1]}
    assert sorted(by_arm) == arms
    for record in records[:-1]:
        assert len(record.tags["kaggle_run_id"]) == 32
        assert {"best_val_mean_auc", "gate_mean_auc", "h2_eval_mean_auc",
                "latency_p95_ms"} <= set(record.metrics)
        assert len(record.steps["val_mean_auc"]) == int(record.params["epochs_run"])
        assert len(record.files()) == 8
    gold, selected = by_arm["distilbert-gold"], by_arm["distilbert-mixed-5ep"]
    assert gold.tags["soft_label_arm"] == "none" and "train_soft" not in gold.steps
    assert selected.params["train.epochs"] == 5
    assert selected.metrics["best_val_mean_auc"] == pytest.approx(0.99368, abs=1e-5)
    assert records[-1].tags["selected_model_version"] == selected.tags["model_version"]


def test_logs_once_then_skips(records, tmp_path, monkeypatch):
    import mlflow

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = local_cfg(tmp_path)
    first = log_records(cfg, records, "test-student")
    assert [r["status"] for r in first] == ["logged"] * len(records)
    run = mlflow.get_run(first[-2]["run_id"])
    assert run.data.tags["kaggle_run_id"] == records[-2].tags["kaggle_run_id"]
    assert run.data.metrics["gate_mean_auc"] == records[-2].metrics["gate_mean_auc"]
    history = mlflow.MlflowClient().get_metric_history(run.info.run_id, "val_mean_auc")
    assert [m.step for m in history] == list(range(len(records[-2].steps["val_mean_auc"])))
    again = log_records(cfg, records, "test-student")
    assert all(r["status"].startswith("skipped") for r in again)
    assert [r["run_id"] for r in again] == [r["run_id"] for r in first]


def test_skips_an_arm_whose_kaggle_run_is_already_there(records, tmp_path, monkeypatch):
    import mlflow

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = local_cfg(tmp_path)
    existing = log_records(cfg, records[:1], "test-student")[0]["run_id"]
    record = records[1]
    record_copy = type(record)(**{**record.__dict__,
                                  "tags": {**record.tags, "kaggle_run_id": existing}})
    result = log_records(cfg, [record_copy], "test-student")[0]
    assert result == {"record": record.name, "status": "skipped (Kaggle run on the server)",
                      "run_id": existing}
    assert len(mlflow.search_runs(output_format="list",
                                  experiment_names=["test-student"])) == 1
