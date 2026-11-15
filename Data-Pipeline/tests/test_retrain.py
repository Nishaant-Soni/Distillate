"""Tests for scripts/retrain.py (the retrain_dag skeleton): no Airflow, network or GCP."""

import json

import pytest

from scripts import retrain


@pytest.fixture
def cfg(tmp_path):
    from scripts.utils import load_config

    cfg = load_config()
    cfg["retrain"]["reports_dir"] = str(tmp_path / "retrain")
    cfg["drift_ingest"]["state_file"] = str(tmp_path / "drift" / "state.json")
    return cfg


def test_steps_are_the_plans_five_in_order():
    assert [name for name, _ in retrain.STEPS] == [
        "pull_data", "submit_training", "gate", "register", "notify"]
    for _, func in retrain.STEPS:
        assert callable(getattr(retrain, func))


def test_a_full_run_submits_nothing_registers_nothing_and_writes_its_record(cfg, tmp_path):
    run = retrain.run_all(cfg, {"reason": "test", "source": "alert"})
    steps = run["steps"]
    assert list(steps) == [name for name, _ in retrain.STEPS]
    assert steps["submit_training"]["submitted"] is False
    assert steps["gate"]["result"] == "not_run"
    assert steps["register"] == {"mode": "stub", "registered": False, "reason": "gate not_run"}
    record = json.loads((tmp_path / "retrain" / f"{run['run_id']}.json").read_text())
    assert record["trigger"] == {"reason": "test", "source": "alert"}
    assert record["steps"]["gate"]["result"] == "not_run"


def test_the_job_spec_matches_model_ci(cfg):
    run = retrain.new_run(run_id="retrain-x")
    run["steps"]["pull_data"] = {"data_rev": "abc1234"}
    spec = retrain.job_spec(cfg, run)
    assert spec["service_account"].startswith("vertex-trainer@")
    assert spec["image"].endswith("/distillate-images/trainer:abc1234")
    # vertex-trainer can write only ci/ in the bucket.
    assert spec["out_dir"] == "/gcs/distillate-mlops-data/ci/retrain/retrain-x"
    script = spec["command"][-1]
    assert script.startswith("python -m src.pipeline.fetch_data --out /tmp/processed")
    assert "python -m src.pipeline.run --config configs/student.yaml" in script
    assert "paths.out_dir=/gcs/distillate-mlops-data/ci/retrain/retrain-x" in script


def test_pull_data_counts_the_delivered_drift_batches(cfg, tmp_path):
    assert retrain.step_pull_data(cfg, retrain.new_run())["drift_batches_delivered"] == 0
    (tmp_path / "drift").mkdir()
    (tmp_path / "drift" / "state.json").write_text(json.dumps({"next_batch": 3}))
    assert retrain.step_pull_data(cfg, retrain.new_run())["drift_batches_delivered"] == 3


def test_register_only_after_a_passed_gate(cfg):
    run = retrain.new_run()
    run["steps"]["gate"] = {"result": "blocked"}
    assert retrain.step_register(cfg, run)["reason"] == "gate blocked"
    run["steps"]["gate"] = {"result": "passed"}
    assert "register_model" in retrain.step_register(cfg, run)["note"]


def test_notify_never_needs_slack(cfg, monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    run = retrain.run_all(cfg)
    assert run["steps"]["notify"]["slack_posted"] is False
    assert run["trigger"] == {"reason": "manual", "source": "manual"}


def test_retrain_dag_wiring():
    pytest.importorskip("airflow", reason="Airflow is only installed in the Docker image")
    try:
        from airflow.dag_processing.dagbag import DagBag
    except ImportError:  # older Airflow
        from airflow.models.dagbag import DagBag

    bag = DagBag(dag_folder="dags")
    assert not bag.import_errors, bag.import_errors
    dag = bag.dags["retrain_dag"]
    assert dag.schedule is None and dag.catchup is False
    order = [name for name, _ in retrain.STEPS]
    wired = {(u, d) for u in dag.task_ids for d in dag.get_task(u).downstream_task_ids}
    assert wired == set(zip(order, order[1:], strict=False))
