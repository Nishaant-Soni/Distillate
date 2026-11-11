"""Tests for src/pipeline/register_model.py and rollback_model.py.

Each test gets its own SQLite MLflow store and artifact folder under tmp_path, so no server,
network or GCP access is needed. The Artifact Registry upload is a fake ``runner``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("mlflow")

from mlflow import MlflowClient  # noqa: E402

from src.pipeline import register_model as reg  # noqa: E402
from src.pipeline import registry  # noqa: E402
from src.pipeline import rollback_model as rb  # noqa: E402

NAME = "distillate-student"


@pytest.fixture
def cfg():
    return registry.load_config()


@pytest.fixture
def store(tmp_path, monkeypatch):
    """(client, artifact root) on a fresh SQLite store; no Slack."""
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = registry.client(uri)
    artifacts = (tmp_path / "artifacts").as_uri()
    client.create_experiment("distillate-student", artifact_location=artifacts)
    client.create_experiment(reg.REGISTRY_EXPERIMENT, artifact_location=artifacts)
    return client


def training_run(client: MlflowClient) -> str:
    experiment = client.get_experiment_by_name("distillate-student")
    run = client.create_run(experiment.experiment_id, run_name="train")
    client.set_terminated(run.info.run_id)
    return run.info.run_id


def make_package(root: Path, model_version: str, run_id: str | None,
                 weights: bool = True) -> Path:
    package = root / model_version / "model"
    package.mkdir(parents=True)
    (package / "config.json").write_text("{}")
    (package / "tokenizer.json").write_text("{}")
    (package / "thresholds.json").write_text(json.dumps({"toxicity": 0.4}))
    if weights:
        (package / "model.safetensors").write_bytes(b"\x00" * 16)
    meta = {"model_version": model_version, "model_id": "distilbert-base-uncased",
            "alpha": 0.5, "temperature": 2.0, "training_data_sha256": "abc",
            "code_commit": "deadbeef", "mlflow_run_id": run_id}
    (package / "meta.json").write_text(json.dumps(meta))
    return package


def make_report(root: Path, model_version: str, passed: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    report = {"model_version": model_version, "passed": passed,
              "failed_checks": [] if passed else ["validation: mean AUC"],
              "metrics": {"gate": {"mean_auc": 0.9876}}}
    path = root / "gate_report.json"
    path.write_text(json.dumps(report))
    (root / "gate_report.md").write_text("# Gate report\n")
    return path


class FakeRunner:
    """Stands in for subprocess.run; records each gcloud command."""

    def __init__(self, returncode: int = 0):
        self.returncode, self.calls = returncode, []

    def __call__(self, command, **_):
        self.calls.append(command)
        return subprocess.CompletedProcess(command, self.returncode, "", "permission denied")


def register(cfg, store, tmp_path, version="student-a", passed=True, runner=None, **kwargs):
    package = make_package(tmp_path / "runs", version, training_run(store))
    report = make_report(tmp_path / "gate" / version, version, passed)
    return reg.register(cfg, package, report, store, runner=runner or FakeRunner(), **kwargs)


# ---------------------------------------------------------------------------
# register_model
# ---------------------------------------------------------------------------

def test_register_sets_staging_and_uploads(cfg, store, tmp_path):
    runner = FakeRunner()
    summary = register(cfg, store, tmp_path, runner=runner)

    assert summary["version"] == "1" and summary["alias"] == "staging"
    mv = store.get_model_version_by_alias(NAME, "staging")
    assert str(mv.version) == "1"
    assert mv.tags["gate_passed"] == "true"
    assert mv.tags["model_version"] == "student-a"
    assert mv.tags["gate_mean_auc"] == "0.987600"
    assert mv.tags["ar_upload"] == "ok"
    assert mv.tags["ar_package"].endswith("/packages/distillate-student/versions/1")

    (command,) = runner.calls
    assert command[:4] == ["gcloud", "artifacts", "generic", "upload"]
    assert "--version=1" in command and "--repository=distillate-models" in command
    assert "--location=us-east1" in command

    logged = {a.path for a in store.list_artifacts(summary["run_id"], "package")}
    assert {"package/meta.json", "package/model.safetensors",
            "package/thresholds.json"} <= logged
    gate = {a.path for a in store.list_artifacts(summary["run_id"], "gate")}
    assert gate == {"gate/gate_report.json", "gate/gate_report.md"}


def test_second_candidate_moves_staging_to_new_version(cfg, store, tmp_path):
    register(cfg, store, tmp_path, "student-a")
    summary = register(cfg, store, tmp_path, "student-b")
    assert summary["version"] == "2"
    assert str(store.get_model_version_by_alias(NAME, "staging").version) == "2"


def test_blocked_gate_is_never_registered(cfg, store, tmp_path):
    with pytest.raises(reg.RegistrationError, match="gate did not pass"):
        register(cfg, store, tmp_path, passed=False)
    assert store.search_registered_models() == []


def test_report_for_another_model_is_refused(cfg, store, tmp_path):
    package = make_package(tmp_path / "runs", "student-a", training_run(store))
    report = make_report(tmp_path / "gate", "student-OTHER")
    with pytest.raises(reg.RegistrationError, match="student-OTHER"):
        reg.register(cfg, package, report, store, runner=FakeRunner())


def test_missing_gate_report_is_refused(cfg, store, tmp_path):
    package = make_package(tmp_path / "runs", "student-a", training_run(store))
    with pytest.raises(reg.RegistrationError, match="run src.eval.gate first"):
        reg.register(cfg, package, tmp_path / "nope.json", store, runner=FakeRunner())


def test_incomplete_package_is_refused(cfg, store, tmp_path):
    package = make_package(tmp_path / "runs", "student-a", None, weights=False)
    report = make_report(tmp_path / "gate", "student-a")
    with pytest.raises(reg.RegistrationError, match="model \\(one of"):
        reg.register(cfg, package, report, store, runner=FakeRunner())


def test_onnx_package_from_src_student_package_is_accepted(cfg, store, tmp_path):
    """The section 2.5 layout: model.onnx, tokenizer/ folder, thresholds.json, meta.json."""
    package = tmp_path / "packages" / "student-onnx"
    (package / "tokenizer").mkdir(parents=True)
    (package / "tokenizer" / "tokenizer.json").write_text("{}")
    (package / "model.onnx").write_bytes(b"\x00" * 16)
    (package / "thresholds.json").write_text(json.dumps({"toxicity": 0.4}))
    (package / "meta.json").write_text(json.dumps(
        {"model_version": "student-onnx", "mlflow_run_id": training_run(store)}))
    report = make_report(tmp_path / "gate", "student-onnx")
    summary = reg.register(cfg, package, report, store, runner=FakeRunner())
    logged = {a.path for a in store.list_artifacts(summary["run_id"], "package")}
    assert {"package/model.onnx", "package/tokenizer"} <= logged


def test_package_without_thresholds_is_refused(cfg, store, tmp_path):
    package = make_package(tmp_path / "runs", "student-a", None)
    (package / "thresholds.json").unlink()
    report = make_report(tmp_path / "gate", "student-a")
    with pytest.raises(reg.RegistrationError, match="thresholds.json"):
        reg.register(cfg, package, report, store, runner=FakeRunner())


def test_pytorch_weights_need_config_json(cfg, store, tmp_path):
    package = make_package(tmp_path / "runs", "student-a", None)
    (package / "config.json").unlink()
    report = make_report(tmp_path / "gate", "student-a")
    with pytest.raises(reg.RegistrationError, match="config.json"):
        reg.register(cfg, package, report, store, runner=FakeRunner())


def test_failed_upload_leaves_no_alias(cfg, store, tmp_path):
    with pytest.raises(reg.RegistrationError, match="permission denied"):
        register(cfg, store, tmp_path, runner=FakeRunner(returncode=1))
    mv = store.get_model_version(NAME, "1")
    assert mv.tags["ar_upload"] == "failed"
    assert registry.version_by_alias(store, NAME, "staging") is None


def test_no_upload_registers_without_gcloud(cfg, store, tmp_path):
    runner = FakeRunner()
    summary = register(cfg, store, tmp_path, runner=runner, upload=False)
    assert runner.calls == []
    assert summary["artifact_registry"] is None
    assert store.get_model_version(NAME, "1").tags["ar_upload"] == "skipped"


def test_unknown_training_run_gets_a_registry_run(cfg, store, tmp_path):
    package = make_package(tmp_path / "runs", "student-a", "0" * 32)
    report = make_report(tmp_path / "gate", "student-a")
    summary = reg.register(cfg, package, report, store, runner=FakeRunner())
    run = store.get_run(summary["run_id"])
    experiment = store.get_experiment(run.info.experiment_id)
    assert experiment.name == reg.REGISTRY_EXPERIMENT


def test_deleted_registry_experiment_is_restored(cfg, store, tmp_path):
    store.delete_experiment(store.get_experiment_by_name(reg.REGISTRY_EXPERIMENT).experiment_id)
    package = make_package(tmp_path / "runs", "student-a", None)
    report = make_report(tmp_path / "gate", "student-a")
    summary = reg.register(cfg, package, report, store, runner=FakeRunner())
    assert summary["version"] == "1"
    assert store.get_experiment_by_name(reg.REGISTRY_EXPERIMENT).lifecycle_stage == "active"


def test_main_exits_1_on_a_blocked_gate(cfg, store, tmp_path, monkeypatch):
    package = make_package(tmp_path / "runs", "student-a", training_run(store))
    report = make_report(tmp_path / "gate", "student-a", passed=False)
    monkeypatch.setattr(reg, "client", lambda _uri: store)
    assert reg.main(["--package", str(package), "--gate-report", str(report)]) == 1


# ---------------------------------------------------------------------------
# rollback_model
# ---------------------------------------------------------------------------

@pytest.fixture
def versions(cfg, store, tmp_path):
    """v1-v3 gate-passed, v4 without a gate record; @production on v3."""
    for name in ("student-a", "student-b", "student-c"):
        register(cfg, store, tmp_path, name, upload=False)
    run_id = training_run(store)
    store.create_model_version(NAME, store.get_run(run_id).info.artifact_uri, run_id=run_id,
                               tags={"model_version": "student-d"})
    store.set_registered_model_alias(NAME, "production", "3")
    return store


def test_previous_moves_production_back_one(versions):
    summary = rb.move_alias(versions, NAME, "production", previous=True, reason="canary drop")
    assert (summary["from"], summary["to"], summary["changed"]) == ("3", "2", True)
    assert str(versions.get_model_version_by_alias(NAME, "production").version) == "2"
    move = json.loads(versions.get_model_version(NAME, "2").tags["last_alias_move"])
    assert move["from"] == "3" and move["reason"] == "canary drop"
    assert versions.get_registered_model(NAME).tags["production_previous"] == "3"


def test_previous_skips_versions_without_a_passed_gate(versions):
    versions.set_registered_model_alias(NAME, "production", "4")
    assert rb.move_alias(versions, NAME, "production", previous=True)["to"] == "3"


def test_to_any_earlier_version(versions):
    assert rb.move_alias(versions, NAME, "production", to="1")["to"] == "1"
    assert str(versions.get_model_version_by_alias(NAME, "production").version) == "1"


def test_version_without_gate_needs_force(versions):
    with pytest.raises(rb.RollbackError, match="--force"):
        rb.move_alias(versions, NAME, "production", to="4")
    assert rb.move_alias(versions, NAME, "production", to="4", force=True)["to"] == "4"


def test_unknown_version_is_refused(versions):
    with pytest.raises(rb.RollbackError, match="no version 99"):
        rb.move_alias(versions, NAME, "production", to="99")


def test_previous_needs_a_current_alias_holder(versions):
    with pytest.raises(rb.RollbackError, match="no version holds @canary"):
        rb.move_alias(versions, NAME, "canary", previous=True)


def test_nothing_older_to_roll_back_to(versions):
    versions.set_registered_model_alias(NAME, "production", "1")
    with pytest.raises(rb.RollbackError, match="older than v1"):
        rb.move_alias(versions, NAME, "production", previous=True)


def test_same_version_is_a_no_op(versions):
    summary = rb.move_alias(versions, NAME, "production", to="3")
    assert summary["changed"] is False


def test_exactly_one_target(versions):
    with pytest.raises(rb.RollbackError, match="exactly one"):
        rb.move_alias(versions, NAME, "production", to="1", previous=True)


def test_version_table_lists_aliases_and_gate(versions):
    table = rb.version_table(versions, NAME)
    assert "@production" in table and "@staging" in table
    line_v4 = next(line for line in table.splitlines() if line.startswith("v4"))
    assert "passed" not in line_v4


def test_main_rollback_and_refusal_exit_codes(versions, monkeypatch):
    monkeypatch.setattr(rb, "client", lambda _uri: versions)
    assert rb.main(["--previous"]) == 0
    assert str(versions.get_model_version_by_alias(NAME, "production").version) == "2"
    assert rb.main(["--to", "99"]) == 1


# ---------------------------------------------------------------------------
# notify
# ---------------------------------------------------------------------------

def test_notify_without_webhook_only_logs(cfg, monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    assert registry.notify(cfg, "ok", "title", {"a": 1}) is False


def test_broken_webhook_never_raises(cfg, monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "http://127.0.0.1:9/unreachable")
    assert registry.notify(cfg, "fail", "title", {"a": 1}) is False


def test_client_needs_a_tracking_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        registry.client(None)
