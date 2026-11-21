"""Tests for scripts/retrain.py (retrain_dag): no Airflow, network, gcloud or MLflow.

The gcloud and Model-Pipeline subprocesses are replaced by a fake ``_run`` that answers from a
script and writes the files the real commands would.
"""

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts import retrain
from scripts.columns import LABELS

DIGEST = "sha256:" + "a" * 64


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    from scripts.utils import load_config

    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    cfg = load_config()
    cfg["retrain"]["reports_dir"] = str(tmp_path / "retrain")
    cfg["retrain"]["records_uri"] = None
    cfg["drift_ingest"]["state_file"] = str(tmp_path / "drift" / "state.json")
    cfg["drift_ingest"]["reports_dir"] = str(tmp_path / "drift")
    return cfg


def set_next_batch(cfg, n):
    path = Path(cfg["drift_ingest"]["state_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"next_batch": n, "batches": {}}))


def done(stdout="", code=0, stderr=""):
    return subprocess.CompletedProcess([], code, stdout, stderr)


class FakeRun:
    """Stands in for retrain._run: records each command and plays the job, gate, promote_check
    and register_model, writing their output files."""

    def __init__(self, tmp_path, gate_passed=True, promote_passed=True, production=True,
                 holdout=5):
        self.calls = []
        self.tmp = tmp_path
        self.gate_passed, self.promote_passed = gate_passed, promote_passed
        self.production, self.holdout = production, holdout
        self.job_config = None

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        text = " ".join(command)
        if "images list" in text:
            return done(DIGEST + "\n")
        if "custom-jobs create" in text:
            config = next(c for c in command if c.startswith("--config=")).split("=", 1)[1]
            self.job_config = json.loads(Path(config).read_text())
            return done("projects/1/locations/us-east1/customJobs/42\n")
        if "custom-jobs describe" in text:
            return done("JOB_STATE_SUCCEEDED\n")
        if command[:3] == ["gcloud", "storage", "ls"]:
            return done(f"{command[3]}student-x/\n")
        if command[:4] == ["gcloud", "storage", "cp", "-r"] and command[4].endswith("/*"):
            dest = Path(command[5])
            (dest / "package").mkdir(parents=True, exist_ok=True)
            (dest / "package" / "meta.json").write_text(json.dumps(
                {"model_version": "cand-1",
                 "retrain": {"drift_batches": list(range(self.holdout)),
                             "holdout_batch": self.holdout}}))
            return done()
        if command[:2] == ["gcloud", "storage"]:
            return done()
        module = command[2] if len(command) > 2 else ""
        if module == "src.pipeline.ci" and command[3] == "fetch-production":
            found = {"version": "1", "model_version": "prod-1", "run_id": "r"}
            return done((json.dumps(found) if self.production else "none") + "\n")
        if module == "src.pipeline.ci":
            return done()
        if module == "src.eval.gate":
            out = Path(command[command.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "gate_report.json").write_text(json.dumps(
                {"passed": self.gate_passed, "model_version": "cand-1",
                 "failed_checks": [] if self.gate_passed else ["regression"]}))
            return done(code=0 if self.gate_passed else 1)
        if module == "src.pipeline.promote_check":
            out = Path(command[command.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            trained = command[command.index("--trained-on") + 1:command.index("--out")]
            (out / "promote_check.json").write_text(json.dumps(
                {"passed": self.promote_passed, "gain": 0.012 if self.promote_passed else -0.01,
                 "trained_on": [int(b) for b in trained]}))
            return done(code=0 if self.promote_passed else 1)
        if module == "src.pipeline.register_model":
            Path(command[command.index("--out") + 1]).write_text(json.dumps({"version": 3}))
            return done()
        raise AssertionError(f"unexpected command {command}")

    def ran(self, needle):
        return [c for c in self.calls if needle in " ".join(c)]


def stored_scores(cfg, k, model_version):
    path = Path(cfg["drift_ingest"]["reports_dir"]) / "predictions" / f"batch_{k:02d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"id": ["a", "b"], "model_version": [model_version] * 2,
                          **{f"score_{label}": [0.1, 0.9] for label in LABELS}})
    frame.to_parquet(path, index=False)
    return path


@pytest.fixture
def live(cfg, tmp_path, monkeypatch):
    """A run that submits: fake gcloud and Model-Pipeline, MLflow and GitHub settings set,
    n = 7 (train on 0..4, hold out 5) and batch 5 scored by @production."""
    cfg["retrain"]["submit"] = True
    cfg["retrain"]["poll_s"] = 0
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.test:5000")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "not-a-real-password")
    monkeypatch.setenv("GITHUB_DISPATCH_TOKEN", "not-a-real-token")
    set_next_batch(cfg, 7)
    stored_scores(cfg, 5, "prod-1")
    fake = FakeRun(tmp_path)
    monkeypatch.setattr(retrain, "_run", fake)
    posts = []
    monkeypatch.setattr(retrain, "_post", lambda url, body, token: posts.append((url, body)) or 204)
    return fake, posts


def test_steps_are_the_seven_in_the_agreed_order():
    # docs/retraining.md section 5: gate -> promote_check -> register @staging -> dispatch -> Slack
    assert [name for name, _ in retrain.STEPS] == [
        "pull_data", "submit_training", "gate", "promote_check", "register", "dispatch",
        "notify"]
    for _, func in retrain.STEPS:
        assert callable(getattr(retrain, func))


@pytest.mark.parametrize(("n", "plan"), [
    (0, None), (2, None),
    (3, {"drift_batches": [0], "holdout_batch": 1, "unlabeled_batch": 2}),
    (7, {"drift_batches": [0, 1, 2, 3, 4], "holdout_batch": 5, "unlabeled_batch": 6}),
    (20, {"drift_batches": list(range(18)), "holdout_batch": 18, "unlabeled_batch": 19}),
])
def test_the_batch_plan_trains_on_0_to_n_minus_3_and_holds_out_n_minus_2(n, plan):
    assert retrain.batch_plan(n) == plan


def test_pull_data_turns_state_json_into_the_set_overrides(cfg):
    set_next_batch(cfg, 7)
    step = retrain.step_pull_data(cfg, retrain.new_run())
    assert step["skipped"] is False and step["holdout_batch"] == 5
    recipe = cfg["retrain"]["recipe"]
    assert step["overrides"][:len(recipe)] == recipe
    assert step["overrides"][len(recipe):len(recipe) + 2] == [
        "retrain.drift_batches=[0,1,2,3,4]", "retrain.holdout_batch=5"]
    assert "retrain.obfuscation_augment.rows=3000" in step["overrides"]


def test_the_job_trains_the_production_recipe_on_the_tables_it_loads(cfg):
    # student.yaml's defaults are the Phase 1 setup: temp data (not in the image), gold-only,
    # no package (check-outputs would fail). The recipe is v1's (docs/tuning.md).
    recipe = dict(s.split("=", 1) for s in cfg["retrain"]["recipe"])
    assert recipe["data.source"] == "official" and recipe["package.enabled"] == "true"
    assert recipe["model.max_length"] == "256" and float(recipe["loss.alpha"]) < 1
    student = yaml.safe_load((retrain.resolve(cfg["retrain"]["model_pipeline_dir"])
                              / cfg["retrain"]["config"]).read_text())
    needed = {"train", "val", *student["pipeline"]["tables"]}
    assert needed <= set(cfg["retrain"]["fetch_tables"])


def test_fewer_than_three_batches_skips_the_whole_run_with_a_message(cfg, monkeypatch):
    alerts = []
    monkeypatch.setattr(retrain, "send_alert", lambda *a: alerts.append(a) or False)
    set_next_batch(cfg, 2)
    run = retrain.run_all(cfg, {"reason": "test", "source": "alert"})
    steps = run["steps"]
    assert steps["pull_data"]["skipped"] is True
    assert steps["submit_training"] == {"skipped": True}
    assert steps["register"]["registered"] is False and steps["dispatch"]["dispatched"] is False
    assert alerts[0][:2] == ("warn", "retrain_dag: retraining skipped")
    assert steps["notify"]["title"].startswith("retrain_dag: skipped")


def test_without_submit_nothing_runs_and_the_record_is_written(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(retrain, "_run", lambda *a, **k: pytest.fail("ran a command"))
    set_next_batch(cfg, 3)
    run = retrain.run_all(cfg, {"reason": "test", "source": "alert"})
    steps = run["steps"]
    assert steps["submit_training"]["submitted"] is False
    assert steps["gate"]["result"] == "not_run"
    assert steps["register"] == {"registered": False, "reason": "gate not_run"}
    record = json.loads((tmp_path / "retrain" / f"{run['run_id']}.json").read_text())
    assert record["trigger"] == {"reason": "test", "source": "alert"}
    assert record["outcome"]["level"] == "warn"


def test_airflow_run_ids_lose_colons_and_plus_signs():
    run = retrain.new_run(run_id="retrain-manual__2026-11-21T18:30:00.123+00:00")
    assert run["run_id"] == "retrain-manual__2026-11-21T18-30-00.123-00-00"


def test_the_job_spec_matches_model_ci(cfg):
    set_next_batch(cfg, 7)
    run = retrain.new_run(run_id="retrain-x")
    run["steps"]["pull_data"] = retrain.step_pull_data(cfg, run)
    spec = retrain.job_spec(cfg, run, image="img@" + DIGEST)
    assert spec["service_account"].startswith("vertex-trainer@")
    # vertex-trainer can write only ci/ in the bucket.
    assert spec["out_dir"] == "/gcs/distillate-mlops-data/ci/retrain/retrain-x"
    script = spec["command"][-1]
    assert script.startswith("python -m src.pipeline.fetch_data --out /tmp/processed")
    assert "python -m src.pipeline.run --config configs/student.yaml" in script
    assert "'retrain.drift_batches=[0,1,2,3,4]' retrain.holdout_batch=5" in script
    assert "paths.out_dir=/gcs/distillate-mlops-data/ci/retrain/retrain-x" in script


def test_a_full_pass_registers_dispatches_and_keeps_secrets_out_of_the_record(cfg, live,
                                                                            tmp_path):
    fake, posts = live
    run = retrain.run_all(cfg, {"reason": "flag_rate PSI 0.31", "source": "alert"})
    steps = run["steps"]
    # The newest trainer image, pinned by digest (the skeleton's tag didn't exist).
    assert steps["submit_training"]["job_spec"]["image"].endswith("/trainer@" + DIGEST)
    assert steps["gate"]["result"] == "passed"
    promote = steps["promote_check"]
    assert promote["result"] == "passed" and promote["rescored"] is False
    assert promote["trained_on"] == [0, 1, 2, 3, 4]      # from the package's meta.json
    assert steps["register"] == {"registered": True, "version": "3", "alias": "staging"}
    url, body = posts[0]
    assert url == "https://api.github.com/repos/Nishaant-Soni/Distillate/dispatches"
    assert body["event_type"] == "deploy-model" and body["client_payload"]["version"] == "3"
    assert steps["notify"]["level"] == "ok"
    # promote_check got the stored production scores and the version to check them against.
    check = fake.ran("src.pipeline.promote_check")[0]
    assert check[check.index("--production-model-version") + 1] == "prod-1"
    assert check[check.index("--batch") + 1] == "5"
    # The job got the MLflow password; the record and the job spec never hold it.
    env = fake.job_config["workerPoolSpecs"][0]["containerSpec"]["env"]
    assert {"name": "MLFLOW_TRACKING_PASSWORD", "value": "not-a-real-password"} in env
    record = (tmp_path / "retrain" / f"{run['run_id']}.json").read_text()
    assert "not-a-real-password" not in record and "not-a-real-token" not in record


def test_the_order_is_gate_then_promote_check_then_register(cfg, live):
    fake, _ = live
    retrain.run_all(cfg)
    modules = [c[2] for c in fake.calls if len(c) > 2 and c[1] == "-m"]
    assert modules.index("src.eval.gate") < modules.index("src.pipeline.promote_check") \
        < modules.index("src.pipeline.register_model")


@pytest.mark.parametrize(("gate", "promote", "reason"), [
    (False, True, "gate blocked"), (True, False, "promote_check failed")])
def test_a_failed_check_keeps_production(cfg, live, tmp_path, monkeypatch, gate, promote, reason):
    fake, posts = live
    fake.gate_passed, fake.promote_passed = gate, promote
    run = retrain.run_all(cfg)
    assert run["steps"]["register"] == {"registered": False, "reason": reason}
    assert not fake.ran("register_model") and not posts
    assert run["steps"]["notify"]["level"] == "warn"
    assert "production kept" in run["steps"]["notify"]["title"]


def test_production_moved_since_the_batch_was_scored_rescores_it(cfg, live, monkeypatch):
    fake, _ = live
    stored_scores(cfg, 5, "older-model")
    scored = []

    def score(batch, service):
        scored.append(len(batch))
        return pd.DataFrame({"id": ["a", "b"], "model_version": ["prod-1"] * 2,
                             **{f"score_{label}": [0.2, 0.8] for label in LABELS}})

    monkeypatch.setattr(retrain.drift_ingest, "release_batch", lambda dcfg, k: pd.DataFrame(
        {"id": ["a", "b"], "text": ["x", "y"]}))
    monkeypatch.setattr(retrain.drift_ingest, "score_batch", score)
    run = retrain.run_all(cfg)
    promote = run["steps"]["promote_check"]
    assert promote["rescored"] is True and scored == [2]
    check = fake.ran("src.pipeline.promote_check")[0]
    assert check[check.index("--production") + 1].endswith("production_batch_05.parquet")


def test_a_live_service_not_yet_on_production_stops_the_check(cfg, live, monkeypatch):
    stored_scores(cfg, 5, "older-model")
    monkeypatch.setattr(retrain.drift_ingest, "release_batch", lambda dcfg, k: pd.DataFrame())
    monkeypatch.setattr(retrain.drift_ingest, "score_batch", lambda b, s: pd.DataFrame(
        {"id": ["a"], "model_version": ["older-model"]}))
    with pytest.raises(retrain.RetrainError, match="wait for the deploy"):
        retrain.run_all(cfg)


def test_a_package_holding_out_another_batch_is_refused(cfg, live):
    fake, _ = live
    fake.holdout = 4
    with pytest.raises(retrain.RetrainError, match="overrides were lost"):
        retrain.run_all(cfg)


def test_no_production_yet_passes_the_promotion_check(cfg, live):
    fake, _ = live
    fake.production = False
    run = retrain.run_all(cfg)
    assert run["steps"]["promote_check"]["result"] == "passed"
    assert not fake.ran("src.pipeline.promote_check")


def test_a_registered_model_without_a_dispatch_token_fails_loudly(cfg, live, monkeypatch):
    monkeypatch.delenv("GITHUB_DISPATCH_TOKEN")
    with pytest.raises(retrain.RetrainError, match="GITHUB_DISPATCH_TOKEN"):
        retrain.run_all(cfg)


def test_submit_needs_the_mlflow_settings(cfg, live, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD")
    with pytest.raises(retrain.RetrainError, match="MLFLOW_TRACKING_PASSWORD"):
        retrain.run_all(cfg)


def test_a_failed_vertex_job_fails_the_step(cfg, live, monkeypatch):
    fake, _ = live
    real = fake.__call__

    def failing(command, **kwargs):
        if "custom-jobs describe" in " ".join(command):
            return done("JOB_STATE_FAILED\n")
        return real(command, **kwargs)

    monkeypatch.setattr(retrain, "_run", failing)
    with pytest.raises(retrain.RetrainError, match="JOB_STATE_FAILED"):
        retrain.run_all(cfg)


def test_the_record_is_copied_for_the_status_page(cfg, live):
    fake, _ = live
    cfg["retrain"]["records_uri"] = "gs://bucket/retrain/"
    run = retrain.run_all(cfg)
    copies = [c[-1] for c in fake.ran("gs://bucket/retrain")]
    assert copies == [f"gs://bucket/retrain/{run['run_id']}.json", "gs://bucket/retrain/latest.json"]
    assert run["steps"]["notify"]["uploaded"] is True


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
    assert dag.get_task("submit_training").execution_timeout.total_seconds() > 5 * 3600
