import copy
import json
import shutil

import pytest

from src.teacher.config import LABELS, PIPELINE_ROOT, load_config
from src.teacher.mlflow_log import (
    Record,
    build_records,
    label_metrics,
    labeling_record,
    log_records,
    read_manifests,
)
from src.teacher.storage import LocalStorage

FIXTURES = PIPELINE_ROOT / "tests" / "fixtures" / "teacher"


def read(path):
    return json.loads((PIPELINE_ROOT / "artifacts" / path).read_text())


def test_records_from_the_committed_files(tmp_path):
    cfg = load_config()
    pilot, stack, calqwen = build_records(cfg, None, tmp_path, ["pilot", "soft_labels"])
    assert [c.name for c in pilot.children] == [
        "pilot/p2-fs-v1", "pilot/p2-zs-v1", "pilot/p2-zs-v2b", "pilot/tfidf-lr-v1",
        "prompt_v2/p2-zs-v2a", "prompt_v2/p2-zs-v2b", "prompt_v2/p2-zs-v2c"]
    decision = read("pilot/pilot_results.json")["decision"]
    assert pilot.metrics["best_zero_shot_mean_auc"] == decision["best_zero_shot_mean_auc"]
    assert pilot.params["adopted_prompt"] == "p2-zs-v2b"
    zs = next(c for c in pilot.children if c.name == "pilot/p2-zs-v1")
    raw = read("pilot/metrics_p2-zs-v1.json")["raw"]
    assert zs.metrics["raw_auc_threat"] == raw["threat"]["auc"]
    assert zs.metrics["raw_mean_auc"] == pytest.approx(decision["best_zero_shot_mean_auc"],
                                                       abs=1e-4)

    check = read("stack/stack_check.json")["soft_labels"]
    assert stack.metrics["gate_subset_mean_auc"] == check["gate_subset"]["mean_auc"]
    assert stack.metrics["train_auc_threat"] == check["train"]["auc"]["threat"]
    assert len(stack.steps["drift_batch_stack_mean_auc"]) == 20
    assert stack.params["stack_version"] == "stack-v1"
    assert calqwen.metrics["gate_subset_ece"] == read("calqwen/calqwen_check.json")[
        "gate_subset"]["ece"]
    assert calqwen.params["calibration_version"] == "platt-v1"


def store_with_nested_run(tmp_path):
    """The valid fixture shards, plus a nested run whose manifests must not be counted."""
    root = tmp_path / "store"
    shutil.copytree(FIXTURES / "valid", root / "run")
    shutil.copytree(FIXTURES / "valid", root / "run" / "gate")
    return LocalStorage(root)


def test_labeling_record_sums_the_manifests(tmp_path):
    manifests = read_manifests(store_with_nested_run(tmp_path), "run/", tmp_path / "cache")
    assert list(manifests) == ["shard_0000", "shard_0001", "shard_0002"]   # not gate/
    record = labeling_record("train", "run/", manifests, tmp_path)
    assert record.name == "labeling-train"
    assert record.metrics["rows"] == sum(m["row_count"] for m in manifests.values()) == 25
    assert record.metrics["shards"] == 3 and record.metrics["failure_rate"] == 0.0
    assert record.params["prompt_version"] == next(iter(manifests.values()))["prompt_version"]
    saved = json.loads(record.artifacts[0].read_text())
    assert saved["prefix"] == "run/" and set(saved["manifests"]) == set(manifests)


def test_labeling_record_refuses_mixed_versions(tmp_path):
    manifests = read_manifests(store_with_nested_run(tmp_path), "run/", tmp_path / "cache")
    manifests["shard_0001"] = {**manifests["shard_0001"], "prompt_version": "other"}
    with pytest.raises(ValueError, match="several prompt/model versions"):
        labeling_record("train", "run/", manifests, tmp_path)
    with pytest.raises(ValueError, match="no manifests"):
        read_manifests(LocalStorage(tmp_path / "empty"), "x/", tmp_path / "cache")


def test_label_metrics_skip_missing_and_nan():
    per_label = {label: {"auc": 0.9, "ece": float("nan")} for label in LABELS}
    out = label_metrics("raw_", per_label, keys=("auc", "ece"))
    assert out["raw_auc_toxicity"] == 0.9 and out["raw_mean_auc"] == pytest.approx(0.9)
    assert not any(k.startswith("raw_ece") or k == "raw_mean_ece" for k in out)


def test_content_hash_follows_artifact_bytes(tmp_path):
    path = tmp_path / "a.json"
    path.write_text("{}")
    first = Record("r", metrics={"x": 1.0}, artifacts=[path]).content_hash()
    path.write_text('{"changed": true}')
    assert Record("r", metrics={"x": 1.0}, artifacts=[path]).content_hash() != first


def test_logs_once_into_the_experiment_and_skips_on_rerun(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = copy.deepcopy(load_config())
    cfg["paths"]["mlflow_db"] = str(tmp_path / "mlruns" / "mlflow.db")
    cfg["paths"]["mlflow_artifacts"] = str(tmp_path / "mlruns" / "artifacts")
    storage = store_with_nested_run(tmp_path)
    cfg["mlflow"]["labeling"] = {"train": "run/", "gate": "run/gate/"}
    records = build_records(cfg, storage, tmp_path)

    first = log_records(cfg, records)
    assert [r["status"] for r in first] == ["logged"] * len(records)
    again = log_records(cfg, build_records(cfg, storage, tmp_path))
    assert [r["status"] for r in again] == ["skipped (already logged)"] * len(records)
    assert [r["run_id"] for r in again] == [r["run_id"] for r in first]

    experiment = mlflow.get_experiment_by_name("distillate-teacher")
    runs = mlflow.search_runs([experiment.experiment_id], output_format="list")
    children = sum(len(r.children) for r in records)
    assert len(runs) == len(records) + children
    assert mlflow.search_runs(["0"], output_format="list") == []          # nothing in Default
    pilot = next(r for r in runs if r.info.run_name == "pilot")
    nested = [r for r in runs if r.data.tags.get("mlflow.parentRunId") == pilot.info.run_id]
    assert len(nested) == 7
    # Artifacts in the configured local folder, not in ./mlruns of the working directory.
    assert experiment.artifact_location == (tmp_path / "mlruns" / "artifacts").as_uri()
    assert all(r.info.artifact_uri.startswith(experiment.artifact_location) for r in runs)
    gate = next(r for r in runs if r.info.run_name == "labeling-gate")
    assert gate.data.metrics["rows"] == 25 and gate.data.tags["lane"] == "teacher"
    names = {a.path for a in mlflow.MlflowClient().list_artifacts(gate.info.run_id)}
    assert names == {"labeling-gate-manifests.json"}

    # A record whose content changed is logged again; the others are still skipped.
    changed = build_records(cfg, storage, tmp_path)
    changed[0].metrics["vllm_load_s"] += 1
    third = log_records(cfg, changed)
    assert third[0]["status"] == "logged" and third[0]["run_id"] != first[0]["run_id"]
    assert all(r["status"].startswith("skipped") for r in third[1:])


def test_a_run_that_fails_partway_is_logged_again(tmp_path, monkeypatch):
    """A crash while logging a child (the server, the network) must not leave a run that a
    rerun skips as already logged: the hash is set only once the run and its children finish."""
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = copy.deepcopy(load_config())
    cfg["paths"]["mlflow_db"] = str(tmp_path / "mlruns" / "mlflow.db")
    cfg["paths"]["mlflow_artifacts"] = str(tmp_path / "mlruns" / "artifacts")
    upload = mlflow.log_artifact

    def fails_on_the_second_child(path, *args, **kwargs):
        if path.endswith("metrics_p2-zs-v1.json"):
            raise ConnectionError("connection reset by the server")
        return upload(path, *args, **kwargs)

    monkeypatch.setattr(mlflow, "log_artifact", fails_on_the_second_child)
    with pytest.raises(ConnectionError):
        log_records(cfg, build_records(cfg, None, tmp_path, ["pilot"]))
    monkeypatch.setattr(mlflow, "log_artifact", upload)

    rerun = log_records(cfg, build_records(cfg, None, tmp_path, ["pilot"]))
    assert rerun[0]["status"] == "logged"                     # not skipped
    experiment = mlflow.get_experiment_by_name("distillate-teacher")
    parents = [r for r in mlflow.search_runs([experiment.experiment_id], output_format="list")
               if r.info.run_name == "pilot"]
    failed = next(r for r in parents if r.info.run_id != rerun[0]["run_id"])
    assert failed.info.status == "FAILED" and "content_sha256" not in failed.data.tags
    done = next(r for r in parents if r.info.run_id == rerun[0]["run_id"])
    assert done.info.status == "FINISHED" and "content_sha256" in done.data.tags
    children = [r for r in mlflow.search_runs([experiment.experiment_id], output_format="list")
                if r.data.tags.get("mlflow.parentRunId") == done.info.run_id]
    assert len(children) == 7
    # Logged in full this time, so a third run skips it.
    assert log_records(cfg, build_records(cfg, None, tmp_path, ["pilot"]))[0]["status"] \
        == "skipped (already logged)"
