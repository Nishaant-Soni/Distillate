"""Tests for src/pipeline/ci.py and register_model's --predictions."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.pipeline import ci
from src.pipeline import register_model as reg
from tests.pipeline.test_registry import NAME, FakeRunner, make_package, make_report, training_run

STEMS = ["val", "gate", "bias"]


def job_output(root, stems=STEMS, model_version="student-a", run_id=None):
    """A folder shaped like the training job's output (docs/ci.md)."""
    package = make_package(root, model_version, run_id)
    out = package.parent
    package.rename(out / "package")
    for sub in ("predictions", "gold"):
        (out / sub).mkdir()
        for stem in stems:
            pd.DataFrame({"id": ["a", "b"], "p_toxicity": [0.1, 0.9]}).to_parquet(
                out / sub / f"{stem}.parquet")
    return out


def test_gate_stems_come_from_gate_yaml():
    assert ci.gate_stems() == STEMS


def test_complete_output_passes(tmp_path):
    assert ci.check_outputs(job_output(tmp_path), STEMS) == []


def test_missing_files_are_named(tmp_path):
    out = job_output(tmp_path, stems=["val", "gate"])
    (out / "package" / "thresholds.json").unlink()
    missing = ci.check_outputs(out, STEMS)
    assert set(missing) == {"package/thresholds.json", "predictions/bias.parquet",
                            "gold/bias.parquet"}


def test_check_outputs_cli_exit_codes(tmp_path):
    out = job_output(tmp_path)
    assert ci.main(["check-outputs", "--dir", str(out)]) == 0
    (out / "gold" / "gate.parquet").unlink()
    assert ci.main(["check-outputs", "--dir", str(out)]) == 1


def test_register_logs_predictions_with_the_version(cfg, store, tmp_path):
    out = job_output(tmp_path, run_id=training_run(store))
    report = make_report(tmp_path / "gate", "student-a")
    summary = reg.register(cfg, out / "package", report, store, runner=FakeRunner(),
                           predictions=out / "predictions")
    logged = {a.path for a in store.list_artifacts(summary["run_id"], "predictions")}
    assert logged == {f"predictions/{s}.parquet" for s in STEMS}


def test_missing_predictions_folder_is_refused_before_registering(cfg, store, tmp_path):
    out = job_output(tmp_path, run_id=training_run(store))
    report = make_report(tmp_path / "gate", "student-a")
    with pytest.raises(reg.RegistrationError, match="predictions folder"):
        reg.register(cfg, out / "package", report, store, runner=FakeRunner(),
                     predictions=tmp_path / "nope")
    assert store.search_registered_models() == []


def test_fetch_production_none_before_any_promotion(cfg, store, tmp_path):
    assert ci.fetch_production(store, cfg, tmp_path / "prod") is None
    assert not (tmp_path / "prod").exists()


def test_fetch_production_downloads_predictions_and_thresholds(cfg, store, tmp_path):
    out = job_output(tmp_path, run_id=training_run(store))
    report = make_report(tmp_path / "gate", "student-a")
    reg.register(cfg, out / "package", report, store, runner=FakeRunner(),
                 predictions=out / "predictions")
    store.set_registered_model_alias(NAME, "production", "1")

    found = ci.fetch_production(store, cfg, tmp_path / "prod")
    assert found["version"] == "1" and found["model_version"] == "student-a"
    files = {p.name for p in (tmp_path / "prod").iterdir()}
    assert files == {"val.parquet", "gate.parquet", "bias.parquet", "thresholds.json"}
    assert json.loads((tmp_path / "prod" / "thresholds.json").read_text()) == {"toxicity": 0.4}


def test_fetch_production_without_predictions_is_skipped(cfg, store, tmp_path):
    out = job_output(tmp_path, run_id=training_run(store))
    report = make_report(tmp_path / "gate", "student-a")
    reg.register(cfg, out / "package", report, store, runner=FakeRunner())   # no predictions
    store.set_registered_model_alias(NAME, "production", "1")
    assert ci.fetch_production(store, cfg, tmp_path / "prod") is None


def test_fetch_production_cli_writes_github_outputs(cfg, store, tmp_path, monkeypatch):
    github_output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    monkeypatch.setattr(ci, "client", lambda _uri: store)
    assert ci.main(["fetch-production", "--out", str(tmp_path / "prod")]) == 0
    assert "has_production=false" in github_output.read_text()


def test_notify_cli_without_webhook(tmp_path, monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    github_output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    assert ci.main(["notify", "--level", "fail", "--title", "Gate blocked",
                    "--field", "reason=mean AUC 0.91 < 0.95", "--field", "run=x=y"]) == 0
    assert "slack_posted=false" in github_output.read_text()
