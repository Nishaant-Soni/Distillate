"""Tests for deploy_tools.py: no GCP, no MLflow server, no ONNX model needed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import deploy_tools as dt

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_package(folder: Path, model_version: str = "student-test-1") -> Path:
    """A package with the section 2.5 layout and correct SHA-256s (dummy model bytes)."""
    (folder / "tokenizer").mkdir(parents=True)
    (folder / "model.onnx").write_bytes(b"not a real model")
    (folder / "tokenizer" / "tokenizer.json").write_text("{}")
    (folder / "thresholds.json").write_text(json.dumps({lab: 0.5 for lab in dt.LABELS}))
    files = {p.relative_to(folder).as_posix(): dt.sha256_file(p)
             for p in sorted(folder.rglob("*")) if p.is_file()}
    meta = {"model_version": model_version, "max_length": 256, "labels": list(dt.LABELS),
            "files": files}
    (folder / "meta.json").write_text(json.dumps(meta))
    return folder


def fake_scorer(texts):
    return [{lab: 0.1 + 0.01 * i for i, lab in enumerate(dt.LABELS)} for _ in texts]


class FakeVersion(SimpleNamespace):
    pass


class FakeClient:
    def __init__(self, versions: dict[str, FakeVersion], aliases: dict[str, str]):
        self.versions, self.aliases = versions, aliases

    def get_model_version(self, name, version):
        if version not in self.versions:
            raise RuntimeError(f"RESOURCE_DOES_NOT_EXIST: {name} v{version}")
        return self.versions[version]

    def get_model_version_by_alias(self, name, alias):
        if alias not in self.aliases:
            raise RuntimeError(f"Registered model alias {alias} not found.")
        return self.versions[self.aliases[alias]]


def version(n: int, gate="true", upload="ok", status="READY") -> FakeVersion:
    return FakeVersion(version=str(n), status=status,
                       tags={"model_version": f"student-v{n}", "gate_passed": gate,
                             "ar_upload": upload})


# ---------------------------------------------------------------------------
# Reference inputs and outputs
# ---------------------------------------------------------------------------


def test_reference_texts_are_fifty_neutral_inputs_of_growing_length():
    texts = dt.reference_texts()
    assert len(texts) == 50
    assert texts == dt.reference_texts()                      # deterministic
    assert len(set(texts)) == 50
    assert max(len(t.split()) for t in texts) > 256          # past the token limit
    words = {w for t in texts for w in t.split()}
    assert words <= {w for s in dt.FILLER for w in s.split()}  # filler only, no comment text


def test_ensure_reference_computes_and_writes(tmp_path):
    package = make_package(tmp_path / "pkg")
    reference, created = dt.ensure_reference(package, scorer=fake_scorer)
    assert created
    assert reference["model_version"] == "student-test-1"
    assert len(reference["texts"]) == len(reference["scores"]) == 50
    saved = json.loads((package / dt.REFERENCE_FILE).read_text())
    assert saved == reference


def test_ensure_reference_keeps_the_packages_own(tmp_path):
    package = make_package(tmp_path / "pkg")
    own = {"texts": ["revert"], "scores": [{lab: 0.2 for lab in dt.LABELS}]}
    (package / dt.REFERENCE_FILE).write_text(json.dumps(own))
    reference, created = dt.ensure_reference(package, scorer=lambda texts: pytest.fail("used"))
    assert not created and reference == own


def test_reference_must_cover_every_label(tmp_path):
    package = make_package(tmp_path / "pkg")
    bad = {"texts": ["revert"], "scores": [{"toxicity": 0.2}]}
    (package / dt.REFERENCE_FILE).write_text(json.dumps(bad))
    with pytest.raises(dt.DeployError, match="no score for"):
        dt.ensure_reference(package)


def test_sigmoid_is_stable_for_large_logits():
    assert dt.sigmoid(0) == pytest.approx(0.5)
    assert dt.sigmoid(1000) == pytest.approx(1.0)
    assert dt.sigmoid(-1000) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Package check
# ---------------------------------------------------------------------------


def test_verify_accepts_an_intact_package(tmp_path):
    package = make_package(tmp_path / "pkg")
    assert dt.verify_package(package) == []
    (package / dt.REFERENCE_FILE).write_text("{}")             # files not in meta are fine
    assert dt.verify_package(package, expect_model_version="student-test-1") == []


def test_verify_catches_a_changed_file(tmp_path):
    package = make_package(tmp_path / "pkg")
    (package / "model.onnx").write_bytes(b"tampered")
    assert dt.verify_package(package) == ["model.onnx: SHA-256 differs from meta.json"]


def test_verify_catches_missing_files_and_a_version_mismatch(tmp_path):
    package = make_package(tmp_path / "pkg")
    (package / "tokenizer" / "tokenizer.json").unlink()
    problems = dt.verify_package(package, expect_model_version="student-other")
    assert "tokenizer/tokenizer.json listed in meta.json but missing" in problems
    assert any("registry says 'student-other'" in p for p in problems)
    (package / "thresholds.json").unlink()
    assert dt.verify_package(package) == ["missing thresholds.json"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_resolve_by_number_and_alias():
    client = FakeClient({"2": version(2), "3": version(3)}, {"production": "2", "staging": "3"})
    assert dt.resolve(client, "m", "3") == {"version": "3", "model_version": "student-v3",
                                            "production_version": "2", "spec": "3"}
    assert dt.resolve(client, "m", "staging")["version"] == "3"
    assert dt.resolve(client, "m", "production")["version"] == "2"


def test_resolve_refuses_unusable_versions():
    client = FakeClient({"4": version(4, gate="false"), "5": version(5, upload="failed")}, {})
    with pytest.raises(dt.DeployError, match="the gate didn't pass"):
        dt.resolve(client, "m", "4")
    with pytest.raises(dt.DeployError, match="not published"):
        dt.resolve(client, "m", "5")
    assert dt.resolve(client, "m", "5", allow_unusable=True)["version"] == "5"


def test_resolve_reports_missing_versions_and_aliases():
    client = FakeClient({"1": version(1)}, {})
    with pytest.raises(dt.DeployError, match="no version of m holds @staging"):
        dt.resolve(client, "m", "staging")
    with pytest.raises(dt.DeployError, match="not found"):
        dt.resolve(client, "m", "9")
    assert dt.resolve(client, "m", "1")["production_version"] == ""


def test_no_tracking_uri_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(dt.DeployError, match="registry unreachable"):
        dt.mlflow_client()


# ---------------------------------------------------------------------------
# Cloud Run
# ---------------------------------------------------------------------------

SERVICE = {"status": {"traffic": [
    {"revisionName": "distillate-api-00007-abc", "percent": 100},
    {"revisionName": "distillate-api-00005-aaa", "tag": "v2-1111111",
     "url": "https://v2-1111111---distillate-api-x.a.run.app"},
    {"revisionName": "distillate-api-00007-abc", "tag": "v2-2222222",
     "url": "https://v2-2222222---distillate-api-x.a.run.app"},
    {"revisionName": "distillate-api-00008-def", "tag": "v3-3333333",
     "url": "https://v3-3333333---distillate-api-x.a.run.app"},
    {"revisionName": "distillate-api-00009-ghi", "tag": "v31-4444444",
     "url": "https://v31-4444444---distillate-api-x.a.run.app"},
]}}


def test_tag_url():
    assert dt.tag_url(SERVICE, "v3-3333333") == "https://v3-3333333---distillate-api-x.a.run.app"
    with pytest.raises(dt.DeployError, match="no URL for tag"):
        dt.tag_url(SERVICE, "v9-0000000")


def test_find_tag_picks_the_newest_revision_of_that_version():
    assert dt.find_tag(SERVICE, "2") == "v2-2222222"
    assert dt.find_tag(SERVICE, "3") == "v3-3333333"           # not v31
    with pytest.raises(dt.DeployError, match="deploy.sh --version 4"):
        dt.find_tag(SERVICE, "4")


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


def record_args(log: Path, **overrides) -> argparse.Namespace:
    base = dict(log=str(log), action="deploy", status="deployed", step="move-alias",
                service="distillate-api", version="3", model_version="student-v3",
                tag="v3-3333333", image="img:v3", from_version="2", detail="ok", smoke="")
    return argparse.Namespace(**{**base, **overrides})


def test_record_appends_jsonl_and_summary(tmp_path, monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps({"passed": True, "n_inputs": 50, "max_abs_diff": 1e-6}))
    log = tmp_path / "logs" / "deployments.jsonl"
    dt.record(record_args(log, smoke=str(smoke)))
    dt.record(record_args(log, status="smoke-failed", detail="scores differ"))
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert [e["status"] for e in lines] == ["deployed", "smoke-failed"]
    assert lines[0]["smoke"] == {"passed": True, "n_inputs": 50, "max_abs_diff": 1e-6}
    text = summary.read_text()
    assert "✅ deploy: deployed" in text and "❌ deploy: smoke-failed" in text


def test_append_to_gcs_rewrites_the_object(tmp_path):
    calls = []

    def runner(command, **_):
        calls.append(command)
        if command[:3] == ["gcloud", "storage", "cat"]:
            return SimpleNamespace(returncode=0, stdout='{"old": 1}\n', stderr="")
        assert Path(command[3]).read_text() == '{"old": 1}\n{"new": 2}\n'
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    dt.append_log("gs://bucket/deployments.jsonl", {"new": 2}, runner=runner)
    assert calls[1][:3] == ["gcloud", "storage", "cp"]
    assert calls[1][4] == "gs://bucket/deployments.jsonl"


def test_slack_is_skipped_without_a_webhook(monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    assert dt.post_slack({"action": "deploy", "status": "deployed"}) is False


def test_cli_verify_exit_codes(tmp_path):
    package = make_package(tmp_path / "pkg")
    assert dt.main(["verify", "--package", str(package)]) == 0
    (package / "model.onnx").write_bytes(b"x")
    assert dt.main(["verify", "--package", str(package)]) == 1
