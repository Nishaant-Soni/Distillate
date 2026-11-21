"""Tests for deploy_tools.py: no GCP, no MLflow server, no ONNX model needed."""

from __future__ import annotations

import argparse
import json
import sys
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


def fake_scorer(package, texts):
    """Stands in for serving's scorer.reference_outputs (same output format)."""
    scores = {lab: 0.1 + 0.01 * i for i, lab in enumerate(dt.LABELS)}
    return {"model_version": "student-test-1", "max_length": 256,
            "items": [{"text": t, "scores": dict(scores)} for t in texts]}


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
    texts, scores = dt.reference_pairs(reference)
    assert texts == dt.reference_texts() and len(scores) == 50
    saved = json.loads((package / dt.REFERENCE_FILE).read_text())
    assert saved == reference


@pytest.mark.parametrize("own", [
    {"model_version": "student-test-1", "max_length": 256,
     "items": [{"text": "revert", "scores": {lab: 0.2 for lab in dt.LABELS}}]},
    {"texts": ["revert"], "scores": [{lab: 0.2 for lab in dt.LABELS}]},
])
def test_ensure_reference_keeps_the_packages_own(tmp_path, own):
    package = make_package(tmp_path / "pkg")
    (package / dt.REFERENCE_FILE).write_text(json.dumps(own))
    reference, created = dt.ensure_reference(package, scorer=lambda *a: pytest.fail("used"))
    assert not created and reference == own
    assert dt.reference_pairs(reference) == (["revert"], [{lab: 0.2 for lab in dt.LABELS}])


def test_reference_must_cover_every_label(tmp_path):
    package = make_package(tmp_path / "pkg")
    bad = {"texts": ["revert"], "scores": [{"toxicity": 0.2}]}
    (package / dt.REFERENCE_FILE).write_text(json.dumps(bad))
    with pytest.raises(dt.DeployError, match="no score for"):
        dt.ensure_reference(package)


def test_missing_serving_scorer_is_a_clear_error(tmp_path, monkeypatch):
    package = make_package(tmp_path / "pkg")
    monkeypatch.setitem(sys.modules, "scorer", None)      # import scorer -> ImportError
    with pytest.raises(dt.DeployError, match="install Deployment/serving/requirements.txt"):
        dt.ensure_reference(package)


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


class FakeGCS:
    """Answers gcloud storage describe / cat / cp like a bucket with generations."""

    def __init__(self, content=None, generation=7, describe_error="", conflicts=0):
        self.content, self.generation = content, generation
        self.describe_error, self.conflicts = describe_error, conflicts
        self.calls = []

    def __call__(self, command, **_):
        self.calls.append(command)
        ok = SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[:4] == ["gcloud", "storage", "objects", "describe"]:
            if self.describe_error:
                return SimpleNamespace(returncode=1, stdout="", stderr=self.describe_error)
            if self.content is None:
                return SimpleNamespace(returncode=1, stdout="", stderr="ERROR: 404 Not Found")
            return SimpleNamespace(returncode=0, stdout=f"{self.generation}\n", stderr="")
        if command[:3] == ["gcloud", "storage", "cat"]:
            assert command[3].endswith(f"#{self.generation}")
            return SimpleNamespace(returncode=0, stdout=self.content, stderr="")
        if command[:3] == ["gcloud", "storage", "cp"]:
            expected = 0 if self.content is None else self.generation
            assert command[5] == f"--if-generation-match={expected}"
            if self.conflicts:
                self.conflicts -= 1
                self.content, self.generation = (self.content or "") + '{"other": 1}\n', \
                    self.generation + 1
                return SimpleNamespace(returncode=1, stdout="",
                                       stderr="ERROR: 412 Precondition Failed")
            self.content, self.generation = Path(command[3]).read_text(), self.generation + 1
            return ok
        raise AssertionError(command)


def test_append_to_gcs_keeps_the_history():
    gcs = FakeGCS(content='{"old": 1}\n')
    dt.append_log("gs://bucket/deployments.jsonl", {"new": 2}, runner=gcs)
    assert gcs.content == '{"old": 1}\n{"new": 2}\n'


def test_append_to_gcs_creates_a_missing_log():
    gcs = FakeGCS(content=None)
    dt.append_log("gs://bucket/deployments.jsonl", {"new": 2}, runner=gcs)
    assert gcs.content == '{"new": 2}\n'


def test_a_read_error_never_overwrites_the_log():
    gcs = FakeGCS(content='{"old": 1}\n', describe_error="ERROR: 403 Permission denied")
    with pytest.raises(dt.DeployError, match="could not read"):
        dt.append_log("gs://bucket/deployments.jsonl", {"new": 2}, runner=gcs)
    assert not any(c[:3] == ["gcloud", "storage", "cp"] for c in gcs.calls)
    assert gcs.content == '{"old": 1}\n'


def test_a_concurrent_write_is_retried_not_lost():
    gcs = FakeGCS(content='{"old": 1}\n', conflicts=1)
    dt.append_log("gs://bucket/deployments.jsonl", {"new": 2}, runner=gcs)
    assert gcs.content == '{"old": 1}\n{"other": 1}\n{"new": 2}\n'


def test_an_unreadable_log_says_so():
    # gcloud reports an unreadable object as 404, so the read looks like "no log yet"; the
    # generation-0 write then keeps failing, and the error should point at permissions.
    with pytest.raises(dt.DeployError, match="exists but can't be read"):
        dt.append_log("gs://bucket/deployments.jsonl", {"new": 2}, runner=_always_conflict)


def _always_conflict(command, **_):
    if command[:4] == ["gcloud", "storage", "objects", "describe"]:
        return SimpleNamespace(returncode=1, stdout="", stderr="ERROR: not found: 404")
    if command[:3] == ["gcloud", "storage", "cp"]:
        assert command[5] == "--if-generation-match=0"
        return SimpleNamespace(returncode=1, stdout="", stderr="ERROR: 412 Precondition Failed")
    raise AssertionError(command)


def test_endless_conflicts_give_up_with_an_error():
    gcs = FakeGCS(content='{"old": 1}\n', conflicts=5)
    with pytest.raises(dt.DeployError, match="kept changing"):
        dt.append_log("gs://bucket/deployments.jsonl", {"new": 2}, runner=gcs, attempts=3)


def test_slack_is_skipped_without_a_webhook(monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    assert dt.post_slack({"action": "deploy", "status": "deployed"}) is False


def test_cli_verify_exit_codes(tmp_path):
    package = make_package(tmp_path / "pkg")
    assert dt.main(["verify", "--package", str(package)]) == 0
    (package / "model.onnx").write_bytes(b"x")
    assert dt.main(["verify", "--package", str(package)]) == 1
