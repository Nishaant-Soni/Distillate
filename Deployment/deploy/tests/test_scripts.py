"""deploy.sh and rollback.sh end to end, with stand-ins for gcloud and the registry.

A fake ``gcloud`` on PATH serves a fake package, records every command, and returns a Cloud
Run service description; a fake ``python3`` wrapper answers the registry subcommands and
``rollback_model`` and forwards everything else (package checks, smoke test, record) to the
real code. The smoke test hits a local fake service. So these check the scripts' control
flow: what runs, in what order, and what happens on a failed smoke test.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from test_deploy_tools import make_package
from test_smoke_test import make_server

HERE = Path(__file__).resolve().parents[1]
DEPLOY, ROLLBACK = HERE / "deploy.sh", HERE / "rollback.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def write_exe(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def env(tmp_path):
    """PATH with fake gcloud and python3, a package to 'download', and a serving folder."""
    bin_dir, calls = tmp_path / "bin", tmp_path / "calls.log"
    bin_dir.mkdir()
    package = make_package(tmp_path / "ar_package", model_version="student-v3")
    # The fake service's scores for each reference text, written as the package's own.
    from test_smoke_test import expected_scores
    texts = ["revert", "thanks for the edit"]
    (package / "reference_outputs.json").write_text(json.dumps(
        {"texts": texts, "scores": [expected_scores(t) for t in texts]}))
    serving = tmp_path / "serving"
    serving.mkdir()
    (serving / "serve.Dockerfile").write_text("FROM scratch\n")

    server, url = make_server(model_version="student-v3")
    service_json = {"status": {"traffic": [
        {"revisionName": "distillate-api-00002-aaa", "tag": "v2-1111111", "url": url},
        {"revisionName": "distillate-api-00003-bbb", "tag": "TAG_PLACEHOLDER", "url": url}]}}

    write_exe(bin_dir / "gcloud", f"""#!/usr/bin/env bash
echo "gcloud $*" >> "{calls}"
case "$*" in
  "secrets versions access"*) echo "pw" ;;
  "artifacts generic download"*)
    dest=$(printf '%s\\n' "$@" | sed -n 's/^--destination=//p')
    mkdir -p "$dest"; cp -R "{package}/." "$dest/" ;;
  "run services describe"*"--format=json"*)
    tag=$(sed -n 's/.*--tag=\\([^ ]*\\).*/\\1/p' "{calls}" | tail -1)
    sed "s/TAG_PLACEHOLDER/${{tag:-none}}/" "{tmp_path}/service.json" ;;
esac
exit 0
""")
    (tmp_path / "service.json").write_text(json.dumps(service_json))
    # python3 stand-in: registry calls are faked, the rest runs the real code.
    write_exe(bin_dir / "python3", f"""#!/usr/bin/env bash
echo "python3 $*" >> "{calls}"
case "$*" in
  *"deploy_tools.py resolve"*)
    echo '{{"version": "3", "model_version": "student-v3", "production_version": "2"}}' ;;
  *"deploy_tools.py previous"*)
    echo '{{"version": "2", "current": "3", "model_version": "student-v2"}}' ;;
  "-m src.pipeline.rollback_model"*) exit "${{ALIAS_EXIT:-0}}" ;;
  *) exec "{sys.executable}" "$@" ;;
esac
""")
    base = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "SERVING_DIR": str(serving), "DEPLOYMENTS_LOG": str(tmp_path / "deployments.jsonl"),
            "PYTHON": str(bin_dir / "python3"), "MLFLOW_TRACKING_URI": "http://fake"}
    base.pop("SLACK_WEBHOOK_URL", None)
    base.pop("GITHUB_STEP_SUMMARY", None)
    yield {"env": base, "calls": calls, "log": tmp_path / "deployments.jsonl",
           "package": package, "tmp": tmp_path}
    server.shutdown()


def run(script: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True,
                          env=env, timeout=120)


def log_entries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", [DEPLOY, ROLLBACK])
def test_help(script):
    result = run(script, "--help")
    assert result.returncode == 0 and "--" in result.stdout


@pytest.mark.parametrize("args", [[], ["--version", "3", "--package-dir", "x"], ["--bogus"],
                                  ["--version"]])
def test_deploy_usage_errors(args):
    assert run(DEPLOY, *args).returncode == 2


@pytest.mark.parametrize("args", [[], ["--to", "abc"], ["--to"]])
def test_rollback_usage_errors(args):
    assert run(ROLLBACK, *args).returncode == 2


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------


def test_deploy_passes_and_promotes(env):
    result = run(DEPLOY, "--version", "staging", env=env["env"])
    assert result.returncode == 0, result.stderr
    calls = env["calls"].read_text()
    order = ["artifacts generic download", "builds submit", "run deploy", "--no-traffic",
             "update-traffic", "rollback_model --to 3 --alias production"]
    positions = [calls.index(step) for step in order]
    assert positions == sorted(positions)
    entry = log_entries(env["log"])[-1]
    assert entry["status"] == "deployed" and entry["version"] == "3"
    assert entry["from_version"] == "2" and entry["smoke"]["passed"] is True


def test_failed_smoke_test_moves_nothing(env):
    reference = env["package"] / "reference_outputs.json"
    data = json.loads(reference.read_text())
    data["scores"][0] = {k: v + 0.2 for k, v in data["scores"][0].items()}
    reference.write_text(json.dumps(data))          # scores the service won't reproduce
    result = run(DEPLOY, "--version", "3", env=env["env"])
    assert result.returncode == 1
    calls = env["calls"].read_text()
    assert "update-traffic" not in calls and "rollback_model" not in calls
    assert log_entries(env["log"])[-1]["status"] == "smoke-failed"


def test_tampered_package_stops_before_build(env):
    (env["package"] / "model.onnx").write_bytes(b"tampered")
    result = run(DEPLOY, "--version", "3", env=env["env"])
    assert result.returncode == 3
    assert "builds submit" not in env["calls"].read_text()
    entry = log_entries(env["log"])[-1]
    assert entry["status"] == "error" and entry["step"] == "verify"


def test_local_test_package_never_gets_traffic(env):
    result = run(DEPLOY, "--package-dir", str(env["package"]), "--label", "Degraded Test",
                 env=env["env"])
    assert result.returncode == 0, result.stderr
    calls = env["calls"].read_text()
    assert "--tag=degraded-test-" in calls
    assert "update-traffic" not in calls and "rollback_model" not in calls
    assert log_entries(env["log"])[-1]["status"] == "smoke-passed-not-promoted"


def test_alias_failure_after_traffic_is_reported(env):
    result = run(DEPLOY, "--version", "3", env={**env["env"], "ALIAS_EXIT": "1"})
    assert result.returncode == 4
    assert log_entries(env["log"])[-1]["status"] == "partial"


def test_rollback_previous_moves_traffic_then_alias(env):
    result = run(ROLLBACK, "--previous", "--reason", "canary AUC dropped", env=env["env"])
    assert result.returncode == 0, result.stderr
    calls = env["calls"].read_text()
    assert "--to-tags=v2-1111111=100" in calls
    assert calls.index("update-traffic") < calls.index("rollback_model --to 2")
    entry = log_entries(env["log"])[-1]
    assert entry["action"] == "rollback" and entry["status"] == "rolled-back"
    assert entry["version"] == "2" and entry["from_version"] == "3"


def test_rollback_to_a_version_without_a_revision_moves_nothing(env):
    result = run(ROLLBACK, "--to", "7", env=env["env"])
    assert result.returncode == 3
    assert "update-traffic" not in env["calls"].read_text()
