"""deploy_frontend.sh with stand-in gcloud and curl: which serving API it picks, and the warning
for an empty MLflow URI. The stand-in build fails, so the script stops before deploying."""

import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "deploy_frontend.sh"
API = "https://distillate-api-xyz.a.run.app"
FAKE_GCLOUD = f"""#!/usr/bin/env bash
case "$*" in
  "auth print-access-token"*) echo token ;;
  "services list"*) printf '%s\\n' run.googleapis.com cloudbuild.googleapis.com \\
                      secretmanager.googleapis.com ;;
  "run services describe distillate-api"*) echo "{API}" ;;
  "builds submit"*) exit 1 ;;
esac
"""
FAKE_CURL = """#!/usr/bin/env bash
printf '%s' "$FAKE_HEALTH"
"""


def run(tmp_path, health, **env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in (("gcloud", FAKE_GCLOUD), ("curl", FAKE_CURL)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    full = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_HEALTH": health,
            "PYTHON": sys.executable, "MLFLOW_TRACKING_URI": "http://mlflow.example:5000",
            **env}
    full.pop("API_URL", None)
    return subprocess.run(["bash", str(SCRIPT)], env=full, capture_output=True, text=True,
                          timeout=30)


def test_the_live_api_is_used_when_it_answers_health(tmp_path):
    result = run(tmp_path, '{"status": "ok", "model_version": "student-v1"}')
    assert f"serving API: {API}" in result.stdout
    assert "built-in mock" not in result.stdout + result.stderr


def test_a_placeholder_service_falls_back_to_the_mock(tmp_path):
    result = run(tmp_path, "<!doctype html><title>Congratulations | Cloud Run</title>")
    assert "serving API: the built-in mock" in result.stdout
    assert "not the serving app yet?" in result.stderr


def test_an_unreachable_api_falls_back_to_the_mock(tmp_path):
    result = run(tmp_path, "")
    assert "serving API: the built-in mock" in result.stdout


def test_an_empty_mlflow_uri_is_warned_about(tmp_path):
    result = run(tmp_path, '{"status": "ok"}', MLFLOW_TRACKING_URI="")
    assert "MLFLOW_TRACKING_URI is empty" in result.stderr
    assert "MLFLOW_TRACKING_URI is empty" not in run(tmp_path, '{"status": "ok"}').stderr


def test_it_stops_at_the_build_and_deploys_nothing(tmp_path):
    result = run(tmp_path, '{"status": "ok"}')
    assert result.returncode == 1 and "the image build failed" in result.stderr
