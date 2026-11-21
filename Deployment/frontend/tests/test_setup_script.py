"""setup_frontend.sh with a stand-in gcloud: the uniform-access check comes before any change."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "setup_frontend.sh"
FAKE_GCLOUD = r"""#!/usr/bin/env bash
echo "gcloud $*" >> "$FAKE_LOG"
case "$*" in
  "storage buckets describe"*)
    [ "$FAKE_UBLA" = error ] && exit 1
    echo "$FAKE_UBLA" ;;
  *"--condition-from-file="*)
    for arg in "$@"; do
      case "$arg" in --condition-from-file=*) cat "${arg#*=}" >> "$FAKE_LOG" ;; esac
    done ;;
esac
"""


def run(tmp_path, ubla, *args):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    gcloud = bin_dir / "gcloud"
    gcloud.write_text(FAKE_GCLOUD)
    gcloud.chmod(0o755)
    log = tmp_path / "calls.log"
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_LOG": str(log),
           "FAKE_UBLA": ubla, "PROJECT": "test-project", "BUCKET": "test-bucket"}
    result = subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True,
                            text=True, timeout=30)
    return result, (log.read_text() if log.exists() else "")


def test_full_setup_when_the_bucket_uses_uniform_access(tmp_path):
    result, calls = run(tmp_path, "True", "--deployer", "user:someone@example.com")
    assert result.returncode == 0, result.stderr
    order = [line for line in calls.splitlines() if line.startswith("gcloud ")]
    assert order[0].startswith("gcloud storage buckets describe gs://test-bucket")
    assert order[1].startswith("gcloud services enable run.googleapis.com")
    assert any("secrets add-iam-policy-binding mlflow-password" in c for c in order)
    grant = next(c for c in order if c.startswith("gcloud storage buckets add-iam-policy-binding"))
    assert "--role=roles/storage.objectViewer" in grant
    # Both clauses: reads by name under deployments/, and listings with that prefix.
    assert 'resource.name.startsWith("projects/_/buckets/test-bucket/objects/deployments/")' \
        in calls
    assert 'api.getAttribute("storage.googleapis.com/objectListPrefix", "")' \
           '.startsWith("deployments/")' in calls
    assert any("iam service-accounts add-iam-policy-binding" in c
               and "user:someone@example.com" in c for c in order)
    # Cloud Build: the staging bucket (exists here, so not created) and the logs.
    assert any("storage buckets add-iam-policy-binding gs://test-project_cloudbuild" in c
               and "--role=roles/storage.admin" in c for c in order)
    assert not any("storage buckets create" in c for c in order)
    assert any("projects add-iam-policy-binding" in c and "roles/logging.viewer" in c
               for c in order)


@pytest.mark.parametrize("ubla, message", [("False", "doesn't use uniform bucket-level access"),
                                           ("error", "can't read gs://test-bucket's settings")])
def test_stops_before_any_change_without_uniform_access(tmp_path, ubla, message):
    result, calls = run(tmp_path, ubla)
    assert result.returncode == 1 and message in result.stderr
    changes = [c for c in calls.splitlines() if c.startswith("gcloud ")
               and "buckets describe" not in c]
    assert changes == []
