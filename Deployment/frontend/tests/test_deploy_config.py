"""The Cloud Run config: service.yaml as deploy_frontend.sh renders it, and the build inputs."""

import fnmatch
import re
from pathlib import Path

import pytest
import yaml

from render_service import fields, render

HERE = Path(__file__).resolve().parents[1]
TEMPLATE = (HERE / "service.yaml").read_text(encoding="utf-8")
ENV = {"SERVICE": "distillate-frontend", "REGION": "us-east1",
       "SERVICE_ACCOUNT": "distillate-frontend@distillate-mlops.iam.gserviceaccount.com",
       "IMAGE": "us-east1-docker.pkg.dev/distillate-mlops/distillate-images/frontend:abc1234",
       "API_URL": "https://distillate-api-xyz.a.run.app",
       "MLFLOW_TRACKING_URI": "http://10.0.0.2:5000", "MLFLOW_TRACKING_USERNAME": "team",
       "MLFLOW_SECRET": "mlflow-password", "GRAFANA_URL": "", "BUCKET": "distillate-mlops-data"}


def rendered(**changes):
    return yaml.safe_load(render(TEMPLATE, {**ENV, **changes}))


def container(service):
    return service["spec"]["template"]["spec"]["containers"][0]


def env_vars(service):
    return {e["name"]: e for e in container(service)["env"]}


def test_session_affinity_websocket_timeout_and_scale():
    service = rendered()
    template = service["spec"]["template"]
    notes = template["metadata"]["annotations"]
    assert notes["run.googleapis.com/sessionAffinity"] == "true"
    assert notes["run.googleapis.com/execution-environment"] == "gen2"   # GCS volumes need it
    assert notes["autoscaling.knative.dev/maxScale"] == "2"
    assert template["spec"]["timeoutSeconds"] == 3600
    assert template["spec"]["serviceAccountName"] == ENV["SERVICE_ACCOUNT"]
    assert container(service)["startupProbe"]["httpGet"]["path"] == "/_stcore/health"


def test_the_password_is_a_secret_reference_only():
    password = env_vars(rendered())["MLFLOW_TRACKING_PASSWORD"]
    assert "value" not in password
    assert password["valueFrom"]["secretKeyRef"] == {"name": "mlflow-password", "key": "latest"}


def test_only_the_deployments_folder_is_mounted_read_only():
    service = rendered()
    volume = service["spec"]["template"]["spec"]["volumes"][0]
    assert volume["csi"]["driver"] == "gcsfuse.run.googleapis.com" and volume["csi"]["readOnly"]
    assert volume["csi"]["volumeAttributes"] == {"bucketName": "distillate-mlops-data",
                                                 "mountOptions": "only-dir=deployments"}
    mount = container(service)["volumeMounts"][0]
    assert mount["name"] == volume["name"] and mount["readOnly"]
    log = env_vars(service)["DEPLOYMENTS_LOG"]["value"]
    assert log == f"{mount['mountPath']}/deployments.jsonl"
    # retrain_dag's records (gs://<bucket>/deployments/retrain/) come through the same mount.
    assert env_vars(service)["RETRAIN_RECORDS"]["value"] == f"{mount['mountPath']}/retrain"


def test_settings_reach_the_app():
    variables = env_vars(rendered())
    assert variables["DISTILLATE_API_URL"]["value"] == ENV["API_URL"]
    assert variables["MLFLOW_TRACKING_URI"]["value"] == ENV["MLFLOW_TRACKING_URI"]
    assert variables["GRAFANA_URL"]["value"] == ""                # empty: "not yet" on Status
    assert container(rendered())["image"] == ENV["IMAGE"]


def test_an_empty_api_url_means_the_mock():
    assert env_vars(rendered(API_URL=""))["DISTILLATE_API_URL"]["value"] == ""


@pytest.mark.parametrize("changes, message", [({"IMAGE": ""}, "empty but required"),
                                              ({"SERVICE_ACCOUNT": "  "}, "empty but required")])
def test_required_fields_cannot_be_empty(changes, message):
    with pytest.raises(ValueError, match=message):
        render(TEMPLATE, {**ENV, **changes})


def test_an_unset_field_stops_the_render():
    env = dict(ENV)
    del env["BUCKET"]
    with pytest.raises(ValueError, match="BUCKET"):
        render(TEMPLATE, env)


def test_values_are_escaped():
    assert env_vars(rendered(GRAFANA_URL='https://g/"x"'))["GRAFANA_URL"]["value"] == \
        'https://g/"x"'


def test_the_build_names_its_staging_folder():
    # The default staging bucket makes gcloud list the project's buckets (storage.buckets.list).
    script = (HERE / "deploy_frontend.sh").read_text(encoding="utf-8")
    assert '--gcs-source-staging-dir="gs://${PROJECT}_cloudbuild/source"' in script


def test_deploy_script_sets_every_field():
    script = (HERE / "deploy_frontend.sh").read_text(encoding="utf-8")
    exported = set(re.findall(r"^export ([A-Z_]+)", script, flags=re.M))
    assert fields(TEMPLATE) <= exported, fields(TEMPLATE) - exported


def test_the_build_uploads_what_the_dockerfile_copies():
    build = yaml.safe_load((HERE / "cloudbuild.yaml").read_text(encoding="utf-8"))
    assert build["steps"][0]["args"][:3] == ["build", "-f", "frontend.Dockerfile"]
    assert build["options"]["logging"] == "CLOUD_LOGGING_ONLY"
    ignored = [line.strip().rstrip("/") for line in
               (HERE / ".gcloudignore").read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.startswith("#")]
    dockerfile = (HERE / "frontend.Dockerfile").read_text(encoding="utf-8")
    copied = [p.rstrip("/") for line in dockerfile.splitlines() if line.startswith("COPY ")
              for p in line.split()[1:-1]]
    for name in ["frontend.Dockerfile", *copied]:
        assert (HERE / name).exists(), name
        assert not any(fnmatch.fnmatch(name, pattern) for pattern in ignored), name
