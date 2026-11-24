#!/usr/bin/env bash
# Everything Distillate needs in a GCP project (work plan v5 6.1). A project owner runs it once;
# it is safe to run again: whatever exists is kept, and role grants that are already there are
# no-ops. Owner: Vaishnavi.
#
#   PROJECT=distillate-mlops Deployment/setup_gcp.sh
#
#  1. APIs
#  2. the data bucket (uniform access) and the Artifact Registry repositories
#  3. service accounts: vertex-trainer (training jobs), github-ci (CI and deploys), mlflow-server,
#     distillate-api (Cloud Run runtime), distillate-ops (the ops VM)
#  4. Workload Identity for GitHub Actions (this repository only)
#  5. roles: each identity gets only what its job needs (Model-Pipeline/docs/ci.md, Deployment/README)
#  6. Secret Manager secrets. Filled by hand once: mlflow-password, github-dispatch-token,
#     slack-webhook-url. Generated here if empty: airflow-admin-password, grafana-admin-password
#  7. the Cloud Run service distillate-api from Deployment/deploy/service.yaml (placeholder image
#     until the first deploy.sh), public
#  8. the demo app's setup (Deployment/frontend/setup_frontend.sh), if it is on this branch
#  9. the MLflow VM (Model-Pipeline/infra/mlflow) and the ops VM (Deployment/ops/startup.sh)
#
# It never prints a secret.
set -euo pipefail

P="${PROJECT:-distillate-mlops}"
REGION="${REGION:-us-east1}"
ZONE="${ZONE:-us-east1-b}"
BUCKET="${BUCKET:-distillate-mlops-data}"
GITHUB_REPO="${GITHUB_REPO:-Nishaant-Soni/Distillate}"
OPS_BRANCH="${OPS_BRANCH:-main}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
B="gs://$BUCKET"

say() { printf '\n== %s\n' "$*"; }
sa() { echo "$1@$P.iam.gserviceaccount.com"; }
exists() { "$@" >/dev/null 2>&1; }

# A new service account takes up to a minute to be visible to every service, so grants retry.
retry() {
  local n
  for n in 1 2 3 4 5 6; do
    if "$@" >/dev/null 2>"${TMPDIR:-/tmp}/setup_gcp.err"; then return 0; fi
    if grep -q "does not exist" "${TMPDIR:-/tmp}/setup_gcp.err" && [ "$n" -lt 6 ]; then
      sleep 15; continue
    fi
    cat "${TMPDIR:-/tmp}/setup_gcp.err" >&2; return 1
  done
}

ensure_sa() {
  exists gcloud iam service-accounts describe "$(sa "$1")" --project "$P" \
    || gcloud iam service-accounts create "$1" --project "$P" --display-name "$2"
}
project_role() {
  retry gcloud projects add-iam-policy-binding "$P" --member="$1" --role="$2" --condition=None
}
bucket_role() {
  retry gcloud storage buckets add-iam-policy-binding "$B" --member="$1" --role="$2" \
    --condition=None
}
# Object access under one prefix only (an IAM condition; needs uniform bucket-level access).
# The expression and title match the Phase 2 grants exactly, so rerunning adds no duplicate.
prefix_role() {
  local member="$1" role="$2" prefix="$3" title="$4" file
  file="$(mktemp)"
  printf 'title: %s\nexpression: resource.name.startsWith("projects/_/buckets/%s/objects/%s/")\n' \
    "$title" "$BUCKET" "$prefix" > "$file"
  retry gcloud storage buckets add-iam-policy-binding "$B" --member="$member" --role="$role" \
    --condition-from-file="$file"
  rm -f "$file"
}
repo_role() {
  retry gcloud artifacts repositories add-iam-policy-binding "$1" --project "$P" \
    --location "$REGION" --member="$2" --role="$3"
}
act_as() {   # $1 may start jobs or services that run as service account $2
  retry gcloud iam service-accounts add-iam-policy-binding "$(sa "$2")" --project "$P" \
    --member="$1" --role=roles/iam.serviceAccountUser
}
secret_access() {
  retry gcloud secrets add-iam-policy-binding "$1" --project "$P" --member="$2" \
    --role=roles/secretmanager.secretAccessor
}

PROJECT_NUMBER="$(gcloud projects describe "$P" --format='value(projectNumber)')"
TRAINER="serviceAccount:$(sa vertex-trainer)"
CI="serviceAccount:$(sa github-ci)"
MLFLOW="serviceAccount:$(sa mlflow-server)"
OPS="serviceAccount:$(sa distillate-ops)"

say "1. APIs"
gcloud services enable compute.googleapis.com iam.googleapis.com iamcredentials.googleapis.com \
  sts.googleapis.com storage.googleapis.com artifactregistry.googleapis.com \
  aiplatform.googleapis.com run.googleapis.com cloudbuild.googleapis.com \
  secretmanager.googleapis.com monitoring.googleapis.com logging.googleapis.com --project "$P"

say "2. Bucket and Artifact Registry"
exists gcloud storage buckets describe "$B" \
  || gcloud storage buckets create "$B" --project "$P" --location "$REGION" \
       --uniform-bucket-level-access
exists gcloud artifacts repositories describe distillate-images --project "$P" --location "$REGION" \
  || gcloud artifacts repositories create distillate-images --project "$P" --location "$REGION" \
       --repository-format docker --description "Training, serving and frontend images"
exists gcloud artifacts repositories describe distillate-models --project "$P" --location "$REGION" \
  || gcloud artifacts repositories create distillate-models --project "$P" --location "$REGION" \
       --repository-format generic --description "Registered model packages"

say "3. Service accounts"
ensure_sa vertex-trainer "Vertex AI training jobs"
ensure_sa github-ci "GitHub Actions (model-ci, deploy.yml)"
ensure_sa mlflow-server "MLflow tracking server VM"
ensure_sa distillate-api "Cloud Run: distillate-api"
ensure_sa distillate-ops "Ops VM: Airflow and monitoring"

say "4. Workload Identity for GitHub Actions ($GITHUB_REPO only)"
exists gcloud iam workload-identity-pools describe github --location global --project "$P" \
  || gcloud iam workload-identity-pools create github --location global --project "$P" \
       --display-name "GitHub Actions"
exists gcloud iam workload-identity-pools providers describe distillate-repo \
    --workload-identity-pool github --location global --project "$P" \
  || gcloud iam workload-identity-pools providers create-oidc distillate-repo \
       --workload-identity-pool github --location global --project "$P" \
       --issuer-uri https://token.actions.githubusercontent.com \
       --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.actor=assertion.actor,attribute.ref=assertion.ref" \
       --attribute-condition "assertion.repository == '$GITHUB_REPO'"
retry gcloud iam service-accounts add-iam-policy-binding "$(sa github-ci)" --project "$P" \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/attribute.repository/$GITHUB_REPO"

say "5. Roles"
# vertex-trainer: read the bucket (DVC data, soft labels), write ci/ only, write its logs.
bucket_role "$TRAINER" roles/storage.objectViewer
bucket_role "$TRAINER" roles/storage.legacyBucketReader
prefix_role "$TRAINER" roles/storage.objectUser ci ci-prefix-only
project_role "$TRAINER" roles/logging.logWriter

# github-ci, model-ci: push images, publish packages, start jobs as vertex-trainer, use ci/.
repo_role distillate-images "$CI" roles/artifactregistry.writer
repo_role distillate-models "$CI" roles/artifactregistry.writer
project_role "$CI" roles/aiplatform.user
act_as "$CI" vertex-trainer
bucket_role "$CI" roles/storage.legacyBucketReader
prefix_role "$CI" roles/storage.objectUser ci ci-prefix-only
# github-ci, deploy.yml (Deployment/deploy/deploy.sh): build the serving image with Cloud Build
# (and stream its logs), deploy revisions and move traffic on distillate-api (run.developer: no
# IAM changes), run it as distillate-api, and append to deployments/deployments.jsonl.
project_role "$CI" roles/run.developer
project_role "$CI" roles/cloudbuild.builds.editor
project_role "$CI" roles/serviceusage.serviceUsageConsumer
project_role "$CI" roles/logging.viewer
act_as "$CI" distillate-api
prefix_role "$CI" roles/storage.objectUser deployments deployments-prefix-only
# gcloud builds submit uploads the build context to Cloud Build's default staging bucket first.
exists gcloud storage buckets describe "gs://${P}_cloudbuild" \
  || gcloud storage buckets create "gs://${P}_cloudbuild" --project "$P" --location "$REGION" \
       --uniform-bucket-level-access
retry gcloud storage buckets add-iam-policy-binding "gs://${P}_cloudbuild" --member="$CI" \
  --role=roles/storage.admin --condition=None
# Before uploading, gcloud lists the project's buckets to find that staging bucket
# (storage.buckets.list; bucket metadata only, no objects).
project_role "$CI" roles/storage.bucketViewer
# Builds run as Cloud Build's runtime account, the Compute default one, and creating a build
# needs actAs on it. act_as builds names under this project, so this one is spelled out.
retry gcloud iam service-accounts add-iam-policy-binding \
  "${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" --project "$P" \
  --member="$CI" --role=roles/iam.serviceAccountUser

# mlflow-server: artifacts under mlflow/ only, plus listing for the artifact browser.
prefix_role "$MLFLOW" roles/storage.objectUser mlflow mlflow-prefix-only
bucket_role "$MLFLOW" roles/storage.legacyBucketReader

# distillate-ops (Airflow's tasks, retrain_dag): read and write the data bucket (DVC, drift
# reports, retrain records), submit Vertex jobs as vertex-trainer, find the newest trainer image,
# publish registered packages, read Cloud Monitoring for Grafana, write its logs.
bucket_role "$OPS" roles/storage.objectUser
project_role "$OPS" roles/aiplatform.user
act_as "$OPS" vertex-trainer
repo_role distillate-images "$OPS" roles/artifactregistry.reader
repo_role distillate-models "$OPS" roles/artifactregistry.writer
project_role "$OPS" roles/monitoring.viewer
project_role "$OPS" roles/logging.logWriter
# distillate-api needs no roles: Cloud Run writes its request logs and metrics itself.

say "6. Secrets"
for name in mlflow-password github-dispatch-token slack-webhook-url airflow-admin-password \
    grafana-admin-password; do
  exists gcloud secrets describe "$name" --project "$P" \
    || gcloud secrets create "$name" --project "$P" --replication-policy automatic
done
for name in airflow-admin-password grafana-admin-password; do
  if ! exists gcloud secrets versions access latest --secret "$name" --project "$P"; then
    python3 -c "import secrets; print(secrets.token_urlsafe(18), end='')" \
      | gcloud secrets versions add "$name" --project "$P" --data-file=- >/dev/null
    echo "generated $name (read it with: gcloud secrets versions access latest --secret $name)"
  fi
done
for name in mlflow-password github-dispatch-token slack-webhook-url airflow-admin-password \
    grafana-admin-password; do
  secret_access "$name" "$OPS"
done
empty=()
for name in mlflow-password github-dispatch-token slack-webhook-url; do
  exists gcloud secrets versions access latest --secret "$name" --project "$P" || empty+=("$name")
done

say "7. Cloud Run service distillate-api"
if exists gcloud run services describe distillate-api --region "$REGION" --project "$P"; then
  echo "exists: kept as it is (deploy.sh owns its image; Deployment/deploy/service.yaml)"
else
  sed "s/distillate-mlops/$P/g" "$ROOT/Deployment/deploy/service.yaml" > "${TMPDIR:-/tmp}/service.yaml"
  retry gcloud run services replace "${TMPDIR:-/tmp}/service.yaml" --region "$REGION" --project "$P"
fi
# Public HTTPS URL (the app has its own rate limit and length cap).
retry gcloud run services add-iam-policy-binding distillate-api --region "$REGION" --project "$P" \
  --member=allUsers --role=roles/run.invoker

say "8. Demo app"
if [ -x "$ROOT/Deployment/frontend/setup_frontend.sh" ]; then
  PROJECT="$P" BUCKET="$BUCKET" REGION="$REGION" "$ROOT/Deployment/frontend/setup_frontend.sh"
else
  echo "Deployment/frontend/setup_frontend.sh isn't on this branch yet; skipped"
fi

say "9. VMs"
if exists gcloud compute instances describe mlflow-server --zone "$ZONE" --project "$P"; then
  echo "mlflow-server exists"
else
  # Model-Pipeline/infra/mlflow/README.md. The admin password is metadata for the first boot
  # only; it is kept in ~/.distillate_mlflow/admin_password. If the new IP differs from
  # 35.227.25.230, update --allowed-hosts in startup.sh first (that README).
  exists gcloud compute addresses describe mlflow-ip --region "$REGION" --project "$P" \
    || gcloud compute addresses create mlflow-ip --region "$REGION" --project "$P"
  exists gcloud compute firewall-rules describe allow-mlflow-5000 --project "$P" \
    || gcloud compute firewall-rules create allow-mlflow-5000 --project "$P" --network default \
         --direction INGRESS --allow tcp:5000 --target-tags mlflow --source-ranges 0.0.0.0/0
  mkdir -p ~/.distillate_mlflow && chmod 700 ~/.distillate_mlflow
  [ -s ~/.distillate_mlflow/admin_password ] || (umask 077; python3 -c \
    "import secrets; print(secrets.token_urlsafe(18), end='')" > ~/.distillate_mlflow/admin_password)
  gcloud compute instances create mlflow-server --project "$P" --zone "$ZONE" \
    --machine-type e2-small --image-family debian-12 --image-project debian-cloud \
    --boot-disk-size 30GB --boot-disk-type pd-standard --address mlflow-ip --tags mlflow \
    --service-account "$(sa mlflow-server)" --scopes cloud-platform \
    --metadata-from-file "startup-script=$ROOT/Model-Pipeline/infra/mlflow/startup.sh,mlflow-admin-password=$HOME/.distillate_mlflow/admin_password"
  echo "mlflow-server created: once /health answers, remove the admin password from its metadata" \
    "(Model-Pipeline/infra/mlflow/README.md) and create the team user"
fi
MLFLOW_IP="$(gcloud compute addresses describe mlflow-ip --region "$REGION" --project "$P" \
  --format='value(address)')"

# Grafana is the only port the ops VM opens; Airflow and Prometheus are reached by SSH tunnel.
exists gcloud compute firewall-rules describe allow-grafana-3000 --project "$P" \
  || gcloud compute firewall-rules create allow-grafana-3000 --project "$P" --network default \
       --direction INGRESS --allow tcp:3000 --target-tags ops-grafana --source-ranges 0.0.0.0/0
if exists gcloud compute instances describe distillate-ops --zone "$ZONE" --project "$P"; then
  echo "distillate-ops exists (startup.sh runs on every boot: sudo reboot picks up changes)"
else
  gcloud compute instances create distillate-ops --project "$P" --zone "$ZONE" \
    --machine-type e2-standard-2 --image-family debian-12 --image-project debian-cloud \
    --boot-disk-size 60GB --boot-disk-type pd-balanced --tags ops-grafana \
    --service-account "$(sa distillate-ops)" --scopes cloud-platform \
    --metadata "repo-url=https://github.com/$GITHUB_REPO.git,repo-branch=$OPS_BRANCH,mlflow-tracking-uri=http://$MLFLOW_IP:5000" \
    --metadata-from-file "startup-script=$ROOT/Deployment/ops/startup.sh"
  echo "distillate-ops created: the first boot builds Airflow's image (about 20 minutes);" \
    "follow it with: gcloud compute ssh distillate-ops --zone $ZONE --project $P" \
    "-- sudo journalctl -fu google-startup-scripts.service"
fi

say "Done"
if [ ${#empty[@]} -gt 0 ]; then
  echo "Fill these secrets once (then reboot distillate-ops so it reads them):"
  for name in "${empty[@]}"; do
    echo "  printf '%s' '<value>' | gcloud secrets versions add $name --project $P --data-file=-"
  done
fi
