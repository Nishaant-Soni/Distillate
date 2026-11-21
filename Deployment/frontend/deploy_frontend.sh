#!/usr/bin/env bash
# Build and deploy the demo app to Cloud Run (work plan v5, section 6). Owner: Nishaant.
#
#   Deployment/frontend/deploy_frontend.sh
#
# 1. checks gcloud is logged in and the Cloud Run, Cloud Build and Secret Manager APIs are on
# 2. builds frontend.Dockerfile with Cloud Build into distillate-images/frontend:<commit>
# 3. fills service.yaml (render_service.py) and applies it (gcloud run services replace)
# 4. makes the service public (allUsers may invoke it; the API it calls is public too)
# 5. checks /_stcore/health on the new URL
#
# Settings (environment; defaults in brackets):
#   PROJECT [distillate-mlops]  REGION [us-east1]  SERVICE [distillate-frontend]
#   SERVICE_ACCOUNT [distillate-frontend@$PROJECT.iam.gserviceaccount.com]
#   BUCKET [distillate-mlops-data]  MLFLOW_SECRET [mlflow-password]
#   API_URL [distillate-api's URL if it answers /health as the serving app does, else empty:
#           the built-in mock]
#   MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, GRAFANA_URL [empty: "not yet" on Status]
#
# One-time setup first (the service account and its access): setup_frontend.sh.
# Exit codes: 0 deployed and healthy; 1 a step failed (the message says which); 2 usage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PROJECT="${PROJECT:-distillate-mlops}"
export REGION="${REGION:-us-east1}"
export SERVICE="${SERVICE:-distillate-frontend}"
export SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-distillate-frontend@$PROJECT.iam.gserviceaccount.com}"
export BUCKET="${BUCKET:-distillate-mlops-data}"
export MLFLOW_SECRET="${MLFLOW_SECRET:-mlflow-password}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-}"
export MLFLOW_TRACKING_USERNAME="${MLFLOW_TRACKING_USERNAME:-}"
export GRAFANA_URL="${GRAFANA_URL:-}"
IMAGE_REPO="${IMAGE_REPO:-$REGION-docker.pkg.dev/$PROJECT/distillate-images/frontend}"
PYTHON="${PYTHON:-python3}"

if [ $# -gt 0 ]; then
  case "$1" in
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "deploy_frontend.sh: takes no arguments (settings come from the environment)" >&2
       exit 2 ;;
  esac
fi

fail() {
  echo "deploy_frontend.sh: $1" >&2
  exit 1
}

# 1. gcloud and the APIs.
gcloud auth print-access-token >/dev/null 2>&1 || fail "gcloud isn't logged in: gcloud auth login"
enabled="$(gcloud services list --enabled --project="$PROJECT" --format='value(config.name)')" \
  || fail "could not list the APIs of $PROJECT"
for api in run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com; do
  grep -qx "$api" <<< "$enabled" \
    || fail "$api is disabled in $PROJECT; a project owner enables it (setup_frontend.sh)"
done

# The serving API: the live service if it exists and answers /health as the serving app does,
# unless API_URL is set (empty means the mock). A service that exists but isn't the app yet
# (Cloud Run's placeholder before the first deploy.sh run) falls back to the mock.
if [ -z "${API_URL+set}" ]; then
  API_URL="$(gcloud run services describe distillate-api --region="$REGION" --project="$PROJECT" \
    --format='value(status.url)' 2>/dev/null || true)"
  if [ -n "$API_URL" ]; then
    health="$(curl -fsS --max-time 15 "$API_URL/health" 2>/dev/null || true)"
    if ! "$PYTHON" -c 'import json, sys; sys.exit(json.loads(sys.argv[1]).get("status") != "ok")' \
        "$health" 2>/dev/null; then
      echo "deploy_frontend.sh: warning: $API_URL/health doesn't answer {\"status\": \"ok\"}" \
        "(not the serving app yet?); using the built-in mock" >&2
      API_URL=""
    fi
  fi
fi
export API_URL
echo "deploy_frontend.sh: serving API: ${API_URL:-the built-in mock}"
if [ -z "$MLFLOW_TRACKING_URI" ]; then
  echo "deploy_frontend.sh: warning: MLFLOW_TRACKING_URI is empty, so the Status page will show" \
    "\"not yet\" for the registry; set it to the team MLflow server to show @production" >&2
fi

# 2. Build.
TAG="$(git -C "$REPO_ROOT" rev-parse --short=7 HEAD 2>/dev/null || date -u +%Y%m%d%H%M%S)"
export IMAGE="$IMAGE_REPO:$TAG"
# The staging folder is named explicitly: with Cloud Build's default one, gcloud first lists the
# project's buckets to check that it owns that bucket, which needs project-wide
# storage.buckets.list. Naming it skips that check (the bucket is created by setup_frontend.sh).
gcloud builds submit "$SCRIPT_DIR" --project="$PROJECT" \
  --gcs-source-staging-dir="gs://${PROJECT}_cloudbuild/source" \
  --config="$SCRIPT_DIR/cloudbuild.yaml" --substitutions="_IMAGE=$IMAGE" --quiet \
  || fail "the image build failed"

# 3. Apply service.yaml.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
"$PYTHON" "$SCRIPT_DIR/render_service.py" "$SCRIPT_DIR/service.yaml" > "$WORK/service.yaml" \
  || fail "service.yaml has an unset field (see above)"
gcloud run services replace "$WORK/service.yaml" --region="$REGION" --project="$PROJECT" \
  --quiet || fail "the Cloud Run service didn't deploy"

# 4. Public, like the API (Try-it has its own rate limits).
gcloud run services add-iam-policy-binding "$SERVICE" --region="$REGION" --project="$PROJECT" \
  --member=allUsers --role=roles/run.invoker --quiet >/dev/null \
  || fail "could not make $SERVICE public (needs run.services.setIamPolicy)"

# 5. Health.
URL="$(gcloud run services describe "$SERVICE" --region="$REGION" --project="$PROJECT" \
  --format='value(status.url)')" || fail "no URL for $SERVICE"
for _ in 1 2 3 4 5 6; do
  if [ "$(curl -fsS --max-time 10 "$URL/_stcore/health" 2>/dev/null)" = "ok" ]; then
    echo "deploy_frontend.sh: $SERVICE is up at $URL ($IMAGE)"
    exit 0
  fi
  sleep 5
done
fail "$URL/_stcore/health didn't answer ok"
