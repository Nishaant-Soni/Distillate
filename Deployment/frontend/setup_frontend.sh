#!/usr/bin/env bash
# One-time project setup for the demo app (a project owner runs it; safe to run again).
# Belongs with setup_gcp.sh (Vaishnavi), which can call it or copy it.
#
#   PROJECT=distillate-mlops Deployment/frontend/setup_frontend.sh [--deployer user:<email>]
#
# 1. enables the Cloud Run, Cloud Build and Secret Manager APIs
# 2. creates the frontend's own service account (distillate-frontend)
# 3. lets it read the mlflow-password secret
# 4. lets it read the bucket's deployments/ prefix only (an IAM condition), for the
#    read-only deployments.jsonl mount in service.yaml; the rest of the bucket holds data
# 5. with --deployer: lets that person deploy (Cloud Build, Cloud Run, pushing the image,
#    acting as the service account)
set -euo pipefail

PROJECT="${PROJECT:-distillate-mlops}"
BUCKET="${BUCKET:-distillate-mlops-data}"
MLFLOW_SECRET="${MLFLOW_SECRET:-mlflow-password}"
REGION="${REGION:-us-east1}"
SA_NAME="distillate-frontend"
SA="$SA_NAME@$PROJECT.iam.gserviceaccount.com"

DEPLOYER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --deployer)
      [ $# -ge 2 ] || { echo "setup_frontend.sh: --deployer needs user:<email>" >&2; exit 2; }
      DEPLOYER="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "setup_frontend.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done

echo "== APIs"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  secretmanager.googleapis.com --project="$PROJECT"

echo "== Service account $SA"
gcloud iam service-accounts describe "$SA" --project="$PROJECT" >/dev/null 2>&1 \
  || gcloud iam service-accounts create "$SA_NAME" --project="$PROJECT" \
       --display-name="Distillate demo app (Cloud Run)"

echo "== Secret $MLFLOW_SECRET: read access"
gcloud secrets add-iam-policy-binding "$MLFLOW_SECRET" --project="$PROJECT" \
  --member="serviceAccount:$SA" --role=roles/secretmanager.secretAccessor >/dev/null

echo "== gs://$BUCKET/deployments/: read access to that prefix only"
# Objects under deployments/, and listings whose prefix is deployments/ (the mount's only-dir
# lists with that prefix). IAM conditions need uniform bucket-level access, which the bucket uses.
# The expression has commas, which --condition would split on, so it goes in a file.
condition_file="$(mktemp)"
trap 'rm -f "$condition_file"' EXIT
cat > "$condition_file" <<EOF
title: frontend-deployments-only
description: The demo app reads deployments.jsonl only
expression: >-
  resource.name.startsWith("projects/_/buckets/$BUCKET/objects/deployments/") ||
  api.getAttribute("storage.googleapis.com/objectListPrefix", "").startsWith("deployments/")
EOF
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member="serviceAccount:$SA" --role=roles/storage.objectViewer \
  --condition-from-file="$condition_file" >/dev/null

if [ -n "$DEPLOYER" ]; then
  echo "== Deployer $DEPLOYER"
  for role in roles/run.admin roles/cloudbuild.builds.editor roles/serviceusage.serviceUsageConsumer; do
    gcloud projects add-iam-policy-binding "$PROJECT" --member="$DEPLOYER" --role="$role" \
      --condition=None >/dev/null
  done
  gcloud artifacts repositories add-iam-policy-binding distillate-images --project="$PROJECT" \
    --location="$REGION" --member="$DEPLOYER" --role=roles/artifactregistry.writer >/dev/null
  gcloud iam service-accounts add-iam-policy-binding "$SA" --project="$PROJECT" \
    --member="$DEPLOYER" --role=roles/iam.serviceAccountUser >/dev/null
fi

echo "Done. Deploy with Deployment/frontend/deploy_frontend.sh"
