#!/bin/bash
# Ops VM (distillate-ops, e2-standard-2): Airflow (Data-Pipeline/docker-compose.yaml) and the
# monitoring stack (Deployment/monitoring/docker-compose.yml). Work plan v5 6.1.
#
# Runs as root on every boot (setup_gcp.sh passes it as the startup-script metadata):
#   first boot  installs Docker, git and DVC, clones the repository, pulls the processed data
#   every boot  updates the clone, writes the .env files from Secret Manager, starts both stacks
#
# No key files: gcloud, DVC and the Airflow tasks authenticate as the VM's service account
# (distillate-ops) through the metadata server. Secrets are read at boot and written only to
# root-owned files (mode 600). Logs: sudo journalctl -u google-startup-scripts.service
set -euo pipefail

MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
attr() { curl -sf -H 'Metadata-Flavor: Google' "$MD/$1" || true; }
secret() { gcloud secrets versions access latest --secret="$1" --quiet 2>/dev/null || true; }

REPO_URL="$(attr repo-url)"
BRANCH="$(attr repo-branch)"
MLFLOW_URI="$(attr mlflow-tracking-uri)"
DIR=/opt/distillate
REPO_URL="${REPO_URL:-https://github.com/Nishaant-Soni/Distillate.git}"
BRANCH="${BRANCH:-main}"

if [ ! -f /opt/.distillate-installed ]; then
  # Airflow's image build and its containers need more than 8 GB at peak.
  if [ ! -f /swapfile ]; then
    fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
  fi
  swapon /swapfile || true

  apt-get update -q
  apt-get install -y -q git python3-venv ca-certificates curl
  curl -fsSL https://get.docker.com | sh            # Docker Engine and the compose plugin
  python3 -m venv /opt/dvcenv
  /opt/dvcenv/bin/pip install -q --no-cache-dir 'dvc[gs]==3.67.1'

  git clone --branch "$BRANCH" "$REPO_URL" "$DIR"
  # The processed data (drift batches, splits, baselines) from the DVC remote in the bucket.
  (cd "$DIR" && /opt/dvcenv/bin/dvc pull Data-Pipeline/data/processed.dvc) \
    || echo "ops startup: dvc pull failed; run it by hand (README)" >&2
  touch /opt/.distillate-installed
fi

# Code updates on reboot; local edits on the VM are not kept.
git -C "$DIR" fetch -q origin "$BRANCH" && git -C "$DIR" checkout -q -B "$BRANCH" "origin/$BRANCH" \
  || echo "ops startup: couldn't update the clone; starting what is there" >&2

# Airflow's settings. The UI isn't exposed: reach it with an SSH tunnel (README).
umask 077
cat > "$DIR/Data-Pipeline/.env" <<EOF
AIRFLOW_UID=50000
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=$(secret airflow-admin-password)
GCP_PROJECT_ID=$(curl -sf -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/project/project-id)
GCS_DATA_BUCKET=distillate-mlops-data
SLACK_WEBHOOK_URL=$(secret slack-webhook-url)
MLFLOW_TRACKING_URI=$MLFLOW_URI
MLFLOW_TRACKING_USERNAME=team
MLFLOW_TRACKING_PASSWORD=$(secret mlflow-password)
GITHUB_DISPATCH_TOKEN=$(secret github-dispatch-token)
EOF
# Grafana is the one service reachable from outside (port 3000, with a login).
cat > "$DIR/Deployment/monitoring/.env" <<EOF
GRAFANA_ADMIN_PASSWORD=$(secret grafana-admin-password)
EOF
umask 022

cd "$DIR/Data-Pipeline"
docker compose -f docker-compose.yaml -f ../Deployment/ops/airflow.ops.yaml up -d --build
cd "$DIR/Deployment/monitoring"
docker compose -f docker-compose.yml -f ../ops/monitoring.ops.yaml up -d
echo "ops startup: Airflow and monitoring are up"
