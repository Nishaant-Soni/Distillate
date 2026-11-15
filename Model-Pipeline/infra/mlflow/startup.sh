#!/bin/bash
# MLflow tracking server for Distillate, on an e2-small (2 GB RAM). An e2-micro (1 GB) is too
# small for MLflow 3.18: it swaps until requests time out.
# Runs on every boot; the install happens only on the first one.
set -euo pipefail

MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
BUCKET=gs://distillate-mlops-data/mlflow

if [ ! -f /opt/mlflow/.installed ]; then
  # 1 GB of RAM is tight for pip and MLflow's workers, so add 1 GB of swap.
  if [ ! -f /swapfile ]; then
    fallocate -l 1G /swapfile && chmod 600 /swapfile && mkswap /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
  fi
  swapon /swapfile || true

  apt-get update -q
  apt-get install -y -q python3-venv sqlite3

  useradd --system --create-home --home-dir /opt/mlflow mlflow || true
  python3 -m venv /opt/mlflow/venv
  /opt/mlflow/venv/bin/pip install -q --no-cache-dir 'mlflow[auth]' google-cloud-storage

  PW=$(curl -sf -H 'Metadata-Flavor: Google' "$MD/mlflow-admin-password")
  cat > /opt/mlflow/basic_auth.ini <<EOF
[mlflow]
default_permission = EDIT
database_uri = sqlite:////opt/mlflow/basic_auth.db
admin_username = admin
admin_password = $PW
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
EOF
  chmod 600 /opt/mlflow/basic_auth.ini

  # MLflow 3.x needs a fixed secret for CSRF protection. Job execution (GenAI scorer workers,
  # about 190 MB each) is not used here and does not fit in 2 GB, so it is turned off.
  ( umask 077; echo "MLFLOW_FLASK_SERVER_SECRET_KEY=$(python3 -c 'import secrets;print(secrets.token_hex(32))')" > /opt/mlflow/env )
  echo MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false >> /opt/mlflow/env

  cat > /etc/systemd/system/mlflow.service <<'EOF'
[Unit]
Description=MLflow tracking server
After=network-online.target
Wants=network-online.target

[Service]
User=mlflow
WorkingDirectory=/opt/mlflow
Environment=MLFLOW_AUTH_CONFIG_PATH=/opt/mlflow/basic_auth.ini
EnvironmentFile=/opt/mlflow/env
ExecStart=/opt/mlflow/venv/bin/mlflow server \
  --host 0.0.0.0 --port 5000 --workers 1 \
  --app-name basic-auth \
  --allowed-hosts 35.227.25.230:5000,35.227.25.230,localhost:5000,localhost \
  --backend-store-uri sqlite:////opt/mlflow/mlflow.db \
  --artifacts-destination gs://distillate-mlops-data/mlflow/artifacts
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

  # Nightly backup of both SQLite databases to the bucket (the VM disk is the only other copy).
  cat > /etc/cron.d/mlflow-backup <<EOF
30 3 * * * root for db in mlflow basic_auth; do sqlite3 /opt/mlflow/\$db.db ".backup /tmp/\$db.db" && gcloud storage cp /tmp/\$db.db $BUCKET/backup/\$db-\$(date +\%a).db -q; done
EOF

  chown -R mlflow:mlflow /opt/mlflow
  systemctl daemon-reload
  systemctl enable mlflow
  touch /opt/mlflow/.installed
fi

systemctl start mlflow
