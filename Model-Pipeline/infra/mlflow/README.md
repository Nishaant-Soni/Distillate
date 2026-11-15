# MLflow tracking server

**Owner:** Vaishnavi · **URI:** `http://35.227.25.230:5000` · **VM:** `mlflow-server`
(`us-east1-b`, project `distillate-mlops`)

This is the team's shared experiment tracking and model registry (work plan sections 2.5 and 5).
Every lane logs here through `MLFLOW_TRACKING_URI`. `register_model.py` and `rollback_model.py`
use its registry (see `docs/registry.md`).

| | Choice | Why |
|---|---|---|
| Machine | e2-small (2 GB), 30 GB standard disk | An e2-micro (1 GB, free tier) swaps until requests time out with MLflow 3.18 |
| Hours | on 9 AM to 2 AM Eastern (schedule `mlflow-hours`) | to stretch the $50 course credits; about $10 through Dec 7 |
| Backend store | SQLite on the VM disk, backed up nightly to `gs://distillate-mlops-data/mlflow/backup/` | Postgres doesn't fit comfortably in 2 GB next to MLflow; SQLite is enough for six users |
| Artifacts | `gs://distillate-mlops-data/mlflow/artifacts/`, served through the server (`--artifacts-destination`) | clients, including Kaggle, need only the URI and login, not a GCP key |
| Login | MLflow basic auth: `admin` (Vaishnavi) and a shared `team` user with EDIT permission | the server is reachable from the internet so that Kaggle can log |
| Server identity | service account `mlflow-server`, object read/write on `mlflow/` only | it cannot touch `dvc/`, `teacher/` or any other prefix |

**Limits:** the connection is plain HTTP, so the password is not encrypted in transit. Don't
reuse it anywhere else. The server holds metrics, plots, and model files, never comment text.
A run that is logging when the server stops at 2 AM loses its connection.

## Use it

```bash
export MLFLOW_TRACKING_URI=http://35.227.25.230:5000
export MLFLOW_TRACKING_USERNAME=team
export MLFLOW_TRACKING_PASSWORD=...        # from Vaishnavi
```

On Kaggle, store the three values as notebook secrets and copy them into `os.environ` before
MLflow is imported.

## Start and stop by hand

The schedule handles the normal hours. Outside them, a project owner can run:

```bash
gcloud compute instances start mlflow-server --zone us-east1-b --project distillate-mlops
gcloud compute instances stop  mlflow-server --zone us-east1-b --project distillate-mlops
```

MLflow takes about a minute after boot before `/health` answers.

## Rebuild from scratch

These are the commands that built the current server, in order. `startup.sh` (this folder)
installs MLflow on the first boot and only starts it on later boots.

```bash
P=distillate-mlops
gcloud services enable compute.googleapis.com iam.googleapis.com --project $P

# Identity: read/write on mlflow/ only, plus bucket listing for the artifact browser.
gcloud iam service-accounts create mlflow-server --project $P
gcloud storage buckets add-iam-policy-binding gs://distillate-mlops-data \
  --member=serviceAccount:mlflow-server@$P.iam.gserviceaccount.com --role=roles/storage.objectUser \
  --condition='expression=resource.name.startsWith("projects/_/buckets/distillate-mlops-data/objects/mlflow/"),title=mlflow-prefix-only'
gcloud storage buckets add-iam-policy-binding gs://distillate-mlops-data \
  --member=serviceAccount:mlflow-server@$P.iam.gserviceaccount.com \
  --role=roles/storage.legacyBucketReader --condition=None

# Network: a fixed IP and port 5000 for VMs tagged mlflow.
gcloud compute addresses create mlflow-ip --region us-east1 --project $P
gcloud compute firewall-rules create allow-mlflow-5000 --project $P --network default \
  --direction INGRESS --allow tcp:5000 --target-tags mlflow --source-ranges 0.0.0.0/0

# The VM. The admin password goes in as metadata for the first boot only.
python3 -c "import secrets;print(secrets.token_urlsafe(18))" > admin_password   # keep it private
gcloud compute instances create mlflow-server --project $P --zone us-east1-b \
  --machine-type e2-small --image-family debian-12 --image-project debian-cloud \
  --boot-disk-size 30GB --boot-disk-type pd-standard --address mlflow-ip --tags mlflow \
  --service-account mlflow-server@$P.iam.gserviceaccount.com --scopes cloud-platform \
  --metadata-from-file startup-script=startup.sh,mlflow-admin-password=admin_password

# After the first boot finishes (about 10 minutes; watch the serial port output):
#   1. create the team user as admin:
#      curl -u admin:<pw> -X POST -H 'Content-Type: application/json' \
#        -d '{"username":"team","password":"<team pw>"}' \
#        http://<ip>:5000/api/2.0/mlflow/users/create
#   2. remove the admin password from the metadata:
gcloud compute instances remove-metadata mlflow-server --zone us-east1-b --project $P \
  --keys mlflow-admin-password

# Daily on/off schedule.
gcloud compute resource-policies create instance-schedule mlflow-hours --project $P \
  --region us-east1 --vm-start-schedule="0 9 * * *" --vm-stop-schedule="0 2 * * *" \
  --timezone=America/New_York
gcloud compute instances add-resource-policies mlflow-server --resource-policies=mlflow-hours \
  --zone us-east1-b --project $P
```

If the IP changes, update `--allowed-hosts` in `startup.sh` and in
`/etc/systemd/system/mlflow.service` on the VM.

## Restore from a backup

```bash
gcloud compute ssh mlflow-server --zone us-east1-b --project distillate-mlops
sudo systemctl stop mlflow
sudo gcloud storage cp gs://distillate-mlops-data/mlflow/backup/mlflow-<Day>.db /opt/mlflow/mlflow.db
sudo gcloud storage cp gs://distillate-mlops-data/mlflow/backup/basic_auth-<Day>.db /opt/mlflow/basic_auth.db
sudo chown mlflow:mlflow /opt/mlflow/*.db && sudo systemctl start mlflow
```

Backups keep one file per weekday, so there are always seven days to choose from.
