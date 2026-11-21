# Demo app (`Deployment/frontend/`)

Streamlit app with two pages (work plan v5, section 6):

* **Try it:** scores a comment through the serving API (contract section 4) and shows per-label
  scores against their thresholds, `flag` and `low_confidence`. Text is shown **masked by
  default** (listed words keep only their first and last letters, `demo_app/masking.py`), with a
  toggle for the original. Presets are shown masked and never in an editable box. Rate limited
  per visitor and per instance.
* **Status:** `@production` and `@staging` from the MLflow registry (version, gate result,
  Artifact Registry upload, last alias move), what is serving (the newest `deployed` or
  `rolled-back` line of `deployments.jsonl`) and any newer attempt that didn't succeed, the last
  retraining, and Grafana links. Anything that doesn't exist yet shows as "not yet". It flags
  `@staging` being newer than `@production` with no deployment line (v5 section 6.1: a replaced
  deploy run never starts, so it leaves no line and no Slack message), and says instead that the
  deploy failed when a line shows it ran and was blocked.

The app never logs or stores the text it scores. The presets (`presets.json`) are written for
the demo; none comes from a dataset.

## Run locally

```bash
cd Deployment/frontend
pip install -r requirements.txt
streamlit run app.py                                   # the built-in mock, no settings needed
```

Against a local HTTP mock of the serving API, or the live service:

```bash
python ../monitoring/mock_service.py --port 8000 &      # Sai's mock, same contract
DISTILLATE_API_URL=http://127.0.0.1:8000 streamlit run app.py
```

## Settings (environment variables)

| Variable | Default | What it does |
|---|---|---|
| `DISTILLATE_API_URL` | unset: the mock | the serving API |
| `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | unset: no registry | the team MLflow server, for Status and the thresholds. On Cloud Run the password comes from Secret Manager (`--set-secrets`) |
| `DEPLOYMENTS_LOG` | unset: "not yet" | `deploy.sh`'s `deployments.jsonl`, a local path (on Cloud Run, the read-only mount of `gs://distillate-mlops-data/deployments/`) |
| `RETRAIN_RECORDS` | unset: "not yet" | a folder of `retrain_dag`'s `<run_id>.json` records |
| `GRAFANA_URL` | unset: "not yet" | the Grafana link |
| `DISTILLATE_SESSION_PER_MINUTE`, `DISTILLATE_INSTANCE_PER_MINUTE` | 10, 60 | the rate limits |
| `DISTILLATE_MAX_CHARS` | 2000 | the longest comment the page sends |

## Image

```bash
docker build -f frontend.Dockerfile -t distillate-frontend .
docker run -p 8080:8080 distillate-frontend
```

## Deploying to Cloud Run

Its own service, `distillate-frontend` (us-east1), defined in [service.yaml](service.yaml):

* **session affinity**, so a visitor's websocket reconnects reach the instance holding their
  session; **request timeout 3600 s**, since each visitor's websocket is one request and the
  300 s default would cut it after 5 minutes; 0 to 2 instances, 1 vCPU, 1 GiB;
* its own service account, `distillate-frontend`; the MLflow password from Secret Manager
  (`mlflow-password`) as a reference, never a value;
* `deployments.jsonl` through a read-only Cloud Storage mount of **only** the bucket's
  `deployments/` folder (`only-dir=deployments`); the service account can read only that prefix,
  since the bucket also holds data. Cloud Storage mounts need the gen2 execution environment.

**Once, by a project owner** (enables the APIs, creates the service account and its access;
safe to rerun; belongs with `setup_gcp.sh`):

```bash
PROJECT=distillate-mlops Deployment/frontend/setup_frontend.sh --deployer user:<email>
```

**Each deploy** (from the repository root; needs `gcloud`, logged in):

```bash
MLFLOW_TRACKING_URI=<team server> MLFLOW_TRACKING_USERNAME=<user> \
  GRAFANA_URL=<optional> Deployment/frontend/deploy_frontend.sh
```

It checks the APIs, builds the image with Cloud Build (`cloudbuild.yaml`; the image is
`distillate-images/frontend:<commit>`), fills `service.yaml` (`render_service.py`; an unset
field stops it), applies it with `gcloud run services replace`, makes the service public, and
waits for `/_stcore/health`. The serving API is `distillate-api`'s URL when that service
exists; `API_URL=` (empty) forces the built-in mock.

Things to know:

* Cloud Run reaches the MLflow server at its external address, with the team login; if MLflow
  ever becomes private to the VPC, the service needs Direct VPC egress.
* Cloud Build pushes the image as the project's build service account, which needs
  Artifact Registry write access on `distillate-images`.
* Retraining records (`RETRAIN_RECORDS`) stay "not yet" until `retrain_dag` writes them to GCS;
  then a second mount like the `deployments/` one shows them.

## Tests

```bash
pip install pytest ruff
ruff check . && python -m pytest tests -v
```

The masking, rate limits, API contract check, registry client (against recorded MLflow answers),
Status logic, and both pages end to end with Streamlit's `AppTest` on the mock. No network.
