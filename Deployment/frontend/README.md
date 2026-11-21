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
| `DEPLOYMENTS_LOG` | unset: "not yet" | `deploy.sh`'s `deployments.jsonl` (a local path for now) |
| `RETRAIN_RECORDS` | unset: "not yet" | a folder of `retrain_dag`'s `<run_id>.json` records |
| `GRAFANA_URL` | unset: "not yet" | the Grafana link |
| `DISTILLATE_SESSION_PER_MINUTE`, `DISTILLATE_INSTANCE_PER_MINUTE` | 10, 60 | the rate limits |
| `DISTILLATE_MAX_CHARS` | 2000 | the longest comment the page sends |

## Image

```bash
docker build -f frontend.Dockerfile -t distillate-frontend .
docker run -p 8080:8080 distillate-frontend
```

Cloud Run deployment (its own service, `distillate-frontend`, with session affinity) comes once
Cloud Run and Cloud Build are enabled in the project.

## Tests

```bash
pip install pytest ruff
ruff check . && python -m pytest tests -v
```

The masking, rate limits, API contract check, registry client (against recorded MLflow answers),
Status logic, and both pages end to end with Streamlit's `AppTest` on the mock. No network.
