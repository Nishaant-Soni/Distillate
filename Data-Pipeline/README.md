# Distillate: Data Pipeline

> **Content warning:** the datasets contain slurs, threats, and hateful language.

`data_prep_dag` takes the project from nothing to training-ready data: download (TFDS), clean, deduplicate, split, schema and statistics (TFDV), anomaly checks with alerts, bias analysis with mitigation, and versioning with DVC. Shared file formats and interfaces are in [CONTRACTS.md](CONTRACTS.md).

## 1. Environment setup
<!-- Owner: Vaishnavi. Prerequisites (Docker with 8 GB memory, Git, optional gcloud), .env values. -->

## 2. Running the pipeline
<!-- Owner: Vaishnavi. Five-command quickstart, where outputs appear, rerunning one task. -->

### Running tests locally (no Docker needed)
```bash
cd Data-Pipeline
pip install -r requirements-dev.txt
pytest
ruff check .
```

## 3. Code structure
<!-- Owner: Vaishnavi. Folder tree with one line per script. -->

## 4. Data acquisition and preprocessing
<!-- Owner: Praniti. Acquisition, cleaning choices, dedupe method and counts. -->

## 5. Splits, schema, and anomaly detection
<!-- Owner: Sai. Splits table, TFDV schema and statistics, anomaly checks. -->

## 6. Bias detection and mitigation
<!-- Owner: Ruchita. Slices, findings, mitigation, link to reports/bias/bias_report.md. -->

## 7. Reproducibility and DVC
<!-- Owner: Vaishnavi. Pinned versions, seed, manifests, dvc pull, committing new data versions. -->

## 8. Pipeline flow optimization
<!-- Owner: Vaishnavi. Gantt before/after table and screenshots. -->

## 9. Alerts
<!-- Owner: Vaishnavi. Slack webhook setup; alerts only log when SLACK_WEBHOOK_URL is unset. -->
