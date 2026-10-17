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
<!-- Owner: Sai. -->

### 5.1 Wikipedia splits (`scripts/split.py`)

Deduplicated Wikipedia train is split with seeded multilabel iterative stratification (`iterative-stratification`, seed 42), so every label, including rare ones such as `threat`, has the same prevalence in train and validation. The canary set is then carved out of validation the same way, at an exact size. The TFDS test split becomes the gate set unchanged. Input rows are sorted by `id` first, so the split depends only on the data and the seed, not on row order.

| Set | Source | Size | Used for |
|---|---|---|---|
| `train` | Wikipedia train after dedupe | about 85% | Student training; carries `sample_weight` (1.0 until the bias step writes it) |
| `val` | Wikipedia train after dedupe | about 15%, minus the canary | Tuning, thresholds, early stopping |
| `canary_wikipedia` | Carved from `val` | exactly 2,000 (`splits.canary_wikipedia_size`) | Replayed against the live service as a health check |
| `gate` | TFDS test, unchanged | 63,978 | CI evaluation gate; never used for training or tuning |

Outputs: `data/processed/wikipedia/{train,val,gate,canary_wikipedia}.parquet`, plus one id list per set in `data/splits/<set>_ids.txt`. Sizes and fractions come from `config/pipeline.yaml`.

Run on its own: `python -m scripts.split --config config/pipeline.yaml`. The summary in `reports/summaries/split.json` records row counts, each set's share of the deduplicated train, and per-label prevalence for every set.

### 5.2 Civil Comments subsets (`scripts/subsets.py`)
<!-- Owner: Sai. H2 eval, canary, drift batches, bias set. -->

### 5.3 Obfuscation set (`scripts/obfuscate.py`)
<!-- Owner: Sai. -->

### 5.4 Schema and statistics (`scripts/schema_stats.py`)
<!-- Owner: Sai. TFDV schema inference, curation, and validation. -->

### 5.5 Anomaly checks (`scripts/anomalies.py`)
<!-- Owner: Sai. Every check with its fail/warn severity. -->

## 6. Bias detection and mitigation
<!-- Owner: Ruchita. Slices, findings, mitigation, link to reports/bias/bias_report.md. -->

## 7. Reproducibility and DVC
<!-- Owner: Vaishnavi. Pinned versions, seed, manifests, dvc pull, committing new data versions. -->

## 8. Pipeline flow optimization
<!-- Owner: Vaishnavi. Gantt before/after table and screenshots. -->

## 9. Alerts
<!-- Owner: Vaishnavi. Slack webhook setup; alerts only log when SLACK_WEBHOOK_URL is unset. -->
