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

Civil Comments is the shifted-domain data: news-site comments, unlike the Wikipedia training data. Each subset comes from a different Civil split, so they share no comments, except that the Identities test split is an annotated subset of Civil test, so some `bias_eval` rows can also appear in `h2_eval`. That overlap is expected; the summary reports its size.

| Subset | Source | Size | Used for |
|---|---|---|---|
| `h2_eval` | Civil test, stratified on toxicity | 10,000 | Shifted-domain evaluation (hypothesis H2) |
| `canary_civil` | Civil validation, stratified on toxicity | 4,000 | Replayed against the live service as a health check, with `canary_wikipedia` |
| `bias_eval` | Full Civil Identities test split, with identity columns | 21,577 | Identity-group bias evaluation |
| `drift/batch_00..19` | Civil train, stratified on toxicity, sorted by `created_date` | 20 batches of 2,000 | Simulated live traffic, oldest to newest, for drift detection |

Drift rows are sampled across the whole train period first and then sorted, so `batch_00` is the oldest traffic and `batch_19` the newest. Sizes come from `splits.civil` in `config/pipeline.yaml`.

Comments the teacher lane used to choose its prompt (its shift check on Civil validation) are listed in `config/civil_teacher_shift_ids.txt` (`paths.civil_exclude_ids`), one id per line. They are removed before sampling, so they never reach `canary_civil` or the drift stream. Until that file exists, nothing is excluded and the stage logs a warning.

Outputs: `data/processed/civil/{h2_eval,canary_civil,bias_eval}.parquet` and `data/processed/civil/drift/batch_XX.parquet`. Id lists go to `data/splits/<subset>_ids.txt`, and `data/splits/civil_drift_ids.csv` (`id, batch, created_date`) lists every drift comment with its batch, for teacher labeling of the drift stream.

Run on its own: `python -m scripts.subsets --config config/pipeline.yaml`. The summary in `reports/summaries/subsets.json` records counts, how many excluded ids were removed, the drift date range, toxicity rates before and after sampling, and the `bias_eval` and `h2_eval` overlap.

### 5.3 Obfuscation set (`scripts/obfuscate.py`)

Tests whether the model can be fooled by the tricks people use to slip past moderation. It takes 3,000 toxic comments from the gate set (`splits.obfuscation_size`) and rewrites each with one of three tricks, chosen per comment:

| Trick | Example | What it simulates |
|---|---|---|
| Substitution | `idiot` becomes `1d10t` | Digits and symbols in place of letters |
| Spacing | `idiot` becomes `i d i o t` | Letters split apart to break up words |
| Homoglyph | `idiot` looks unchanged | Latin letters swapped for identical-looking Cyrillic ones |

Labels are unchanged, since the comment is still toxic. Ids become `<original_id>_obf`, and the derived columns (`char_len`, `has_identity_term`, `has_obfuscation`) are recomputed on the new text. Every choice is seeded per comment id, so the output is identical on every run regardless of row order.

Output: `data/processed/obfuscation.parquet`, with the same columns as the Wikipedia files.

Run on its own: `python -m scripts.obfuscate --config config/pipeline.yaml`. The summary in `reports/summaries/obfuscate.json` records how many comments used each trick and the share flagged as `has_obfuscation`.

### 5.4 Schema and statistics (`scripts/schema_stats.py`)
<!-- Owner: Sai. TFDV schema inference, curation, and validation. -->

### 5.5 Anomaly checks (`scripts/anomalies.py`)

Runs after every data stage and checks all of their outputs. Fail-level checks stop the pipeline; warn-level checks post one Slack alert and let it continue. Every check runs even after one fails, so a single run reports everything.

| Check | Severity | What it catches |
|---|---|---|
| Raw row counts | Fail | Raw files whose row counts differ from `datasets.*.expected_rows` (skipped while `acquire` is still a stub) |
| Id overlap | Fail | The same comment in two sets, e.g. gate comments leaking into train. Allowed: `bias_eval` and `h2_eval`, since Identities test is a subset of Civil test |
| Missing text or labels | Fail above 0.1%, warn below | Empty text or missing labels (`validation.max_missing_fraction`) |
| Label domain | Fail | A label outside {0, 1}, or a `*_frac` value outside [0, 1] |
| Duplicate ids | Fail | An id repeated within one file |
| Long text | Warn | Comments over 5,000 characters (`preprocess.max_char_len_flag`) |
| Prevalence gap | Warn | Train and validation label rates differing by more than 1 point (`validation.prevalence_tolerance_pts`) |
| `created_date` | Fail | Missing or unparseable dates in Civil files. Parsed with `format="mixed"`, since some Kaggle timestamps have no fractional seconds |
| Expected file present | Fail | An output file an upstream stage should have written |

Run on its own: `python -m scripts.anomalies --config config/pipeline.yaml`. Every result, with its severity and a detail message, is recorded in `reports/summaries/anomalies.json`.

## 6. Bias detection and mitigation
<!-- Owner: Ruchita. Slices, findings, mitigation, link to reports/bias/bias_report.md. -->

## 7. Reproducibility and DVC
<!-- Owner: Vaishnavi. Pinned versions, seed, manifests, dvc pull, committing new data versions. -->

## 8. Pipeline flow optimization
<!-- Owner: Vaishnavi. Gantt before/after table and screenshots. -->

## 9. Alerts
<!-- Owner: Vaishnavi. Slack webhook setup; alerts only log when SLACK_WEBHOOK_URL is unset. -->
