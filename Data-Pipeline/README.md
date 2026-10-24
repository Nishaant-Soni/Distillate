# Distillate: Data Pipeline

> **Content warning:** the datasets contain slurs, threats, and hateful language.

`data_prep_dag` takes the project from nothing to training-ready data: download (Kaggle), clean, deduplicate, split, schema and statistics (TFDV), anomaly checks with alerts, bias analysis with mitigation, and versioning with DVC. Shared file formats and interfaces are in [CONTRACTS.md](CONTRACTS.md).

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
### 4.1 Data source: Kaggle instead of TFDS

The TFDS copies of both datasets now return HTTP 403, so the pipeline downloads the
original Kaggle competition files the TFDS versions were built from. Ids and row
counts are identical.

| Dataset | Kaggle competition | Files used |
|---|---|---|
| Wikipedia Toxicity | `jigsaw-toxic-comment-classification-challenge` | `train.csv`, `test.csv` + `test_labels.csv` |
| Civil Comments | `jigsaw-unintended-bias-in-toxicity-classification` | `train.csv`, `test_public_expanded.csv` (validation), `test_private_expanded.csv` (test) |

**One-time setup:** accept both competitions' rules on kaggle.com, create an API token
(Kaggle → Settings → API), and save it in `~/.kaggle/access_token` or the
`KAGGLE_API_TOKEN` environment variable. Never commit the token. The competition rules
allow non-commercial use and forbid redistribution, so raw data lives only in our
private DVC remote, never in Git.

### 4.2 Running these stages on their own

```bash
python -m scripts.stage --stage acquire
python -m scripts.stage --stage preprocess
python -m scripts.stage --stage dedupe
```

### 4.3 acquire: download and reshape

Writes six raw files to `data/raw/` and a `manifest.json` with row counts and SHA-256
checksums. A rerun verifies the files against the manifest and skips the download
(68 s first run, under 1 s afterwards).

| File | Rows |
|---|---|
| `wikipedia/train.parquet` | 159,571 |
| `wikipedia/test.parquet` (unscored `-1` rows removed) | 63,978 |
  | `civil/train.parquet` (in `data/cache/`, not DVC-tracked) | 1,804,874 |
  The full Civil train split is too large to version, so its raw and clean files live in the untracked `data/cache/` folder. It can be rebuilt from the pinned Kaggle source; only the sampled subsets are versioned with DVC.
| `civil/validation.parquet` | 97,320 |
| `civil/test.parquet` | 97,320 |
| `civil/identities_test.parquet` (test rows with `identity_annotator_count > 0`) | 21,577 |

CSVs are read with `keep_default_na=False`. Without it, pandas turns comments that are
literally a "missing value" word into empty values: Civil train has 3 comments that
are exactly `None`, and they are kept as text.

### 4.4 preprocess: labels and light cleaning

| Step | Choice | Why |
|---|---|---|
| Wikipedia labels | must be 0/1, cast to `int8`; anything else stops the run | never guess a label |
| Civil labels | rater fraction kept as `<label>_frac`; `<label>` = fraction ≥ 0.5 | standard convention, fraction kept for later use |
| `sexual_explicit` | dropped | not one of our six labels |
| `created_date` | parsed to UTC with `format="mixed"` | 3 dates have no fractional seconds |
| Text | Unicode NFKC → HTML entities → whitespace collapsed | removes web noise; same order as the teacher lane |
| Not done | no lowercasing, punctuation removal or spelling fixes | CAPS, "!!!" and "1d10t" are toxicity signals |
| Derived columns | `char_len`, `has_identity_term`, `has_obfuscation` | used by validation, bias slicing and drift monitoring |

Result: 2,244,640 rows in, 2,244,640 out (no empty comments, no unparseable dates).
`has_obfuscation` (non-letter share above 0.3) flags 1.9% of Wikipedia train;
`has_identity_term` flags 6.9%.

### 4.5 dedupe: remove duplicates from Wikipedia train

Exact duplicates are found by comparing text after lowercasing and removing punctuation.
Near duplicates are found with MinHash LSH on word 3-grams, and every candidate is
confirmed with the real Jaccard similarity (≥ 0.9). Only train rows are removed; the
gate (test) set is never touched. Rows are processed in id order, so results are
identical on every run.

| Reason | Removed | Share toxic |
|---|---|---|
| Exact copy of a gate comment | 388 | 9.5% |
| Near copy of a gate comment | 1,486 | 1.6% |
| Exact duplicate within train | 664 | 14.9% |
| Near duplicate within train | 1,282 | 9.4% |
| **Total** | **3,820 of 159,571 (2.4%)** | |

- 1,874 train comments (1.2%) overlapped the gate set. Without removing them, test
  scores would be inflated.
- Most near overlaps with the gate set are Wikipedia template messages (welcome notes,
  deletion notices, spam warnings), which is why they are rarely toxic.
- 81 duplicate pairs had conflicting labels; the first by id is kept.
- The toxic rate stays about 9.6% after deduplication, so the label balance is unchanged.
- Every removal is listed in `reports/dedupe/removed_ids.csv` (ids only, never text).

### 4.6 Errors and alerts

| Problem | What happens |
|---|---|
| Download fails | `DataUnavailableError` with a hint about the rules and token; Airflow retries |
| Corrupted zip | the zip is deleted and `FileCorruptionError` raised, so the retry downloads it again |
| Wrong row count or changed file | `ManifestMismatchError` naming the file and both numbers |
| Missing or invalid label | `LabelError` with example ids |
  | More than 0.1% of a file dropped (`validation.max_missing_fraction`) | `DataQualityError`: the stage fails and the file is not written |
  | Fewer rows dropped, or unparseable dates | warning alert through `send_alert`; rows dropped (dates kept) |`send_alert` |
| Identity lexicon missing or empty | warning alert; `has_identity_term` is then False |

### 4.7 Tests

51 unit tests (`tests/test_acquire.py`, `tests/test_preprocess.py`, `tests/test_dedupe.py`)
use tiny made-up data and never touch the network. They cover the "None" comment trap,
label edge cases (0.5 vs 0.4999), missing labels, both date formats, HTML and Unicode
cleanup, empty and missing text, exact and near duplicates, gate protection, corrupted
zips, wrong row counts, and run-to-run determinism.

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
