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
<!-- Owner: Sai. Splits table, TFDV schema and statistics, anomaly checks. -->

## 6. Bias detection and mitigation
<!-- Owner: Ruchita. Slices, findings, mitigation, link to reports/bias/bias_report.md. -->

## 7. Reproducibility and DVC
<!-- Owner: Vaishnavi. Pinned versions, seed, manifests, dvc pull, committing new data versions. -->

## 8. Pipeline flow optimization
<!-- Owner: Vaishnavi. Gantt before/after table and screenshots. -->

## 9. Alerts
<!-- Owner: Vaishnavi. Slack webhook setup; alerts only log when SLACK_WEBHOOK_URL is unset. -->
