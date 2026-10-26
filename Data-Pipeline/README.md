# Distillate: Data Pipeline

> **Content warning:** the datasets contain slurs, threats, and hateful language.

`data_prep_dag` takes the project from nothing to training-ready data: download (Kaggle), clean, deduplicate, split, schema and statistics (TFDV), anomaly checks with alerts, bias analysis with mitigation, and versioning with DVC. Shared file formats and interfaces are in [CONTRACTS.md](CONTRACTS.md).

## 1. Environment setup

**Prerequisites**

| | |
|---|---|
| Docker Desktop | **8 GB memory or more** (Settings → Resources). On Apple Silicon also turn on Settings → General → *Use Rosetta for x86/amd64 emulation* — the image is `linux/amd64` because TFDV publishes no Linux arm64 wheel |
| Git | any recent version |
| Python 3.12 | only to run the tests outside Docker |
| Disk | about 15 GB: competition downloads, raw and processed Parquet, and the DVC cache |

**Accept both Kaggle competitions' rules.** Do this first, in a browser, on the
account whose credentials you are about to use. Downloads return 403 until you do,
and the rules are accepted per competition:

- <https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/rules>
- <https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/rules>

**Credentials.** Compose reads `.env` from this folder only, so the copy lives in
`Data-Pipeline/`, not the repo root:

```bash
cd Data-Pipeline
cp .env.example .env
```

Fill in the Kaggle block from <https://www.kaggle.com/settings/api>. Either route works:

- **`KAGGLE_API_TOKEN`** — the access token that page shows. Checked first, so this is the simplest.
- **`KAGGLE_USERNAME` + `KAGGLE_KEY`** — from *Legacy API Credentials → Create Legacy API Key*,
  which downloads `kaggle.json` holding both values. If you use these, leave
  `KAGGLE_API_TOKEN` empty, because a value there takes precedence.

Everything else in `.env` has a working default. With no `SLACK_WEBHOOK_URL` alerts are
logged instead of posted, and `DVC_PUSH=false` keeps a fresh clone green without any GCP
credentials. `.env` is git-ignored; never commit it.

## 2. Running the pipeline

Five commands from a fresh clone:

```bash
cd Data-Pipeline
cp .env.example .env                 # 1. then add your Kaggle credentials (section 1)
docker compose build                 # 2. 20-40 min first time; TensorFlow is large
docker compose up -d                 # 3. Airflow UI on http://localhost:8080
docker compose exec airflow-scheduler airflow dags trigger data_prep_dag   # 4.
cat reports/run_summary.json         # 5. what the run did
```

The UI login comes from `_AIRFLOW_WWW_USER_*` in `.env`. The build prints
`TFDV checkpoint OK` when the TensorFlow environment is healthy; if that line is missing
the build has failed, and nothing after it will work.

A full run on real data takes roughly five minutes of task time after the downloads are
cached. **Keep the machine awake** — everything runs in Docker on your laptop, so sleep
suspends the run and tasks can exceed their timeouts on wake.

**Where the outputs go**

| Path | Contents |
|---|---|
| `data/kaggle/` | competition downloads (a Docker volume, so reruns skip them) |
| `data/raw/` | raw Parquet plus `manifest.json` with row counts and checksums |
| `data/cache/` | the 1.8M-row Civil train file: untracked, not in DVC |
| `data/processed/` | the contract files in [CONTRACTS.md](CONTRACTS.md) section 2 |
| `data/splits/` | id lists per split, and the drift manifest |
| `reports/summaries/<task>.json` | one summary per task: rows in, rows out, drops, duration |
| `reports/run_summary.json` | all of them merged, written by `notify_success` |
| `logs/` | Airflow task logs |

**Rerunning one task.** In the UI, select the task and press *Clear*; it reruns with
everything downstream. To run a single stage outside Airflow, use the CLI from
[CONTRACTS.md](CONTRACTS.md) section 3:

```bash
docker compose exec --workdir /opt/airflow/pipeline airflow-scheduler \
  python -m scripts.stage --stage dedupe --config config/pipeline.yaml
```

Stages are safe to rerun: every write goes through `utils.atomic_write`, so an
interrupted task leaves the previous output intact rather than a half-written file.

### Running tests locally (no Docker needed)
```bash
cd Data-Pipeline
pip install -r requirements-dev.txt
pytest
ruff check .
```

## 3. Code structure

```text
Data-Pipeline/
├── dags/data_prep_dag.py     the DAG: 11 tasks built from the graph below
├── config/
│   ├── pipeline.yaml         every tunable value; seed 42; no script hard-codes these
│   └── identity_terms.txt    identity lexicon (owner: Ruchita; others only read it)
├── scripts/
│   ├── stage.py              one entry point per stage: real module if merged, stub if not
│   ├── pipeline_graph.py     the task graph as data, so it is testable without Airflow
│   ├── stubs.py              contract-shaped placeholder output for unmerged stages
│   ├── utils.py              get_logger, load_config, atomic_write, write_summary, timed
│   ├── alerts.py             send_alert plus the Airflow on_failure_callback
│   ├── columns.py            label names and the Kaggle-to-contract column renames
│   ├── errors.py             the pipeline's typed errors
│   ├── parquet_io.py         chunked Parquet writing and SHA-256 helpers
│   ├── acquire.py            download both competitions, reshape, write the manifest
│   ├── preprocess.py         label casting, NFKC and entity cleanup, derived columns
│   ├── dedupe.py             exact and MinHash-LSH near-duplicate removal
│   ├── split.py              seeded multilabel train/val split, canary carve-out, gate
│   ├── subsets.py            Civil H2, canary, bias set, 20 drift batches
│   ├── obfuscate.py          3,000 disguised toxic gate comments
│   ├── anomalies.py          every data check, each with a fail or warn severity
│   ├── schema_stats.py       TFDV statistics and schema validation (runs in /opt/tfenv)
│   ├── dvc_version.py        dvc add, then dvc push when DVC_PUSH is true
│   └── run_summary.py        merges the task summaries and posts one Slack message
├── tests/                    pytest; the Airflow DAG tests skip outside the image
├── Dockerfile                Airflow 3.3.2 on Python 3.12, plus TFDV on 3.11 at /opt/tfenv
└── docker-compose.yaml       LocalExecutor, Postgres, and the Kaggle download cache
```

**Two Python environments, one image.** Airflow and most stages run on Python 3.12.
`schema_stats` runs on Python 3.11 at `/opt/tfenv`, because TFDV 1.16.1 is the last release
built for this base image's glibc and it publishes no cp312 wheel. It also pins
`pandas<2`, which would drag the main environment backwards. The two never share memory:
Parquet on disk is the only thing that crosses between them.

**Adding a stage.** Add it to `TASKS` and the edges in `scripts/pipeline_graph.py`, and map
it to its module in `STAGE_MODULES` in `scripts/stage.py`. The DAG builds itself from that
graph, so no DAG edit is needed, and `tests/test_dag.py` checks the two agree.

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

Validates every raw and processed file against a TFDV schema, and fails the run on any anomaly. It runs in the TensorFlow environment (`/opt/tfenv`, Python 3.11), which is why it is a BashOperator in the DAG.

**Two schemas**, because raw and processed files give the same columns different types (raw labels are float rater fractions, processed labels are `int8`):

| File | Built from | Validates |
|---|---|---|
| `data/schema/raw_schema.pbtxt` | raw Wikipedia train and raw Civil identities test | the 6 raw files |
| `data/schema/schema.pbtxt` | processed Wikipedia train and `bias_eval` | the 16 processed files, including the obfuscation set and all drift batches |

Files with different columns share a schema through **TFDV environments**, so a column can be required in one kind of file and absent from another:

| Environment | Files | Extra columns |
|---|---|---|
| `RAW_WIKI` | raw Wikipedia train and test | none |
| `RAW_CIVIL` | raw Civil train, validation, test, identities test | dates, publication and article ids, `sexual_explicit`, identity scores |
| `WIKI` | processed Wikipedia files, obfuscation set | none |
| `WIKI_TRAIN` | processed Wikipedia train | `sample_weight` |
| `CIVIL` | processed Civil files, H2, canary, drift | `created_date`, publication and article ids, `*_frac` |
| `CIVIL_BIAS` | `bias_eval`, `identities_test_clean` | `identity_annotator_count` and 24 identity scores |

**Curation on top of inference:**

| Rule | Why |
|---|---|
| Labels and the two flags in {0, 1}; `*_frac` and processed identity scores in [0, 1] | the contract's value ranges |
| `id` and `text` required in every file; `char_len` at least 1 | no row without an id or text |
| `created_date` required in every Civil file | drift batches are ordered by it |
| Inferred string vocabularies removed | otherwise every new publication or article id would be an anomaly |
| Raw identity scores may be missing | most raw Civil rows were never shown to identity annotators (about 78% are empty) |

The work plan's "language is English" rule is not checked: neither Kaggle source has a language field. Both datasets are English-only by construction.

Timestamps are converted to ISO strings and booleans to 0/1 before TFDV reads them, since TFDV handles neither. The two 1.8M-row Civil train files are validated on a seeded sample of `validation.raw_civil_stats_sample` (100,000) rows.

**Running it:**

```bash
# Validate (what the DAG runs); writes reports/schema/validation.json, fails on any anomaly
docker compose exec airflow-scheduler bash -c \
  "cd /opt/airflow/repo/Data-Pipeline && /opt/tfenv/bin/python -m scripts.schema_stats"

# Rebuild both schemas after an intentional contract change, then review the diff and commit
docker compose exec airflow-scheduler bash -c \
  "cd /opt/airflow/repo/Data-Pipeline && /opt/tfenv/bin/python -m scripts.schema_stats --infer"
```

If a schema file is missing, the stage stops and says to run `--infer`. On the first full run on real data, all 22 files matched their schemas, in about 25 seconds.

**Tests:** `tests/test_schema_stats.py` (11 tests) builds contract-shaped raw and processed files, infers schemas from them, and checks that clean data passes and that each kind of anomaly is caught: a label out of range, missing text, a missing required column, an unexpected column, a missing file, and a missing schema. They skip outside the TFDV environment; run them inside the image with `/opt/tfenv/bin/python -m pytest tests/test_schema_stats.py`.

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

**What makes a rerun identical**

- **Seed 42** everywhere, from `config/pipeline.yaml`. `split.py` sorts by `id` before
  splitting, so results depend on the data and the seed, never on row order.
- **Pinned versions.** The image is `apache/airflow:3.3.2-python3.12` at an exact patch,
  and `requirements.txt` installs against Airflow's official constraints file for that
  version, so a rebuild months from now resolves the same dependency set.
- **Manifests.** `acquire` writes `data/raw/manifest.json` with row counts and a SHA-256
  per downloaded file, and fails if a later run's counts or checksums differ.
- **Expected counts are asserted**, not assumed: 159,571 and 63,978 for Wikipedia,
  1,804,874 and 97,320 twice for Civil, 21,577 identity-annotated rows.

**DVC.** Git holds only the small `.dvc` pointer files; the Parquet lives in the bucket.

```bash
dvc pull                 # fetch the data for the current commit
```

The `dvc_version` task runs `dvc add data/raw data/processed`, then `dvc push` only when
`DVC_PUSH=true`. It defaults to false so a fresh clone and CI stay green with no GCP
credentials, and it logs a reminder naming the pointer files to commit:

```bash
git add Data-Pipeline/data/raw.dvc Data-Pipeline/data/processed.dvc
git commit -m "Data version for <what changed>"
```

Commit the pointer files in the same commit as the code that produced them — that pairing
is what makes a past result reproducible.

## 8. Pipeline flow optimization

The DAG runs in two shapes, set by `PIPELINE_MODE` in `.env`. `baseline` wires all eleven
tasks into one sequential chain; `optimized` (the default) splits the Wikipedia and Civil
work into parallel branches that rejoin at `schema_stats`. Both are built from the same
`scripts/pipeline_graph.py`, so the comparison is the same tasks in two orders, not two
different pipelines.

Measured on an M-series MacBook running the `linux/amd64` image under emulation, with the
competition downloads already cached. Durations are seconds.

| Task | Baseline (sequential) | Optimized (parallel) |
|---|---|---|
| acquire | 4.9 | 2.7 |
| preprocess | 147.7 | 145.6 |
| dedupe | 61.7 | 71.7 |
| subsets | 7.3 | 12.9 *(runs beside `dedupe`)* |
| split | 3.0 | 3.2 |
| obfuscate | 1.5 | 1.2 |
| schema_stats | 0.9 | 1.4 |
| anomalies | 1.5 | 1.8 |
| bias | 0.9 | 1.0 |
| dvc_version | 0.1 | 0.1 |
| notify_success | 0.1 | 0.1 |
| **Wall clock** | **235.7** | **235.2** |

**The parallel branches bought almost nothing: 0.5 s, which is inside run-to-run noise.**
Two reasons, and both are visible in the table:

1. **The branches are badly unbalanced.** The Civil branch is `subsets` alone, about 7 s.
   The Wikipedia branch is `dedupe` → `split` → `obfuscate`, about 66 s. Overlapping them
   can never save more than the shorter branch, so roughly 7 s was the ceiling.
2. **Concurrency made both tasks slower.** Run side by side, `dedupe` went from 61.7 s to
   71.7 s and `subsets` from 7.3 s to 12.9 s. On an emulated image with a small CPU
   allocation, the contention cost about as much as the overlap saved.

**The optimization that did work is caching the downloads.** `acquire` writes a manifest of
row counts and per-file checksums, and skips the download when it still matches:

| `acquire` | Duration |
|---|---|
| Cold, first run, about 2 GB over the network | **68.7 s** |
| Warm, manifest matches | **2.7 – 9.7 s** |

That is a real ~65 s saved on every rerun, and it is why `data/kaggle` is a named Docker
volume: the cache survives `docker compose down`. To force a cold run, remove the volume.

**Where the time actually goes.** `preprocess` is 147 s, about 62% of the run, and it is a
single task cleaning both datasets, so no amount of reordering touches it. The honest
next step is to split it per dataset — `preprocess_wikipedia` alongside `preprocess_civil`
— which would put real work on both sides of the fork instead of 7 s against 66 s. That
needs a change to the stage contract in [CONTRACTS.md](CONTRACTS.md) section 3, which is
one function per script, so it is a team decision rather than a local edit.

Reproduce either shape:

```bash
sed -i '' 's/^PIPELINE_MODE=.*/PIPELINE_MODE=baseline/' .env   # or optimized
docker compose up -d
docker compose exec airflow-scheduler airflow dags trigger data_prep_dag
```

The Graph and Gantt views in the Airflow UI show the two shapes; per-task durations are in
`reports/run_summary.json` after each run.

## 9. Alerts

Every alert goes through `alerts.send_alert(level, title, details)`, where level is
`"warn"` or `"fail"`. **With no `SLACK_WEBHOOK_URL` set, alerts are only logged** and
nothing fails, which is why a grader or a teammate without Slack can run the pipeline
unchanged.

To turn on Slack, create an Incoming Webhook for your channel
(<https://api.slack.com/messaging/webhooks>) and put the URL in `.env`:

```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T000/B000/xxxx
ALERT_EMAIL=                      # optional
```

Restart so the containers pick it up: `docker compose up -d`.

**What posts a message**

| Trigger | Level |
|---|---|
| Any task failure, through `on_failure_callback` on every task | `fail` — DAG id, task id, error, log link |
| `anomalies` warn-level checks | `warn` — one message listing every warning |
| `dvc push` failure | `warn` — the data is still versioned locally |
| End of run, from `notify_success` | `ok`, or `warn` if any stage was still a stub |

A broken webhook never fails the task that called it: `send_alert` catches its own errors
and falls back to logging. Failures are posted per task, but success is posted once per
run, so a green run produces one message rather than eleven.

