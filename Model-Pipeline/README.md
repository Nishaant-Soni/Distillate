# Distillate: Model Pipeline

This README documents Phase 2 of Distillate (the model pipeline): teacher labels, the student's training, evaluation, the gate, the model registry and CI. The data pipeline is documented in `Data-Pipeline/README.md`.

## Contents

In pipeline order; each lane fills in its section.

1. Teacher, soft labels and sensitivity analysis (Nishaant)
2. Teacher labels, data loading and EDA (Praniti)
3. Student training and tuning (Shivam)
4. Evaluation, gate and bias (Ruchita)
5. CI/CD, registry and rollback (Vaishnavi)
6. Monitoring preparation (Sai)

## 1. Teacher, soft labels and sensitivity analysis (Nishaant)

The teacher lane labels every training comment with soft probabilities for the six labels, which the student is distilled from. The labels come from a *stack*: the zero-shot LLM teacher `Qwen/Qwen3-4B-Instruct-2507` (pinned revision `cdbee75f…`) combined with a TF-IDF baseline trained on the gold labels, so they are **a gold-trained baseline corrected by Qwen**, not independent LLM judgments. The model card's section 3 says what the labels are and how good they are; `docs/teacher_prompts_and_shard_spec.md` is the full contract (prompts, shard format, checks, paths). Paths here are relative to `Model-Pipeline/` unless they start with `Data-Pipeline/`.

### 1.1 What the other lanes use

| You want | Use | Where |
|---|---|---|
| The student's targets | `p_<label>` from `soft_labels/stack-v1/p2-zs-v2b/` | `gs://distillate-mlops-data/` (Hugging Face `distillate-team/distillate-teacher-shards` is a backup copy, same paths) |
| The same, joined onto the splits | `teacher_labels.parquet` (`p_<label>`, `p_<label>_raw`, `tfidf_oof_<label>`, `p_<label>_calqwen`) | `Data-Pipeline/data/processed/`, built by the data pipeline's `merge_teacher_labels`, read with `src/data/load_data.py` (section 2) |
| The ablation's attribution arms | calibrated Qwen: `soft_labels/calqwen-v1/p2-zs-v2b/`; TF-IDF out-of-fold: `tfidf/tfidf-lr-v1-oof/tfidf_oof.parquet` | same bucket |
| Teacher and soft labels for the gate subset and the Civil drift stream | `…/gate/` and `…/drift/` under each prefix | same bucket |
| Gate subset weights (gate-subset metrics must be weighted) | `artifacts/gate_subset_ids.parquet` (`weight`, `stratum`) | this repository |

Never train on the `p_*` of the raw `teacher/` shards: they are uncalibrated.

### 1.2 Results

| | |
|---|---|
| Rows labeled | 159,571 training comments, 20,000 test comments (the gate subset), 40,000 Civil drift comments; 0 failed rows; about 9.4 T4 GPU-hours |
| Held-out quality, gate subset (weighted mean AUC / PR-AUC / ECE) | stack **0.982 / 0.659 / 0.010**; TF-IDF 0.980 / 0.623 / 0.010; raw Qwen 0.959 / 0.475 / 0.086 |
| Civil drift stream (five labels) | stack 0.924 mean AUC, TF-IDF 0.887, raw Qwen 0.882; the stack is best in all 20 batches |
| What the LLM adds to the student (Phase 2 ablation, DistilBERT, same loss) | stack soft labels against TF-IDF soft labels: **+0.0022** mean AUC on the gate set, **+0.033** on Civil H2. Calibrated Qwen alone is not a better teacher than TF-IDF (−0.0043 on the gate set) |

**Validation caveat:** the TF-IDF out-of-fold models saw the validation rows' gold labels, so students trained on soft labels score optimistically on validation (by about 0.003 to 0.006 mean AUC against gold-only). Gold-only and soft-label arms are compared on the gate set and H2; the selection compares only arms with the same teacher. Details: model card section 3.

### 1.3 Setup

1. **Python 3.12 environment:** `pip install -r requirements-dev.txt` from `Model-Pipeline/`.
2. **Read access to the bucket** (from Vaishnavi), then `gcloud auth application-default login`, which the validator and the store code use. On Kaggle, a `GCS_TEACHER_KEY` secret (a service-account key) instead.
3. **Optional, for the Hugging Face backup:** membership of the `MLOps-Distillate-Project` organization and a read token via `hf auth login` (or `HF_TOKEN`). Tokens and keys go in secrets or the environment, never in code or notebooks.
4. **Local data copy** (git-ignored, it holds comment text): `data/wiki_train_raw.parquet` and `data/wiki_test_raw.parquet`, made by `kaggle/00_prepare_data.ipynb`.

### 1.4 How the labels were made

Every step reads and writes committed records (`artifacts/`), so its result can be checked without rerunning it. GPU steps ran on Kaggle (two T4s); the rest run on a laptop CPU.

| Step | Command or notebook | Writes |
|---|---|---|
| 1. Pilot sample and gate subset | `python -m src.teacher.sampling pilot`, `python -m src.teacher.sampling gate` | `artifacts/pilot/pilot_ids.parquet`, `artifacts/gate_subset_ids.parquet` |
| 2. TF-IDF baseline on the pilot | `python -m src.teacher.baseline_tfidf` | `artifacts/pilot/metrics_tfidf.json` |
| 3. Pilot: prompts compared, calibration, prompt-injection test (GPU) | `kaggle/01_smoke_test.ipynb`, then `kaggle/02_pilot.ipynb` | `artifacts/pilot/` |
| 4. Obscene prompt follow-up | `python -m src.teacher.prompt_tuning halves`, a GPU run, then `python -m src.teacher.prompt_tuning evaluate` | `artifacts/prompt_v2/` |
| 5. Civil Comments shift check | `python -m src.teacher.shift sample`, then `evaluate` | `artifacts/shift/` |
| 6. Out-of-fold TF-IDF scores (about 20 min CPU) | `python -m src.teacher.tfidf_oof` | `artifacts/oof/` and `tfidf/tfidf-lr-v1-oof/` in the store |
| 7. Label the shards (GPU): `RUN = "train"` (about 3.6 h), `"gate"` (about 45 min), `"drift"` | `kaggle/label_shards.ipynb`, pinned to a commit (`REPO_REF`); reruns resume | `teacher/p2-zs-v2b/` (and `gate/`, `drift/`) in the store |
| 8. Stack and soft labels | `python -m src.teacher.soft_labels [--upload]`; `--drift` for the drift stream | `artifacts/stack/`, `soft_labels/stack-v1/p2-zs-v2b/` |
| 9. Calibrated-Qwen labels (ablation arm) | `python -m src.teacher.calqwen [--upload]` | `artifacts/calqwen/`, `soft_labels/calqwen-v1/p2-zs-v2b/` |

`--upload` refuses to run unless `src/`, `configs/` and the step's `artifacts/` folder are committed, so every shard's manifest names a code commit that exists. Without `--upload`, steps 8 and 9 recompute and check everything but write nothing to the store.

### 1.5 Checking the shards

Every shard has a manifest (row count, id hash, prompt and model versions, code commit). The validator checks a run against the spec, including that every id is present exactly once:

```bash
python -m src.teacher.validate_local --prefix soft_labels/stack-v1/p2-zs-v2b/ \
    --ids data/wiki_train_raw.parquet --plan artifacts/teacher_shards.json \
    --model-version "stack-v1+Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554+tfidf-lr-v1"
```

The commands for all eight runs are in the spec (§8); `--backend hf` checks the backup copy. The data pipeline's `validate_teacher_shards` runs the same checks on all eight, plus the stack and calibration fingerprints, before merging the labels into `teacher_labels.parquet` (section 2.1).

### 1.6 Tracking

The lane's runs are in MLflow on the team server, experiment `distillate-teacher` (13 runs): the pilot with one child run per configuration compared, the three labeling runs, and both soft-label sets. They are logged from the committed records and manifests, so nothing is rerun (`src/teacher/mlflow_log.py`, PR #27):

```bash
python -m src.teacher.mlflow_log --dry-run   # what would be logged
python -m src.teacher.mlflow_log             # MLFLOW_TRACKING_URI if set, else mlruns/
```

Rerunning skips any record already logged with the same content.

### 1.7 Sensitivity analysis

How the selected student (`distilbert-mixed-5ep`, chosen by the student lane's selection rule, section 3) depends on its hyperparameters, which words drive its predictions, and how it holds up on short inputs and obfuscated text, all on Wikipedia validation (`src/sensitivity/`, model card section 7):

1. **Hyperparameter importance** (fANOVA and MDI) from the merged Optuna study `distilbert-mixed-v1` (28 trials), each recomputed under 11 seeds. Maximum length explains the most (fANOVA 0.548, range 0.367 to 0.548), then epochs (0.238) and α (0.141); learning rate and T explain little. The objective varies little (validation mean AUC 0.988 to 0.9936).
2. **Token attributions** (SHAP Partition explainer) per label for caught, missed and wrongly flagged comments, reported as top words and masked examples, never comment text.
3. **Input length**: validation rescored at 64, 128 and 256 tokens, and results by token-length band.
4. **Obfuscation strength**: the data pipeline's character substitutions at rates 0 to 1.

```bash
python -m src.sensitivity.run --model-dir <selected package> --rev <commit>
python -m src.sensitivity.run --model-dir <dir> --sample 2000 --no-mlflow   # a quick look
```

`optuna.storage` and `optuna.study_name` in `configs/sensitivity.yaml` point at the study (`sqlite:///` of `gs://distillate-mlops-data/optuna/distilbert-mixed-v1.db`, downloaded). The report goes to `reports/sensitivity.md`, and the run to MLflow experiment `distillate-sensitivity`. **Status:** item 1 is done; items 2 to 4 run once the selected student's package is available.

### 1.8 Tests

```bash
cd Model-Pipeline
pytest tests/teacher       # no network or GPU: synthetic data and fixtures
pytest tests/sensitivity   # with PR D
ruff check src tests
```

CI (`.github/workflows/model-pipeline-ci.yml`) runs `ruff check src tests` and the whole `pytest` suite on every pull request that touches `Model-Pipeline/`.

### 1.9 Code

| Module | What it does |
|---|---|
| `src/teacher/config.py`, `data.py`, `compat.py` | `configs/teacher.yaml` and the shared names; loading and light cleaning of the raw Wikipedia data; workarounds for the Kaggle GPU image |
| `src/teacher/prompts.py`, `scoring.py` | the prompts and `prompt_version`; reading yes/no probabilities and log-odds from the token log-probabilities |
| `src/teacher/sampling.py` | the pilot sample and the gate subset, with inclusion weights |
| `src/teacher/baseline_tfidf.py`, `tfidf_oof.py` | the TF-IDF baseline and its out-of-fold scores |
| `src/teacher/pilot.py`, `calibrate.py`, `injection.py`, `metrics.py` | the pilot's evaluation, Platt calibration, the prompt-injection test, weighted metrics |
| `src/teacher/shift.py`, `prompt_tuning.py` | the Civil Comments shift check and the obscene prompt follow-up |
| `src/teacher/label_worker.py`, `shards.py`, `storage.py` | GPU labeling, shard plan and manifests, the GCS and Hugging Face stores |
| `src/teacher/validate_local.py` | the shard checks |
| `src/teacher/stack.py`, `soft_labels.py`, `calqwen.py` | the stack and the two soft-label sets |
| `src/teacher/mlflow_log.py` | the runs in MLflow (PR #27) |
| `src/sensitivity/` | hyperparameter importance, SHAP attributions, length and obfuscation sweeps, the report (PR D) |

## 2. Teacher labels, data loading and EDA (Praniti)

Everything the student trains and is evaluated on goes through three steps: the teacher's shards are **validated**, **merged** into one table, and **loaded** with the processed splits by a single function. An EDA notebook compares the teacher's labels with the gold labels, and `drift_ingest_dag` starts the Phase 3 drift stream.

### 2.1 Validating the teacher's shards (`Data-Pipeline/scripts/validate_teacher_shards.py`)

Before anything reads the teacher's output, the stage checks every run in `gs://distillate-mlops-data` (Hugging Face `distillate-team/distillate-teacher-shards` is the backup) against the teacher shard spec (`Model-Pipeline/docs/teacher_prompts_and_shard_spec.md`, section 6). It reuses the teacher lane's own validator (`src.teacher.validate_local`), so the checks have one implementation.

| Run | Shards | Rows |
|---|---|---|
| raw teacher: train / gate / drift | 32 / 4 / 20 | 159,571 / 20,000 / 40,000 |
| stack soft labels: train / gate / drift | 32 / 4 / 20 | 159,571 / 20,000 / 40,000 |
| calibrated Qwen: train / gate | 32 / 4 | 159,571 / 20,000 |

All eight runs pass with 0 errors. On top of the spec's checks, every manifest's fingerprint must match the committed frozen artifact: `stack_sha256` against `artifacts/stack/stack_p2-zs-v2b.json`, and `calibration_sha256` against `artifacts/pilot/calibration_p2-zs-v2b.json` (hashed with line endings normalized, so a Windows checkout gives the same hash). Training runs are checked against the committed shard plan, and drift runs against `Data-Pipeline/data/splits/civil_drift_ids.csv`, one shard per batch.

The result is written to `Data-Pipeline/reports/teacher/validation.json`. Any error fails the stage with `TeacherShardError`. If the store can't be reached (no credentials, as on a fresh clone or in CI), the stage warns and skips instead of failing the DAG; `teacher.required: true` makes that a failure. A missing package fails loudly rather than counting as an unreachable store. All settings are in the `teacher:` section of `Data-Pipeline/config/pipeline.yaml`.

### 2.2 The merged table (`Data-Pipeline/scripts/merge_teacher_labels.py`)

`Data-Pipeline/data/processed/teacher_labels.parquet` has one row per raw Wikipedia training id (159,571), joined by id from four sources, and is versioned with `Data-Pipeline/data/processed.dvc`:

| Columns | Source | Use |
|---|---|---|
| `p_<label>` | stack-v1 soft labels | **the student's targets** |
| `p_<label>_raw` | raw Qwen | audit; never a training target |
| `tfidf_oof_<label>` | out-of-fold TF-IDF | the TF-IDF ablation arm |
| `p_<label>_calqwen` | calibrated Qwen | the attribution arm |
| `prompt_version`, `model_version`, `teacher_version` | the soft labels | lineage |

The merge runs only if validation passed (it reads the validation report), checks the TF-IDF file against its committed manifest (SHA-256, row count, id hash, version), requires every source to hold exactly the same ids once each, and fails without writing anything if any training row lacks a soft label. There are no gate or drift rows. Sanity check: the mean `p_toxicity` is 0.096, the same as the gold rate. The columns are also in `Data-Pipeline/CONTRACTS.md`.

### 2.3 Loading data (`src/data/load_data.py`)

One call returns the tables, with the teacher labels joined onto train and validation by id (rows dedupe removed drop out):

```python
from src.data.load_data import load_data

data = load_data(tables=("train", "val"))   # training: only what it needs
data = load_data()                          # every table
data = load_data(rev="<git commit>")        # the data exactly as that commit versioned it, through DVC

data["train"], data["val"]      # gold <label> (0/1), targets p_<label>, sample_weight, the arms' columns
data["gate"], data["h2"], data["bias"], data["obfuscation"]   # evaluation tables, no teacher columns
data.train_data_hash            # SHA-256 over train, val and teacher_labels: log it with data.rev to MLflow
data.data_hash                  # SHA-256 over every file read
data.teacher_versions           # prompt / model / teacher versions
```

| Table | Rows |
|---|---|
| train / val | 132,388 / 21,363 (train + val + the 2,000-row canary = the 155,751 deduplicated rows) |
| gate | 63,978 |
| h2 / bias | 10,000 / 21,577 |
| obfuscation | 3,000 |

`train_data_hash` covers only what training reads, so changing an evaluation table doesn't change a training run's lineage. Reading through DVC at a commit gives the same hash as reading the same files from disk. Loading stops with a clear error if any train or validation row is missing any teacher column (a count per column is reported), and a missing file raises `FileNotFoundError` with or without `rev=`.

**Setup.** `load_data()` reads `Data-Pipeline/data/processed/`, so get the data first, from `Data-Pipeline/`:

```bash
gcloud auth application-default login   # once, with an account that has bucket access
dvc pull
```

`rev=` also needs DVC in the environment (`pip install "dvc[gs]"`). The Google Cloud Storage client needs a project: set `GOOGLE_CLOUD_PROJECT` to the team project if it complains that no project could be determined. On macOS with the python.org Python, DVC may report "object ... is corrupted" when the real cause is an SSL certificate failure; this fixes it:

```bash
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
```

Linux, Docker and Kaggle aren't affected.

### 2.4 EDA: teacher vs gold (`notebooks/eda_teacher_vs_gold.ipynb`)

Compares the four score sources with the gold labels on train + validation (153,751 rows). Figures are in `docs/images/eda/`; the notebook shows no comment text.

- **Ranking:** the stack has the highest AUC and PR-AUC on every label (mean AUC 0.990; TF-IDF 0.985; raw Qwen 0.974), matching the teacher lane's reported results. Its lead is largest on rare labels (threat PR-AUC 0.589 vs 0.272 for raw Qwen).
- **Calibration:** the stack is the best calibrated source overall (ECE 0.003 vs 0.076 for raw Qwen). Raw Qwen over-predicts every label except obscene (toxicity 0.224 vs a gold rate of 0.096).
- **Length:** longer comments are harder for every source; raw Qwen's false-alarm rate at 0.5 doubles from short to long comments (10.1% to 20.1%).
- **Identity mention:** the stack's ranking holds up best on comments that mention an identity term (AUC drop 0.016 vs 0.057 for raw Qwen). At thresholds matched to the same overall false-alarm rate, every source flags clean identity-mentioning comments 2.3–4.1× as often as other clean comments (at 1%: stack 2.9×, raw Qwen 3.7×). The stack's skew is somewhat smaller than raw Qwen's, but substantial for every source, so the student's bias check should watch this slice. (A fixed 0.5 comparison would mix calibration with bias.)
- **Truncation (distilbert-base-uncased):** 18.3% of comments exceed 128 tokens (11.2% of toxic ones) and 6.4% exceed 256 (4.1%), consistent with the sweep's choice of `max_length = 256`.

### 2.5 Drift ingestion skeleton (`Data-Pipeline/dags/drift_ingest_dag.py`)

`drift_ingest_dag` releases the Civil drift stream (20 time-ordered batches of 2,000) one batch per run, as six tasks:

![drift_ingest_dag in the Airflow Graph view](../Data-Pipeline/docs/images/drift_ingest_graph.png)

| Task | Phase 2 (now) | Phase 3 |
|---|---|---|
| `release_batch` | checks batch k's schema, size and ids against `civil_drift_ids.csv`; once all 20 are done, evaluates the last batch (once) and skips the rest | same |
| `compute_drift` | basic statistics (stub) | `src/monitoring/drift.py` (PSI) |
| `score_batch` | `POST /predict` (v4 §2.4) against a mock with the same response format; stores ids, text hashes and scores, never text | the live endpoint (`drift_ingest.service_url`), with batched or parallel requests and per-request retries |
| `evaluate_previous` | AUC and F1 per label for the previous batch, with its delayed gold labels | same |
| `merge_overrides` | stub | moderation-queue overrides from the GCS feedback log |
| `finish_batch` | writes the batch record and moves the stream on | same, with the stream state kept in GCS |

The DAG is created paused and is scheduled only when `drift_ingest.enabled` is true (`schedule: demo` every 5 minutes, `real` weekly), so it never advances the stream by itself. Run one batch outside Airflow with `python -m scripts.drift_ingest` (from `Data-Pipeline/`). Outputs go to `Data-Pipeline/reports/drift/` (git-ignored). Rare labels have very few positives per batch (for example 4 threats in 2,000), so per-batch AUCs for them are noisy.

### 2.6 Tests

- `Data-Pipeline/tests/test_validate_teacher_shards.py` (8) and `test_merge_teacher_labels.py` (7): the teacher lane's fixtures (valid shards pass, a broken one gives exactly one error), wrong fingerprints, unreachable store and missing package, the contract table, TF-IDF manifest mismatches, missing soft labels, and failed, skipped or missing validation reports.
- `Model-Pipeline/tests/data/test_load_data.py`: every table, teacher columns only on train/val, missing teacher scores per column, `tables=`, the two hashes, a missing file, and `rev=` through a stubbed DVC.
- `Data-Pipeline/tests/test_drift_ingest.py` (8): one batch per run and evaluation of the previous one, the last batch evaluated once the stream is finished, no text in predictions, the §2.4 response format, flags and `low_confidence`, wrong ids, missing columns, a missing batch file.

All use small made-up data and no network.
