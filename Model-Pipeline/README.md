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
| What the LLM adds to the student (Phase 2 ablation, DistilBERT, same loss) | stack soft labels against TF-IDF soft labels (paired bootstrap): **+0.0022** [+0.0017, +0.0027] mean AUC on the gate set, **+0.033** [+0.024, +0.044] on Civil H2. Calibrated Qwen alone is not a better teacher than TF-IDF (−0.0043 on the gate set) |

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

1. **Hyperparameter importance** (fANOVA and MDI) from the merged Optuna study `distilbert-mixed-v1` (28 trials), each recomputed under 11 seeds. Maximum length explains the most and ranks first under every seed (fANOVA 0.37 to 0.55 over the seeds), then epochs (0.24 to 0.34) and α (0.10 to 0.23); learning rate and T explain little. The objective varies little (validation mean AUC 0.988 to 0.9936).
2. **Token attributions** (SHAP Partition explainer) per label for caught, missed and wrongly flagged comments, reported as top words and masked examples, never comment text. Identity words drive false alarms (for identity_attack: a racial slur, "gay", "nazi", "racist"), consistent with the bias report's race-slice finding but not an explanation of it (model card section 7.2); missed toxic comments are mostly implicit, such as insults without profanity.
3. **Input length**: validation rescored at 64, 128 and 256 tokens, and results by token-length band. 256 tokens truncates 6.4% of comments at little cost (mean AUC 0.994); 64 tokens gives 0.990 overall but drops comments of 129 to 256 tokens from 0.993 to 0.970.
4. **Obfuscation strength**: the data pipeline's character substitutions at rates 0 to 1. The clearest weakness: changing 4.3% of characters makes the student miss 14.5% of the toxic comments it caught when clean, and 21% makes it miss 54% (threat 98%, 44 of 45). The flag rate falls too, so obfuscated comments pass as clean.

```bash
python -m src.sensitivity.run --model-dir <selected package> --rev <commit>
python -m src.sensitivity.run --model-dir <dir> --sample 2000 --no-mlflow   # a quick look
```

`optuna.storage` and `optuna.study_name` in `configs/sensitivity.yaml` point at the study (`sqlite:///` of `gs://distillate-mlops-data/optuna/distilbert-mixed-v1.db`, downloaded). The report goes to `reports/sensitivity.md`, and the run to MLflow experiment `distillate-sensitivity`. **Status:** run on 2026-11-13 on all 21,363 validation rows (DVC revision `5808fdf`, 3 h 20 min on CPU); the report has plots and hand-written findings (`reports/sensitivity_findings.md`).

### 1.8 Tests

```bash
cd Model-Pipeline
pytest tests/teacher       # no network or GPU: synthetic data and fixtures
pytest tests/sensitivity
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
| `src/sensitivity/` | hyperparameter importance, SHAP attributions, length and obfuscation sweeps, the report (PR D, #46) |

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

### 2.5 Drift ingestion (`Data-Pipeline/dags/drift_ingest_dag.py`)

`drift_ingest_dag` releases the Civil drift stream (20 time-ordered batches of 2,000) one batch per run, as six tasks:

![drift_ingest_dag in the Airflow Graph view](../Data-Pipeline/docs/images/drift_ingest_graph.png)

| Task | Phase 2 (now) | Phase 3 |
|---|---|---|
| `release_batch` | checks batch k's schema, size and ids against `civil_drift_ids.csv`; once all 20 are done, evaluates the last batch (once) and skips the rest | same |
| `compute_drift` | PSI of length, out-of-vocabulary rate and non-alphabetic share against the Wikipedia-train baseline (`Data-Pipeline/scripts/drift.py`, section 6.1) | adds a second reference from the first Civil batch |
| `score_batch` | `POST /predict` (v4 §2.4) against a mock with the same response format; stores ids, text hashes and scores, never text; plus flag-rate drift (status unknown until a Civil reference rate is set) | the live endpoint (`drift_ingest.service_url`), with batched or parallel requests and per-request retries |
| `evaluate_previous` | AUC and F1 per label for the previous batch, with its delayed gold labels | same |
| `merge_overrides` | stub | moderation-queue overrides from the GCS feedback log |
| `finish_batch` | writes the batch record and moves the stream on | same, with the stream state kept in GCS |

The DAG is created paused and is scheduled only when `drift_ingest.enabled` is true (`schedule: demo` every 5 minutes, `real` weekly), so it never advances the stream by itself. Run one batch outside Airflow with `python -m scripts.drift_ingest` (from `Data-Pipeline/`). Outputs go to `Data-Pipeline/reports/drift/` (git-ignored). Rare labels have very few positives per batch (for example 4 threats in 2,000), so per-batch AUCs for them are noisy. The mock's scores carry no signal, so with six labels each flagged about half the time it flags about 98% of comments (1 − 0.5⁶): a plumbing artefact, not a real flag rate.

On the real data, batch 0 gives input drift `warn` (length PSI 0.013, OOV 0.083, non-alpha 0.137): the Wikipedia-to-Civil shift described in section 6.1, not a change within the stream.

### 2.6 Tests

- `Data-Pipeline/tests/test_validate_teacher_shards.py` (8) and `test_merge_teacher_labels.py` (7): the teacher lane's fixtures (valid shards pass, a broken one gives exactly one error), wrong fingerprints, unreachable store and missing package, the contract table, TF-IDF manifest mismatches, missing soft labels, and failed, skipped or missing validation reports.
- `Model-Pipeline/tests/data/test_load_data.py`: every table, teacher columns only on train/val, missing teacher scores per column, `tables=`, the two hashes, a missing file, and `rev=` through a stubbed DVC.
- `Data-Pipeline/tests/test_drift_ingest.py` (12): one batch per run and evaluation of the previous one, the last batch evaluated once the stream is finished, input and flag-rate drift recorded, a missing drift baseline, zero latency kept, no text in predictions, the §2.4 response format, flags and `low_confidence`, wrong ids, missing columns, a missing batch file.

All use small made-up data and no network.

## 3. Student training and tuning (Shivam)

The student is a small encoder with six sigmoid outputs, one per label. It is distilled from the stack's soft labels (section 1) and the gold labels, with the mixed recipe tuned once in an Optuna study. Eight arms are then trained under identical conditions, and one is selected by a fixed rule. The details are in four documents:

| Document | What it covers |
|---|---|
| [`docs/student.md`](docs/student.md) | The training harness: the loss, data and split, run outputs, thresholds and the ONNX model package |
| [`docs/tuning.md`](docs/tuning.md) | The Optuna study: search space, sampler, budget, the two workers' merged study, the best values and the 5-epoch check |
| [`docs/pipeline.md`](docs/pipeline.md) | The single entry point `python -m src.pipeline.run`, the selection rule, the results across arms and where it all runs (Kaggle, Docker, locally) |
| [`reports/student/phase2/README.md`](reports/student/phase2/README.md) | The reported run: every arm on every evaluation set, the selection and the findings |

### 3.1 Training (`src/student/`)

```text
loss = α · BCE(student, gold) + (1 − α) · T² · BCE(student / T, stack / T)
```

The gold term carries `sample_weight`, and rows with a failed soft label are left out of the soft term. Each run keeps its best epoch on validation and tunes one threshold per label on validation (the best F1). It then builds the ONNX package (`model.onnx`, `tokenizer/`, `thresholds.json`, `meta.json` with lineage and file hashes) and stops if ONNX and PyTorch differ by more than 1e-3. Everything is logged to MLflow. Settings are in `configs/student.yaml`.

### 3.2 Tuning (`src/student/sweep.py`, `kaggle/11_student_runs.ipynb`)

One Optuna study (TPE) tuned α, T, learning rate, epochs and maximum length on DistilBERT with the mixed loss. The objective was validation mean AUC; the gate set was never read. The study ran 28 trials over two workers on Kaggle T4s (Shivam and Nishaant) and is stored at `gs://distillate-mlops-data/optuna/distilbert-mixed-v1.db`.

**Best values (trial 26):** α 0.343, T 3.71, learning rate 9.23e-5, 4 epochs, 256 tokens; validation mean AUC 0.99364. 256 tokens won clearly (the top 21 trials). The top three trials are within 0.00004 AUC of each other, so the best values mark a region rather than one point. Every trial peaked at its last epoch, so a check run at 5 epochs was added. It gained +0.0002 on validation (within the one-seed noise) and nothing on the gate set or H2.

### 3.3 The reported run (`kaggle/13_student_ablation_all.ipynb`)

All eight arms ran through `src.pipeline.run` with the tuned values in one Kaggle session on T4 x2 (5.9 hours, code `80a4502`, seed 42, the official split). Each arm was trained, then scored on validation, the gate set (H1), Civil H2, the Civil bias set and the obfuscated comments, then checked for bias, timed on CPU and plotted. Mean AUC with 95% bootstrap intervals:

| Arm | Validation | Gate (H1) | Civil (H2) |
|---|---|---|---|
| gold only (α = 1) | 0.98937 | 0.9841 [0.9831, 0.9849] | 0.9139 [0.8993, 0.9285] |
| stack only (α = 0) | 0.99303 | 0.9835 [0.9826, 0.9842] | 0.9359 [0.9225, 0.9471] |
| mixed (the tuned recipe) | 0.99348 | 0.9848 [0.9840, 0.9854] | 0.9410 [0.9280, 0.9510] |
| TF-IDF soft labels | 0.99426 | 0.9826 [0.9815, 0.9835] | 0.9081 [0.8895, 0.9252] |
| calibrated-Qwen soft labels | 0.98834 | 0.9783 [0.9771, 0.9793] | 0.9206 [0.9055, 0.9350] |
| **mixed, 5 epochs (selected)** | 0.99368 | 0.9845 [0.9836, 0.9852] | 0.9399 [0.9263, 0.9502] |
| MiniLM-L12, mixed | 0.99187 | 0.9849 [0.9841, 0.9857] | 0.9438 [0.9336, 0.9524] |
| ModernBERT-base, mixed | 0.99349 | 0.9846 [0.9839, 0.9853] | 0.9384 [0.9233, 0.9498] |

- **Distillation (H2):** mixed vs gold only, +0.027 [+0.013, +0.041] on Civil. On the gate set the AUC gain is small (+0.0007) and within the effect of the shared recipe; the PR-AUC gain (+0.064) is large.
- **What the LLM adds:** stack vs TF-IDF soft labels, both with the mixed loss at the tuned α (`distilbert-mixed` − `distilbert-tfidf`, paired bootstrap): +0.033 [+0.024, +0.044] on H2 and +0.0022 [+0.0017, +0.0027] on the gate set (section 1.2, model card section 3).
- **Every arm passes the bias check.**
- **The thresholds don't transfer to Civil:** H2 AUC is 0.94, but toxicity recall at the validation thresholds is 38%. Phase 3 sets thresholds for news-comment traffic.
- **Latency:** no fp32 package meets the 50 ms p95 budget (DistilBERT 144 ms, MiniLM 103 ms). The budget applies to the int8 package in Phase 3.

### 3.4 Selection (`src/student/select_best.py`)

The rule: among the candidates that pass the bias check, the highest validation mean AUC, with runs within 0.0002 counted as tied and the tie broken by PR-AUC. The gate set and H2 never choose. Two settings were decided after the run and are recorded in `configs/student.yaml`:

1. Only the tuned recipe competes: DistilBERT at 4 and 5 epochs, MiniLM-L12 and ModernBERT. The other arms are the ablation's controls.
2. Latency is reported, not enforced.

**Selected: `distilbert-mixed-5ep`** (`student-distilbert-mixed-5ep-20261112T210931Z`, MLflow run `ab34619a27e84b07aae044e3a414c0be` on the team server). It ties ModernBERT on validation AUC (within 0.0002) and wins on PR-AUC (0.7678 vs 0.7615); DistilBERT at 4 epochs is just outside the tie and MiniLM 0.0018 lower. The top three are within noise of each other, so the choice is the rule's, not a measured difference. Its package and every arm's prediction files are in `gs://distillate-mlops-data/models/student-distilbert-mixed-5ep-20261112T210931Z/` (`selected_package/` and `phase2_results.zip`). The model card's section 4 describes the training in full.

### 3.5 Running it

```bash
# from Model-Pipeline/, offline on the tiny model
python -m src.student.synthetic --out outputs/synthetic
python -m src.student.train --config configs/smoke.yaml
pytest tests/student tests/pipeline

# one arm through the whole pipeline (needs the data, section 2.3)
python -m src.pipeline.run --set run_name=distilbert-mixed

# the training image, from the repository root
docker build -f Model-Pipeline/Dockerfile -t distillate-train .
```

The image pins every version (`requirements-pipeline.txt`, CUDA 12.6 torch). CI builds it whenever it can change and runs `src.pipeline.container_check` and `tests/pipeline` inside it (`.github/workflows/model-pipeline-image.yml`).

### 3.6 Tests

`tests/student/` and `tests/pipeline/` (100 tests): the loss, thresholds, the package and its parity check, the sweep's shared-study guard, MLflow logging, the evaluation tables, the selection rule, and every pipeline step end to end on a tiny model. All run offline with made-up data.

## 4. Evaluation, gate and bias (Ruchita)

Every model is scored the same way: a prediction file (`id`, `p_<label>`, `model_version`, CONTRACTS §2.2) and the gold labels go into one evaluation harness, the **gate** decides whether a candidate may be registered, and the **bias analysis** checks identity slices on the Civil bias set. Every student arm goes through the same code, and its metric definitions match the teacher's (4.1), so the numbers are comparable.

| Document | What it covers |
|---|---|
| [`reports/student/phase2/results.md`](reports/student/phase2/results.md) | H1 and H2 tables for every arm, paired differences, robustness and latency, from the harness |
| [`configs/gate.yaml`](configs/gate.yaml) | The gate's checks and tolerances, with where each value comes from |
| [`reports/bias/bias_report_phase2.md`](reports/bias/bias_report_phase2.md) | The Phase 2 bias report: per-group metrics, paired comparisons, the gate verdict and the mitigation decision |
| [`../docs/model_card.md`](../docs/model_card.md) | Sections 6 (results), 8 (gate) and 9 (bias) summarise all three |

### 4.1 Evaluation harness (`src/eval/metrics.py`)

Per label: ROC-AUC, PR-AUC, F1, precision, recall and FPR at the tuned threshold, ECE and Brier. Across labels: mean AUC, mean PR-AUC and macro-F1, plus the flag decision (any label at or above its threshold) against gold "any label". Intervals come from 1,000 bootstrap resamples, and `paired_bootstrap` compares two models on the same resampled rows.

The definitions match the teacher's (`src/teacher/metrics.py`), and `tests/eval/test_metrics.py -k reference` reproduces the teacher's gate-subset numbers to within 2e-5. On the gate subset, rows are weighted by their inclusion weight and resampled within stratum; AUC and PR-AUC rank on `logit_<label>` when a file has it; rows that aren't `ok` are dropped and reported as the failure rate. Civil Comments metrics use five labels (no severe_toxicity).

### 4.2 Gate (`src/eval/gate.py`, `configs/gate.yaml`)

A candidate is registered only if the gate passes (exit code 0); it writes `gate_report.json` and `gate_report.md`. Three checks:

1. **Validation floors:** mean AUC at least 0.98, and a PR-AUC floor per label (0.86 toxicity, 0.87 obscene, 0.76 insult, 0.38 severe_toxicity, 0.45 threat, 0.45 identity_attack): the gold-only student's validation PR-AUC minus 0.05, or 0.10 for the rare labels.
2. **No regression** against the production model on the gate set: mean AUC may drop at most 0.005, and PR-AUC at most 0.02 (toxicity, obscene, insult) or 0.05 (the rare labels), set above the measured bootstrap noise. Skipped while there is no production model.
3. **Bias:** on the Civil bias set, no identity slice (n ≥ 100) more than 10 points of FPR above overall, or rising more than 2 points over production. The bias report flags gaps from 5 points.

The selected student passes ([`reports/student/phase2/runs/distilbert-mixed-5ep/gate_report.md`](reports/student/phase2/runs/distilbert-mixed-5ep/gate_report.md)): every floor is met and 0 of 18 judged slices are over a limit.

### 4.3 Bias analysis (`src/eval/bias.py`)

For each arm, per identity attribute and group on the Civil bias set (gender, sexual orientation, religion, race or ethnicity, disability): subgroup, BPSN and BNSP AUC, and FPR, FNR and F1 at the tuned threshold. The Wikipedia-tuned thresholds give a low overall Civil FPR (about 2%), so the analysis also reports the FPR ratio, compares arms by BPSN AUC (threshold-free) and at a matched 5% overall FPR, and gives paired bootstrap intervals for the differences. The code also covers Wikipedia lexicon slices (identity mention, length in characters, obfuscation flag), but they weren't computed in Phase 2: the final gold validation file lacks the derived columns (a Phase 3 follow-up).

**Findings** (selected student vs gold-only, paired BPSN AUC):

- **Soft labels raise false positives on Black and White identity slices:** white −0.053, black −0.039, race:any −0.041 [−0.051, −0.032]. Every soft-label arm shows it, including TF-IDF, so it isn't specific to the LLM.
- **Sexual orientation improves:** +0.026 [+0.007, +0.042]. Religion and disability are unchanged.
- **Mitigation:** no variant in Phase 2 (Phase 1's weights are all 1.0, and choosing a new factor would tune on the bias set). Phase 3 monitors race-slice FPR and runs a proper mitigation experiment.

### 4.4 Running it

```bash
# from Model-Pipeline/: score one prediction file (gate subset: weighted, with intervals)
python -m src.eval.metrics --pred <predictions>.parquet --gold <gold>.parquet \
  --thresholds <run>/thresholds.json --design artifacts/gate_subset_ids.parquet \
  --bootstrap 1000 --out metrics.json

# the gate on a candidate's prediction folder (val, gate, bias_eval + thresholds.json)
python -m src.eval.gate --config configs/gate.yaml --candidate <preds dir> \
  --gold <gold dir> [--production <prod preds dir>] --out <out dir>

# the bias analysis across arms (the exact command for the report is in its header)
python -m src.eval.bias --gold <gold dir> --arm gold=<dir> --arm mixed=<dir> \
  --matched-fpr 0.05 --compare mixed gold --plots --out reports/bias/phase2
```

### 4.5 Tests

`tests/eval/`: the metric definitions against the teacher's and against hand-computed examples, weighting and within-stratum resampling, logit ranking, failed rows, every gate check (including a degraded candidate blocked and a biased slice blocked), and the bias metrics (BPSN and BNSP checked by hand, a planted biased slice flagged, matched FPR, the paired bootstrap and the missing-data warnings). All run offline with made-up data; the reference check runs only when the private gate data is available.

## 6. Monitoring preparation (Sai)

Builds on #45 (drift metrics), #50 (canary) and #52 (dashboards and alert rules); this section describes their code, so it merges after them.

Phase 2 builds the monitoring pieces against mocks (v4 §5), so Phase 3 only points them at the live endpoint: **drift metrics** for each batch of the Civil stream, a **canary** that replays labeled comments through the service, and **dashboards and alert rules** with documented thresholds.

### 6.1 Drift metrics (`Data-Pipeline/scripts/drift.py`)

Each drift batch is compared with a baseline from the model's training data (Wikipedia train) using the population stability index (PSI). For the three input features, the baseline is split into equal-count bins (deciles), and PSI compares the batch's share of comments in each bin with the baseline's; `flag_rate` is a two-bin PSI (flagged or not) against a reference rate. 0 means no change; status is `ok` below 0.1, `warn` from 0.1, and `alert` from 0.25 (the usual rules of thumb, in the `drift:` section of `Data-Pipeline/config/pipeline.yaml`).

| Feature | Measures | When |
|---|---|---|
| `char_len` | comment length | before scoring (`input_drift`) |
| `oov_rate` | share of a comment's words not seen at least twice in training | before scoring |
| `nonalpha_share` | share of non-space characters that aren't letters | before scoring |
| `flag_rate` | share of comments the model flags | after scoring (`flag_rate_drift`) |

How `drift_ingest_dag` calls it (section 2.5):

```python
from scripts import drift

baseline = drift.load_baseline(cfg)                        # built by data_prep_dag
drift.input_drift(batch, baseline, cfg)                    # in compute_drift
drift.flag_rate_drift(predictions, reference_rate, cfg)    # in score_batch, after scoring
```

`reference_rate` comes from `drift_ingest.reference_flag_rate`, which is unset until a Civil reference exists, so flag-rate drift reports `unknown` for now; the flag it measures uses the five evaluated Civil labels (no `severe_toxicity`).

The baseline is built by the `drift_baseline` stage of `data_prep_dag`, after `split` and before `dvc_version`, so it is versioned in the same DVC commit as the data it describes; the drift DAG only loads it, and a missing baseline is an error. It stores bin edges, shares and means, and the vocabulary as truncated BLAKE2b hashes of single words: no comment text. The file lives at `Data-Pipeline/scripts/drift.py`, not the plan's `src/monitoring/drift.py`, so it imports as `scripts.drift` like the rest of Data-Pipeline and doesn't collide with Model-Pipeline's `src` package.

**On the real data:** the baseline from 132,388 Wikipedia train comments has 76,508 known words and builds in 17 s. Against it, all 20 Civil drift batches show:

| Feature | PSI across the 20 batches | Civil vs Wikipedia | Status |
|---|---|---|---|
| Length | 0.004 to 0.035 | ~295 vs 380 characters, similar distribution | ok |
| OOV rate | 0.05 to 0.13 | 1.9% vs 1.3% unfamiliar words | ok to warn |
| Non-alpha share | 0.10 to 0.18 | 5.4% vs 7.3% (talk pages carry markup, timestamps, signatures) | warn in every batch |

Every batch is `warn`, none reaches `alert`, and there is no trend from batch 0 to batch 19 (the teacher lane's `Model-Pipeline/artifacts/stack/drift_check.json` shows the same). That is the Wikipedia-to-Civil domain shift, not drift within the stream, which shapes the alert rules (6.4).

### 6.2 Canary replay (`Data-Pipeline/scripts/canary.py`)

Real traffic has no labels, so the canary replays the labeled canary sets (`canary_wikipedia`, 2,000 comments, and `canary_civil`, 4,000) through any service following the serving API (v4 §2.4) and compares the answers with the known labels: AUC (from `scores`) and F1 (from `flagged`) per label, plus the precision, recall, F1 and false positive rate of the service's `flag`. Civil covers five labels (no `severe_toxicity`, v4 §2.2).

```bash
python -m scripts.canary --url http://127.0.0.1:8000    # from Data-Pipeline/
```

- **Checks first:** `GET /health` must answer `ok`, so a down service fails before any comment is sent.
- **Bad answers:** a failed call, or a response that breaks the API (including a score that isn't a finite number in [0, 1]), counts as an error and is left out of the metrics. More than 1% errors fails the run, and the CLI then sends a `fail` alert.
- **Outputs (no comment text):** `reports/canary/canary_report.json`, and the same metrics in Prometheus format, `reports/canary/canary.prom`, for the dashboards. A run that sees two model versions (a rollout mid-run) logs a warning.
- **Use `127.0.0.1`, not `localhost`,** on Windows: `localhost` tries IPv6 first, about 2 s per request.

Against the mock (6.3), all 6,000 real canary comments are scored in about 4 s with no errors. The mock's scores carry no signal, so the AUCs (Wikipedia 0.509) only test the plumbing. They come from a hash of each comment's text, so they're deterministic, and the canary's AUCs are identical on every run, as the alert design relies on.

### 6.3 Dashboards and alert rules (`Deployment/monitoring/`)

| File | What it is |
|---|---|
| `mock_service.py` | Mock of the §2.4 service: `/predict`, `/health`, and `/metrics` with exactly the §2.4 metric names, counted from real traffic, plus the canary's `.prom` file. Flags about 11% of comments; bad input is a 422, not counted as a service error |
| `prometheus/alert_rules.yml` | The **active** rules (9, service health and canary), using only metrics that exist today; Prometheus loads it |
| `prometheus/alert_rules_planned.yml` | The **planned** rules (7), not loaded until Phase 3 provides what they need: latency (the int8 package's p95), the canary retrain rule (the promoted model's reference AUC) and drift (`drift_ingest_dag`'s metrics). All three `action: retrain` rules are here |
| `prometheus/alert_rules_test.yml` | `promtool` unit tests: no flag-rate alert when traffic goes idle, service down fires, a missing canary fires |
| `grafana/dashboards/distillate_dashboard.json` | v4 §6's panels: p50/p90/p95 latency, RPS, errors, flag rates, low confidence, canary AUC and flag precision/recall, drift PSI, delayed-label AUC; panels that need Phase 3 metrics are titled "(planned)", and instances and gate history are placeholders |
| `thresholds.md` | every threshold and why (6.4) |
| `docker-compose.yml` | Prometheus and Grafana, to view the drafts locally |

**Verified end to end:** `promtool check rules` finds 9 valid active and 7 valid planned rules, and `promtool test rules` passes. On the local stack, Prometheus scrapes the mock (target UP), evaluates the active rules with no errors (all inactive on the healthy mock), and Grafana loads the dashboard with data in every service, decision and canary panel; the planned drift panels stay empty until `drift_ingest_dag` exports its metrics in Phase 3. `.github/workflows/monitoring-ci.yml` runs the tests, ruff and both `promtool` checks on every PR that touches the folder. How to run it: `Deployment/monitoring/README.md`.

### 6.4 Alert thresholds (`Deployment/monitoring/thresholds.md`)

| Alert | Threshold | Why |
|---|---|---|
| Canary AUC drop | 0.02 below a reference, for 3 runs | The canary replays the same comments, so an unchanged model gives an identical AUC every run; a drop means the service changed. 0.02 is about 2 standard errors of the Wikipedia canary's AUC; 3 runs rides out a transient failure. Active against the 7-day average, **notify only** (after a legitimate promotion it would keep firing for days); the **retrain** version (planned) compares with the promoted model's own canary AUC |
| Delayed-label AUC drop (planned) | 0.03 below its 7-day average | Each drift batch is new comments, so noise is real (about ±0.04 for one batch); the rule uses the mean of two batches (about ±0.03). Retrains |
| Input drift (planned) | warn 0.1, alert 0.25, against the **start of the Civil stream** | Against Wikipedia every Civil batch is already `warn` (6.1), so a warning there would fire permanently. Alert retrains |
| Drift beyond the known shift (planned) | 0.25 against Wikipedia | The known shift tops out near 0.18, so 0.25 means traffic unlike anything measured |
| Flag-rate drift (planned) | 0.25, against a Civil reference | Wikipedia thresholds don't transfer to Civil (toxicity recall 38% on H2, 3.3) |
| Service (active) | errors 2% / 5% (service-side only); flag rate outside 0.5x to 2x its 7-day average, only with real traffic | Bad client input doesn't count as an error; with no traffic the flag-rate ratio is 0, so without the traffic guard it would fire every quiet night. A halving or doubling of what gets flagged is worth a look before any labels arrive |
| Latency (planned) | p95 50 ms (the int8 package, 3.3) | v4 §6's target applies to the int8 package: the measured fp32 p95 is 144 ms (3.3), so against fp32 it would fire constantly. Activates once the int8 p95 is measured |
| Canary missing or stale (active) | no run in 2 hours, or no canary series at all | `absent()` catches a series that disappears entirely, which the staleness check can't see |

**Open for Phase 3:** the canary's mean AUC is fragile on rare labels. `canary_civil` has only 3 threat, 12 obscene and 14 identity_attack positives (against 286 for toxicity); with the mock's signal-free scores, threat landed at AUC 0.25 by chance and pulled the Civil mean to 0.42. Before the canary alert retrains anything, it should use a positives-weighted mean or toxicity AUC. The `civil_start` drift reference should be built from the first few batches or `canary_civil`, since 2,000 comments is a thin vocabulary for the OOV feature.

### 6.5 Tests

- `Data-Pipeline/tests/test_drift.py` (16): PSI is 0 for identical distributions and stays finite with empty bins; each feature on known text; the baseline stores no words and is deterministic; a missing baseline is an error by default; a batch like training data shows no drift, while a shifted one alerts on every feature; flag-rate drift; JSON-ready results.
- `Data-Pipeline/tests/test_canary.py` (23), against a local HTTP server following §2.4: a perfect service scores perfectly; five Civil labels; the Prometheus file parses; no comment text in the outputs; failed calls and bad responses, including NaN, out-of-range and non-numeric scores; a down service fails the health check before any replay; the CLI alerts on failure; mixed model versions warn.
- `Deployment/monitoring/tests/` (22): the mock follows §2.4 (response shape, every metric name, counters following the traffic, cumulative histograms, no text in metrics), rejects bad input such as a lone surrogate as a counted 422, counts its own failures as 500s, and resolves a p95 near 144 ms; active rules and unplanned panels use only metrics that exist, with the canary's names read from `canary.py`; no active rule can retrain; the thresholds doc covers every rule. Plus `prometheus/alert_rules_test.yml` (`promtool test rules`).

All use small made-up data and no network beyond the local machine.
