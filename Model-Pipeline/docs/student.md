# Student training harness

**Owner:** Shivam · **Code:** `src/student/` · **Configs:** `configs/student.yaml`, `configs/smoke.yaml` · **Tests:** `tests/student/`

The student is a small encoder (DistilBERT by default) with six sigmoid outputs, one per label
in the fixed order `toxicity, severe_toxicity, obscene, threat, insult, identity_attack`. It
learns from the gold labels and, when `loss.alpha < 1`, from the stack's soft labels
(`soft_labels/stack-v1/p2-zs-v2b/`, [teacher spec §7](teacher_prompts_and_shard_spec.md)).

## Quickstart (CPU, offline, under a minute)

From `Model-Pipeline/`:

```bash
pip install -r requirements-dev.txt
python -m src.student.synthetic --out outputs/synthetic     # fake rows + soft-label store, no real text
python -m src.student.train --config configs/smoke.yaml     # tiny model, about 100 steps
pytest tests/student
```

## Real runs

| Run | Command |
|---|---|
| Gold-only baseline (Phase 1) | `python -m src.student.train` |
| Mixed: gold + stack soft labels | `python -m src.student.train --set run_name=distilbert-mixed loss.alpha=0.5 loss.temperature=2` |
| Soft labels only | `python -m src.student.train --set run_name=distilbert-soft loss.alpha=0` |
| Another student | `--set model.name=minilm` or `model.name=modernbert` |
| An attribution arm (TF-IDF or calibrated-Qwen soft labels) | `--set soft_labels.arm=tfidf` or `soft_labels.arm=calqwen` |
| The official split without DVC | `--set data.source=ids` (raw train restricted to `Data-Pipeline/data/splits/{train,val}_ids.txt`) |
| Optuna sweep, ablation grid | `python -m src.student.sweep`, see [tuning.md](tuning.md); on Kaggle `kaggle/11_student_runs.ipynb` |
| Truncation share at 64/128/256/512 tokens | `python -m src.student.token_lengths` |
| Prediction file for any Parquet with `id`, `text` | `python -m src.student.predict --model-dir <run>/model --input X.parquet --output Y.parquet` |

Any config value can be overridden with `--set key=value` (values are parsed as YAML, and
unknown keys fail). On a GPU, use the Kaggle notebook `kaggle/10_student_baseline.ipynb`
(instructions in its first cell). It builds raw train from the competition files with the
teacher lane's cleaning and keeps the text in `/tmp`, so no text is saved as output.

**What a real run needs**

* `paths.raw_train`: `wiki_train_raw.parquet` (`id`, `text`, six int8 labels), the output of
  `kaggle/00_prepare_data.ipynb` or the notebook's first step. Comment text never goes into git.
* Mixed and soft runs also need read access to the team bucket `gs://distillate-mlops-data/`
  (`storage.backend: gcs`): `gcloud auth application-default login` locally, or the
  `GCS_TEACHER_KEY` secret (a service-account key, the whole JSON) on Kaggle. The Hugging Face
  dataset `distillate-team/distillate-teacher-shards` is a backup with the same paths
  (`--set storage.backend=hf`, with `HF_TOKEN`). Gold-only runs never read the soft labels.

## The loss

```text
loss = alpha * BCE(sigmoid(z_s), y) + (1 - alpha) * T^2 * BCE(sigmoid(z_s / T), sigmoid(z_t / T))
```

* `z_t` is the shard's `logit_<label>` (the stack's log-odds), not `logit(p_<label>)`, because
  float32 `p` saturates. At `T = 1`, `sigmoid(z_t)` is exactly `p_<label>`.
* `sample_weight` (written by the bias step into `train.parquet`) scales the gold term only.
* Rows whose soft label has `status = failed` are masked out of the soft term.

Unit tests in `tests/student/test_loss.py` check each of these points.

## Data and split

* `data.source: temp` (Phase 1): a seeded (42) multilabel iterative stratification of raw train
  into 85% train and 15% validation, the same method as `Data-Pipeline/scripts/split.py`.
  Rows with empty text are dropped. The split ids are saved with every run.
* `data.source: ids` (Phase 2 sweeps and ablations): raw train restricted to the data pipeline's
  official split ids (`Data-Pipeline/data/splits/train_ids.txt`, 132,388 rows; `val_ids.txt`,
  21,363 rows), so Kaggle runs use the official split without DVC. Raw train has the teacher
  lane's cleaning and no `sample_weight`. More than 0.1% of either list missing from raw train
  is an error.
* `data.source: official` (Phase 2): the data pipeline's `train.parquet` and `val.parquet`.
  Reported numbers are rerun on these.
* Every source checks every soft-label row (the teacher validator: values, status, versions,
  shard layout and manifests). Coverage (every raw train id has a label) is checked too when raw
  train is on the machine (`paths.raw_train`: always for temp and ids); the summary records it
  as `soft_labels.coverage_checked`.
* The gate set is never read. The run fails if any training or validation id is in
  `artifacts/gate_subset_ids.parquet`, and fails if that file is missing (the check can't be
  skipped).

**Soft-label arms** (`soft_labels.arm`; the arm's entry in `soft_labels.arms` replaces the stack's values):

| Arm | Store path | Integrity check |
|---|---|---|
| `stack` (main targets) | `soft_labels/stack-v1/p2-zs-v2b/` | each manifest's `stack_sha256` = SHA-256 of `artifacts/stack/stack_p2-zs-v2b.json` |
| `calqwen` (ablation) | `soft_labels/calqwen-v1/p2-zs-v2b/` | each manifest's `calibration_sha256` = SHA-256 of `artifacts/pilot/calibration_p2-zs-v2b.json` |
| `tfidf` (ablation) | `tfidf/tfidf-lr-v1-oof/tfidf_oof.parquet` | the file's SHA-256 = `artifacts/oof/tfidf_oof.manifest.json`; `p_<label>` becomes `logit_<label>` (clipped at 1e-6) |

**Soft-label checks before training:**
* the `model_version` and `prompt_version` named in the config;
* the arm's integrity check above;
* a failure rate of at most 1%;
* with `temp` and `ids`, which read raw train and so know every raw train id: the teacher validator's full §6 checks (plan, ids, coverage, and types) for shards, and full id coverage for the TF-IDF file.

Any failure stops the run before training.

## Outputs

Each run writes `outputs/student/<run_name>-<UTC time>/` (gitignored):

| Path | What |
|---|---|
| `model/` | weights and tokenizer (`save_pretrained`) from the epoch with the best validation mean AUC |
| `model/meta.json` | lineage: `model_version`, model id, `max_length`, `labels`, `alpha`, `temperature`, `teacher_version`, `teacher_model_version`, `prompt_version`, `stack_sha256`, soft-label arm and check, training-data SHA-256, seed, code commit, MLflow run id |
| `model/thresholds.json` | per-label decision thresholds chosen on validation: `{"toxicity": 0.41, ...}` and nothing else (the format the gate, sensitivity analysis, and serving read) |
| `model/thresholds_report.json` | the method, and per label F1, precision, recall, and flag rate at the threshold and at 0.5 |
| `package/` | with `package.enabled: true`: the §2.5 model package (below) |
| `predictions/val.parquet` | prediction file per CONTRACTS.md §2.2: `id`, `p_<label>` ×6, `model_version` |
| `splits/{train,val}_ids.txt` | the exact rows used |
| `metrics.json`, `summary.json` | per-label AUC, PR-AUC, ECE, and Brier on validation; run summary with history |

The summary is also written to `reports/summaries/train.json`. MLflow logs params, per-epoch
metrics, and the files above (not the weights, unless `mlflow.log_model: true`) to a local
SQLite store (`mlruns/mlflow.db`). When `MLFLOW_TRACKING_URI` is set, it uses that server
instead, which will be the team VM in Phase 2. To browse the local store:
`mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.

## Thresholds and the model package

**Thresholds** (`src/student/thresholds.py`, `thresholds:` in the config): for each label, the
cut-off that maximizes F1 on the validation split, searched over the predicted scores within
[0.01, 0.99] and the two bounds, so a best cut above 0.99 is capped at 0.99 rather than
dropped. Ties go to the higher threshold. Thresholds are stored unrounded (rounding could move
one past the score it was chosen at). Validation sets thresholds (v4 §5.1); the gate set never
does. To recompute for any prediction file:
`python -m src.student.thresholds --pred <val predictions> --gold <val parquet> --out <folder>`.

**Package** (`src/student/package.py`, work plan v4 §2.5), built from a run's `model/` folder:

| File | What |
|---|---|
| `model.onnx` | one self-contained file; inputs `input_ids`, `attention_mask` (int64, dynamic batch and sequence, up to the model's 512 positions), output `logits` (batch, 6) in label order; torch's dynamo exporter, opset 18 |
| `tokenizer/` | the `save_pretrained` tokenizer |
| `thresholds.json` | copied from the run |
| `meta.json` | the §2.5 fields (`model_version`, `training_data_dvc_hash`, `alpha`, `temperature`, `teacher_version`, `teacher_model_version`, `prompt_version`, `stack_sha256`) plus `max_length`, `labels`, `thresholds`, the ONNX export record with its parity check, and the SHA-256 of every other file |

The build compares ONNX Runtime with PyTorch on the same inputs at lengths 16, 64, 128 and 256,
batch sizes 1 and 4, and stops if any logit differs by more than 1e-3. A pretrained DistilBERT
exports to a 256 MB `model.onnx` in about 8 seconds on a laptop CPU, with a maximum difference of
4e-7. Because the sequence axis is dynamic, the package scores at any length up to 512, not only
the training `max_length` (the sensitivity analysis's truncation sweep). `verify_package()`
rechecks the hashes, for example after a registry download. int8 quantization is Phase 3.

Training builds the package last, after `metrics.json`, `summary.json` and the MLflow logging.
If the build fails (export, parity, a missing package), the run keeps all its results and
records the failure: `summary.json` has `package: {"status": "failed", "error": ...}` and MLflow
the tag `package_status: failed`. Rebuild with the command below once it's fixed.

```bash
python -m src.student.package --run-dir outputs/student/<run> --out outputs/packages/<run>
python -m src.student.predict --model-dir outputs/packages/<run> --input X.parquet --output Y.parquet
```

`model_version` is `student-<run_name>-<UTC time>`, for example
`student-distilbert-gold-20261022T150000Z`.

## Phase 1 results

Kaggle, one T4, fp16, `kaggle/10_student_baseline.ipynb`. Numbers are on the validation half
of the temporary split (23,936 rows), unweighted. They are development numbers only. Reported
numbers come from the Phase 2 rerun on the official splits, scored by the eval harness.

| Run | Model | alpha / T | Mean AUC | Mean PR-AUC | Best epoch (of 3) | Minutes |
|---|---|---|---|---|---|---|
| Gold-only baseline | DistilBERT, max length 128 | 1 / – | 0.9882 | 0.7340 | 1 | 25.3 |
| Mixed | DistilBERT, max length 128 | 0.5 / 2 | not run yet | | | |

Per-label AUC for the gold-only baseline:

| Label | toxicity | severe_toxicity | obscene | threat | insult | identity_attack |
|---|---|---|---|---|---|---|
| AUC | 0.9870 | 0.9899 | 0.9921 | 0.9860 | 0.9893 | 0.9852 |

Validation mean AUC by epoch: 0.9878, 0.9882, 0.9880. It peaks at epoch 1, so 2 epochs are
enough at this learning rate. Full record: `artifacts/student/phase1_gold_baseline.json`.

The teacher lane reports TF-IDF at 0.986 and the stack at 0.990 mean AUC on the training rows
(out-of-fold; spec §7). Those are different rows, so this is a rough sanity check, not a
comparison.

**Truncation** (DistilBERT tokenizer, all 159,571 raw train rows;
`artifacts/student/token_lengths_distilbert.json`):

| Max length | 64 | 128 | 256 | 512 |
|---|---|---|---|---|
| Comments longer than this | 41.4% | 19.3% | 6.9% | 2.2% |

The median comment is 52 tokens and the 95th percentile is 311. At 128 tokens, about one in
five comments is cut. Phase 2 compares 128 with 256
(`--set model.max_length=256 run_name=distilbert-gold-256`) and weighs any AUC gain against
serving latency (the 50 ms target).
