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
| Truncation share at 64/128/256/512 tokens | `python -m src.student.token_lengths` |
| Prediction file for any Parquet with `id`, `text` | `python -m src.student.predict --model-dir <run>/model --input X.parquet --output Y.parquet` |

Any config value can be overridden with `--set key=value` (values are parsed as YAML, and
unknown keys fail). On a GPU, use the Kaggle notebook `kaggle/10_student_baseline.ipynb`
(instructions in its first cell). It builds raw train from the competition files with the
teacher lane's cleaning and keeps the text in `/tmp`, so no text is saved as output.

**What a real run needs**

* `paths.raw_train`: `wiki_train_raw.parquet` (`id`, `text`, six int8 labels), the output of
  `kaggle/00_prepare_data.ipynb` or the notebook's first step. Comment text never goes into git.
* Mixed and soft runs also need Hugging Face read access to `distillate-team/distillate-teacher-shards`
  (`hf auth login`, or `HF_TOKEN`). Gold-only runs never read the soft labels.

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
* `data.source: official` (Phase 2): the data pipeline's `train.parquet` and `val.parquet`.
  Reported numbers are rerun on these.
* The gate set is never read. The run fails if any training or validation id is in
  `artifacts/gate_subset_ids.parquet`.

**Soft-label checks before training:**
* the `model_version` and `prompt_version` named in the config;
* every manifest's `stack_sha256` equal to the SHA-256 of the committed `artifacts/stack/stack_p2-zs-v2b.json`;
* a failure rate of at most 1%;
* with `temp`, which has every raw train id, the teacher validator's full §6 checks: plan, ids, coverage, and types.

Any failure stops the run before training.

## Outputs

Each run writes `outputs/student/<run_name>-<UTC time>/` (gitignored):

| Path | What |
|---|---|
| `model/` | weights and tokenizer (`save_pretrained`) from the epoch with the best validation mean AUC |
| `model/meta.json` | lineage: `model_version`, model id, `max_length`, `alpha`, `temperature`, `teacher_version`, `teacher_model_version`, `prompt_version`, `stack_sha256`, training-data SHA-256, seed, code commit, MLflow run id |
| `predictions/val.parquet` | prediction file per CONTRACTS.md §2.2: `id`, `p_<label>` ×6, `model_version` |
| `splits/{train,val}_ids.txt` | the exact rows used |
| `metrics.json`, `summary.json` | per-label AUC, PR-AUC, ECE, and Brier on validation; run summary with history |

The summary is also written to `reports/summaries/train.json`. MLflow logs params, per-epoch
metrics, and the files above (not the weights, unless `mlflow.log_model: true`) to a local
SQLite store (`mlruns/mlflow.db`). When `MLFLOW_TRACKING_URI` is set, it uses that server
instead, which will be the team VM in Phase 2. To browse the local store:
`mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.

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
