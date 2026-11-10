# Student tuning (Optuna)

How the student's distillation recipe is tuned (work plan v4 §5): the search space, sampler,
trial plan, budget, how the two halves of the sweep share one study, and the best values.
Code: `src/student/sweep.py`. Notebook: `kaggle/11_student_runs.ipynb` (`MODE = "sweep"`).

## What is tuned

Once, on **DistilBERT with the mixed loss on the stack's soft labels**. The best values are then
reused for every other arm of the ablation grid and for the architecture comparison, so the
arms differ only in what they compare (v4 §5).

| Parameter | Config key | Range | Scale |
|---|---|---|---|
| α (gold weight) | `loss.alpha` | 0.1 to 0.9 | linear |
| T (temperature) | `loss.temperature` | 1 to 5 | linear |
| Learning rate | `train.lr` | 1e-5 to 1e-4 | log |
| Epochs (maximum) | `train.epochs` | 1 to 4 | integer |
| Maximum length | `model.max_length` | 128 or 256 | categorical |

* α = 0 and α = 1 are left out: they are their own arms of the grid (stack only, gold only).
* 128 vs 256: 19.3% of training comments exceed 128 tokens and 6.9% exceed 256
  (`artifacts/student/token_lengths_distilbert.json`).
* Epochs is a maximum: every run keeps its best epoch on validation (early stopping with
  `train.patience: 1`), but the epoch count also sets the learning-rate schedule.
* Everything else is fixed by `configs/student.yaml` (batch size 32, weight decay 0.01, warmup
  6%, fp16, seed 42).

**Objective:** the run's best validation mean ROC-AUC over the six labels (unweighted), on the
official validation split (`data.source: ids`, 21,363 rows). Validation chooses; the gate set is
never read during tuning.

**Sampler:** TPE (`optuna.samplers.TPESampler`), seeded `42 + worker`, with the first 8 trials of
the study random (`sweep.n_startup_trials`). Worker 0's first trial is fixed at the Phase 1 recipe
with a middle α and T (α 0.5, T 2, lr 5e-5, 3 epochs, 128), so the study always contains one
known-reasonable point.

## Trial plan and budget

| | |
|---|---|
| Trials | 30: 15 per worker (Shivam worker 0, Nishaant worker 1) |
| Hardware | Kaggle, one T4 per worker, fp16 |
| Time per trial | about 8 minutes per epoch at 128 tokens (Phase 1: 25.3 min for 3 epochs), about twice that at 256: 10 to 60 minutes, about 30 on average |
| Budget | about 15 GPU-hours in all, about 7.5 per worker: one 12-hour Kaggle session each (`TIMEOUT_H = 11` stops new trials in time) |

Disk: only the best trial's weights are kept (`sweep.keep_models: best`); every trial keeps its
`meta.json`, validation prediction file, metrics and summary.

## One study, two workers

Both halves must end in **one** Optuna study (`distilbert-mixed-v1`), because Nishaant's
hyperparameter importance (`src/sensitivity/hyperparameters.py`) reads a single study. Each worker
records the setup on the study (search space, a hash of the base recipe, data source, arm,
model) and refuses to add trials to a study that was started with another setup.

Two ways to share it:

1. **Separate files, merged afterwards (default, no infrastructure).** Each worker writes
   `student/optuna.db` in its Kaggle output. Afterwards:

   ```bash
   python -m src.student.sweep merge --into sqlite:///outputs/optuna/distilbert-mixed-v1.db \
       --from sqlite:///path/to/worker0/optuna.db sqlite:///path/to/worker1/optuna.db
   ```

   Merging twice adds nothing. The halves don't see each other's trials while they run (each
   TPE learns from its own 15), which costs a little efficiency but nothing in validity. Upload
   the merged file to `gs://distillate-mlops-data/optuna/distilbert-mixed-v1.db`.
2. **One shared database** (an Optuna RDB URL, such as Postgres on the MLflow VM) in the
   `OPTUNA_STORAGE` Kaggle secret. Both workers then sample from all trials as they finish. It
   needs the database reachable from Kaggle, whose addresses aren't fixed, so only with a
   password and TLS, never an open port.

For the importance analysis: `optuna.storage` = the merged file's `sqlite:///` URL (or the
shared database), `optuna.study_name` = `distilbert-mixed-v1`.

**MLflow:** every trial is a normal training run in experiment `distillate-student-sweep`, tagged
`study`, `trial` and `sweep_worker`. With `MLFLOW_TRACKING_URI` set (the team server), trials log
there; otherwise to the session's local store (`/kaggle/working/mlruns`), kept as output.

## Running it

```bash
# Locally or in the notebook: worker 0, 15 trials, the official split
python -m src.student.sweep --set data.source=ids paths.raw_train=<raw parquet> \
    mlflow.experiment=distillate-student-sweep --worker 0 --n-trials 15
# The best values as --set flags for train.py
python -m src.student.sweep best --storage sqlite:///outputs/optuna/distilbert-mixed-v1.db
```

## The ablation grid with the tuned values

`MODE = "ablation"` in the same notebook trains each arm with the tuned values (`TUNED`) and the
same data, seed and settings. Only the listed key changes:

| Run | Change from the tuned mixed recipe | Soft labels |
|---|---|---|
| `distilbert-gold` | `loss.alpha=1.0` | none |
| `distilbert-stack` | `loss.alpha=0.0` | stack |
| `distilbert-mixed` | none | stack (`soft_labels/stack-v1/p2-zs-v2b/`) |
| `distilbert-tfidf` | `soft_labels.arm=tfidf` | TF-IDF out-of-fold (`tfidf/tfidf-lr-v1-oof/`) |
| `distilbert-calqwen` | `soft_labels.arm=calqwen` | calibrated Qwen (`soft_labels/calqwen-v1/p2-zs-v2b/`) |
| `minilm-mixed`, `modernbert-mixed` | `model.name=...` | stack |

The TF-IDF baseline classifier itself (the grid's first row) is scored by the evaluation code,
not trained here.

**Caveat (applies equally to every soft-label arm):** the TF-IDF out-of-fold models, and so the
stack, were fitted on raw train, which includes the validation rows' gold labels. A training
row's soft label never comes from a model that saw *that row*, but it can carry a little
information from validation labels. Validation numbers may be slightly optimistic for the
soft-label arms. The gate set (the Kaggle test split) was never used by any teacher, so the gate
numbers are clean.

## Best values

To be filled in after the sweep: the trial count, the best trial's values and validation mean
AUC and PR-AUC, and the study's storage location.

| Parameter | Best |
|---|---|
| α | |
| T | |
| Learning rate | |
| Epochs | |
| Maximum length | |
| Validation mean AUC | |
