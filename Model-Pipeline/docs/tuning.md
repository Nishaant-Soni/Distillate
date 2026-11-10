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

**Sampler:** TPE (`optuna.samplers.TPESampler`), seeded `42 + worker + 1000 × session`, with the
first 8 trials of the study random (`sweep.n_startup_trials`, counted over the trials already in
the study). Worker 0's first trial on an empty study is fixed at the Phase 1 recipe with a
middle α and T (α 0.5, T 2, lr 5e-5, 3 epochs, 128), so the study always contains one
known-reasonable point.

**Re-running a worker** (a dead Kaggle session): set `SESSION = 1` (then 2, ...). With the same
session on a fresh study file the sampler repeats the same random trials.

## Trial plan and budget

| | |
|---|---|
| Trials | 30: 15 per worker (Shivam worker 0, Nishaant worker 1) |
| Hardware | Kaggle, one T4 per worker, fp16 |
| Time per trial | about 8 minutes per epoch at 128 tokens (Phase 1: 25.3 min for 3 epochs), about twice that at 256: 10 to 60 minutes, about 30 on average |
| Budget | about 15 GPU-hours in all, about 7.5 per worker: one 12-hour Kaggle session each |
| Timeout | `TIMEOUT_H = 11` is the session's budget from the sweep's start: a trial starts only if, at the length of the longest trial so far, it would end within it, so a 65-minute trial can't start at hour 10.5 and hit Kaggle's 12-hour limit |

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

   Merging twice adds nothing: every trial has a `trial_key` that its copies keep, failed
   trials included (studies from before keys existed are matched by trial number and start
   time). Upload the merged file to `gs://distillate-mlops-data/optuna/distilbert-mixed-v1.db`.

   Run in parallel, the halves don't see each other's trials (each TPE learns from its own 15,
   and each draws its own 8 random trials: about 16 of 30 random). **When one half is done
   first, start the other from it** (`sweep.warm_start`, `WARM_START` in the notebook): its
   finished trials are copied in before the first trial, so the second half's TPE starts from
   them with no new random trials. The copies keep their keys, so merging both files afterwards
   counts each trial once. A warm start from a study with another setup is refused.
2. **One shared database** (an Optuna RDB URL, such as Postgres on the MLflow VM) in the
   `OPTUNA_STORAGE` Kaggle secret. Both workers then sample from all trials as they finish. It
   needs the database reachable from Kaggle, whose addresses aren't fixed, so only with a
   password and TLS, never an open port.

Every trial records `worker`, `session`, `sampler_seed` and `code_commit` (user attributes),
and the study's setup (search space, recipe hash, data source, arm, model). Optuna is pinned
(`optuna==5.0.0`) because the study file is read by `merge` and the importance analysis.

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
stack, were fitted on raw train, which includes the validation rows' gold labels; and 263
pilot rows (used to choose the prompt) are in the official validation split. A training row's
soft label never comes from a model that saw *that row*, but it can carry a little
information from validation labels. Validation numbers may be slightly optimistic for the
soft-label arms. The gate set (the Kaggle test split) was never used by any teacher, so the gate
numbers are clean. **Check after the ablation:** compare the validation-to-gate drop of
`distilbert-mixed` with that of `distilbert-gold` (which uses no teacher). A clearly larger drop
for the mixed arm would mean the leak matters.

## Results so far

**Worker 1 (Nishaant), Nov 10:** 15 of 15 trials complete, no failures, at commit `264731c8bd36`
with the soft labels from the Hugging Face copy (identical to GCS). Study:
`gs://distillate-mlops-data/optuna/worker1.db`, recipe hash `ee9c9934…`.

* Best: trial 8, α 0.384, T 3.74, lr 7.9e-5, 4 epochs, 256 tokens: val mean AUC 0.9936,
  PR-AUC 0.766.
* 256 tokens won every time: all of the top 9 trials; every 128 trial is in the bottom 6.
* The top 5 are within 0.0005 AUC of each other with α from 0.16 to 0.52, so their order is
  probably noise.
* Trial 8 peaked at its last epoch with lr near the top of the range: 4 epochs (the space's
  maximum) may be capping it. The space can't change within the study, so this is checked
  with one run outside it after the sweep (the best values at 5 epochs).

**Worker 0 (Shivam):** warm-started from `worker1.db`, so all of its trials are TPE's.

## Best values

To be filled in after the merge: the trial count, the best trial's values and validation mean
AUC and PR-AUC, and the merged study's location.

| Parameter | Best |
|---|---|
| α | |
| T | |
| Learning rate | |
| Epochs | |
| Maximum length | |
| Validation mean AUC | |
