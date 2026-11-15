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
| Timeout | `TIMEOUT_H = 11` is the session's budget from the sweep's start: a trial starts only if, at the length of the longest trial so far, it would end within it, so a 65-minute trial can't start at hour 10.5 and hit Kaggle's 12-hour limit. The guard is only as long as the longest trial *so far*: a trial longer than all before it (more epochs, 256 tokens) can still overrun, which is why the budget is 11 hours and not 12 |

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
   time, in UTC: Optuna returns local time, which differs between Kaggle and a laptop). Upload the merged file to `gs://distillate-mlops-data/optuna/distilbert-mixed-v1.db`.

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

**Same conditions for every arm:** code `80a4502` for the reported run (`d07c861` for the first one), seed 42, one run per arm, `data.source=ids`.
`ids` gives the same training data as `official` (the pipeline's `train.parquet`): the same
132,388 / 21,363 ids, the same cleaning function, and `sample_weight` 1.0 everywhere (the bias
step chose reweighting factor 1, `Data-Pipeline/reports/bias/bias_metrics.json`). Run-to-run
noise at one seed: three runs of the tuned recipe gave val mean AUC 0.99364 (sweep trial 26),
0.99349 (the first ablation run) and 0.99348 (the reported `distilbert-mixed`), a span of
0.00016, so 0.0002 is at least the one-seed noise, not a measure of it: three runs can easily
understate the spread between seeds, and larger gaps between arms are not shown to be real by
this alone.

**Scoring:** in ablation mode the notebook scores each arm right after training on the gate set
(63,978), the Civil bias set (21,577) and H2 (10,000), by the id lists in
`Data-Pipeline/data/splits/`, writing `predictions/{val,gate,bias_eval,h2_eval}.parquet` and
`predictions/thresholds.json` (the stems in `configs/gate.yaml` as of PR #33).

**The reported run** is `kaggle/13_student_ablation_all.ipynb`: all eight arms through the
pipeline entry point (`src/pipeline/run.py`), two at a time on Kaggle's T4 x2, then latency,
selection, the truncation sweep and the results across arms, in one committed session
(`docs/pipeline.md`).

**Caveat (applies equally to every soft-label arm):** the TF-IDF out-of-fold models, and so the
stack, were fitted on raw train, which includes the validation rows' gold labels; and 263
pilot rows (used to choose the prompt) are in the official validation split. A training row's
soft label never comes from a model that saw *that row*, but it can carry a little
information from validation labels. Validation numbers may be slightly optimistic for the
soft-label arms. The gate set (the Kaggle test split) was never used by any teacher, so the gate
numbers are clean. **Check after the ablation:** compare the validation-to-gate drop of
`distilbert-mixed` with that of `distilbert-gold` (which uses no teacher). A clearly larger drop
for the mixed arm would mean the leak matters.

## Results

**Worker 1 (Nishaant), Nov 6:** 15 of 15 trials complete, no failures, at commit `264731c8bd36`
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

**Worker 0 (Shivam), Nov 10:** warm-started from `worker1.db` (15 trials copied), so all 13 of
its trials are TPE's; at commit `ddf7fff35245`, 10.4 GPU-hours on Kaggle (T4), stopped by the
11-hour budget. 13 of 13 complete. Best: trial 26, val mean AUC 0.99364; median 0.99333.
Study: `gs://distillate-mlops-data/optuna/worker0.db` (it also holds the 15 copied trials).

**The merged study:** 28 trials, all complete (15 worker 1, 13 worker 0), one per key (merging
again copies none). `gs://distillate-mlops-data/optuna/distilbert-mixed-v1.db`, SHA-256
`d26cd782ef1c…`, recipe hash `ee9c99344c83…`.

* 256 tokens: the top 21 trials are all at 256; the best 128 trial (0.9918) ranks 22nd. The
  one 256 trial below a 128 trial is trial 0 (1 epoch).
* The top 3 (trials 26, 8, 22) are within 0.00004 AUC, with α 0.34 to 0.50, T 3.4 to 3.7, and
  lr 7.9e-5 to 9.95e-5: one region, not one point. The order inside it is noise.
* **Two edges of the space bind.** Every trial (28 of 28) peaked at its last epoch, and the
  best learning rates are near the top of the range (1e-4). The study can't change its space,
  so the 5-epoch check run outside it (the best values, `train.epochs=5`) is the test; its
  result is reported next to the ablation, and the tuned values below stay as chosen.

## Best values

Trial 26 (worker 0), the highest validation mean AUC in the merged study
(`python -m src.student.sweep best --storage sqlite:///outputs/optuna/distilbert-mixed-v1.db`):

| Parameter | Best |
|---|---|
| α | 0.3433 (`0.3432987447660936`) |
| T | 3.713 (`3.71295432376268`) |
| Learning rate | 9.23e-5 (`9.22542696771891e-05`) |
| Epochs | 4 (best epoch: the 4th) |
| Maximum length | 256 |
| Validation mean AUC | 0.99364 |
| Validation mean PR-AUC | 0.7706 |

`TUNED` for the ablation (notebook) or `--set` for `train.py`:

```
loss.alpha=0.3432987447660936 loss.temperature=3.71295432376268
train.lr=9.22542696771891e-05 train.epochs=4 model.max_length=256
```

## The 5-epoch check and the reported run (Nov 12)

The reported run (notebook 13, all arms at code `80a4502`;
`reports/student/phase2/README.md`):

| Run | Val mean AUC | Gate mean AUC | H2 mean AUC | Epochs (best) |
|---|---|---|---|---|
| `distilbert-mixed` (the tuned values) | 0.99348 | 0.9848 | 0.9410 | 4 (4th) |
| `distilbert-mixed-5ep` (`train.epochs=5`) | 0.99368 | 0.9845 | 0.9399 | 5 (5th) |

* The tuned recipe reproduced: 0.99348 here, 0.99349 in the first ablation, 0.99364 as trial 26.
* The fifth epoch adds +0.0002 on validation (within the one-seed noise) and nothing on the gate set or
  H2: the 4-epoch edge of the space was not costing much. Its validation AUC was still rising at
  the fifth epoch (0.99355 after the fourth, 0.99368 after the fifth), so the epoch limit still
  binds, but by no more than the one-seed noise. The tuned values stay as chosen;
  the selection rule picked the 5-epoch run on its validation AUC and PR-AUC
  (`reports/student/phase2/selection.md`).

