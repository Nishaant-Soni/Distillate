# Retraining on the drift stream

**Status:** agreed Nov 21 (work plan v5, 6.1 and Nishaant's section), for `retrain_dag` by Sun Nov 29.

What a retraining run trains on, what it holds out, and how the candidate is compared with production. `src/pipeline/promote_check.py` is the comparison (section 5).

## 1. Which batches are labeled when a retrain starts

`drift_ingest_dag` releases one Civil drift batch per run (20 batches of 2,000 rows). Its *evaluate* step attaches the gold labels to the **previous** batch's predictions: the labels arrive one batch late. So when `check_thresholds` triggers a retrain after batch *n−1* is released (`state.json` has `next_batch = n`):

| Batches | State | Use |
|---|---|---|
| 0 … n−3 | released, labels arrived | **training** |
| n−2 | released, labels arrived (the newest labeled batch) | **held out:** the promotion comparison |
| n−1 | released, labels not yet arrived | not used |

`retrain_dag` computes these from `state.json`, and refuses to start (Slack message, no Vertex job) when `n < 3`, i.e. until three batches are out (batches are numbered from 0, so `n = 3` means batches 0, 1 and 2 are released): before that there is no drift batch to train on. The first retrain, at `n = 3`, trains on batch 0, holds out batch 1, and skips batch 2.

## 2. Labels for the drift rows

* **Gold:** the batch file's own label columns (Civil Comments, positive at a rater fraction ≥ 0.5): the "delayed labels".
* **Soft labels:** the stack's, already produced for all 20 batches in `soft_labels/stack-v1/p2-zs-v2b/drift/shard_<k>` (one shard per batch). On the drift stream the stack's mean AUC is 0.924, against 0.887 for TF-IDF and 0.882 for raw Qwen; it beats both in all 20 batches (`artifacts/stack/drift_check.json`).
* The same loss as for Wikipedia rows (α · gold BCE + (1 − α) · T² · soft BCE), with the production recipe's α and T.
* **`severe_toxicity` on drift rows:** no drift comment is positive for it (Civil marks a label positive only when at least half the raters chose it), so the drift rows add only negatives for that label through the gold loss. That is Civil's real rate, so they are kept as they are, not masked. Two things still guard the label: the gate checks it on Wikipedia (validation PR-AUC floor, no-regression on the gate set), and `promote_check` doesn't score it (five Civil labels). Revisit (mask its gold loss on drift rows) if a retrained candidate is blocked on `severe_toxicity`.

## 3. What the retraining job trains on

1. **Wikipedia train**, unchanged (with `teacher_labels.parquet`).
2. **Drift batches 0 … n−3**, gold plus soft labels, `sample_weight = drift_weight`.
3. **Obfuscation augmentation from training rows:** the rewrites of `Data-Pipeline/scripts/obfuscate.py` applied to a seeded sample of toxic Wikipedia **train** rows, keeping the original's gold and soft labels (`id` = `<id>_aug`). The obfuscation evaluation set is built from gate comments, so this doesn't leak into it.

**Validation, early stopping and per-label thresholds** stay on Wikipedia validation, as in CI, so the gate's floors (set on Wikipedia validation) still apply unchanged.

**Values:**

| Setting | Value | Why |
|---|---|---|
| `drift_weight` | 1.0 | Simplest. The drift share grows from about 1% (n = 3) to 20% (n = 20). Raise it only if retrained models don't move on Civil. |
| obfuscation augmentation | 3,000 rows, seed 42 | The same size as the obfuscation evaluation set. |

## 4. Configuration

A `retrain` section in `configs/student.yaml` (and `smoke.yaml`), implemented by `src/student/retrain.py`. The defaults change nothing in CI; `retrain_dag` sets it through `--set`:

```yaml
retrain:
  drift_batches: []        # training batches 0, 1, ..., m-1 (no gaps), e.g. [0, 1, 2, 3, 4]
  holdout_batch: null      # scored whole into predictions/drift_holdout.parquet; must be the
                           # batch right after the training ones (n-2 = len(drift_batches))
  drift_weight: 1.0
  drift_dir: null          # the batch files; null = <pipeline.data_dir>/civil/drift
  obfuscation_augment: {rows: 0, seed: 42}   # 3000 in a retraining run
```

For example, after batch 6 is released (n = 7): `retrain.drift_batches=[0,1,2,3,4] retrain.holdout_batch=5 retrain.obfuscation_augment.rows=3000`.

* Drift rows' soft labels come from `<soft_labels.prefix>drift/shard_<k>`, checked like the training shards: the manifests' `stack_sha256` against the committed stack, versions, each shard's ids against its batch, failure rate.
* The job needs these tables: `train val gate bias h2 obfuscation` (Praniti's data-fetch fix), plus the drift batch files 0 … n−2 under `civil/drift/` in the same layout. A missing batch file stops the run before training. The drift soft-label shards come from the store, as the training shards do.
* The run's and the package's `meta.json` record a `retrain` block (`drift_batches`, `holdout_batch`, `drift_weight`, the row counts, and the augmentation's tricks), so `promote_check.py --trained-on` can be given the batches trained on.
* `predictions/drift_holdout.parquet` and `gold/drift_holdout.parquet` are written next to the other splits; `eval/metrics.json` scores the held-out batch on the five Civil labels.

## 5. Promotion: beat production on the held-out batch

After the gate passes, `python -m src.pipeline.promote_check` compares the candidate with `@production` on batch n−2:

* **Candidate's predictions:** written by the job (`predictions/drift_holdout.parquet`).
* **Production's predictions:** already on the ops VM: `drift_ingest` stored the live service's scores for that batch (`reports/drift/predictions/batch_<k>.parquet`, no text). If `@production` has moved since the batch was scored, the check refuses that file, and `retrain_dag` rescores the batch through the live service first.
* **Rule (work plan v5):** the candidate's mean AUC over the five Civil labels minus production's is above `min_gain`. Both models are scored on the same rows, which must cover at least 95% of the batch (`min_coverage`). Exit 0 = pass; 1 = keep production; 2 = the check couldn't decide (bad or mismatched inputs, too few rows scored), with the reason on stderr and no report, so a crash is never read as a verdict.
* **Order in `retrain_dag`:** gate → promote_check → register `@staging` → `repository_dispatch` → Slack.

**How noisy one batch is.** A 2,000-row batch has few positives for some labels:

| Label | Positives per batch | 95% interval width on one model's AUC |
|---|---|---|
| toxicity | 130–201 | 0.06 |
| insult | 94–145 | 0.07 |
| identity_attack | 4–28 | 0.11 |
| obscene | 5–14 | 0.14 |
| threat | 2–10 | 0.24 (up to 0.69) |

Widths are the median over the 20 batches (TF-IDF scores, bootstrap).

**The rule, tried on the 20 real batches.** `promote_check` itself, on the real gold labels, with two pairs of real models: the drift scores in the store (paired bootstrap, 500 replicates).

| Pair (candidate vs "production") | Overall gain | Batches where the candidate passes | Paired 95% interval width (median) |
|---|---|---|---|
| Stack vs TF-IDF: clearly better | +0.039 | **20 / 20** (interval above 0 in all 20) | 0.047 |
| TF-IDF vs raw Qwen: about equal | +0.006 | **11 / 20** | 0.099 |
| Same pairs, `min_positives: 20` (threat and obscene left out) | +0.032 / +0.020 | 20 / 20 and 17 / 20 | 0.035 / 0.067 |

So:

* **A real improvement gets promoted reliably,** even on one batch.
* **Between two models of about the same quality, the result is a coin flip.**
* **`min_positives: 20` narrows the interval by about a third,** but it also changes what is measured: threat and obscene never count. The near-equal pair then goes 17 / 20 for TF-IDF, because those two labels are where raw Qwen does better.

**Rule kept as the plan states it:** all five labels, `min_gain: 0`, `min_positives: null` (`configs/promotion.yaml`), and the paired bootstrap recorded in the report.

* **A coin flip between near-equal models is harmless.** The gate already blocks regressions: no-regression on the 64k-row Wikipedia gate set, plus the bias limits. So the drift comparison can only promote a model about as good as production, or keep production.
* **Dropping threat and obscene would hide a candidate that got worse on exactly the rare harms on Civil text.**

## 6. Who does what

| Piece | Owner |
|---|---|
| `retrain` config section, drift rows + soft labels + augmentation in the student's data loading, holdout scoring in `run.py`, `meta.json` fields | Nishaant (touches the student's data code: review by Shivam) |
| `promote_check.py` | Nishaant (this PR) |
| Fetching `h2`, `obfuscation`, the drift batches and shards in the job | Praniti (the data-fetch gap) |
| `retrain_dag`: batch numbers from `state.json`, the `n < 3` refusal, rescoring when production moved, calling `promote_check`, then register and dispatch | Vaishnavi |

## 7. Decisions (agreed Nov 21)

1. Hold out the newest labeled batch (n−2); the unlabeled one (n−1) isn't used.
2. Drift rows get gold and soft labels with the same loss; `drift_weight` 1.0.
3. Obfuscation augmentation: 3,000 toxic Wikipedia train rows, seed 42.
4. Promotion: all five Civil labels, `min_gain` 0, as the plan says (section 5 has the tests on the real batches).
5. No retrain until three batches are out (`n < 3`, i.e. before batch 2 is released). The first retrain is at `n = 3`: it trains on batch 0, holds out batch 1, and skips batch 2.
