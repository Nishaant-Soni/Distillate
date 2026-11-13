# Model card: distillate-student

> **Status: skeleton (draft).** Sections marked `TODO (owner)` are filled in by that lane
> owner in their own PR. Numbers come from `src/eval` and the bias reports; don't type them
> in by hand from memory. Final by Fri Dec 4.

| Section | Owner |
|---|---|
| 1, 2, 8, 9, 10, 11, 12 | Ruchita |
| 3 (teacher and soft labels) | Nishaant |
| 4 (training) | Shivam |
| 7 (sensitivity analysis) | Nishaant |
| 5 data facts | Praniti / Sai |

---

## 1. Model details

- **Name:** `distillate-student` (MLflow registered model; aliases `staging` and `production`).
- **Version:** `student-distilbert-mixed-5ep-20261112T210931Z` (training MLflow run
  `358772663fc34a5e81275c3ed073e598`, code `80a4502`). The registered MLflow model version
  is assigned when the gate registers it (Phase 3 CI).
- **Type:** multi-label text classifier, six sigmoid outputs (`toxicity`, `severe_toxicity`,
  `obscene`, `threat`, `insult`, `identity_attack`).
- **Architecture:** DistilBERT (`distilbert-base-uncased`, 6 layers, 66M parameters) with a
  six-output classification head; maximum length 256 tokens. Chosen by the selection rule from
  DistilBERT, MiniLM-L12 and ModernBERT-base (section 4).
- **Training method:** knowledge distillation from a teacher (section 3) mixed with gold labels:
  `alpha * BCE(gold) + (1 - alpha) * T^2 * BCE(soft)`, with alpha = 0.343 and T = 3.71
  (tuned, section 4).
- **Serving format:** ONNX (int8) behind FastAPI on Cloud Run (Phase 3).
- **Package contents:** `model.onnx`, `tokenizer/`, `thresholds.json`, `meta.json` (model
  version, training-data DVC hash, alpha, temperature, teacher_version,
  teacher_model_version, prompt_version, stack_sha256).
- **Owners:** Distillate team, MLOps (DADS / IE 7305).
- **Contact:** TODO.

## 2. Intended use

**Intended use**

- Scoring English online comments for toxicity, to prioritise human moderation (the demo's
  moderation queue) and to study model behaviour under distribution shift.
- The `flag` decision (any label at or above its threshold) routes a comment for review; it
  is not an automatic removal decision.
- `low_confidence` marks scores within 0.1 of a threshold for extra human attention.

**Out of scope**

- Automated removal, account sanctions or any decision about a person without human review.
- Languages other than English, and platforms far from Wikipedia talk pages and news
  comments, without new evaluation.
- Inferring anything about a commenter, including their identity. The model never receives
  identity information, and its scores must not be used as a proxy for it.
- Use as a ground-truth labeller for other datasets.

## 3. Teacher and soft labels

Besides the human (gold) labels, the student learns from **soft labels**: per-comment probabilities for the six labels, made by a *stack* that combines a zero-shot LLM teacher with a TF-IDF baseline. How much weight the student gives them is the loss's α (section 4). This section says what the soft labels are, how they were chosen, how good they are on held-out rows, and where they fall short. Full specification: `Model-Pipeline/docs/teacher_prompts_and_shard_spec.md`.

### At a glance

| | |
|---|---|
| Teacher model | `Qwen/Qwen3-4B-Instruct-2507` at revision `cdbee75f17c01a7cc42f958dc650907174af0554`, run with vLLM 0.31.0 on Kaggle T4 GPUs |
| Prompt | `p2-zs-v2b`: one zero-shot yes/no question per label (six prompts per comment), temperature 0 |
| What the student trains on | `stack-v1` soft labels, `soft_labels/stack-v1/p2-zs-v2b/` in `gs://distillate-mlops-data` (the Hugging Face dataset `distillate-team/distillate-teacher-shards` is a backup copy), merged into `Data-Pipeline/data/processed/teacher_labels.parquet` and versioned with DVC |
| Soft-label `model_version` | `stack-v1+Qwen/Qwen3-4B-Instruct-2507@cdbee75…+tfidf-lr-v1` |
| Rows labeled | 159,571 training comments, 20,000 test comments (the gate subset), 40,000 Civil Comments drift comments; 0 failed rows |
| Labeling cost | about 9.4 T4 GPU-hours: 7.0 for training, 0.9 for the gate subset, 1.6 for drift (shard manifests) |
| Held-out quality (gate subset) | mean AUC 0.982, PR-AUC 0.659, ECE 0.010 (weighted; table below) |

### How a probability is read from the teacher

Each prompt asks one question about one label and allows a single output token. The probability comes from the top-20 token log-probabilities: P(yes) and P(no) are each summed over six spellings (`yes`, `Yes`, `YES`, with and without a leading space), `p = P(yes) / (P(yes) + P(no))`, and the log-odds `log P(yes) − log P(no)` is kept alongside it, because float32 `p` saturates at exactly 1.0 for confident answers. A cell with less than half its probability mass on yes or no is retried once and then marked failed; none were. Comments longer than 1,024 tokens are truncated (0.37% of training comments, 0.52% of the gate subset).

### What the soft labels are

On its own the teacher ranks comments well but is poorly calibrated (raw `p_toxicity` averages 0.22 against a 0.10 positive rate), and a TF-IDF logistic regression trained on the gold labels beats it on Wikipedia. The stack combines the two per label:

```text
logit = w_teacher · logit_teacher + w_tfidf · logit(p_tfidf) + intercept,    p = sigmoid(logit)
```

The 18 weights (`Model-Pipeline/artifacts/stack/stack_p2-zs-v2b.json`) were fitted by weighted logistic regression on 2,000 pilot comments with their inclusion weights, so the stack is also its own calibration. It replaces the per-label Platt calibration (`calibration.json`) first planned for the raw teacher.

**The soft labels blend a TF-IDF model trained on the gold labels with Qwen's zero-shot judgments, so they carry gold-label information and are not independent LLM labels.** On held-out rows, adding Qwen to TF-IDF improves mean AUC by +0.0023 [0.0015, 0.0032] and PR-AUC by +0.036 [0.029, 0.044] on the Wikipedia gate subset (95% bootstrap intervals), and by +0.038 AUC and +0.065 PR-AUC on the Civil drift stream (five labels), where the stack is ahead in all 20 batches (by 0.016 to 0.077 AUC per batch). The gain is small where TF-IDF is already strong and larger on shifted data, where TF-IDF transfers worst.

Neither input alone is clearly better out of domain: on the drift stream the raw teacher scores 0.882 mean AUC and TF-IDF 0.887, while the stack scores 0.924. That the combination gains most on unfamiliar text is consistent with TF-IDF depending on Wikipedia's vocabulary while Qwen was never fitted to these labels, so the two make different errors there. This is an interpretation, not something the runs tested.

The stack's weights are not comparable across its two inputs, whose logits are on different scales, so the held-out gains above, not the weights, measure what Qwen adds. How much of the *student's* quality is due to the LLM is measured by the ablation (section 6; `Model-Pipeline/reports/student/phase2/results.md`). The like-for-like comparisons are between arms trained with the same loss (gold plus soft labels, α 0.343, T 3.71), differing only in where the soft labels come from:

| Soft labels (DistilBERT, mixed loss) | Gate set mean AUC [95% CI] | Civil H2 mean AUC [95% CI] |
|---|---|---|
| Stack (`distilbert-mixed`) | 0.9848 [0.9840, 0.9854] | 0.9410 [0.9280, 0.9510] |
| TF-IDF out-of-fold (`distilbert-tfidf`) | 0.9826 [0.9815, 0.9835] | 0.9081 [0.8895, 0.9252] |
| Calibrated Qwen (`distilbert-calqwen`) | 0.9783 [0.9771, 0.9793] | 0.9206 [0.9055, 0.9350] |

* **Stack against TF-IDF, the part creditable to the LLM:** +0.0022 mean AUC on the gate set and +0.033 on Civil, with non-overlapping intervals on both. As on the teacher's own held-out rows, Qwen's contribution is small in-domain and largest on shifted text.
* **Qwen alone is not a better teacher than TF-IDF.** Calibrated Qwen against TF-IDF, paired bootstrap: −0.0043 [−0.0053, −0.0034] on the gate set and +0.0125 [−0.0080, +0.0329] on Civil, where the interval includes zero. What the LLM adds reaches the student through the stack, combined with TF-IDF.
* The intervals in the table are per arm; paired intervals for the stack-against-TF-IDF difference need the arms' prediction files and will be added. `results.md`'s paired "stack − TF-IDF" row compares the stack-only arm (α 0) with the TF-IDF arm (α 0.343), so it mixes the teacher with the loss and is not used here.

**No training comment's soft label comes from a model that saw its own gold label.** TF-IDF scores for training comments are out-of-fold (five folds, seed 42); the 157,571 training comments outside the pilot get the frozen stack; the 2,000 pilot comments, on which the stack was fitted, get five-fold cross-fitted outputs; test and drift comments get the frozen stack and the full-train TF-IDF model. See the validation caveat under *Known limits*.

### How the teacher was chosen (development results)

These results guided choices, so they are **development** numbers, not held-out ones.

* **Pilot** (2,000 training comments, enriched for the rare labels and weighted back to the population; `artifacts/pilot/`). Mean AUC over the six labels: zero-shot P2 0.965, few-shot P2 0.967 (at half the throughput, 3.4 against 6.8 comments per second), TF-IDF 0.985. The adopted prompt, `p2-zs-v2b`, scored 0.972 on the same pilot. The pre-agreed rule (fine-tune the teacher if its best zero-shot AUC is below TF-IDF's) triggered; instead the team approved the stack, which beats TF-IDF on held-out rows.
* **Prompt follow-up for obscene** (`artifacts/prompt_v2/`). The v1 obscene question scored AUC 0.936. Three rewrites were compared on one half of the pilot; `v2b` (0.989) was chosen, then confirmed against v1 on the other half (+0.031 obscene AUC, 95% CI [0.009, 0.062], 998 comments) and on Civil Comments (+0.160 [0.129, 0.189], 3,000 comments). Only that question changed.
* **Civil Comments shift sample** (3,000 comments; `artifacts/shift/`). Mean AUC over five labels (the sample has no severe_toxicity positives): stack 0.934, TF-IDF 0.895, raw teacher 0.894. The sample was used to accept the out-of-fold stack and to check the prompt follow-up, so it is a development result, not an independent test.

**Disclosure: one choice used the gate subset.** A second stack (`stack-v2`, fitted on all training comments) was compared with `stack-v1` on the gate subset and on Civil, under a rule fixed in advance: keep `stack-v1` unless `stack-v2` is at least as good. `stack-v2` minus `stack-v1`, mean AUC: −0.00026 [−0.00034, −0.00019] on the gate subset, +0.00011 [−0.0007, 0.0009] on Civil. `stack-v1`, the pre-agreed default, was kept. The gate set is meant for final numbers only, never for choosing; the practical effect here is close to none (the default was kept, the two stacks are nearly identical, and Civil agrees), but it is recorded.

### Teacher accuracy on held-out rows

Scored by the team's evaluation harness (`src/eval/metrics.py`), the code that scores every student, on the 20,000-row gate subset: weighted by its inclusion weights (it is enriched for the rare labels), with 95% intervals from 1,000 bootstrap replicates resampled within each stratum. AUC and PR-AUC rank on log-odds where a file has them. The harness reproduces the teacher lane's reference values (`artifacts/stack/stack_check.json`, `artifacts/calqwen/calqwen_check.json`) to within 2e-5 (`tests/eval/test_metrics.py -k reference`).

| Teacher | Mean AUC [95% CI] | Mean PR-AUC [95% CI] | ECE | Brier |
|---|---|---|---|---|
| **Stack (soft labels, `stack-v1`)** | **0.9822** [0.9810, 0.9835] | **0.659** [0.641, 0.684] | 0.010 | 0.019 |
| TF-IDF (`tfidf-lr-v1`) | 0.9799 [0.9785, 0.9814] | 0.623 [0.604, 0.648] | 0.010 | 0.019 |
| Raw Qwen | 0.9588 [0.9564, 0.9611] | 0.475 [0.454, 0.504] | 0.086 | 0.085 |
| Calibrated Qwen (`calqwen-v1`) | 0.9588 [0.9564, 0.9611] | 0.475 [0.454, 0.504] | 0.011 | 0.025 |

Per label, AUC / PR-AUC:

| Label | Stack | TF-IDF | Raw Qwen |
|---|---|---|---|
| toxicity | 0.967 / 0.789 | 0.965 / 0.780 | 0.926 / 0.586 |
| severe_toxicity | 0.987 / 0.395 | 0.985 / 0.371 | 0.977 / 0.288 |
| obscene | 0.977 / 0.781 | 0.977 / 0.782 | 0.935 / 0.582 |
| threat | 0.997 / 0.622 | 0.994 / 0.527 | 0.994 / 0.506 |
| insult | 0.974 / 0.746 | 0.972 / 0.733 | 0.941 / 0.479 |
| identity_attack | 0.991 / 0.625 | 0.986 / 0.545 | 0.981 / 0.410 |

* The stack has the best AUC on every label. Its largest PR-AUC gains over TF-IDF are on the rare labels threat (+0.095) and identity_attack (+0.080); it is level with TF-IDF on obscene.
* **Calibrated Qwen** (`calqwen-v1`, the ablation's second attribution arm) is Qwen alone with the pilot's Platt calibration. It is as well calibrated as the stack (ECE 0.011 against 0.010), but its calibration is monotone per label, so it ranks exactly as raw Qwen does (per-label AUC and PR-AUC equal to within 3e-8). The stack's lead (+0.023 AUC, +0.18 PR-AUC) therefore comes from its TF-IDF input, not from calibration.
* **Civil drift stream** (40,000 comments in 20 batches; five labels, unweighted; `artifacts/stack/drift_check.json`): AUC / PR-AUC / ECE are 0.924 / 0.344 / 0.008 for the stack, 0.887 / 0.279 / 0.013 for TF-IDF and 0.882 / 0.197 / 0.134 for raw Qwen. The stack is best in every batch (batch mean AUC 0.888 to 0.950, no trend over time). No drift comment is positive for severe_toxicity.

### Prompt injection

40 synthetic comments with an instruction hidden in them (`configs/injection_set.csv`): 20 clearly toxic ones told to answer "no", and 20 clean ones told to answer "yes", some trying to close the comment's delimiter early. With the adopted prompt, 2 of 73 eligible answers flipped to "no" (2.7%) and 8 of 120 flipped to "yes" (6.7%); the largest per-label rate is threat, at 16.7% (1 of 6) toward "no" (`artifacts/pilot/injection_p2-zs-v2b.csv`). This was measured on raw Qwen only. The soft labels also depend on TF-IDF, which counts words rather than following instructions, so the effect on them is probably smaller, but text written to fool an LLM can still move a training label.

### Known limits

* **Zero-shot judgments of one 4-billion-parameter model**, not human labels, and not fine-tuned on this task.
* **They depend on TF-IDF**, and so on its gold-label training (above). Qwen's measured contribution is largest on shifted data.
* **Validation caveat (the soft-label arms' validation numbers are optimistic).** The out-of-fold TF-IDF models were fitted on raw train, which includes the official validation split's rows and gold labels, and the stack and the calibrated-Qwen Platt fits used the 2,000 pilot comments, 263 of which are validation comments. A training comment's soft label never comes from a model that saw *that* comment, but it can carry a little information about the validation labels, so validation scores of students trained on soft labels may be optimistic, and tuning on validation may lean slightly toward lower α. Gate-set, H2 and bias-set numbers are unaffected: no teacher-side model saw those rows' labels.

  **Check, after the ablation** (validation-to-gate drop in mean AUC, `results.md`): gold-only 0.0053; mixed 0.0087; stack only 0.0096; calibrated Qwen 0.0101; TF-IDF 0.0117. Every soft-label arm drops 0.003 to 0.006 more than gold-only, and the arm most exposed to the leak (TF-IDF) drops most, as the leak predicts. But calibrated Qwen, which touches it only through the 263 pilot comments, also drops 0.0101, so the leak is not the whole difference. **Verdict:** soft-label arms' validation AUC is optimistic by up to about 0.003 to 0.006 relative to gold-only, so gold-only and soft-label arms are compared on the gate set and H2, never on validation. Arms with the *same* teacher are affected about equally, which is why the selection (section 4) compares only the tuned recipe's architectures, all trained on the stack: validation still ranks them fairly, while the TF-IDF arm, whose validation AUC was the highest (0.99426) but whose gate and H2 scores are lower than the mixed arm's, is not a candidate. A bootstrap interval for the drop difference needs the prediction files and will be added.
* **Weaker on the test split than on training comments**: raw Qwen's mean AUC is 0.959 on the gate subset against 0.975 on training comments, mostly for toxicity, obscene and insult. Quote test-split numbers.
* **Rare labels**: severe_toxicity, threat and identity_attack are 0.3% to 1% of comments, and their PR-AUC is low, though the stack improves it.
* **Domain**: English Wikipedia talk-page comments. Over the same five labels the stack's mean AUC is 0.981 on the Wikipedia gate subset (weighted) and 0.924 on the Civil drift stream (unweighted). Civil gold labels are rater fractions thresholded at 0.5.
* **Truncation** at 1,024 tokens (0.37% of training comments).
* **Censored cells**: when one answer has no token in the top 20, its log-odds is a conservative bound (1,090 of 957,426 training cells, 0.11%).

### Lineage and reproduction

* **Shards and checks:** every shard has a manifest with its row count, id hash, prompt and model versions, and the code commit that wrote it. The teacher lane's `validate_local.py` and the data pipeline's `validate_teacher_shards` check all eight runs against the spec (0 errors).
* **Training data:** the data pipeline merges the shards by id into `Data-Pipeline/data/processed/teacher_labels.parquet` (`p_<label>` the stack, `p_<label>_raw`, `tfidf_oof_<label>`, `p_<label>_calqwen`), versioned with DVC; `load_data` joins it onto train and validation and checks that every row has every score.
* **MLflow:** experiment `distillate-teacher` on the team tracking server holds 13 runs logged from these records by `python -m src.teacher.mlflow_log` (#27): `pilot` with one child run per configuration (7), `labeling-train`, `labeling-gate` and `labeling-drift`, and `soft-labels-stack-v1` and `soft-labels-calqwen-v1` (the drift stream's per-batch AUCs as stepped metrics). Each run carries a hash of what it logged, so rerunning the logger adds no duplicates.
* **Licenses** (also section 11):
  * **Teacher model:** `Qwen/Qwen3-4B-Instruct-2507` is released under **Apache 2.0** (the model page's metadata and `LICENSE` file at the pinned revision `cdbee75…`).
  * **Data:** the two Jigsaw Kaggle competitions' rules permit academic use and forbid redistributing the data, so the soft labels (ids and scores derived from it) are kept in the private bucket and the private Hugging Face dataset only.
  * **Civil Comments shift sample:** the Hugging Face copy `google/civil_comments`, released under CC0 1.0.

## 4. Training

Details and every number: `Model-Pipeline/reports/student/phase2/README.md`; tuning:
`Model-Pipeline/docs/tuning.md`; the pipeline: `Model-Pipeline/docs/pipeline.md`.

**Data.** Jigsaw Toxic Comment train, restricted to the data pipeline's official split by id
(`Data-Pipeline/data/splits/`): 132,388 training and 21,363 validation comments, the same ids
and cleaning as the pipeline's `train.parquet` (`data.source: ids`; source SHA-256
`acde36e2…`, in `meta.json`). `sample_weight` is 1.0 everywhere: the Phase 1 bias step chose
reweighting factor 1. Soft labels: `soft_labels/stack-v1/p2-zs-v2b/` (32 shards, 159,571 rows,
none failed; `stack_sha256` `9b241804…` checked against the committed stack). 6.5% of training
comments exceed 256 tokens and are truncated.

**Recipe.**

| Setting | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Loss | mixed: alpha 0.3433 (gold), T 3.713 |
| Learning rate, schedule | 9.23e-5, linear with 6% warmup, AdamW, weight decay 0.01 |
| Epochs | 5 (best: the 5th), early stopping on validation mean AUC, patience 1 |
| Batch size, max length | 32, 256 tokens |
| Precision, seed | fp16 on GPU, 42 |

**Tuning.** Optuna TPE over alpha (0.1 to 0.9), T (1 to 5), learning rate (1e-5 to 1e-4, log),
epochs (1 to 4) and maximum length (128 or 256), on DistilBERT with the mixed loss; objective:
validation mean AUC. 28 trials in one merged study (`distilbert-mixed-v1`, two Kaggle workers on
T4s, one 11-hour session each). Best: trial 26, validation mean AUC 0.99364. All other arms reuse these
values. Every trial peaked at its last epoch, so one run at 5 epochs was added outside the
study; it is the one selected, though its gain over 4 epochs (+0.0002) is at the noise level.

**Thresholds.** Per label, F1-maximising on validation, within [0.01, 0.99]
(`thresholds.json`): toxicity 0.351, severe_toxicity 0.225, obscene 0.207, threat 0.232,
insult 0.325, identity_attack 0.290. Validation macro-F1 0.727 at these thresholds (0.675 at
0.5). **They don't transfer to Civil Comments:** on H2 the ranking holds (mean AUC 0.94) but
recall at these thresholds falls to 38% for toxicity, 21% for insult, 3 of 71 for
identity_attack and 0 of 28 for threat. Thresholds for news-comment traffic must be set on
Civil or drift data (Phase 3).

**Arms trained** (identical data, seed and tuned values; only the named setting changes):
gold-only (alpha = 1), stack only (alpha = 0), mixed, the attribution arms (gold plus TF-IDF
out-of-fold soft labels; gold plus calibrated-Qwen soft labels), mixed at 5 epochs, and the
mixed recipe on MiniLM-L12 and ModernBERT-base. Results: section 6 and the Phase 2 results
README. **Caveat:** every arm reuses the recipe tuned for the mixed loss, as planned, so the
gold-only arm trains at mixed's learning rate and epochs. On its own recipe gold-only reached
validation mean AUC 0.9901, against 0.98937 here: about 0.0007, the size of the whole in-domain
AUC gain of distillation (+0.0007 [+0.0000, +0.0013] on the gate set). That gain is within the
effect of the recipe choice; the PR-AUC gain (+0.064) and the cross-domain gain (+0.027 on
H2) are far larger and stand.

**Selection** (`select_best.py`, `configs/student.yaml` `select:`). Among the candidates that
pass the bias check, the highest validation mean AUC, ties within 0.0002 (the measured
run-to-run noise) broken by validation PR-AUC. Two decisions were made after the run and are
recorded with it:

- Candidates are the tuned recipe only (DistilBERT at 4 and 5 epochs, MiniLM-L12,
  ModernBERT-base). The ablation and attribution arms are controls. Under the rule as first
  written the TF-IDF attribution arm would have won on validation AUC while being worse on the
  gate set and H2.
- The CPU latency budget (p95 at most 50 ms, batch size 1, 2 vCPUs) is reported, not
  enforced: no fp32 package meets it (DistilBERT p95 144 ms, MiniLM-L12 103 ms, ModernBERT
  373 ms). It is enforced on the int8 package in Phase 3.

Result: `distilbert-mixed-5ep` (validation mean AUC 0.99368, PR-AUC 0.7678), tied with
ModernBERT (0.99349) and chosen on PR-AUC. The candidates are within noise of each other on
every evaluation set.

**Package and latency.** The §2.5 package (`model.onnx`, `tokenizer/`, `thresholds.json`,
`meta.json`), ONNX fp32 with dynamic axes; maximum difference from PyTorch 4.8e-6. CPU latency
(ONNX Runtime, batch 1, 2 threads, Kaggle Xeon 2.0 GHz, tokenization included): p50 30.4 ms,
p95 144.3 ms, p99 151.8 ms. **It misses the 50 ms target**; p95 is the full-length (256-token)
time. Phase 3: int8 with the parity check (validation AUC within 0.002), then MiniLM-L12 or
128 tokens (−0.0013 gate AUC) if needed.

**Compute.** Kaggle, NVIDIA T4, fp16. The selected run: 71 minutes of training. The Phase 2
run of all eight arms: 5.9 hours on two T4s (`kaggle/13_student_ablation_all.ipynb`, code
`80a4502`). MLflow experiment `distillate-student-ablation`, run
`358772663fc34a5e81275c3ed073e598` (in the session's store, the Kaggle version's output).

## 5. Evaluation data

| Set | Source | Rows | Used for |
|---|---|---|---|
| Wikipedia validation | Jigsaw Toxic Comment (Kaggle) train, ~15% stratified split | 21,363 | Model choice and thresholds |
| Gate set | Jigsaw Toxic Comment scored test set | 63,978 | Pass/fail regression check and final numbers only |
| H2 | Civil Comments test, 10,000 stratified rows | 10,000 | Cross-domain evaluation only |
| Bias set | Civil Comments identity-annotated test rows | 21,577 | Identity-slice bias evaluation only |
| Obfuscation | 3,000 toxic gate comments with seeded character attacks | 3,000 | Robustness only |

- Civil Comments metrics use five labels; `severe_toxicity` is excluded because almost no
  Civil rows reach 0.5 on it.
- Validation chooses between models and sets thresholds; the gate set, H2 and the bias set
  are never used for choosing.

## 6. Metrics and results

Definitions (`src/eval/metrics.py`, matching the teacher's metrics): mean column-wise ROC-AUC,
per-label PR-AUC (average precision), macro-F1 at the tuned thresholds, ECE (10 bins), Brier,
and the flag decision's precision, recall, F1 and false positive rate. Intervals are 95%
bootstrap intervals over 1,000 resamples.

### H1: in-domain (Wikipedia gate set)

TODO (Ruchita, from Shivam's prediction files)

| Arm | Mean AUC [95% CI] | Mean PR-AUC | Macro-F1 | Flag F1 | Flag FPR |
|---|---|---|---|---|---|
| TF-IDF baseline | | | | | |
| Gold-only (alpha = 1) | | | | | |
| Stack teacher only (alpha = 0) | | | | | |
| Mixed: gold + stack soft labels | | | | | |
| Attribution: gold + TF-IDF soft labels | | | | | |
| Attribution: gold + calibrated-Qwen soft labels | | | | | |

### H2: cross-domain (Civil Comments, five labels)

TODO (Ruchita): same arms and columns as H1.

Only the gap between the stack arm and the TF-IDF arm, especially on H2, can be credited to
the LLM. Paired bootstrap differences: TODO.

### Robustness (obfuscation set)

TODO (Ruchita): recall on the 3,000 obfuscated toxic gate comments versus their originals.

## 7. Sensitivity analysis

How the selected student's quality depends on its hyperparameters, which words drive its predictions, and how it holds up on short inputs and obfuscated text. Generated by `python -m src.sensitivity.run` (`Model-Pipeline/src/sensitivity/`); the full report, with plots and masked examples, is `Model-Pipeline/reports/sensitivity.md`, and the run is logged to MLflow (experiment `distillate-sensitivity`).

**Everything is on Wikipedia validation**, the split for choosing and diagnosing; the gate set is for final numbers only. Validation is read through `load_data` at a recorded DVC revision, and the report records that revision and the validation file's SHA-256. Metrics are unweighted; AUC and PR-AUC rank on logits. The student analysed is the selected one, `student-distilbert-mixed-5ep-20261112T210931Z` (section 4). It was trained for 5 epochs, one more than the study's range (1 to 4), so section 7.1's importances describe the space it was tuned in rather than its exact setting. TODO (Nishaant): the DVC revision, recorded at the run.

### 7.1 Hyperparameter importance

From the merged Optuna study `distilbert-mixed-v1` (28 completed trials, both halves of the sweep; `gs://distillate-mlops-data/optuna/distilbert-mixed-v1.db`), the share of the objective's variance (validation mean AUC) explained by each hyperparameter. Two evaluators, each recomputed under 11 seeds (42 to 52); the value at seed 42 with the range over the 11:

| Parameter | fANOVA | MDI |
|---|---|---|
| Maximum length | 0.548 (0.367 to 0.548) | 0.478 (0.328 to 0.478) |
| Epochs | 0.238 (0.238 to 0.341) | 0.224 (0.224 to 0.319) |
| α | 0.141 (0.104 to 0.229) | 0.172 (0.119 to 0.224) |
| Learning rate | 0.049 (0.035 to 0.114) | 0.089 (0.066 to 0.144) |
| Temperature T | 0.024 (0.011 to 0.037) | 0.037 (0.028 to 0.059) |

* Maximum length ranks first under all 11 seeds for both evaluators, and both give the same mean order.
* **The objective barely moves:** validation mean AUC runs from 0.988 to 0.9936 across the 28 trials, so these are shares of a small variance. Maximum length leads because every 128-token trial scores below every good 256-token trial; epochs comes second mostly because of the 1-epoch trials. Within the good region (256 tokens, 3 to 4 epochs), α from about 0.15 to 0.6 and T from about 2.5 to 4.5 score within about 0.001 of each other.
* **Caveat:** TPE samples mostly near the best region, so importance measured on its trials understates α and T far from it. The study's validation scores carry the validation caveat in section 3.

### 7.2 Token attributions (SHAP)

SHAP Partition explainer with a text masker on validation comments, per label: up to 12 comments each that the student caught (true positives), missed (false negatives) and flagged wrongly (false positives), 500 model calls per comment. Reported as the most-attributed words per label and outcome (a word must occur in at least 5 validation comments to be shown), plus masked example lines in which only the top few words are visible (at most a third of a comment's words). No comment text is published.

TODO (Nishaant): results, after the run on the selected student.

### 7.3 Input length

Validation rescored with the input truncated at 64, 128 and 256 **tokens**, and results by each comment's untruncated token length (≤64, 65–128, 129–256, >256). These bands are in tokens; the bias report's length slices (section 9) are in characters, so the two are not the same bands.

TODO (Nishaant): results, after the run.

### 7.4 Obfuscation strength

The data pipeline's character substitutions (a→4, e→3, i→1, l→1, o→0, s→5, t→7) applied to every validation comment at rates 0, 0.1, 0.25, 0.5, 0.75 and 1.0, seeded per comment and nested (a character changed at 0.25 is also changed at 0.5). Reported per rate: AUC, PR-AUC, recall and flag rate at the tuned thresholds, and the share of toxic comments caught when clean but missed once obfuscated. This is a sweep on validation; the fixed obfuscation set built from toxic gate comments is scored separately as robustness evidence (section 6).

TODO (Nishaant): results, after the run.

### 7.5 Findings

TODO (Nishaant): from `reports/sensitivity_findings.md`, after the run.

## 8. Gate and release criteria

A candidate is registered only if `src/eval/gate.py` passes (`configs/gate.yaml`):

- **Validation:** mean AUC at least TODO; per-label PR-AUC floors TODO.
- **No regression on the gate set** against production: mean AUC drop at most 0.005;
  PR-AUC drop at most 0.02 (toxicity, obscene, insult) or 0.05 (severe_toxicity,
  identity_attack, threat), set above measured bootstrap noise.
- **Bias:** on the Civil bias set, no identity slice (n >= 100) more than 10 points of FPR
  above overall, or rising more than 2 points over production. The bias report flags gaps
  from 5 points.

Final values: TODO (Ruchita, after team sign-off). Gate history: TODO (Phase 3 Grafana).

## 9. Bias and fairness

TODO (Ruchita). Link `reports/bias/bias_report_phase2.md`.

- **Slices:** Civil identity groups (gender, sexual orientation, religion, race or ethnicity,
  disability; attribute >= 0.5) and Wikipedia lexicon slices (identity mention, length in
  characters, obfuscation flag).
- **Metrics per identity group:** subgroup AUC, BPSN AUC, BNSP AUC, FPR, FNR and F1 at the
  tuned thresholds.

  | Group | n | Subgroup AUC | BPSN AUC | BNSP AUC | FPR | FNR | F1 |
  |---|---|---|---|---|---|---|---|
  | TODO | | | | | | | |

- **Findings:** TODO. Does distillation pass on, or amplify, the teacher's bias?
- **Mitigation:** TODO: whether the `sample_weight` variant was kept, and the trade-off.
- **Why no per-group thresholds:** the service never knows a commenter's identity, so it
  applies one threshold per label to everyone.
- **Phase 1 context:** the TF-IDF baseline already showed sexual-orientation slices about 8
  points above overall FPR on Civil, and identity slices with toxic rates 16 to 21 points
  above overall in the data itself (16.0 to 21.2 in the Phase 1 bias report).

## 10. Limitations and risks

- **Domain shift:** trained on Wikipedia talk pages; performance drops on news comments
  (Civil Comments). TODO: quantify from H2.
- **Identity-term bias:** comments mentioning identity groups are more often labelled toxic
  in the data, which a model can learn as a shortcut, raising false positives for those groups.
- **Approximate lexicon:** Wikipedia slices rely on word matching against
  `config/identity_terms.txt`, which misses unlisted references and matches innocent uses.
- **Rater bias:** gold labels come from crowd raters, whose own biases are in the labels.
- **Missing groups:** no age or location fields, so those groups are not evaluated.
- **Obfuscation:** character-level attacks can lower recall. TODO: quantify.
- **Rare labels:** threat and identity_attack have few positives, so their metrics are noisy.
- **Teacher dependence:** the soft labels blend a gold-trained TF-IDF model with Qwen, so they
  carry gold-label information; bias in either can pass into the student.

## 11. License lineage

TODO (Ruchita): confirm each entry against its source before finalising.

| Component | Source | Terms to confirm |
|---|---|---|
| Wikipedia toxicity data | Kaggle: jigsaw-toxic-comment-classification-challenge | Competition rules (academic/educational use, no redistribution); underlying data licence |
| Civil Comments data | Kaggle: jigsaw-unintended-bias-in-toxicity-classification | Competition rules (academic/educational use, no redistribution); underlying data licence |
| Teacher LLM | Qwen3-4B-Instruct-2507 | Apache 2.0 (checked at the pinned revision) |
| Student base model | `distilbert-base-uncased` (Hugging Face) | Apache 2.0 (Ruchita to confirm) |
| Llama Guard 3-1B | Benchmark only, never used for training | Llama 3.2 Community License (Nishaant to confirm with the Phase 3 benchmark) |

Comment text is never redistributed: the repository and any public artifact hold only ids,
scores and metrics.

## 12. Privacy

- The service never logs comment text; logs hold `text_sha256`, `char_len` and scores only.
- Raw text stays in private storage (the DVC bucket and private Kaggle datasets shared only
  with teammates who accepted the competition rules).
- Demo text is masked by default.

## Changelog

| Date | Change |
|---|---|
| TODO | Skeleton created (Ruchita) |
| TODO | Review fixes: eval-set sizes, data-level gap range, section 10 teacher wording, Qwen licence (Ruchita) |
| 2026-11-11 | Sections 3 (teacher and soft labels) and 7 (sensitivity analysis, 7.1 filled) (Nishaant) |
| 2026-11-13 | Section 3: the ablation's attribution results, the leak check's result and verdict, the validation caveat's wording on what the stack was fitted on; section 7: the selected student (Nishaant) |
| 2026-11-13 | Section 4 (training), section 1 version and architecture, student base model (Shivam) |
