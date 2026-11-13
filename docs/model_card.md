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

TODO (Nishaant). Suggested content:

- Teacher `stack-v1`: per-label logistic regression on Qwen3-4B-Instruct-2507's zero-shot
  logit (prompt `p2-zs-v2b`) and an out-of-fold TF-IDF logit; frozen in
  `Model-Pipeline/artifacts/stack/stack_p2-zs-v2b.json`.
- **Description to use:** the soft labels are *a gold-trained baseline corrected by Qwen*,
  not independent LLM judgments (the TF-IDF weight is 6 to 16 times Qwen's per unit of logit).
- Cross-fitting: no training row's soft label comes from a model that saw its gold label.
- Teacher accuracy on the gate subset (weighted, within-stratum bootstrap, from `src/eval`):

  | Teacher | Mean AUC [95% CI] | Mean PR-AUC | ECE |
  |---|---|---|---|
  | Stack (soft labels) | TODO | TODO | TODO |
  | TF-IDF | TODO | TODO | TODO |
  | Raw Qwen | TODO | TODO | TODO |
  | Calibrated Qwen | TODO | TODO | TODO |

- Prompt-injection results.
- Test-split vs training-split accuracy (the raw teacher is weaker on test).

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
pass the bias check, the highest validation mean AUC, ties within 0.0002 (a lower bound on
run-to-run noise, from one pair of runs) broken by validation PR-AUC. Two decisions were made after the run and are
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

TODO (Nishaant): hyperparameter importance (Optuna), SHAP token attributions, performance
against input length (truncation at 64, 128 and 256 **tokens**) and against obfuscation
strength. Link `reports/sensitivity.md`.

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
| 2026-11-13 | Section 4 (training), section 1 version and architecture, student base model (Shivam) |
