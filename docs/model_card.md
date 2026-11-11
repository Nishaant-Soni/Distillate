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
- **Version:** TODO (Shivam): MLflow model version of the selected candidate.
- **Type:** multi-label text classifier, six sigmoid outputs (`toxicity`, `severe_toxicity`,
  `obscene`, `threat`, `insult`, `identity_attack`).
- **Architecture:** TODO (Shivam): the architecture chosen by the selection rule
  (candidates: DistilBERT, MiniLM-L12, ModernBERT-base), maximum length.
- **Training method:** knowledge distillation from a teacher (section 3) mixed with gold labels:
  `alpha * BCE(gold) + (1 - alpha) * T^2 * BCE(soft)`. TODO (Shivam): tuned alpha and T.
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

TODO (Shivam). Suggested content:

- Training data: Wikipedia train split (DVC hash), soft-label columns from
  `teacher_labels.parquet`, `sample_weight` if the mitigation variant is kept (section 9).
- Tuning: Optuna (TPE) search space, trial count, budget and best values (link
  `docs/tuning.md`).
- Thresholds: per label, F1-maximising on Wikipedia validation, within [0.01, 0.99]
  (`thresholds.json`, `thresholds_report.json`).
- Selection rule (`select_best.py`): among candidates passing the bias check and the CPU
  latency budget, highest mean validation AUC, ties broken by mean PR-AUC.
- Compute: hardware, run time, MLflow run id.

## 5. Evaluation data

| Set | Source | Rows | Used for |
|---|---|---|---|
| Wikipedia validation | Jigsaw Toxic Comment (Kaggle) train, ~15% stratified split | TODO | Model choice and thresholds |
| Gate set | Jigsaw Toxic Comment scored test set | 63,978 | Pass/fail regression check and final numbers only |
| H2 | Civil Comments test, 10,000 stratified rows | 10,000 | Cross-domain evaluation only |
| Bias set | Civil Comments identity-annotated test rows | TODO | Identity-slice bias evaluation only |
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
  points above overall FPR on Civil, and identity slices with toxic rates 17 to 21 points
  above overall in the data itself.

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
- **Teacher dependence:** the soft labels are a gold-trained baseline corrected by Qwen; any
  bias in either part can pass into the student.

## 11. License lineage

TODO (Ruchita): confirm each entry against its source before finalising.

| Component | Source | Terms to confirm |
|---|---|---|
| Wikipedia toxicity data | Kaggle: jigsaw-toxic-comment-classification-challenge | Competition rules (academic/educational use, no redistribution); underlying data licence |
| Civil Comments data | Kaggle: jigsaw-unintended-bias-in-toxicity-classification | Competition rules (academic/educational use, no redistribution); underlying data licence |
| Teacher LLM | Qwen3-4B-Instruct-2507 | Model licence |
| Student base model | TODO (selected architecture) | Model licence |
| Llama Guard 3-1B | Benchmark only, never used for training | Llama licence (Nishaant to confirm) |

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
