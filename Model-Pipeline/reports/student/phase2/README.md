# Phase 2 student results

The reported run of the student lane (work plan v4 §5): eight arms trained under identical
conditions, scored on every evaluation set, checked for bias, timed on CPU, compared with
bootstrap intervals, and one selected.

* **Run:** `kaggle/13_student_ablation_all.ipynb`, one committed Kaggle session (T4 x2), Nov 12,
  2026, 5.88 hours, code `80a4502`. Every arm through `python -m src.pipeline.run` with the tuned
  values (trial 26, `docs/tuning.md`), seed 42, official split by id (132,388 train / 21,363
  validation).
* **Selection:** rerun from the run's outputs with the rule as decided after the run
  (`configs/student.yaml` `select:`, below); the results tables are unchanged by it (they were
  rebuilt from `phase2_results.zip` and `results.md` matches the session's exactly). `results.md`
  was then rebuilt from the same zip with the teacher comparisons paired like for like (mixed
  against TF-IDF and calibrated Qwen, same loss; `src/eval/results.py` `PAIRS`) and the best
  epoch counted from 1; every other number is unchanged.

## Files

| File | What it is |
|---|---|
| `selection.md`, `selection.json` | The selection with the decided rule: every arm, eligible or not, and why |
| `selection_as_run.md` | The selection as the notebook ran it (rule as first written): no arm passed |
| `results.md`, `results.json` | H1 and H2 tables (1,000-resample bootstrap), paired differences, leak check, obfuscation, latency, truncation |
| `plots/` | Per-label AUC (dots) and PR-AUC (bars) per split, the H1 / H2 figure, the selected arm's confusion matrices |
| `runs/<arm>/` | Each arm's `summary.json`, `meta.json`, thresholds, train and eval metrics, `latency.json`, gate report |
| `arm_status.json`, `eval_tables.json`, `run_info.json` | The session: minutes per arm, evaluation-table row and positive counts, code and tuned values |

Not in git (in the Kaggle version's output): the prediction files (ids and scores) and gold
files, in `phase2_results.zip` (`runs/<arm>/predictions/`, `gold/`); the weights and the ONNX
packages, in `student/<run>/`.

## Selected: `distilbert-mixed-5ep`

`student-distilbert-mixed-5ep-20261112T210931Z`: DistilBERT (`distilbert-base-uncased`), the
tuned mixed recipe (α 0.343, T 3.71, lr 9.23e-5, 256 tokens) trained for 5 epochs. Validation
mean AUC 0.99368, PR-AUC 0.7678. Gate set 0.9845 [0.9836, 0.9852]; Civil H2 0.9399 [0.9263,
0.9502]. Bias check passed (worst slice +4.6 FPR points, `race_or_ethnicity:white`, under the
5-point flag). Gate report: passed (validation floors and bias; no production model yet).
MLflow run `ab34619a27e84b07aae044e3a414c0be` (team server, experiment
`distillate-student-ablation`; Kaggle run `358772663fc34a5e81275c3ed073e598`).

**The rule** (decided Nov 13, after the run, and recorded in `configs/student.yaml`):

1. **Candidates:** the tuned recipe only: `distilbert-mixed`, `distilbert-mixed-5ep`,
   `minilm-mixed`, `modernbert-mixed`. The gold, stack, TF-IDF and calibrated-Qwen arms are the
   loss ablation and the attribution controls; they answer H1, H2 and what the LLM adds, not
   what ships. Under the rule as first written, `distilbert-tfidf` would have been chosen
   (highest validation AUC, 0.99426) although it is worse on the gate set (0.9826 vs 0.9848)
   and H2 (0.9081 vs 0.9410), and has the largest validation-to-gate drop.
2. **Bias check:** every arm passes.
3. **Latency, reported, not enforced:** no fp32 package meets the 50 ms p95 budget (below).
   The budget is enforced on the int8 package in Phase 3.
4. **Highest validation mean AUC, ties within 0.0002 by PR-AUC:** `distilbert-mixed-5ep`
   (0.99368) ties `modernbert-mixed` (0.99349) and wins on PR-AUC (0.7678 vs 0.7615).
   `distilbert-mixed` (0.99348) sits exactly at the tie line, with a lower PR-AUC (0.7646).

The three DistilBERT and ModernBERT candidates are within noise of each other on every set;
the choice among them is the rule's, not a measured difference.

## Findings

**H1, in-domain (gate set).** Distillation adds little AUC and a lot of PR-AUC: mixed vs gold
+0.0007 [+0.0000, +0.0013] mean AUC, +0.064 [+0.051, +0.075] mean PR-AUC. **Caveat on the AUC
gain:** every arm reuses the recipe tuned for the mixed loss (the v4 plan), so the gold arm
trains at mixed's learning rate and epochs. On its own recipe, gold-only reached validation
mean AUC 0.9901 (`distilbert-gold-20261111T073929Z`, the run behind the `gate.yaml` floors),
against 0.98937 here: about 0.0007, the size of the whole H1 AUC gain. The in-domain AUC gain
is within the effect of the recipe choice; the PR-AUC gain and H2 are far larger and stand.

**H2, cross-domain (Civil Comments).** The gain is large: mixed vs gold +0.027 [+0.013,
+0.041] mean AUC (0.9410 vs 0.9139).

**What the LLM adds** (mixed vs TF-IDF: stack against TF-IDF soft labels, both with the mixed
loss at the tuned α, the only gap creditable to the LLM): +0.0022 [+0.0017, +0.0027] on the gate
set and **+0.033 [+0.024, +0.044] on H2**. Calibrated Qwen alone is worse than TF-IDF in-domain
(−0.0043) and not clearly better on H2 (+0.013 [−0.008, +0.033]); the stack beats both of its
parts (mixed vs calibrated Qwen: +0.020 [+0.006, +0.036] on H2). The stack-only arm (α 0) is
compared only with mixed, since against the α 0.343 arms it would mix the teacher with the loss:
gold labels on top of the stack help on both sets (mixed vs stack: +0.0013 gate, +0.0051 H2).

**Architectures** (same recipe and tuned values, tuned on DistilBERT):

| | Val AUC | Gate AUC | H2 AUC | p95 latency (fp32) | Training |
|---|---|---|---|---|---|
| DistilBERT (4 epochs) | 0.99348 | 0.9848 | 0.9410 | 146 ms | 54 min |
| MiniLM-L12 | 0.99187 | 0.9849 | 0.9438 | 103 ms | 45 min |
| ModernBERT-base | 0.99349 | 0.9846 | 0.9384 | 373 ms | 184 min |

MiniLM-L12 is as good as DistilBERT on the gate set and H2 (intervals overlap) with a lower
validation AUC, and is the fastest. ModernBERT brings nothing for 2.5 times the latency.

**Five epochs** (the check outside the study, `docs/tuning.md`): validation +0.0002 (within
the one-seed noise), gate −0.0003, H2 −0.0011. Every mixed run still peaked at its last epoch, but the
fifth epoch adds about nothing.

**Leak check** (validation-to-gate drop in mean AUC): gold 0.0053; mixed 0.0087; stack 0.0096;
calibrated Qwen 0.0101; TF-IDF 0.0117; MiniLM 0.0069. Mixed drops 0.0034 more than gold, so the
validation gain of distillation overstates its in-domain gain (validation +0.0041, gate
+0.0007). It is not clearly the leak: the calibrated-Qwen arm, whose soft labels carry almost
no validation-label information, drops as much (0.0101). The TF-IDF arm, whose soft labels do,
has the largest drop and the highest validation AUC, consistent with a small leak there. Gate
and H2 numbers are clean either way (no teacher saw them).

**Gold-only overfits early:** its validation AUC peaks at its second epoch and falls after; every
distilled arm still improves at its last epoch.

**Robustness.** Toxicity recall on the 3,000 obfuscated toxic gate comments falls from about
0.88 to 0.65 for DistilBERT (0.877 → 0.718 for ModernBERT, 0.905 → 0.705 for MiniLM).

**The thresholds don't transfer to Civil.** The validation-tuned thresholds keep their
precision on H2 but lose most of the recall, while H2 AUC is 0.94: the ranking transfers, the
operating point doesn't. Selected model on H2: toxicity recall 38% (306 of 799, FPR 1.8%),
obscene 37% (21 of 57), insult 21% (125 of 583), identity_attack 3 of 71, threat 0 of 28
(`plots/confusion_distilbert-mixed-5ep_h2_eval.png`). Phase 3: thresholds set on Civil or the
drift data for news-comment traffic.

**Truncation** (DistilBERT mixed, gate set): 256 tokens 0.9848, 128 tokens 0.9835 (−0.0013),
64 tokens 0.9792.

**Latency** (fp32 ONNX, batch 1, 2 threads, Kaggle's Xeon 2.0 GHz, tokenization included, 300
validation comments): DistilBERT p50 30 ms, p95 144 to 147 ms; MiniLM 20 / 103 ms; ModernBERT
75 / 373 ms. p95 is the time at the full 256 tokens: 6.5% of training comments reach it, so p95 and p99
are almost equal. Phase 3's levers, in the work plan's order: int8 (with the parity check
under 0.002 AUC), then MiniLM-L12 or 128 tokens (−0.0013 gate AUC).
