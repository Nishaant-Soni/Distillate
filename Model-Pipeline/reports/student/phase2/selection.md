# Student selection

Rule: among candidates (distilbert-mixed, distilbert-mixed-5ep, minilm-mixed, modernbert-mixed) that pass the bias check (configs/gate.yaml bias section (slice FPR gap), no production model), highest validation mean AUC; ties by validation mean PR-AUC (tie: within 0.0002 AUC). The CPU latency budget (p95_ms <= 50 ms, batch size 1) is reported, not enforced.

Note: Candidates and latency enforcement were set on 2026-11-13, after the Phase 2 run: the ablation and attribution arms are controls, and no fp32 ONNX package meets the 50 ms p95 budget (2 threads, batch 1), so the budget is reported here and enforced on the int8 package in Phase 3 (work plan v4: int8, then MiniLM-L12 or a shorter maximum length).

**Selected: `distilbert-mixed-5ep`** (`student-distilbert-mixed-5ep-20261112T210931Z`): validation mean AUC 0.99368, PR-AUC 0.7678, latency p95 144.3 ms on 2 threads.

Tied within 0.0002 AUC: distilbert-mixed-5ep, modernbert-mixed; the higher validation PR-AUC wins.

**The selected model misses the latency budget** (p95_ms <= 50 ms, batch size 1).

| Arm | Val mean AUC | Val mean PR-AUC | Bias check | Worst slice gap (pts) | Latency p50 / p95 (ms) | Eligible | Reasons and notes |
|---|---|---|---|---|---|---|---|
| distilbert-mixed-5ep | 0.99368 | 0.7678 | pass | +4.6 (race_or_ethnicity:white) | 30.4 / 144.3 | yes | latency p95_ms 144.3 ms over 50 ms (reported, not enforced) |
| modernbert-mixed | 0.99349 | 0.7615 | pass | +5.1 (race_or_ethnicity:white) | 75.1 / 373.4 | yes | latency p95_ms 373.4 ms over 50 ms (reported, not enforced) |
| distilbert-mixed | 0.99348 | 0.7646 | pass | +3.9 (race_or_ethnicity:white) | 29.8 / 146.3 | yes | latency p95_ms 146.3 ms over 50 ms (reported, not enforced) |
| minilm-mixed | 0.99187 | 0.7305 | pass | +5.7 (race_or_ethnicity:white) | 19.5 / 102.9 | yes | latency p95_ms 102.9 ms over 50 ms (reported, not enforced) |
| distilbert-tfidf | 0.99426 | 0.7788 | pass | +8.0 (sexual_orientation:any) | 30.8 / 146.6 | no | control arm, not a candidate (select.candidates); latency p95_ms 146.6 ms over 50 ms (reported, not enforced) |
| distilbert-stack | 0.99303 | 0.7559 | pass | +2.6 (race_or_ethnicity:white) | 31.0 / 145.7 | no | control arm, not a candidate (select.candidates); latency p95_ms 145.7 ms over 50 ms (reported, not enforced) |
| distilbert-gold | 0.98937 | 0.7033 | pass | +6.0 (gender:transgender) | 30.2 / 146.3 | no | control arm, not a candidate (select.candidates); latency p95_ms 146.3 ms over 50 ms (reported, not enforced) |
| distilbert-calqwen | 0.98834 | 0.6982 | pass | +5.9 (race_or_ethnicity:white) | 30.1 / 146.4 | no | control arm, not a candidate (select.candidates); latency p95_ms 146.4 ms over 50 ms (reported, not enforced) |
