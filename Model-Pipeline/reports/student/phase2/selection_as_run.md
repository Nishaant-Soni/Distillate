# Student selection

Rule: among candidates that pass the bias check (configs/gate.yaml bias section (slice FPR gap), no production model) and the CPU latency budget (p95_ms <= 50 ms, batch size 1), highest validation mean AUC; ties by validation mean PR-AUC (tie: within 0.0002 AUC).

**No candidate passes the rule.**

| Arm | Val mean AUC | Val mean PR-AUC | Bias check | Worst slice gap (pts) | Latency p50 / p95 (ms) | Eligible | Reasons |
|---|---|---|---|---|---|---|---|
| distilbert-tfidf | 0.99426 | 0.7788 | pass | +8.0 (sexual_orientation:any) | 30.8 / 146.6 | no | latency p95_ms 146.6 ms over 50 ms |
| distilbert-mixed-5ep | 0.99368 | 0.7678 | pass | +4.6 (race_or_ethnicity:white) | 30.4 / 144.3 | no | latency p95_ms 144.3 ms over 50 ms |
| modernbert-mixed | 0.99349 | 0.7615 | pass | +5.1 (race_or_ethnicity:white) | 75.1 / 373.4 | no | latency p95_ms 373.4 ms over 50 ms |
| distilbert-mixed | 0.99348 | 0.7646 | pass | +3.9 (race_or_ethnicity:white) | 29.8 / 146.3 | no | latency p95_ms 146.3 ms over 50 ms |
| distilbert-stack | 0.99303 | 0.7559 | pass | +2.6 (race_or_ethnicity:white) | 31.0 / 145.7 | no | latency p95_ms 145.7 ms over 50 ms |
| minilm-mixed | 0.99187 | 0.7305 | pass | +5.7 (race_or_ethnicity:white) | 19.5 / 102.9 | no | latency p95_ms 102.9 ms over 50 ms |
| distilbert-gold | 0.98937 | 0.7033 | pass | +6.0 (gender:transgender) | 30.2 / 146.3 | no | latency p95_ms 146.3 ms over 50 ms |
| distilbert-calqwen | 0.98834 | 0.6982 | pass | +5.9 (race_or_ethnicity:white) | 30.1 / 146.4 | no | latency p95_ms 146.4 ms over 50 ms |
