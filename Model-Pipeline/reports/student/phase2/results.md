# Student results

Bootstrap: 1000 resamples (95% percentile intervals). Thresholds: each arm's own, chosen on validation.

## H1: in-domain (Wikipedia gate set)

| Arm | Mean AUC [95% CI] | Mean PR-AUC | Macro-F1 | Flag F1 | Flag FPR |
|---|---|---|---|---|---|
| distilbert-gold | 0.9841 [0.9831, 0.9849] | 0.6231 | 0.5875 | 0.6894 | 0.0757 |
| distilbert-stack | 0.9835 [0.9826, 0.9842] | 0.6724 | 0.6124 | 0.6986 | 0.0563 |
| distilbert-mixed | 0.9848 [0.9840, 0.9854] | 0.6866 | 0.6182 | 0.6834 | 0.0751 |
| distilbert-tfidf | 0.9826 [0.9815, 0.9835] | 0.6509 | 0.6048 | 0.6733 | 0.0803 |
| distilbert-calqwen | 0.9783 [0.9771, 0.9793] | 0.6380 | 0.5904 | 0.6764 | 0.0531 |
| distilbert-mixed-5ep | 0.9845 [0.9836, 0.9852] | 0.6814 | 0.6238 | 0.6814 | 0.0757 |
| minilm-mixed | 0.9849 [0.9841, 0.9857] | 0.6884 | 0.6171 | 0.6800 | 0.0809 |
| modernbert-mixed | 0.9846 [0.9839, 0.9853] | 0.6827 | 0.6167 | 0.6855 | 0.0728 |

## H2: cross-domain (Civil Comments, five labels)

| Arm | Mean AUC [95% CI] | Mean PR-AUC | Macro-F1 | Flag F1 | Flag FPR |
|---|---|---|---|---|---|
| distilbert-gold | 0.9139 [0.8993, 0.9285] | 0.3711 | 0.2873 | 0.4862 | 0.0179 |
| distilbert-stack | 0.9359 [0.9225, 0.9471] | 0.3757 | 0.2515 | 0.4262 | 0.0097 |
| distilbert-mixed | 0.9410 [0.9280, 0.9510] | 0.4001 | 0.2895 | 0.4861 | 0.0172 |
| distilbert-tfidf | 0.9081 [0.8895, 0.9252] | 0.3453 | 0.2577 | 0.4814 | 0.0172 |
| distilbert-calqwen | 0.9206 [0.9055, 0.9350] | 0.3076 | 0.2110 | 0.3997 | 0.0159 |
| distilbert-mixed-5ep | 0.9399 [0.9263, 0.9502] | 0.3956 | 0.2603 | 0.4812 | 0.0183 |
| minilm-mixed | 0.9438 [0.9336, 0.9524] | 0.3878 | 0.2865 | 0.5134 | 0.0248 |
| modernbert-mixed | 0.9384 [0.9233, 0.9498] | 0.3871 | 0.2931 | 0.4920 | 0.0179 |

## Validation, the leak check, robustness, latency

| Arm | Val mean AUC | Val mean PR-AUC | Val-to-gate AUC drop | Obfuscation toxicity recall (orig -> obf) | Latency p50 / p95 (ms) | Best epoch |
|---|---|---|---|---|---|---|
| distilbert-gold | 0.98937 | 0.7033 | 0.0053 | 0.900 -> 0.649 | 30.2 / 146.3 | 1 |
| distilbert-stack | 0.99303 | 0.7559 | 0.0096 | 0.808 -> 0.559 | 31.0 / 145.7 | 3 |
| distilbert-mixed | 0.99348 | 0.7646 | 0.0087 | 0.880 -> 0.653 | 29.8 / 146.3 | 3 |
| distilbert-tfidf | 0.99426 | 0.7788 | 0.0117 | 0.884 -> 0.680 | 30.8 / 146.6 | 3 |
| distilbert-calqwen | 0.98834 | 0.6982 | 0.0101 | 0.746 -> 0.647 | 30.1 / 146.4 | 1 |
| distilbert-mixed-5ep | 0.99368 | 0.7678 | 0.0092 | 0.880 -> 0.655 | 30.4 / 144.3 | 4 |
| minilm-mixed | 0.99187 | 0.7305 | 0.0069 | 0.905 -> 0.705 | 19.5 / 102.9 | 3 |
| modernbert-mixed | 0.99349 | 0.7615 | 0.0089 | 0.877 -> 0.718 | 75.1 / 373.4 | 3 |

**Leak check** (docs/tuning.md): validation-to-gate mean AUC drop 0.0087 for mixed vs 0.0053 for gold (difference +0.0034; PR-AUC difference -0.0022). A clearly larger drop for mixed would mean the validation labels in the teacher's inputs matter.

## Paired bootstrap differences (a - b)

| Comparison | Set | Mean AUC diff [95% CI] | P(diff > 0) | Mean PR-AUC diff [95% CI] |
|---|---|---|---|---|
| distilbert-mixed - distilbert-gold | gate | +0.0007 [+0.0000, +0.0013] | 0.983 | +0.0635 [+0.0510, +0.0748] |
| distilbert-mixed - distilbert-gold | h2_eval | +0.0271 [+0.0125, +0.0414] | 1.000 | +0.0290 [-0.0141, +0.0645] |
| distilbert-stack - distilbert-tfidf | gate | +0.0009 [+0.0005, +0.0014] | 1.000 | +0.0215 [+0.0145, +0.0286] |
| distilbert-stack - distilbert-tfidf | h2_eval | +0.0278 [+0.0181, +0.0382] | 1.000 | +0.0304 [+0.0061, +0.0506] |
| distilbert-calqwen - distilbert-tfidf | gate | -0.0043 [-0.0053, -0.0034] | 0.000 | -0.0129 [-0.0256, -0.0011] |
| distilbert-calqwen - distilbert-tfidf | h2_eval | +0.0125 [-0.0080, +0.0329] | 0.882 | -0.0377 [-0.0814, +0.0085] |
| distilbert-stack - distilbert-calqwen | gate | +0.0052 [+0.0046, +0.0060] | 1.000 | +0.0344 [+0.0239, +0.0454] |
| distilbert-stack - distilbert-calqwen | h2_eval | +0.0153 [+0.0010, +0.0304] | 0.984 | +0.0682 [+0.0266, +0.1007] |
| distilbert-mixed - distilbert-stack | gate | +0.0013 [+0.0011, +0.0015] | 1.000 | +0.0142 [+0.0098, +0.0185] |
| distilbert-mixed - distilbert-stack | h2_eval | +0.0051 [+0.0030, +0.0072] | 1.000 | +0.0244 [+0.0124, +0.0367] |

## Truncation (gate set, same model, shorter inputs)

| Arm | Max tokens | Mean AUC | Mean PR-AUC | Macro-F1 |
|---|---|---|---|---|
| distilbert-mixed | 64 | 0.9792 | 0.6675 | 0.6139 |
| distilbert-mixed | 128 | 0.9835 | 0.6810 | 0.6179 |
| distilbert-mixed | 256 | 0.9848 | 0.6866 | 0.6182 |
