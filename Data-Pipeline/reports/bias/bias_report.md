# Bias report: TF-IDF + logistic regression toxicity baseline

Model `tfidf-lr-bias-v1`, label `toxicity`. Slices with at least 100 rows are judged; a slice is flagged when its FPR is 5 or more points above the overall FPR. The decision threshold maximizes F1 on Wikipedia validation.

## 1. Data-level and model-level slices before mitigation

### Civil identity rows (bias set)

n = 21577, toxic rate 11.1%, threshold 0.25, AUC 0.830, F1 0.442, FPR 3.8%, FNR 63.1%

| Slice | n | Toxic rate | Rate gap (pts) | FPR | FPR gap (pts) | FNR | AUC | Status |
|---|---|---|---|---|---|---|---|---|
| gender:female | 2602 | 13.5% | +2.3 | 5.1% | +1.3 | 64.0% | 0.811 | ok |
| gender:male | 2112 | 15.2% | +4.0 | 5.3% | +1.5 | 58.1% | 0.802 | ok |
| gender:transgender | 129 | 20.2% | +9.0 | 7.8% | +4.0 | 84.6% | 0.724 | ok |
| gender:other_gender | 0 | n/a | n/a | n/a | n/a | n/a | n/a | too small |
| gender:any | 3960 | 14.5% | +3.4 | 5.4% | +1.7 | 62.7% | 0.803 | ok |
| sexual_orientation:homosexual_gay_or_lesbian | 538 | 28.6% | +17.5 | 12.2% | +8.5 | 59.1% | 0.763 | **flagged** |
| sexual_orientation:bisexual | 21 | 14.3% | +3.2 | 0.0% | -3.8 | 66.7% | 0.889 | too small |
| sexual_orientation:heterosexual | 66 | 19.7% | +8.6 | 13.2% | +9.4 | 53.8% | 0.729 | too small |
| sexual_orientation:other_sexual_orientation | 1 | 0.0% | -11.1 | 0.0% | -3.8 | n/a | n/a | too small |
| sexual_orientation:any | 570 | 28.1% | +17.0 | 11.7% | +7.9 | 60.0% | 0.763 | **flagged** |
| religion:christian | 2109 | 10.0% | -1.2 | 2.3% | -1.5 | 70.0% | 0.805 | ok |
| religion:jewish | 411 | 16.5% | +5.4 | 8.5% | +4.7 | 69.1% | 0.784 | ok |
| religion:muslim | 1054 | 22.1% | +11.0 | 2.9% | -0.9 | 73.4% | 0.744 | ok |
| religion:hindu | 28 | 14.3% | +3.2 | 0.0% | -3.8 | 75.0% | 0.635 | too small |
| religion:buddhist | 27 | 14.8% | +3.7 | 0.0% | -3.8 | 100.0% | 0.565 | too small |
| religion:atheist | 149 | 6.7% | -4.4 | 2.9% | -0.9 | 70.0% | 0.929 | ok |
| religion:other_religion | 15 | 6.7% | -4.5 | 0.0% | -3.8 | 100.0% | 0.071 | too small |
| religion:any | 3316 | 13.4% | +2.3 | 2.9% | -0.9 | 71.2% | 0.796 | ok |
| race_or_ethnicity:black | 761 | 32.3% | +21.2 | 7.2% | +3.4 | 72.4% | 0.736 | ok |
| race_or_ethnicity:white | 1178 | 30.0% | +18.8 | 7.3% | +3.5 | 70.0% | 0.746 | ok |
| race_or_ethnicity:asian | 217 | 9.7% | -1.4 | 1.0% | -2.8 | 66.7% | 0.855 | ok |
| race_or_ethnicity:latino | 121 | 23.1% | +12.0 | 3.2% | -0.6 | 85.7% | 0.722 | ok |
| race_or_ethnicity:other_race_or_ethnicity | 16 | 37.5% | +26.4 | 0.0% | -3.8 | 50.0% | 0.717 | too small |
| race_or_ethnicity:any | 1911 | 27.2% | +16.0 | 5.9% | +2.1 | 70.1% | 0.762 | ok |
| disability:physical_disability | 3 | 0.0% | -11.1 | 0.0% | -3.8 | n/a | n/a | too small |
| disability:intellectual_or_learning_disability | 14 | 7.1% | -4.0 | 0.0% | -3.8 | 100.0% | 0.923 | too small |
| disability:psychiatric_or_mental_illness | 238 | 22.7% | +11.6 | 5.4% | +1.7 | 57.4% | 0.834 | ok |
| disability:other_disability | 0 | n/a | n/a | n/a | n/a | n/a | n/a | too small |
| disability:any | 252 | 21.8% | +10.7 | 5.1% | +1.3 | 58.2% | 0.835 | ok |

### Wikipedia validation

n = 21363, toxic rate 9.6%, threshold 0.25, AUC 0.975, F1 0.793, FPR 2.1%, FNR 21.2%

| Slice | n | Toxic rate | Rate gap (pts) | FPR | FPR gap (pts) | FNR | AUC | Status |
|---|---|---|---|---|---|---|---|---|
| identity_mention | 1445 | 18.3% | +8.7 | 4.7% | +2.6 | 20.4% | 0.966 | ok |
| no_identity_mention | 19918 | 9.0% | -0.6 | 2.0% | -0.2 | 21.3% | 0.975 | ok |
| obfuscation | 387 | 14.2% | +4.6 | 4.8% | +2.7 | 9.1% | 0.982 | ok |
| no_obfuscation | 20976 | 9.6% | -0.1 | 2.1% | -0.0 | 21.6% | 0.975 | ok |
| length_0_99 | 5731 | 15.9% | +6.3 | 3.3% | +1.2 | 15.5% | 0.974 | ok |
| length_100_499 | 11408 | 8.2% | -1.4 | 1.9% | -0.2 | 24.5% | 0.974 | ok |
| length_500+ | 4224 | 4.9% | -4.7 | 1.2% | -0.9 | 31.6% | 0.973 | ok |

## 2. Biases found

Flagged on the Civil identity rows before mitigation: `sexual_orientation:homosexual_gay_or_lesbian`, `sexual_orientation:any`.

## 3. Mitigation

Non-toxic Wikipedia training rows that mention an identity term were reweighted by each factor in [1, 2, 3, 5]. A factor passes when the validation `identity_mention` slice's FPR gap is under 5 points and validation AUC drops by at most 0.005. The smallest passing factor is kept. Chosen factor: **1**.

| Factor | Val AUC | AUC drop | Identity FPR gap (pts) | Threshold | Passes |
|---|---|---|---|---|---|
| 1 | 0.9747 | 0.0000 | +2.6 | 0.25 | yes |
| 2 | 0.9735 | 0.0013 | +1.0 | 0.27 | yes |
| 3 | 0.9730 | 0.0017 | +0.6 | 0.31 | yes |
| 5 | 0.9722 | 0.0025 | +0.1 | 0.27 | yes |

## 4. Before and after on the Civil identity rows

Before: n = 21577, toxic rate 11.1%, threshold 0.25, AUC 0.830, F1 0.442, FPR 3.8%, FNR 63.1%

After: n = 21577, toxic rate 11.1%, threshold 0.25, AUC 0.830, F1 0.442, FPR 3.8%, FNR 63.1%

| Slice | n | FPR before | FPR after | Gap before (pts) | Gap after (pts) | Status |
|---|---|---|---|---|---|---|
| gender:female | 2602 | 5.1% | 5.1% | +1.3 | +1.3 | ok → ok |
| gender:male | 2112 | 5.3% | 5.3% | +1.5 | +1.5 | ok → ok |
| gender:transgender | 129 | 7.8% | 7.8% | +4.0 | +4.0 | ok → ok |
| gender:other_gender | 0 | n/a | n/a | n/a | n/a | too small |
| gender:any | 3960 | 5.4% | 5.4% | +1.7 | +1.7 | ok → ok |
| sexual_orientation:homosexual_gay_or_lesbian | 538 | 12.2% | 12.2% | +8.5 | +8.5 | **flagged** → **flagged** |
| sexual_orientation:bisexual | 21 | 0.0% | 0.0% | -3.8 | -3.8 | too small |
| sexual_orientation:heterosexual | 66 | 13.2% | 13.2% | +9.4 | +9.4 | too small |
| sexual_orientation:other_sexual_orientation | 1 | 0.0% | 0.0% | -3.8 | -3.8 | too small |
| sexual_orientation:any | 570 | 11.7% | 11.7% | +7.9 | +7.9 | **flagged** → **flagged** |
| religion:christian | 2109 | 2.3% | 2.3% | -1.5 | -1.5 | ok → ok |
| religion:jewish | 411 | 8.5% | 8.5% | +4.7 | +4.7 | ok → ok |
| religion:muslim | 1054 | 2.9% | 2.9% | -0.9 | -0.9 | ok → ok |
| religion:hindu | 28 | 0.0% | 0.0% | -3.8 | -3.8 | too small |
| religion:buddhist | 27 | 0.0% | 0.0% | -3.8 | -3.8 | too small |
| religion:atheist | 149 | 2.9% | 2.9% | -0.9 | -0.9 | ok → ok |
| religion:other_religion | 15 | 0.0% | 0.0% | -3.8 | -3.8 | too small |
| religion:any | 3316 | 2.9% | 2.9% | -0.9 | -0.9 | ok → ok |
| race_or_ethnicity:black | 761 | 7.2% | 7.2% | +3.4 | +3.4 | ok → ok |
| race_or_ethnicity:white | 1178 | 7.3% | 7.3% | +3.5 | +3.5 | ok → ok |
| race_or_ethnicity:asian | 217 | 1.0% | 1.0% | -2.8 | -2.8 | ok → ok |
| race_or_ethnicity:latino | 121 | 3.2% | 3.2% | -0.6 | -0.6 | ok → ok |
| race_or_ethnicity:other_race_or_ethnicity | 16 | 0.0% | 0.0% | -3.8 | -3.8 | too small |
| race_or_ethnicity:any | 1911 | 5.9% | 5.9% | +2.1 | +2.1 | ok → ok |
| disability:physical_disability | 3 | 0.0% | 0.0% | -3.8 | -3.8 | too small |
| disability:intellectual_or_learning_disability | 14 | 0.0% | 0.0% | -3.8 | -3.8 | too small |
| disability:psychiatric_or_mental_illness | 238 | 5.4% | 5.4% | +1.7 | +1.7 | ok → ok |
| disability:other_disability | 0 | n/a | n/a | n/a | n/a | too small |
| disability:any | 252 | 5.1% | 5.1% | +1.3 | +1.3 | ok → ok |

Still flagged after mitigation: `sexual_orientation:homosexual_gay_or_lesbian`, `sexual_orientation:any`.

## 5. Trade-off

AUC 0.830 → 0.830, F1 0.442 → 0.442 on the Civil identity rows.

## 6. Limits

- **Approximate lexicon.** Wikipedia slices rely on word matching against `config/identity_terms.txt`, which misses identity references it doesn't list and matches innocent uses ("black car").
- **Rater bias.** Gold labels come from crowd raters, whose own biases are in the labels the model is measured against.
- **No age or location fields.** Neither dataset records them, so those groups can't be checked.
- **No per-group thresholds.** They would need the commenter's identity at prediction time, which the service doesn't have and shouldn't infer, and they treat groups differently by design. One threshold plus reweighting is used instead.
