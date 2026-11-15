# Gate report: PASSED

Candidate `student-distilbert-stack-20261112T181004Z`; production `none`.

| Check | Result | Value | Limit | Detail |
|---|---|---|---|---|
| validation: mean AUC | pass | 0.9930 | 0.9800 | at least the minimum |
| validation: PR-AUC toxicity | pass | 0.9285 | 0.8600 | at least the minimum |
| validation: PR-AUC severe_toxicity | pass | 0.5319 | 0.3800 | at least the minimum |
| validation: PR-AUC obscene | pass | 0.9368 | 0.8700 | at least the minimum |
| validation: PR-AUC threat | pass | 0.6617 | 0.4500 | at least the minimum |
| validation: PR-AUC insult | pass | 0.8393 | 0.7600 | at least the minimum |
| validation: PR-AUC identity_attack | pass | 0.6372 | 0.4500 | at least the minimum |
| gate set: no regression | skipped | n/a | n/a | skipped: no production model to compare against |
| bias: slice FPR limits | pass | 18 | n/a | 18 slices judged (n >= 100), 0 over a limit |

## Validation metrics

n = 21363, mean AUC 0.9930, mean PR-AUC 0.7559, macro-F1 0.7155, flag precision 0.9194, recall 0.7747, F1 0.8408, FPR 0.0077.

## Gate metrics

n = 63978, mean AUC 0.9835, mean PR-AUC 0.6724, macro-F1 0.6124, flag precision 0.6104, recall 0.8164, F1 0.6986, FPR 0.0563.

## Bias slices (`toxicity`, Civil bias set)

Overall FPR 0.0106. Slices with at least 100 rows are judged. The bias report flags a slice 5+ points above overall; the gate blocks only above 10 points, so it stops large gaps while smaller ones are reported and tracked. It also blocks a slice whose FPR rises more than 2 points over production.

| Slice | n | FPR | Gap (pts) | Production FPR | Rise (pts) | Result |
|---|---|---|---|---|---|---|
| gender:female | 2602 | 0.0142 | 0.4 | n/a | n/a | pass |
| gender:male | 2112 | 0.0190 | 0.8 | n/a | n/a | pass |
| gender:transgender | 129 | 0.0194 | 0.9 | n/a | n/a | pass |
| gender:any | 3960 | 0.0168 | 0.6 | n/a | n/a | pass |
| sexual_orientation:homosexual_gay_or_lesbian | 538 | 0.0339 | 2.3 | n/a | n/a | pass |
| sexual_orientation:bisexual | 21 | 0.0000 | -1.1 | n/a | n/a | too small |
| sexual_orientation:heterosexual | 66 | 0.0755 | 6.5 | n/a | n/a | too small |
| sexual_orientation:other_sexual_orientation | 1 | 0.0000 | -1.1 | n/a | n/a | too small |
| sexual_orientation:any | 570 | 0.0341 | 2.4 | n/a | n/a | pass |
| religion:christian | 2109 | 0.0068 | -0.4 | n/a | n/a | pass |
| religion:jewish | 411 | 0.0233 | 1.3 | n/a | n/a | pass |
| religion:muslim | 1054 | 0.0110 | 0.0 | n/a | n/a | pass |
| religion:hindu | 28 | 0.0000 | -1.1 | n/a | n/a | too small |
| religion:buddhist | 27 | 0.0000 | -1.1 | n/a | n/a | too small |
| religion:atheist | 149 | 0.0000 | -1.1 | n/a | n/a | pass |
| religion:other_religion | 15 | 0.0000 | -1.1 | n/a | n/a | too small |
| religion:any | 3316 | 0.0077 | -0.3 | n/a | n/a | pass |
| race_or_ethnicity:black | 761 | 0.0311 | 2.0 | n/a | n/a | pass |
| race_or_ethnicity:white | 1178 | 0.0364 | 2.6 | n/a | n/a | pass |
| race_or_ethnicity:asian | 217 | 0.0051 | -0.5 | n/a | n/a | pass |
| race_or_ethnicity:latino | 121 | 0.0323 | 2.2 | n/a | n/a | pass |
| race_or_ethnicity:other_race_or_ethnicity | 16 | 0.0000 | -1.1 | n/a | n/a | too small |
| race_or_ethnicity:any | 1911 | 0.0266 | 1.6 | n/a | n/a | pass |
| disability:physical_disability | 3 | 0.0000 | -1.1 | n/a | n/a | too small |
| disability:intellectual_or_learning_disability | 14 | 0.0000 | -1.1 | n/a | n/a | too small |
| disability:psychiatric_or_mental_illness | 238 | 0.0109 | 0.0 | n/a | n/a | pass |
| disability:any | 252 | 0.0102 | -0.0 | n/a | n/a | pass |
