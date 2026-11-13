# Gate report: PASSED

Candidate `student-distilbert-mixed-20261112T171037Z`; production `none`.

| Check | Result | Value | Limit | Detail |
|---|---|---|---|---|
| validation: mean AUC | pass | 0.9935 | 0.9800 | at least the minimum |
| validation: PR-AUC toxicity | pass | 0.9370 | 0.8600 | at least the minimum |
| validation: PR-AUC severe_toxicity | pass | 0.5479 | 0.3800 | at least the minimum |
| validation: PR-AUC obscene | pass | 0.9386 | 0.8700 | at least the minimum |
| validation: PR-AUC threat | pass | 0.6650 | 0.4500 | at least the minimum |
| validation: PR-AUC insult | pass | 0.8442 | 0.7600 | at least the minimum |
| validation: PR-AUC identity_attack | pass | 0.6551 | 0.4500 | at least the minimum |
| gate set: no regression | skipped | n/a | n/a | skipped: no production model to compare against |
| bias: slice FPR limits | pass | 18 | n/a | 18 slices judged (n >= 100), 0 over a limit |

## Validation metrics

n = 21363, mean AUC 0.9935, mean PR-AUC 0.7646, macro-F1 0.7216, flag precision 0.8893, recall 0.8334, F1 0.8605, FPR 0.0118.

## Gate metrics

n = 63978, mean AUC 0.9848, mean PR-AUC 0.6866, macro-F1 0.6182, flag precision 0.5588, recall 0.8795, F1 0.6834, FPR 0.0751.

## Bias slices (`toxicity`, Civil bias set)

Overall FPR 0.0204. Slices with at least 100 rows are judged. The bias report flags a slice 5+ points above overall; the gate blocks only above 10 points, so it stops large gaps while smaller ones are reported and tracked. It also blocks a slice whose FPR rises more than 2 points over production.

| Slice | n | FPR | Gap (pts) | Production FPR | Rise (pts) | Result |
|---|---|---|---|---|---|---|
| gender:female | 2602 | 0.0289 | 0.8 | n/a | n/a | pass |
| gender:male | 2112 | 0.0357 | 1.5 | n/a | n/a | pass |
| gender:transgender | 129 | 0.0485 | 2.8 | n/a | n/a | pass |
| gender:any | 3960 | 0.0325 | 1.2 | n/a | n/a | pass |
| sexual_orientation:homosexual_gay_or_lesbian | 538 | 0.0521 | 3.2 | n/a | n/a | pass |
| sexual_orientation:bisexual | 21 | 0.0000 | -2.0 | n/a | n/a | too small |
| sexual_orientation:heterosexual | 66 | 0.0943 | 7.4 | n/a | n/a | too small |
| sexual_orientation:other_sexual_orientation | 1 | 0.0000 | -2.0 | n/a | n/a | too small |
| sexual_orientation:any | 570 | 0.0512 | 3.1 | n/a | n/a | pass |
| religion:christian | 2109 | 0.0116 | -0.9 | n/a | n/a | pass |
| religion:jewish | 411 | 0.0262 | 0.6 | n/a | n/a | pass |
| religion:muslim | 1054 | 0.0158 | -0.5 | n/a | n/a | pass |
| religion:hindu | 28 | 0.0000 | -2.0 | n/a | n/a | too small |
| religion:buddhist | 27 | 0.0000 | -2.0 | n/a | n/a | too small |
| religion:atheist | 149 | 0.0000 | -2.0 | n/a | n/a | pass |
| religion:other_religion | 15 | 0.0000 | -2.0 | n/a | n/a | too small |
| religion:any | 3316 | 0.0122 | -0.8 | n/a | n/a | pass |
| race_or_ethnicity:black | 761 | 0.0447 | 2.4 | n/a | n/a | pass |
| race_or_ethnicity:white | 1178 | 0.0594 | 3.9 | n/a | n/a | pass |
| race_or_ethnicity:asian | 217 | 0.0102 | -1.0 | n/a | n/a | pass |
| race_or_ethnicity:latino | 121 | 0.0323 | 1.2 | n/a | n/a | pass |
| race_or_ethnicity:other_race_or_ethnicity | 16 | 0.0000 | -2.0 | n/a | n/a | too small |
| race_or_ethnicity:any | 1911 | 0.0453 | 2.5 | n/a | n/a | pass |
| disability:physical_disability | 3 | 0.0000 | -2.0 | n/a | n/a | too small |
| disability:intellectual_or_learning_disability | 14 | 0.0000 | -2.0 | n/a | n/a | too small |
| disability:psychiatric_or_mental_illness | 238 | 0.0217 | 0.1 | n/a | n/a | pass |
| disability:any | 252 | 0.0203 | -0.0 | n/a | n/a | pass |
