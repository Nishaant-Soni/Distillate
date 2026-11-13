# Gate report: PASSED

Candidate `student-distilbert-calqwen-20261112T200941Z`; production `none`.

| Check | Result | Value | Limit | Detail |
|---|---|---|---|---|
| validation: mean AUC | pass | 0.9883 | 0.9800 | at least the minimum |
| validation: PR-AUC toxicity | pass | 0.8909 | 0.8600 | at least the minimum |
| validation: PR-AUC severe_toxicity | pass | 0.4427 | 0.3800 | at least the minimum |
| validation: PR-AUC obscene | pass | 0.8914 | 0.8700 | at least the minimum |
| validation: PR-AUC threat | pass | 0.6329 | 0.4500 | at least the minimum |
| validation: PR-AUC insult | pass | 0.7909 | 0.7600 | at least the minimum |
| validation: PR-AUC identity_attack | pass | 0.5402 | 0.4500 | at least the minimum |
| gate set: no regression | skipped | n/a | n/a | skipped: no production model to compare against |
| bias: slice FPR limits | pass | 18 | n/a | 18 slices judged (n >= 100), 0 over a limit |

## Validation metrics

n = 21363, mean AUC 0.9883, mean PR-AUC 0.6982, macro-F1 0.6670, flag precision 0.8837, recall 0.7531, F1 0.8132, FPR 0.0113.

## Gate metrics

n = 63978, mean AUC 0.9783, mean PR-AUC 0.6380, macro-F1 0.5904, flag precision 0.6083, recall 0.7618, F1 0.6764, FPR 0.0531.

## Bias slices (`toxicity`, Civil bias set)

Overall FPR 0.0202. Slices with at least 100 rows are judged. The bias report flags a slice 5+ points above overall; the gate blocks only above 10 points, so it stops large gaps while smaller ones are reported and tracked. It also blocks a slice whose FPR rises more than 2 points over production.

| Slice | n | FPR | Gap (pts) | Production FPR | Rise (pts) | Result |
|---|---|---|---|---|---|---|
| gender:female | 2602 | 0.0377 | 1.8 | n/a | n/a | pass |
| gender:male | 2112 | 0.0329 | 1.3 | n/a | n/a | pass |
| gender:transgender | 129 | 0.0777 | 5.7 | n/a | n/a | pass |
| gender:any | 3960 | 0.0357 | 1.6 | n/a | n/a | pass |
| sexual_orientation:homosexual_gay_or_lesbian | 538 | 0.0547 | 3.5 | n/a | n/a | pass |
| sexual_orientation:bisexual | 21 | 0.0000 | -2.0 | n/a | n/a | too small |
| sexual_orientation:heterosexual | 66 | 0.0755 | 5.5 | n/a | n/a | too small |
| sexual_orientation:other_sexual_orientation | 1 | 0.0000 | -2.0 | n/a | n/a | too small |
| sexual_orientation:any | 570 | 0.0512 | 3.1 | n/a | n/a | pass |
| religion:christian | 2109 | 0.0095 | -1.1 | n/a | n/a | pass |
| religion:jewish | 411 | 0.0496 | 2.9 | n/a | n/a | pass |
| religion:muslim | 1054 | 0.0231 | 0.3 | n/a | n/a | pass |
| religion:hindu | 28 | 0.0000 | -2.0 | n/a | n/a | too small |
| religion:buddhist | 27 | 0.0000 | -2.0 | n/a | n/a | too small |
| religion:atheist | 149 | 0.0000 | -2.0 | n/a | n/a | pass |
| religion:other_religion | 15 | 0.0000 | -2.0 | n/a | n/a | too small |
| religion:any | 3316 | 0.0153 | -0.5 | n/a | n/a | pass |
| race_or_ethnicity:black | 761 | 0.0427 | 2.3 | n/a | n/a | pass |
| race_or_ethnicity:white | 1178 | 0.0788 | 5.9 | n/a | n/a | pass |
| race_or_ethnicity:asian | 217 | 0.0510 | 3.1 | n/a | n/a | pass |
| race_or_ethnicity:latino | 121 | 0.0538 | 3.4 | n/a | n/a | pass |
| race_or_ethnicity:other_race_or_ethnicity | 16 | 0.2000 | 18.0 | n/a | n/a | too small |
| race_or_ethnicity:any | 1911 | 0.0596 | 3.9 | n/a | n/a | pass |
| disability:physical_disability | 3 | 0.0000 | -2.0 | n/a | n/a | too small |
| disability:intellectual_or_learning_disability | 14 | 0.0000 | -2.0 | n/a | n/a | too small |
| disability:psychiatric_or_mental_illness | 238 | 0.0272 | 0.7 | n/a | n/a | pass |
| disability:any | 252 | 0.0254 | 0.5 | n/a | n/a | pass |
