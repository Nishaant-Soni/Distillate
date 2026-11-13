# Gate report: PASSED

Candidate `student-minilm-mixed-20261112T180732Z`; production `none`.

| Check | Result | Value | Limit | Detail |
|---|---|---|---|---|
| validation: mean AUC | pass | 0.9919 | 0.9800 | at least the minimum |
| validation: PR-AUC toxicity | pass | 0.9273 | 0.8600 | at least the minimum |
| validation: PR-AUC severe_toxicity | pass | 0.4589 | 0.3800 | at least the minimum |
| validation: PR-AUC obscene | pass | 0.9261 | 0.8700 | at least the minimum |
| validation: PR-AUC threat | pass | 0.6126 | 0.4500 | at least the minimum |
| validation: PR-AUC insult | pass | 0.8244 | 0.7600 | at least the minimum |
| validation: PR-AUC identity_attack | pass | 0.6339 | 0.4500 | at least the minimum |
| gate set: no regression | skipped | n/a | n/a | skipped: no production model to compare against |
| bias: slice FPR limits | pass | 18 | n/a | 18 slices judged (n >= 100), 0 over a limit |

## Validation metrics

n = 21363, mean AUC 0.9919, mean PR-AUC 0.7305, macro-F1 0.7063, flag precision 0.8600, recall 0.8458, F1 0.8528, FPR 0.0156.

## Gate metrics

n = 63978, mean AUC 0.9849, mean PR-AUC 0.6884, macro-F1 0.6171, flag precision 0.5461, recall 0.9007, F1 0.6800, FPR 0.0809.

## Bias slices (`toxicity`, Civil bias set)

Overall FPR 0.0283. Slices with at least 100 rows are judged. The bias report flags a slice 5+ points above overall; the gate blocks only above 10 points, so it stops large gaps while smaller ones are reported and tracked. It also blocks a slice whose FPR rises more than 2 points over production.

| Slice | n | FPR | Gap (pts) | Production FPR | Rise (pts) | Result |
|---|---|---|---|---|---|---|
| gender:female | 2602 | 0.0409 | 1.3 | n/a | n/a | pass |
| gender:male | 2112 | 0.0497 | 2.1 | n/a | n/a | pass |
| gender:transgender | 129 | 0.0583 | 3.0 | n/a | n/a | pass |
| gender:any | 3960 | 0.0467 | 1.8 | n/a | n/a | pass |
| sexual_orientation:homosexual_gay_or_lesbian | 538 | 0.0807 | 5.2 | n/a | n/a | pass |
| sexual_orientation:bisexual | 21 | 0.0000 | -2.8 | n/a | n/a | too small |
| sexual_orientation:heterosexual | 66 | 0.1509 | 12.3 | n/a | n/a | too small |
| sexual_orientation:other_sexual_orientation | 1 | 0.0000 | -2.8 | n/a | n/a | too small |
| sexual_orientation:any | 570 | 0.0805 | 5.2 | n/a | n/a | pass |
| religion:christian | 2109 | 0.0142 | -1.4 | n/a | n/a | pass |
| religion:jewish | 411 | 0.0496 | 2.1 | n/a | n/a | pass |
| religion:muslim | 1054 | 0.0280 | -0.0 | n/a | n/a | pass |
| religion:hindu | 28 | 0.0000 | -2.8 | n/a | n/a | too small |
| religion:buddhist | 27 | 0.0000 | -2.8 | n/a | n/a | too small |
| religion:atheist | 149 | 0.0072 | -2.1 | n/a | n/a | pass |
| religion:other_religion | 15 | 0.0000 | -2.8 | n/a | n/a | too small |
| religion:any | 3316 | 0.0188 | -0.9 | n/a | n/a | pass |
| race_or_ethnicity:black | 761 | 0.0524 | 2.4 | n/a | n/a | pass |
| race_or_ethnicity:white | 1178 | 0.0848 | 5.7 | n/a | n/a | pass |
| race_or_ethnicity:asian | 217 | 0.0153 | -1.3 | n/a | n/a | pass |
| race_or_ethnicity:latino | 121 | 0.0323 | 0.4 | n/a | n/a | pass |
| race_or_ethnicity:other_race_or_ethnicity | 16 | 0.1000 | 7.2 | n/a | n/a | too small |
| race_or_ethnicity:any | 1911 | 0.0603 | 3.2 | n/a | n/a | pass |
| disability:physical_disability | 3 | 0.0000 | -2.8 | n/a | n/a | too small |
| disability:intellectual_or_learning_disability | 14 | 0.0000 | -2.8 | n/a | n/a | too small |
| disability:psychiatric_or_mental_illness | 238 | 0.0217 | -0.7 | n/a | n/a | pass |
| disability:any | 252 | 0.0203 | -0.8 | n/a | n/a | pass |
