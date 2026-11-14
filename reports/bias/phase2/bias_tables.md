# Phase 2 bias tables (generated)

Slices with n >= 100 are judged; flagged (⚑) when FPR is 5+ points above overall. FPR, FNR and F1 use each arm's tuned threshold from Wikipedia validation; on Civil that threshold gives a low overall FPR, so read the FPR ratio (slice / overall) alongside the gap. BPSN AUC doesn't depend on the threshold, so it leads the arm comparison. Wikipedia length bands are in characters.

## Warnings

- wikipedia: val.parquet has no has_identity_term, char_len, has_obfuscation; those slices are skipped (load_data's validation table has them)
- wikipedia section skipped: no slices could be built

## Arms compared (Civil, toxicity, tuned thresholds)

| Slice | gold BPSN | gold FPR ratio | gold FPR gap | stack BPSN | stack FPR ratio | stack FPR gap | tfidf BPSN | tfidf FPR ratio | tfidf FPR gap | calqwen BPSN | calqwen FPR ratio | calqwen FPR gap | mixed5ep BPSN | mixed5ep FPR ratio | mixed5ep FPR gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender:female | 0.860 | 1.2x | +0.3 | 0.849 | 1.3x | +0.4 | 0.836 | 1.4x | +0.8 | 0.816 | 1.9x | +1.8 | 0.851 | 1.3x | +0.7 |
| gender:male | 0.848 | 1.3x | +0.6 | 0.834 | 1.8x | +0.8 | 0.825 | 1.6x | +1.3 | 0.816 | 1.6x | +1.3 | 0.838 | 1.6x | +1.3 |
| gender:transgender | 0.793 | 4.3x | +6.0 ⚑ | 0.813 | 1.8x | +0.9 | 0.792 | 3.0x | +4.5 | 0.780 | 3.8x | +5.7 ⚑ | 0.822 | 2.2x | +2.7 |
| gender:any | 0.851 | 1.3x | +0.5 | 0.838 | 1.6x | +0.6 | 0.826 | 1.5x | +1.1 | 0.813 | 1.8x | +1.6 | 0.841 | 1.5x | +1.1 |
| sexual_orientation:homosexual_gay_or_lesbian | 0.736 | 3.5x | +4.5 | 0.758 | 3.2x | +2.3 | 0.709 | 4.5x | +7.9 ⚑ | 0.798 | 2.7x | +3.5 | 0.766 | 2.5x | +3.3 |
| sexual_orientation:any | 0.744 | 3.3x | +4.1 | 0.761 | 3.2x | +2.4 | 0.713 | 4.6x | +8.0 ⚑ | 0.801 | 2.5x | +3.1 | 0.769 | 2.5x | +3.2 |
| religion:christian | 0.912 | 0.4x | -1.1 | 0.911 | 0.6x | -0.4 | 0.906 | 0.5x | -1.2 | 0.888 | 0.5x | -1.1 | 0.914 | 0.5x | -1.2 |
| religion:jewish | 0.834 | 1.6x | +1.1 | 0.842 | 2.2x | +1.3 | 0.833 | 1.8x | +1.8 | 0.796 | 2.5x | +2.9 | 0.848 | 1.6x | +1.3 |
| religion:muslim | 0.862 | 1.1x | +0.2 | 0.856 | 1.0x | +0.0 | 0.866 | 0.7x | -0.8 | 0.806 | 1.1x | +0.3 | 0.865 | 1.1x | +0.3 |
| religion:atheist | 0.878 | 0.4x | -1.1 | 0.911 | 0.0x | -1.1 | 0.895 | 0.3x | -1.5 | 0.896 | 0.0x | -2.0 | 0.913 | 0.0x | -2.2 |
| religion:any | 0.899 | 0.6x | -0.6 | 0.897 | 0.7x | -0.3 | 0.895 | 0.6x | -0.8 | 0.863 | 0.8x | -0.5 | 0.902 | 0.7x | -0.6 |
| race_or_ethnicity:black | 0.847 | 1.0x | -0.0 | 0.792 | 2.9x | +2.0 | 0.787 | 2.9x | +4.2 | 0.767 | 2.1x | +2.3 | 0.808 | 2.3x | +2.9 |
| race_or_ethnicity:white | 0.825 | 2.1x | +2.0 | 0.757 | 3.4x | +2.6 | 0.772 | 2.8x | +4.1 | 0.689 | 3.9x | +5.9 ⚑ | 0.773 | 3.1x | +4.6 |
| race_or_ethnicity:asian | 0.866 | 1.4x | +0.8 | 0.894 | 0.5x | -0.5 | 0.883 | 0.5x | -1.2 | 0.839 | 2.5x | +3.1 | 0.894 | 0.7x | -0.6 |
| race_or_ethnicity:latino | 0.902 | 0.0x | -1.8 | 0.862 | 3.0x | +2.2 | 0.875 | 1.4x | +1.0 | 0.806 | 2.7x | +3.4 | 0.874 | 1.5x | +1.1 |
| race_or_ethnicity:any | 0.848 | 1.5x | +0.9 | 0.794 | 2.5x | +1.6 | 0.800 | 2.3x | +2.9 | 0.737 | 3.0x | +3.9 | 0.807 | 2.4x | +2.9 |
| disability:psychiatric_or_mental_illness | 0.828 | 1.2x | +0.4 | 0.827 | 1.0x | +0.0 | 0.805 | 0.7x | -0.6 | 0.838 | 1.3x | +0.7 | 0.832 | 1.3x | +0.5 |
| disability:any | 0.836 | 1.1x | +0.2 | 0.835 | 1.0x | -0.0 | 0.814 | 0.7x | -0.7 | 0.844 | 1.3x | +0.5 | 0.839 | 1.2x | +0.4 |

## Arms compared at matched overall FPR (5%)

Each arm at the threshold giving the same overall Civil FPR, for comparing arms only (not a deployment threshold).

| Slice | gold BPSN | gold FPR ratio | gold FPR gap | stack BPSN | stack FPR ratio | stack FPR gap | tfidf BPSN | tfidf FPR ratio | tfidf FPR gap | calqwen BPSN | calqwen FPR ratio | calqwen FPR gap | mixed5ep BPSN | mixed5ep FPR ratio | mixed5ep FPR gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gender:female | 0.860 | 1.1x | +0.4 | 0.849 | 1.3x | +1.7 | 0.836 | 1.3x | +1.6 | 0.816 | 1.7x | +3.7 | 0.851 | 1.4x | +1.9 |
| gender:male | 0.848 | 1.4x | +1.8 | 0.834 | 1.7x | +3.5 | 0.825 | 1.5x | +2.6 | 0.816 | 1.9x | +4.4 | 0.838 | 1.6x | +3.1 |
| gender:transgender | 0.793 | 2.3x | +6.7 ⚑ | 0.813 | 2.9x | +9.6 ⚑ | 0.792 | 2.3x | +6.6 ⚑ | 0.780 | 2.3x | +6.6 ⚑ | 0.822 | 2.3x | +6.6 ⚑ |
| gender:any | 0.851 | 1.2x | +1.0 | 0.838 | 1.6x | +2.8 | 0.826 | 1.5x | +2.3 | 0.813 | 1.8x | +3.9 | 0.841 | 1.5x | +2.5 |
| sexual_orientation:homosexual_gay_or_lesbian | 0.736 | 3.2x | +11.2 ⚑ | 0.758 | 3.3x | +11.7 ⚑ | 0.709 | 3.7x | +13.7 ⚑ | 0.798 | 1.9x | +4.6 | 0.766 | 2.9x | +9.3 ⚑ |
| sexual_orientation:any | 0.744 | 3.2x | +10.9 ⚑ | 0.761 | 3.2x | +11.1 ⚑ | 0.713 | 3.7x | +13.3 ⚑ | 0.801 | 1.9x | +4.3 | 0.769 | 2.8x | +8.9 ⚑ |
| religion:christian | 0.912 | 0.4x | -2.9 | 0.911 | 0.7x | -1.7 | 0.906 | 0.5x | -2.3 | 0.888 | 0.7x | -1.5 | 0.914 | 0.7x | -1.7 |
| religion:jewish | 0.834 | 1.3x | +1.7 | 0.842 | 1.6x | +2.9 | 0.833 | 1.5x | +2.3 | 0.796 | 2.1x | +5.5 ⚑ | 0.848 | 1.6x | +2.9 |
| religion:muslim | 0.862 | 1.1x | +0.4 | 0.856 | 1.1x | +0.5 | 0.866 | 0.9x | -0.3 | 0.806 | 1.5x | +2.3 | 0.865 | 1.0x | +0.1 |
| religion:atheist | 0.878 | 0.7x | -1.4 | 0.911 | 0.4x | -2.8 | 0.895 | 0.6x | -2.1 | 0.896 | 0.6x | -2.1 | 0.913 | 0.6x | -2.1 |
| religion:any | 0.899 | 0.7x | -1.5 | 0.897 | 0.8x | -1.0 | 0.895 | 0.7x | -1.4 | 0.863 | 1.0x | -0.2 | 0.902 | 0.8x | -1.0 |
| race_or_ethnicity:black | 0.847 | 1.4x | +1.8 | 0.792 | 2.4x | +7.0 ⚑ | 0.787 | 2.2x | +6.1 ⚑ | 0.767 | 2.5x | +7.4 ⚑ | 0.808 | 2.3x | +6.5 ⚑ |
| race_or_ethnicity:white | 0.825 | 1.7x | +3.7 | 0.757 | 3.0x | +10.0 ⚑ | 0.772 | 2.6x | +7.8 ⚑ | 0.689 | 3.8x | +14.2 ⚑ | 0.773 | 2.8x | +8.9 ⚑ |
| race_or_ethnicity:asian | 0.866 | 1.4x | +2.1 | 0.894 | 0.6x | -1.9 | 0.883 | 0.6x | -1.9 | 0.839 | 1.9x | +4.7 | 0.894 | 0.8x | -0.9 |
| race_or_ethnicity:latino | 0.902 | 0.2x | -3.9 | 0.862 | 0.9x | -0.7 | 0.875 | 0.6x | -1.8 | 0.806 | 2.2x | +5.8 ⚑ | 0.874 | 0.9x | -0.7 |
| race_or_ethnicity:any | 0.848 | 1.5x | +2.6 | 0.794 | 2.4x | +7.0 ⚑ | 0.800 | 2.1x | +5.4 ⚑ | 0.737 | 3.0x | +10.0 ⚑ | 0.807 | 2.3x | +6.3 ⚑ |
| disability:psychiatric_or_mental_illness | 0.828 | 1.4x | +2.1 | 0.827 | 1.5x | +2.6 | 0.805 | 2.0x | +4.8 | 0.838 | 1.0x | -0.1 | 0.832 | 1.8x | +4.2 |
| disability:any | 0.836 | 1.3x | +1.6 | 0.835 | 1.4x | +2.1 | 0.814 | 1.8x | +4.1 | 0.844 | 0.9x | -0.4 | 0.839 | 1.7x | +3.6 |

## Paired bootstrap: BPSN AUC, mixed5ep minus gold (1000 replicates)

| Slice | n | mixed5ep BPSN | gold BPSN | Difference [95% CI] | Interval excludes 0 |
|---|---|---|---|---|---|
| gender:female | 2602 | 0.851 | 0.860 | -0.009 [-0.017, -0.000] | yes |
| gender:male | 2112 | 0.838 | 0.848 | -0.010 [-0.018, -0.001] | yes |
| gender:transgender | 129 | 0.822 | 0.793 | +0.028 [-0.007, +0.067] | no |
| gender:any | 3960 | 0.841 | 0.851 | -0.009 [-0.017, -0.001] | yes |
| sexual_orientation:homosexual_gay_or_lesbian | 538 | 0.766 | 0.736 | +0.030 [+0.010, +0.046] | yes |
| sexual_orientation:any | 570 | 0.769 | 0.744 | +0.026 [+0.007, +0.042] | yes |
| religion:christian | 2109 | 0.914 | 0.912 | +0.002 [-0.005, +0.009] | no |
| religion:jewish | 411 | 0.848 | 0.834 | +0.014 [-0.001, +0.030] | no |
| religion:muslim | 1054 | 0.865 | 0.862 | +0.002 [-0.009, +0.013] | no |
| religion:atheist | 149 | 0.913 | 0.878 | +0.035 [+0.012, +0.056] | yes |
| religion:any | 3316 | 0.902 | 0.899 | +0.003 [-0.004, +0.009] | no |
| race_or_ethnicity:black | 761 | 0.808 | 0.847 | -0.039 [-0.052, -0.026] | yes |
| race_or_ethnicity:white | 1178 | 0.773 | 0.825 | -0.053 [-0.065, -0.042] | yes |
| race_or_ethnicity:asian | 217 | 0.894 | 0.866 | +0.028 [+0.012, +0.044] | yes |
| race_or_ethnicity:latino | 121 | 0.874 | 0.902 | -0.028 [-0.059, -0.001] | yes |
| race_or_ethnicity:any | 1911 | 0.807 | 0.848 | -0.041 [-0.051, -0.032] | yes |
| disability:psychiatric_or_mental_illness | 238 | 0.832 | 0.828 | +0.004 [-0.014, +0.022] | no |
| disability:any | 252 | 0.839 | 0.836 | +0.003 [-0.013, +0.020] | no |

## Paired bootstrap: BPSN AUC, tfidf minus gold (1000 replicates)

| Slice | n | tfidf BPSN | gold BPSN | Difference [95% CI] | Interval excludes 0 |
|---|---|---|---|---|---|
| gender:female | 2602 | 0.836 | 0.860 | -0.024 [-0.033, -0.015] | yes |
| gender:male | 2112 | 0.825 | 0.848 | -0.023 [-0.032, -0.014] | yes |
| gender:transgender | 129 | 0.792 | 0.793 | -0.001 [-0.036, +0.039] | no |
| gender:any | 3960 | 0.826 | 0.851 | -0.025 [-0.033, -0.016] | yes |
| sexual_orientation:homosexual_gay_or_lesbian | 538 | 0.709 | 0.736 | -0.027 [-0.047, -0.010] | yes |
| sexual_orientation:any | 570 | 0.713 | 0.744 | -0.031 [-0.050, -0.013] | yes |
| religion:christian | 2109 | 0.906 | 0.912 | -0.007 [-0.014, -0.000] | yes |
| religion:jewish | 411 | 0.833 | 0.834 | -0.001 [-0.017, +0.015] | no |
| religion:muslim | 1054 | 0.866 | 0.862 | +0.004 [-0.007, +0.015] | no |
| religion:atheist | 149 | 0.895 | 0.878 | +0.017 [-0.005, +0.038] | no |
| religion:any | 3316 | 0.895 | 0.899 | -0.004 [-0.012, +0.003] | no |
| race_or_ethnicity:black | 761 | 0.787 | 0.847 | -0.059 [-0.073, -0.046] | yes |
| race_or_ethnicity:white | 1178 | 0.772 | 0.825 | -0.054 [-0.066, -0.042] | yes |
| race_or_ethnicity:asian | 217 | 0.883 | 0.866 | +0.017 [+0.000, +0.035] | yes |
| race_or_ethnicity:latino | 121 | 0.875 | 0.902 | -0.027 [-0.056, -0.001] | yes |
| race_or_ethnicity:any | 1911 | 0.800 | 0.848 | -0.048 [-0.057, -0.038] | yes |
| disability:psychiatric_or_mental_illness | 238 | 0.805 | 0.828 | -0.023 [-0.041, -0.004] | yes |
| disability:any | 252 | 0.814 | 0.836 | -0.022 [-0.039, -0.004] | yes |

## Paired bootstrap: BPSN AUC, stack minus gold (1000 replicates)

| Slice | n | stack BPSN | gold BPSN | Difference [95% CI] | Interval excludes 0 |
|---|---|---|---|---|---|
| gender:female | 2602 | 0.849 | 0.860 | -0.010 [-0.019, -0.002] | yes |
| gender:male | 2112 | 0.834 | 0.848 | -0.015 [-0.024, -0.006] | yes |
| gender:transgender | 129 | 0.813 | 0.793 | +0.020 [-0.016, +0.058] | no |
| gender:any | 3960 | 0.838 | 0.851 | -0.013 [-0.021, -0.004] | yes |
| sexual_orientation:homosexual_gay_or_lesbian | 538 | 0.758 | 0.736 | +0.022 [+0.001, +0.040] | yes |
| sexual_orientation:any | 570 | 0.761 | 0.744 | +0.018 [-0.003, +0.034] | no |
| religion:christian | 2109 | 0.911 | 0.912 | -0.001 [-0.009, +0.006] | no |
| religion:jewish | 411 | 0.842 | 0.834 | +0.008 [-0.008, +0.024] | no |
| religion:muslim | 1054 | 0.856 | 0.862 | -0.006 [-0.018, +0.005] | no |
| religion:atheist | 149 | 0.911 | 0.878 | +0.032 [+0.008, +0.055] | yes |
| religion:any | 3316 | 0.897 | 0.899 | -0.002 [-0.010, +0.005] | no |
| race_or_ethnicity:black | 761 | 0.792 | 0.847 | -0.055 [-0.069, -0.040] | yes |
| race_or_ethnicity:white | 1178 | 0.757 | 0.825 | -0.068 [-0.081, -0.056] | yes |
| race_or_ethnicity:asian | 217 | 0.894 | 0.866 | +0.028 [+0.010, +0.046] | yes |
| race_or_ethnicity:latino | 121 | 0.862 | 0.902 | -0.040 [-0.073, -0.009] | yes |
| race_or_ethnicity:any | 1911 | 0.794 | 0.848 | -0.054 [-0.065, -0.044] | yes |
| disability:psychiatric_or_mental_illness | 238 | 0.827 | 0.828 | -0.001 [-0.021, +0.019] | no |
| disability:any | 252 | 0.835 | 0.836 | -0.001 [-0.020, +0.018] | no |

## Paired bootstrap: BPSN AUC, calqwen minus gold (1000 replicates)

| Slice | n | calqwen BPSN | gold BPSN | Difference [95% CI] | Interval excludes 0 |
|---|---|---|---|---|---|
| gender:female | 2602 | 0.816 | 0.860 | -0.044 [-0.054, -0.034] | yes |
| gender:male | 2112 | 0.816 | 0.848 | -0.033 [-0.044, -0.021] | yes |
| gender:transgender | 129 | 0.780 | 0.793 | -0.013 [-0.053, +0.023] | no |
| gender:any | 3960 | 0.813 | 0.851 | -0.037 [-0.047, -0.027] | yes |
| sexual_orientation:homosexual_gay_or_lesbian | 538 | 0.798 | 0.736 | +0.062 [+0.039, +0.083] | yes |
| sexual_orientation:any | 570 | 0.801 | 0.744 | +0.057 [+0.035, +0.078] | yes |
| religion:christian | 2109 | 0.888 | 0.912 | -0.024 [-0.033, -0.015] | yes |
| religion:jewish | 411 | 0.796 | 0.834 | -0.038 [-0.058, -0.019] | yes |
| religion:muslim | 1054 | 0.806 | 0.862 | -0.056 [-0.070, -0.043] | yes |
| religion:atheist | 149 | 0.896 | 0.878 | +0.018 [-0.008, +0.042] | no |
| religion:any | 3316 | 0.863 | 0.899 | -0.036 [-0.045, -0.027] | yes |
| race_or_ethnicity:black | 761 | 0.767 | 0.847 | -0.080 [-0.097, -0.063] | yes |
| race_or_ethnicity:white | 1178 | 0.689 | 0.825 | -0.137 [-0.153, -0.121] | yes |
| race_or_ethnicity:asian | 217 | 0.839 | 0.866 | -0.027 [-0.054, -0.002] | yes |
| race_or_ethnicity:latino | 121 | 0.806 | 0.902 | -0.096 [-0.135, -0.060] | yes |
| race_or_ethnicity:any | 1911 | 0.737 | 0.848 | -0.111 [-0.124, -0.098] | yes |
| disability:psychiatric_or_mental_illness | 238 | 0.838 | 0.828 | +0.010 [-0.013, +0.033] | no |
| disability:any | 252 | 0.844 | 0.836 | +0.009 [-0.014, +0.031] | no |

## gold (`student-distilbert-gold-20261112T171037Z`)

### Civil bias set: toxicity (threshold 0.62)

| Slice | n | Toxic rate | Subgroup AUC | BPSN AUC | BNSP AUC | FPR | FPR ratio | FPR gap (pts) | FNR | F1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **overall** | 21577 | 0.111 | 0.874 |  |  | 0.018 | 1.0x |  | 0.644 | 0.475 |  |
| gender:female | 2602 | 0.135 | 0.865 | 0.860 | 0.880 | 0.021 | 1.2x | +0.3 | 0.643 | 0.479 | ok |
| gender:male | 2112 | 0.152 | 0.842 | 0.848 | 0.872 | 0.024 | 1.3x | +0.6 | 0.644 | 0.478 | ok |
| gender:transgender | 129 | 0.202 | 0.820 | 0.793 | 0.905 | 0.078 | 4.3x | +6.0 | 0.692 | 0.381 | **flagged** |
| gender:any | 3960 | 0.145 | 0.856 | 0.851 | 0.883 | 0.023 | 1.3x | +0.5 | 0.639 | 0.482 | ok |
| sexual_orientation:homosexual_gay_or_lesbian | 538 | 0.286 | 0.787 | 0.736 | 0.912 | 0.062 | 3.5x | +4.5 | 0.578 | 0.535 | ok |
| sexual_orientation:bisexual | 21 | 0.143 | 0.907 | 0.909 | 0.878 | 0.000 | 0.0x | -1.8 | 1.000 | 0.000 | too small |
| sexual_orientation:heterosexual | 66 | 0.197 | 0.724 | 0.743 | 0.859 | 0.094 | 5.3x | +7.6 | 0.692 | 0.364 | too small |
| sexual_orientation:other_sexual_orientation | 1 | 0.000 | n/a | 0.544 | n/a | 0.000 | 0.0x | -1.8 | n/a | 0.000 | too small |
| sexual_orientation:any | 570 | 0.281 | 0.788 | 0.744 | 0.909 | 0.059 | 3.3x | +4.1 | 0.588 | 0.528 | ok |
| religion:christian | 2109 | 0.100 | 0.860 | 0.912 | 0.812 | 0.007 | 0.4x | -1.1 | 0.776 | 0.347 | ok |
| religion:jewish | 411 | 0.165 | 0.831 | 0.834 | 0.876 | 0.029 | 1.6x | +1.1 | 0.750 | 0.358 | ok |
| religion:muslim | 1054 | 0.221 | 0.778 | 0.862 | 0.811 | 0.019 | 1.1x | +0.2 | 0.803 | 0.312 | ok |
| religion:hindu | 28 | 0.143 | 0.750 | 0.908 | 0.709 | 0.000 | 0.0x | -1.8 | 1.000 | 0.000 | too small |
| religion:buddhist | 27 | 0.148 | 0.554 | 0.892 | 0.552 | 0.000 | 0.0x | -1.8 | 1.000 | 0.000 | too small |
| religion:atheist | 149 | 0.067 | 0.878 | 0.878 | 0.873 | 0.007 | 0.4x | -1.1 | 0.500 | 0.625 | ok |
| religion:other_religion | 15 | 0.067 | 0.214 | 0.825 | 0.425 | 0.000 | 0.0x | -1.8 | 1.000 | 0.000 | too small |
| religion:any | 3316 | 0.134 | 0.841 | 0.899 | 0.821 | 0.011 | 0.6x | -0.6 | 0.775 | 0.347 | ok |
| race_or_ethnicity:black | 761 | 0.323 | 0.749 | 0.847 | 0.808 | 0.017 | 1.0x | -0.0 | 0.821 | 0.294 | ok |
| race_or_ethnicity:white | 1178 | 0.300 | 0.766 | 0.825 | 0.844 | 0.038 | 2.1x | +2.0 | 0.765 | 0.355 | ok |
| race_or_ethnicity:asian | 217 | 0.097 | 0.842 | 0.866 | 0.856 | 0.026 | 1.4x | +0.8 | 0.810 | 0.267 | ok |
| race_or_ethnicity:latino | 121 | 0.231 | 0.835 | 0.902 | 0.797 | 0.000 | 0.0x | -1.8 | 0.857 | 0.250 | ok |
| race_or_ethnicity:other_race_or_ethnicity | 16 | 0.375 | 0.750 | 0.848 | 0.803 | 0.100 | 5.6x | +8.2 | 0.500 | 0.600 | too small |
| race_or_ethnicity:any | 1911 | 0.272 | 0.775 | 0.848 | 0.837 | 0.027 | 1.5x | +0.9 | 0.775 | 0.347 | ok |
| disability:physical_disability | 3 | 0.000 | n/a | 0.905 | n/a | 0.000 | 0.0x | -1.8 | n/a | 0.000 | too small |
| disability:intellectual_or_learning_disability | 14 | 0.071 | 0.923 | 0.967 | 0.794 | 0.000 | 0.0x | -1.8 | 1.000 | 0.000 | too small |
| disability:psychiatric_or_mental_illness | 238 | 0.227 | 0.888 | 0.828 | 0.915 | 0.022 | 1.2x | +0.4 | 0.611 | 0.532 | ok |
| disability:any | 252 | 0.218 | 0.890 | 0.836 | 0.912 | 0.020 | 1.1x | +0.2 | 0.618 | 0.525 | ok |

## stack (`student-distilbert-stack-20261112T181004Z`)

### Civil bias set: toxicity (threshold 0.47)

| Slice | n | Toxic rate | Subgroup AUC | BPSN AUC | BNSP AUC | FPR | FPR ratio | FPR gap (pts) | FNR | F1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **overall** | 21577 | 0.111 | 0.887 |  |  | 0.011 | 1.0x |  | 0.720 | 0.411 |  |
| gender:female | 2602 | 0.135 | 0.873 | 0.849 | 0.909 | 0.014 | 1.3x | +0.4 | 0.743 | 0.381 | ok |
| gender:male | 2112 | 0.152 | 0.860 | 0.834 | 0.910 | 0.019 | 1.8x | +0.8 | 0.684 | 0.444 | ok |
| gender:transgender | 129 | 0.202 | 0.782 | 0.813 | 0.865 | 0.019 | 1.8x | +0.9 | 0.808 | 0.303 | ok |
| gender:any | 3960 | 0.145 | 0.866 | 0.838 | 0.913 | 0.017 | 1.6x | +0.6 | 0.716 | 0.411 | ok |
| sexual_orientation:homosexual_gay_or_lesbian | 538 | 0.286 | 0.795 | 0.758 | 0.910 | 0.034 | 3.2x | +2.3 | 0.721 | 0.410 | ok |
| sexual_orientation:bisexual | 21 | 0.143 | 0.907 | 0.886 | 0.912 | 0.000 | 0.0x | -1.1 | 0.667 | 0.500 | too small |
| sexual_orientation:heterosexual | 66 | 0.197 | 0.787 | 0.705 | 0.905 | 0.075 | 7.1x | +6.5 | 0.462 | 0.583 | too small |
| sexual_orientation:other_sexual_orientation | 1 | 0.000 | n/a | 0.918 | n/a | 0.000 | 0.0x | -1.1 | n/a | 0.000 | too small |
| sexual_orientation:any | 570 | 0.281 | 0.800 | 0.761 | 0.911 | 0.034 | 3.2x | +2.4 | 0.719 | 0.411 | ok |
| religion:christian | 2109 | 0.100 | 0.870 | 0.911 | 0.843 | 0.007 | 0.6x | -0.4 | 0.743 | 0.390 | ok |
| religion:jewish | 411 | 0.165 | 0.850 | 0.842 | 0.895 | 0.023 | 2.2x | +1.3 | 0.794 | 0.311 | ok |
| religion:muslim | 1054 | 0.221 | 0.791 | 0.856 | 0.847 | 0.011 | 1.0x | +0.0 | 0.820 | 0.296 | ok |
| religion:hindu | 28 | 0.143 | 0.823 | 0.931 | 0.730 | 0.000 | 0.0x | -1.1 | 0.750 | 0.400 | too small |
| religion:buddhist | 27 | 0.148 | 0.761 | 0.935 | 0.595 | 0.000 | 0.0x | -1.1 | 1.000 | 0.000 | too small |
| religion:atheist | 149 | 0.067 | 0.905 | 0.911 | 0.877 | 0.000 | 0.0x | -1.1 | 0.800 | 0.333 | ok |
| religion:other_religion | 15 | 0.067 | 0.429 | 0.891 | 0.473 | 0.000 | 0.0x | -1.1 | 1.000 | 0.000 | too small |
| religion:any | 3316 | 0.134 | 0.856 | 0.897 | 0.855 | 0.008 | 0.7x | -0.3 | 0.777 | 0.350 | ok |
| race_or_ethnicity:black | 761 | 0.323 | 0.771 | 0.792 | 0.882 | 0.031 | 2.9x | +2.0 | 0.789 | 0.331 | ok |
| race_or_ethnicity:white | 1178 | 0.300 | 0.775 | 0.757 | 0.906 | 0.036 | 3.4x | +2.6 | 0.737 | 0.391 | ok |
| race_or_ethnicity:asian | 217 | 0.097 | 0.919 | 0.894 | 0.909 | 0.005 | 0.5x | -0.5 | 0.762 | 0.370 | ok |
| race_or_ethnicity:latino | 121 | 0.231 | 0.843 | 0.862 | 0.871 | 0.032 | 3.0x | +2.2 | 0.893 | 0.176 | ok |
| race_or_ethnicity:other_race_or_ethnicity | 16 | 0.375 | 0.800 | 0.778 | 0.879 | 0.000 | 0.0x | -1.1 | 0.667 | 0.500 | too small |
| race_or_ethnicity:any | 1911 | 0.272 | 0.796 | 0.794 | 0.900 | 0.027 | 2.5x | +1.6 | 0.757 | 0.370 | ok |
| disability:physical_disability | 3 | 0.000 | n/a | 0.888 | n/a | 0.000 | 0.0x | -1.1 | n/a | 0.000 | too small |
| disability:intellectual_or_learning_disability | 14 | 0.071 | 0.769 | 0.961 | 0.706 | 0.000 | 0.0x | -1.1 | 1.000 | 0.000 | too small |
| disability:psychiatric_or_mental_illness | 238 | 0.227 | 0.863 | 0.827 | 0.912 | 0.011 | 1.0x | +0.0 | 0.741 | 0.400 | ok |
| disability:any | 252 | 0.218 | 0.864 | 0.835 | 0.908 | 0.010 | 1.0x | -0.0 | 0.745 | 0.394 | ok |

## tfidf (`student-distilbert-tfidf-20261112T190956Z`)

### Civil bias set: toxicity (threshold 0.30)

| Slice | n | Toxic rate | Subgroup AUC | BPSN AUC | BNSP AUC | FPR | FPR ratio | FPR gap (pts) | FNR | F1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **overall** | 21577 | 0.111 | 0.868 |  |  | 0.022 | 1.0x |  | 0.645 | 0.462 |  |
| gender:female | 2602 | 0.135 | 0.863 | 0.836 | 0.893 | 0.031 | 1.4x | +0.8 | 0.666 | 0.437 | ok |
| gender:male | 2112 | 0.152 | 0.851 | 0.825 | 0.891 | 0.035 | 1.6x | +1.3 | 0.613 | 0.489 | ok |
| gender:transgender | 129 | 0.202 | 0.721 | 0.792 | 0.821 | 0.068 | 3.0x | +4.5 | 0.731 | 0.350 | ok |
| gender:any | 3960 | 0.145 | 0.853 | 0.826 | 0.894 | 0.034 | 1.5x | +1.1 | 0.641 | 0.461 | ok |
| sexual_orientation:homosexual_gay_or_lesbian | 538 | 0.286 | 0.793 | 0.709 | 0.918 | 0.102 | 4.5x | +7.9 | 0.565 | 0.515 | **flagged** |
| sexual_orientation:bisexual | 21 | 0.143 | 0.926 | 0.848 | 0.919 | 0.000 | 0.0x | -2.2 | 0.667 | 0.500 | too small |
| sexual_orientation:heterosexual | 66 | 0.197 | 0.761 | 0.649 | 0.926 | 0.170 | 7.6x | +14.7 | 0.385 | 0.533 | too small |
| sexual_orientation:other_sexual_orientation | 1 | 0.000 | n/a | 0.315 | n/a | 1.000 | 44.5x | +97.8 | n/a | 0.000 | too small |
| sexual_orientation:any | 570 | 0.281 | 0.794 | 0.713 | 0.918 | 0.102 | 4.6x | +8.0 | 0.569 | 0.509 | **flagged** |
| religion:christian | 2109 | 0.100 | 0.854 | 0.906 | 0.810 | 0.011 | 0.5x | -1.2 | 0.733 | 0.392 | ok |
| religion:jewish | 411 | 0.165 | 0.836 | 0.833 | 0.872 | 0.041 | 1.8x | +1.8 | 0.750 | 0.343 | ok |
| religion:muslim | 1054 | 0.221 | 0.776 | 0.866 | 0.795 | 0.015 | 0.7x | -0.8 | 0.781 | 0.345 | ok |
| religion:hindu | 28 | 0.143 | 0.740 | 0.907 | 0.710 | 0.000 | 0.0x | -2.2 | 0.750 | 0.400 | too small |
| religion:buddhist | 27 | 0.148 | 0.739 | 0.917 | 0.640 | 0.000 | 0.0x | -2.2 | 1.000 | 0.000 | too small |
| religion:atheist | 149 | 0.067 | 0.876 | 0.895 | 0.847 | 0.007 | 0.3x | -1.5 | 0.800 | 0.308 | ok |
| religion:other_religion | 15 | 0.067 | 0.214 | 0.855 | 0.504 | 0.000 | 0.0x | -2.2 | 1.000 | 0.000 | too small |
| religion:any | 3316 | 0.134 | 0.838 | 0.895 | 0.815 | 0.014 | 0.6x | -0.8 | 0.755 | 0.367 | ok |
| race_or_ethnicity:black | 761 | 0.323 | 0.759 | 0.787 | 0.859 | 0.064 | 2.9x | +4.2 | 0.752 | 0.359 | ok |
| race_or_ethnicity:white | 1178 | 0.300 | 0.768 | 0.772 | 0.877 | 0.063 | 2.8x | +4.1 | 0.720 | 0.393 | ok |
| race_or_ethnicity:asian | 217 | 0.097 | 0.887 | 0.883 | 0.869 | 0.010 | 0.5x | -1.2 | 0.714 | 0.414 | ok |
| race_or_ethnicity:latino | 121 | 0.231 | 0.836 | 0.875 | 0.828 | 0.032 | 1.4x | +1.0 | 0.857 | 0.229 | ok |
| race_or_ethnicity:other_race_or_ethnicity | 16 | 0.375 | 0.767 | 0.779 | 0.843 | 0.000 | 0.0x | -2.2 | 0.500 | 0.667 | too small |
| race_or_ethnicity:any | 1911 | 0.272 | 0.788 | 0.800 | 0.873 | 0.052 | 2.3x | +2.9 | 0.730 | 0.383 | ok |
| disability:physical_disability | 3 | 0.000 | n/a | 0.864 | n/a | 0.000 | 0.0x | -2.2 | n/a | 0.000 | too small |
| disability:intellectual_or_learning_disability | 14 | 0.071 | 0.923 | 0.969 | 0.697 | 0.000 | 0.0x | -2.2 | 1.000 | 0.000 | too small |
| disability:psychiatric_or_mental_illness | 238 | 0.227 | 0.856 | 0.805 | 0.903 | 0.016 | 0.7x | -0.6 | 0.611 | 0.538 | ok |
| disability:any | 252 | 0.218 | 0.858 | 0.814 | 0.899 | 0.015 | 0.7x | -0.7 | 0.618 | 0.532 | ok |

## calqwen (`student-distilbert-calqwen-20261112T200941Z`)

### Civil bias set: toxicity (threshold 0.39)

| Slice | n | Toxic rate | Subgroup AUC | BPSN AUC | BNSP AUC | FPR | FPR ratio | FPR gap (pts) | FNR | F1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **overall** | 21577 | 0.111 | 0.877 |  |  | 0.020 | 1.0x |  | 0.687 | 0.424 |  |
| gender:female | 2602 | 0.135 | 0.848 | 0.816 | 0.905 | 0.038 | 1.9x | +1.8 | 0.631 | 0.457 | ok |
| gender:male | 2112 | 0.152 | 0.835 | 0.816 | 0.897 | 0.033 | 1.6x | +1.3 | 0.694 | 0.411 | ok |
| gender:transgender | 129 | 0.202 | 0.845 | 0.780 | 0.917 | 0.078 | 3.8x | +5.7 | 0.654 | 0.419 | **flagged** |
| gender:any | 3960 | 0.145 | 0.846 | 0.813 | 0.907 | 0.036 | 1.8x | +1.6 | 0.660 | 0.438 | ok |
| sexual_orientation:homosexual_gay_or_lesbian | 538 | 0.286 | 0.759 | 0.798 | 0.854 | 0.055 | 2.7x | +3.5 | 0.740 | 0.372 | ok |
| sexual_orientation:bisexual | 21 | 0.143 | 0.741 | 0.894 | 0.738 | 0.000 | 0.0x | -2.0 | 0.667 | 0.500 | too small |
| sexual_orientation:heterosexual | 66 | 0.197 | 0.772 | 0.779 | 0.865 | 0.075 | 3.7x | +5.5 | 0.769 | 0.300 | too small |
| sexual_orientation:other_sexual_orientation | 1 | 0.000 | n/a | 0.969 | n/a | 0.000 | 0.0x | -2.0 | n/a | 0.000 | too small |
| sexual_orientation:any | 570 | 0.281 | 0.766 | 0.801 | 0.857 | 0.051 | 2.5x | +3.1 | 0.744 | 0.369 | ok |
| religion:christian | 2109 | 0.100 | 0.861 | 0.888 | 0.851 | 0.009 | 0.5x | -1.1 | 0.762 | 0.360 | ok |
| religion:jewish | 411 | 0.165 | 0.824 | 0.796 | 0.896 | 0.050 | 2.5x | +2.9 | 0.603 | 0.482 | ok |
| religion:muslim | 1054 | 0.221 | 0.775 | 0.806 | 0.869 | 0.023 | 1.1x | +0.3 | 0.798 | 0.314 | ok |
| religion:hindu | 28 | 0.143 | 0.885 | 0.898 | 0.852 | 0.000 | 0.0x | -2.0 | 0.750 | 0.400 | too small |
| religion:buddhist | 27 | 0.148 | 0.783 | 0.908 | 0.700 | 0.000 | 0.0x | -2.0 | 1.000 | 0.000 | too small |
| religion:atheist | 149 | 0.067 | 0.899 | 0.896 | 0.868 | 0.000 | 0.0x | -2.0 | 0.900 | 0.182 | ok |
| religion:other_religion | 15 | 0.067 | 0.571 | 0.881 | 0.657 | 0.000 | 0.0x | -2.0 | 1.000 | 0.000 | too small |
| religion:any | 3316 | 0.134 | 0.848 | 0.863 | 0.869 | 0.015 | 0.8x | -0.5 | 0.761 | 0.357 | ok |
| race_or_ethnicity:black | 761 | 0.323 | 0.778 | 0.767 | 0.887 | 0.043 | 2.1x | +2.3 | 0.732 | 0.395 | ok |
| race_or_ethnicity:white | 1178 | 0.300 | 0.762 | 0.689 | 0.919 | 0.079 | 3.9x | +5.9 | 0.623 | 0.483 | **flagged** |
| race_or_ethnicity:asian | 217 | 0.097 | 0.881 | 0.839 | 0.919 | 0.051 | 2.5x | +3.1 | 0.667 | 0.368 | ok |
| race_or_ethnicity:latino | 121 | 0.231 | 0.837 | 0.806 | 0.902 | 0.054 | 2.7x | +3.4 | 0.857 | 0.216 | ok |
| race_or_ethnicity:other_race_or_ethnicity | 16 | 0.375 | 0.717 | 0.704 | 0.900 | 0.200 | 9.9x | +18.0 | 0.667 | 0.400 | too small |
| race_or_ethnicity:any | 1911 | 0.272 | 0.782 | 0.737 | 0.910 | 0.060 | 3.0x | +3.9 | 0.663 | 0.450 | ok |
| disability:physical_disability | 3 | 0.000 | n/a | 0.895 | n/a | 0.000 | 0.0x | -2.0 | n/a | 0.000 | too small |
| disability:intellectual_or_learning_disability | 14 | 0.071 | 0.769 | 0.931 | 0.730 | 0.000 | 0.0x | -2.0 | 1.000 | 0.000 | too small |
| disability:psychiatric_or_mental_illness | 238 | 0.227 | 0.891 | 0.838 | 0.915 | 0.027 | 1.3x | +0.7 | 0.556 | 0.578 | ok |
| disability:any | 252 | 0.218 | 0.891 | 0.844 | 0.912 | 0.025 | 1.3x | +0.5 | 0.564 | 0.571 | ok |

## mixed5ep (`student-distilbert-mixed-5ep-20261112T210931Z`)

### Civil bias set: toxicity (threshold 0.35)

| Slice | n | Toxic rate | Subgroup AUC | BPSN AUC | BNSP AUC | FPR | FPR ratio | FPR gap (pts) | FNR | F1 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **overall** | 21577 | 0.111 | 0.887 |  |  | 0.022 | 1.0x |  | 0.625 | 0.484 |  |
| gender:female | 2602 | 0.135 | 0.874 | 0.851 | 0.908 | 0.028 | 1.3x | +0.7 | 0.660 | 0.447 | ok |
| gender:male | 2112 | 0.152 | 0.864 | 0.838 | 0.910 | 0.035 | 1.6x | +1.3 | 0.600 | 0.501 | ok |
| gender:transgender | 129 | 0.202 | 0.786 | 0.822 | 0.862 | 0.049 | 2.2x | +2.7 | 0.692 | 0.410 | ok |
| gender:any | 3960 | 0.145 | 0.868 | 0.841 | 0.912 | 0.032 | 1.5x | +1.1 | 0.632 | 0.472 | ok |
| sexual_orientation:homosexual_gay_or_lesbian | 538 | 0.286 | 0.800 | 0.766 | 0.907 | 0.055 | 2.5x | +3.3 | 0.604 | 0.517 | ok |
| sexual_orientation:bisexual | 21 | 0.143 | 0.926 | 0.900 | 0.908 | 0.000 | 0.0x | -2.2 | 0.667 | 0.500 | too small |
| sexual_orientation:heterosexual | 66 | 0.197 | 0.801 | 0.718 | 0.907 | 0.075 | 3.5x | +5.4 | 0.462 | 0.583 | too small |
| sexual_orientation:other_sexual_orientation | 1 | 0.000 | n/a | 0.837 | n/a | 0.000 | 0.0x | -2.2 | n/a | 0.000 | too small |
| sexual_orientation:any | 570 | 0.281 | 0.805 | 0.769 | 0.908 | 0.054 | 2.5x | +3.2 | 0.613 | 0.508 | ok |
| religion:christian | 2109 | 0.100 | 0.869 | 0.914 | 0.838 | 0.010 | 0.5x | -1.2 | 0.700 | 0.432 | ok |
| religion:jewish | 411 | 0.165 | 0.843 | 0.848 | 0.885 | 0.035 | 1.6x | +1.3 | 0.735 | 0.367 | ok |
| religion:muslim | 1054 | 0.221 | 0.787 | 0.865 | 0.836 | 0.024 | 1.1x | +0.3 | 0.764 | 0.357 | ok |
| religion:hindu | 28 | 0.143 | 0.792 | 0.921 | 0.717 | 0.000 | 0.0x | -2.2 | 0.750 | 0.400 | too small |
| religion:buddhist | 27 | 0.148 | 0.674 | 0.934 | 0.570 | 0.000 | 0.0x | -2.2 | 1.000 | 0.000 | too small |
| religion:atheist | 149 | 0.067 | 0.904 | 0.913 | 0.868 | 0.000 | 0.0x | -2.2 | 0.800 | 0.333 | ok |
| religion:other_religion | 15 | 0.067 | 0.286 | 0.885 | 0.442 | 0.000 | 0.0x | -2.2 | 1.000 | 0.000 | too small |
| religion:any | 3316 | 0.134 | 0.852 | 0.902 | 0.846 | 0.015 | 0.7x | -0.6 | 0.727 | 0.397 | ok |
| race_or_ethnicity:black | 761 | 0.323 | 0.774 | 0.808 | 0.873 | 0.050 | 2.3x | +2.9 | 0.740 | 0.381 | ok |
| race_or_ethnicity:white | 1178 | 0.300 | 0.772 | 0.773 | 0.897 | 0.068 | 3.1x | +4.6 | 0.669 | 0.445 | ok |
| race_or_ethnicity:asian | 217 | 0.097 | 0.902 | 0.894 | 0.892 | 0.015 | 0.7x | -0.6 | 0.667 | 0.452 | ok |
| race_or_ethnicity:latino | 121 | 0.231 | 0.847 | 0.874 | 0.865 | 0.032 | 1.5x | +1.1 | 0.786 | 0.324 | ok |
| race_or_ethnicity:other_race_or_ethnicity | 16 | 0.375 | 0.783 | 0.782 | 0.864 | 0.100 | 4.6x | +7.8 | 0.500 | 0.600 | too small |
| race_or_ethnicity:any | 1911 | 0.272 | 0.795 | 0.807 | 0.892 | 0.051 | 2.4x | +2.9 | 0.692 | 0.427 | ok |
| disability:physical_disability | 3 | 0.000 | n/a | 0.855 | n/a | 0.000 | 0.0x | -2.2 | n/a | 0.000 | too small |
| disability:intellectual_or_learning_disability | 14 | 0.071 | 0.769 | 0.960 | 0.707 | 0.000 | 0.0x | -2.2 | 1.000 | 0.000 | too small |
| disability:psychiatric_or_mental_illness | 238 | 0.227 | 0.865 | 0.832 | 0.909 | 0.027 | 1.3x | +0.5 | 0.630 | 0.506 | ok |
| disability:any | 252 | 0.218 | 0.865 | 0.839 | 0.906 | 0.025 | 1.2x | +0.4 | 0.636 | 0.500 | ok |
