# Alert thresholds

Every threshold in `prometheus/alert_rules.yml`, what it watches, and why it is set where it is
(work plan v4 section 5: "document the alert thresholds"). These are Phase 2 drafts against the
mock; Phase 3 revisits them once the live endpoint and real traffic exist.

Alerts with `action: retrain` are the ones Phase 3 routes to Airflow's REST API to trigger
`retrain_dag` (v4 section 6). The rest notify only.

## Summary

| Alert | Fires when | Severity | Action |
|---|---|---|---|
| `DistillateServiceDown` | Scrape fails for 2 min | critical | notify |
| `DistillateHighErrorRate` | Errors above 2% of requests for 5 min | warning | notify |
| `DistillateErrorRateCritical` | Errors above 5% for 5 min | critical | notify (consider rollback) |
| `DistillateLatencyP95High` | p95 latency above 50 ms for 10 min | warning | notify |
| `DistillateFlagRateShift` | Hourly flag rate under 0.5x or over 2x its 7-day average, for 30 min | warning | notify |
| `DistillateCanaryAUCDrop` | Canary mean AUC more than 0.02 below its 7-day average, for 3 runs | critical | **retrain** |
| `DistillateCanaryFlagRecallDrop` | Canary flag recall more than 5 points below its 7-day average, for 3 runs | warning | notify |
| `DistillateCanaryStale` | No canary run in 2 hours | warning | notify |
| `DistillateCanaryErrors` | Any canary call failed | warning | notify |
| `DistillateInputDriftWarning` | PSI at or above 0.1 against the start of the Civil stream | warning | notify |
| `DistillateInputDriftAlert` | PSI at or above 0.25 against the start of the Civil stream | critical | **retrain** |
| `DistillateDomainShiftBeyondKnown` | PSI at or above 0.25 against Wikipedia train | warning | notify |
| `DistillateOutputDrift` | Flag-rate PSI at or above 0.25 against the start of the stream | warning | notify |
| `DistillateDelayedLabelAUCDrop` | Delayed-label mean AUC more than 0.03 below its 7-day average | critical | **retrain** |

## Service health

**Error rate: 2% warning, 5% critical, over 5 minutes.** Below 2% is ordinary noise (timeouts,
malformed requests); above 5% means a real fraction of comments are going unmoderated, which is
worth a rollback decision. The 5-minute window keeps a single burst from firing it.

**p95 latency: 50 ms, for 10 minutes.** The 50 ms target comes from v4 section 6 (batch size 1
on 2 vCPUs). Ten minutes rides out cold starts on Cloud Run, which are expected after scaling up.

**Flag rate: 0.5x to 2x its 7-day average, for 30 minutes.** The flag rate is the one signal
available instantly, with no labels. A halving or doubling within an hour is a large change for
moderation traffic, and the 30-minute hold avoids firing on a single unusual thread.

## Canary (`scripts/canary.py`)

The canary replays the same fixed comments every run (`canary_wikipedia`, 2,000, and
`canary_civil`, 4,000). So for an unchanged model, the canary AUC is identical on every run:
there is no sampling noise between runs. A drop means the service itself changed (a different
model version, a tokenizer or preprocessing mismatch, a partial outage).

**Mean AUC: more than 0.02 below the 7-day average, for 3 runs.** 0.02 is about the size of the
difference that matters: it is roughly two standard errors of the Wikipedia canary's AUC (about
0.0086 at AUC 0.97, Hanley-McNeil), so a 0.02 drop also generalises beyond these particular
comments. Requiring three consecutive runs (with hourly runs, `for: 3h`) means a transient
problem, such as a run during a partial outage, doesn't trigger a retrain.

**Flag recall: more than 5 points below the 7-day average, for 3 runs.** Recall is the share of
toxic canary comments the service flags. A drop is the failure moderation cares about most
(toxic comments getting through), so it notifies even when AUC holds, for example after a
threshold change.

**Staleness: 2 hours.** Twice the planned hourly schedule: one missed run is tolerated, two means
the schedule stopped and accuracy is unmonitored.

**Any canary error.** The canary already refuses to report metrics when more than 1% of calls
fail (`canary.max_error_fraction`); any failure at all is worth knowing about, since the canary
sends the same comments every time.

**Rare labels make the canary's mean AUC noisy (open for Phase 3).** Replaying the real canary
sets through the mock (random scores) showed how few positives some labels have:
`canary_civil` has 286 toxicity and 221 insult positives, but only 14 identity_attack, 12
obscene and **3 threat**. With random scores the common labels sat at AUC 0.48, while threat
landed at 0.25 by chance, pulling the Civil mean down to 0.42. A real model's AUC on 3 threat
comments is just as fragile, and the unweighted mean over five labels gives those 3 comments the
same weight as 286. Before the canary AUC alert retrains anything in Phase 3, it should use a
mean weighted by positives, or toxicity AUC alone (with flag recall alongside), rather than the
plain mean. The canary run is deterministic for a given model, so this is about what the number
means, not about noise between runs.

## Drift (`scripts/drift.py`, via `drift_ingest_dag`)

PSI thresholds are the usual rules of thumb: below 0.1 no meaningful change, 0.1 to 0.25 a
moderate shift, 0.25 or more a large one (`drift.psi_warn`, `drift.psi_alert` in config).

**Two references, because of the domain shift.** On the real data (PR #45), every one of the 20
Civil batches sits at `warn` against the Wikipedia training baseline: non-alphabetic share PSI
0.11 to 0.18 (Civil has fewer non-letter characters, 5.4% vs 7.4%, since Wikipedia talk pages
carry markup, timestamps and signatures) and OOV-rate PSI 0.05 to 0.13 (1.9% vs 1.3% unfamiliar
words), with no trend across the batches. The teacher lane's `drift_check.json` shows the same:
no trend across the 20 batches. So:

- **Against the start of the Civil stream** (`reference="civil_start"`): warn at 0.1, alert and
  retrain at 0.25. This reference separates "the stream is changing" from "Civil differs from
  Wikipedia". It should be built from the first few drift batches or from `canary_civil` (4,000
  comments), not batch 0 alone: 2,000 comments is a thin vocabulary for the OOV feature (a word
  counts as known if it appears twice). This needs a second baseline entry in config and a
  second `input_drift` call per batch.
- **Against Wikipedia train** (`reference="wikipedia_train"`): only alert-level PSI is acted on.
  The known Civil shift tops out around 0.18, so 0.25 means traffic further from the training
  data than anything measured so far. Warning on this reference would fire on every batch,
  permanently, and a constant alert gets ignored.

**Flag rate, 0.25 against the start of the stream.** The flag-rate reference must be Civil too:
thresholds tuned on Wikipedia validation don't transfer to Civil (toxicity recall is 38% on H2,
#42), so a Wikipedia flag rate would differ from every Civil batch for that reason alone.

**Delayed-label mean AUC: more than 0.03 below its 7-day average.** Unlike the canary, each drift
batch is different comments, so its AUC does carry sampling noise. For a 2,000-comment batch at
AUC 0.83 with about 8% positives, the standard error is about 0.020, so a single batch moves by
up to about 0.04 by chance alone. The exporter therefore publishes the mean over the last two
batches (standard error about 0.014), and 0.03 is about two standard errors of that mean: a drop
larger than chance, without waiting many batches. The mean is over the five Civil labels; no
alert uses a single rare label per batch (a batch has about 4 threats, so threat AUC moves by
about 0.25 by chance).

## Metric names

The service metrics are exactly the v4 section 2.4 names. The others:

| Metric | Labels | Source |
|---|---|---|
| `distillate_canary_auc`, `distillate_canary_f1` | `set`, `label` | `scripts/canary.py` (`canary.prom`) |
| `distillate_canary_mean_auc`, `distillate_canary_flag_{precision,recall,f1,false_positive_rate}`, `distillate_canary_rows_scored`, `distillate_canary_errors` | `set` | `scripts/canary.py` |
| `distillate_canary_last_run_timestamp_seconds` | none | `scripts/canary.py` |
| `distillate_drift_psi` | `feature` (`char_len`, `oov_rate`, `nonalpha_share`, `flag_rate`), `reference` (`wikipedia_train`, `civil_start`) | `drift_ingest_dag`, exported in Phase 3 |
| `distillate_delayed_label_mean_auc` | none | `drift_ingest_dag`, mean over the last two batches, exported in Phase 3 |

## What changes in Phase 3

- Run hourly canaries against the live endpoint, then confirm the 0.02 canary drop and the 50 ms
  latency target against real behaviour.
- Switch the canary AUC alert to a positives-weighted mean, or toxicity AUC (see "Rare labels").
- Build the `civil_start` drift reference and set the flag-rate reference from the production
  model on it.
- Route `action: retrain` alerts through Alertmanager to Airflow's REST API (`retrain_dag`).
