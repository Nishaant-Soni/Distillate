# Alert thresholds

Every threshold in the alert rules, what it watches, and why it is set where it is (work plan
v4 section 5: "document the alert thresholds"). These are Phase 2 drafts against the mock;
Phase 3 revisits them once the live endpoint and real traffic exist.

The rules are in two files:

- `prometheus/alert_rules.yml`, **active**: Prometheus loads it. Every rule uses only metrics
  that exist today (the service's section 2.4 metrics and `scripts/canary.py`'s).
- `prometheus/alert_rules_planned.yml`, **planned**: not loaded yet. Each rule waits on a metric
  nothing emits yet, or on a measurement not taken yet (the int8 package's latency). Move a rule
  into the active file once what it needs exists.

Alerts with `action: retrain` are the ones Phase 3 routes to Airflow's REST API to trigger
`retrain_dag` (v4 section 6). All three are planned, so nothing active can trigger a retrain.

## Summary

| Alert | Fires when | Severity | Status |
|---|---|---|---|
| `DistillateServiceDown` | Scrape fails for 2 min | critical | active |
| `DistillateHighErrorRate` | Service-side errors above 2% of requests for 5 min | warning | active |
| `DistillateErrorRateCritical` | Service-side errors above 5% for 5 min | critical | active |
| `DistillateFlagRateShift` | Hourly flag rate under 0.5x or over 2x its 7-day average, for 30 min, with real traffic | warning | active |
| `DistillateCanaryAUCDrop` | Canary mean AUC more than 0.02 below its 7-day average, for 3 runs | warning | active (notify only) |
| `DistillateCanaryFlagRecallDrop` | Canary flag recall more than 5 points below its 7-day average, for 3 runs | warning | active |
| `DistillateCanaryStale` | Last canary run over 2 hours ago | warning | active |
| `DistillateCanaryMissing` | No canary series at all for 2 hours | warning | active |
| `DistillateCanaryErrors` | The last canary run had failed calls | warning | active |
| `DistillateLatencyP95High` | p95 latency above 50 ms for 10 min | warning | planned (int8 package) |
| `DistillateCanaryAUCDropVsPromoted` | Canary mean AUC more than 0.02 below the promoted model's, for 3 runs | critical | planned, **retrain** |
| `DistillateInputDriftWarning` | PSI at or above 0.1 against the start of the Civil stream | warning | planned |
| `DistillateInputDriftAlert` | PSI at or above 0.25 against the start of the Civil stream | critical | planned, **retrain** |
| `DistillateDomainShiftBeyondKnown` | PSI at or above 0.25 against Wikipedia train | warning | planned |
| `DistillateOutputDrift` | Flag-rate PSI at or above 0.25 against the start of the stream | warning | planned |
| `DistillateDelayedLabelAUCDrop` | Delayed-label mean AUC more than 0.03 below its 7-day average | critical | planned, **retrain** |

Tests: `prometheus/alert_rules_test.yml` (`promtool test rules`) covers the misfires found in
review; `tests/test_rules_and_dashboard.py` checks that active rules and unplanned panels only use
metrics that exist, reading the canary's metric names from `canary.py` itself.

## Service health

**Error rate: 2% warning, 5% critical, over 5 minutes.** `distillate_errors_total` counts failures
on the service's side only. Bad client input (a 422, such as a body without `text` or text that
can't be encoded) counts as a request but not as an error, so malformed requests can't trip these
alerts. Below 2% is ordinary noise (timeouts); above 5% means a real fraction of comments are going
unmoderated, which is worth a rollback decision. The 5-minute window keeps a single burst from
firing it.

**Flag rate: 0.5x to 2x its 7-day average, for 30 minutes, with real traffic.** The flag rate is
the one signal available instantly, with no labels. A halving or doubling within an hour is a large
change for moderation traffic, and the 30-minute hold avoids firing on a single unusual thread.
The rule only applies above one request a minute over the hour: with no traffic the hourly ratio
is 0, which would otherwise read as "flag rate halved" and fire every quiet night (found in review
with a promtool test: 6 hours of traffic then 3 hours idle fired with ratio 0.00; now covered by
`alert_rules_test.yml`).

**p95 latency: 50 ms, for 10 minutes (planned).** The 50 ms budget is v4 section 6's target for
the **int8** package (README 3.3, model card section 4). The fp32 model's measured CPU p95 is
144 ms, so against fp32 this alert would fire constantly; it becomes active once the int8
package's p95 is measured, with the threshold set from that measurement if it differs. The latency
histogram has buckets at 0.15 and 0.2 s, so a p95 near 144 ms resolves correctly (with only 0.1 and
0.25, it interpolated to about 242 ms).

## Canary (`scripts/canary.py`)

The canary replays the same fixed comments every run (`canary_wikipedia`, 2,000, and
`canary_civil`, 4,000). So for an unchanged model, the canary AUC is identical on every run:
there is no sampling noise between runs. A drop means the service itself changed (a different
model version, a tokenizer or preprocessing mismatch, a partial outage).

**Mean AUC: more than 0.02 below a reference, for 3 runs.** 0.02 is about the size of the
difference that matters: it is roughly two standard errors of the Wikipedia canary's AUC (about
0.0086 at AUC 0.97, Hanley-McNeil), so a 0.02 drop also generalises beyond these particular
comments. Three consecutive runs (with hourly runs, `for: 3h`) mean a transient problem doesn't
trigger anything. Two versions of the rule:

- **Active, notify only** (`DistillateCanaryAUCDrop`): against the 7-day average. It can't be what
  triggers a retrain: after a legitimately promoted model with a slightly lower canary AUC, it would
  keep firing for days.
- **Planned, retrain** (`DistillateCanaryAUCDropVsPromoted`): against the promoted model's own
  canary value, recorded at promotion as `distillate_canary_reference_mean_auc{set}`. A new
  promotion sets a new reference instead of firing.

**Flag recall: more than 5 points below the 7-day average, for 3 runs.** Recall is the share of
toxic canary comments the service flags. A drop is the failure moderation cares about most
(toxic comments getting through), so it notifies even when AUC holds, for example after a
threshold change.

**Staleness and absence.** `DistillateCanaryStale` fires when the last run is over 2 hours old
(twice the planned hourly schedule), but it needs the series to exist. If the series disappears
entirely (the canary never ran, or its metrics come through a service that is down),
`DistillateCanaryMissing` (`absent()`, for 2 hours) notices instead.

**Errors.** `DistillateCanaryErrors` fires when the last run had any failed calls. The gauge holds
the last run's count until the next run, so the alert stays active until a clean run (up to the
canary interval): it reports "the last run was not clean", which matters because the canary sends
the same comments every time. The canary itself refuses to report metrics when more than 1% of
calls fail (`canary.max_error_fraction`).

**Rare labels make the canary's mean AUC fragile (open for Phase 3).** Replaying the real canary
sets through the mock showed how few positives some labels have: `canary_civil` has 286 toxicity
and 221 insult positives, but only 14 identity_attack, 12 obscene and **3 threat**. With the mock's
signal-free scores the common labels sat at AUC 0.48, while threat landed at 0.25 by chance,
pulling the Civil mean down to 0.42. A real model's AUC on 3 threat comments is just as fragile,
and the unweighted mean gives those 3 comments the same weight as 286. Before the canary AUC alert
retrains anything, it should use a mean weighted by positives, or toxicity AUC alone (with flag
recall alongside).

## Drift (`scripts/drift.py`, via `drift_ingest_dag`; planned)

All drift rules are planned: `distillate_drift_psi{feature, reference}` and
`distillate_delayed_label_mean_auc` are exported by `drift_ingest_dag` in Phase 3, and the
`civil_start` reference isn't built yet (#45 builds the Wikipedia baseline only).

PSI thresholds are the usual rules of thumb: below 0.1 no meaningful change, 0.1 to 0.25 a
moderate shift, 0.25 or more a large one (`drift.psi_warn`, `drift.psi_alert` in config).

**Two references, because of the domain shift.** On the real data (#45), every one of the 20
Civil batches sits at `warn` against the Wikipedia training baseline: non-alphabetic share PSI
0.11 to 0.18 (Civil has fewer non-letter characters, 5.4% vs 7.4%, since Wikipedia talk pages
carry markup, timestamps and signatures) and OOV-rate PSI 0.05 to 0.13 (1.9% vs 1.3% unfamiliar
words), with no trend across the batches. The teacher lane's
`Model-Pipeline/artifacts/stack/drift_check.json` shows the same. So:

- **Against the start of the Civil stream** (`reference="civil_start"`): warn at 0.1, alert and
  retrain at 0.25 (`DistillateInputDriftWarning`, `DistillateInputDriftAlert`). This reference
  separates "the stream is changing" from "Civil differs from Wikipedia". It should be built from
  the first few drift batches or from `canary_civil` (4,000 comments), not batch 0 alone: 2,000
  comments is a thin vocabulary for the OOV feature. This needs a second baseline entry in config
  and a second `input_drift` call per batch.
- **Against Wikipedia train** (`reference="wikipedia_train"`): only alert-level PSI is acted on
  (`DistillateDomainShiftBeyondKnown`). The known Civil shift tops out around 0.18, so 0.25 means
  traffic further from the training data than anything measured so far. Warning on this reference
  would fire on every batch, permanently.

**Flag rate, 0.25 against the start of the stream** (`DistillateOutputDrift`). The flag-rate
reference must be Civil too: thresholds tuned on Wikipedia validation don't transfer to Civil
(toxicity recall is 38% on H2), so a Wikipedia flag rate would differ from every Civil batch for
that reason alone.

**Delayed-label mean AUC: more than 0.03 below its 7-day average** (`DistillateDelayedLabelAUCDrop`).
Unlike the canary, each drift batch is different comments, so its AUC does carry sampling noise.
For a 2,000-comment batch at AUC 0.83 with about 8% positives, the standard error is about 0.020,
so a single batch moves by up to about 0.04 by chance alone. The exporter will publish the mean
over the last two batches (standard error about 0.014), and 0.03 is about two standard errors of
that mean. The mean is over the five Civil labels; no alert uses a single rare label per batch (a
batch has about 4 threats, so threat AUC moves by about 0.25 by chance).

## Metric names

The service metrics are exactly the v4 section 2.4 names. The others:

| Metric | Labels | Source | Status |
|---|---|---|---|
| `distillate_canary_auc`, `distillate_canary_f1` | `set`, `label` | `scripts/canary.py` (`canary.prom`) | exists |
| `distillate_canary_mean_auc`, `distillate_canary_flag_{precision,recall,f1,false_positive_rate}`, `distillate_canary_rows_scored`, `distillate_canary_errors` | `set` | `scripts/canary.py` | exists |
| `distillate_canary_last_run_timestamp_seconds` | none | `scripts/canary.py` | exists |
| `distillate_canary_reference_mean_auc` | `set` | recorded at promotion (Phase 3) | planned |
| `distillate_drift_psi` | `feature` (`char_len`, `oov_rate`, `nonalpha_share`, `flag_rate`), `reference` (`wikipedia_train`, `civil_start`) | `drift_ingest_dag` (Phase 3) | planned |
| `distillate_delayed_label_mean_auc` | none | `drift_ingest_dag`, mean over the last two batches (Phase 3) | planned |

## What changes in Phase 3

- **A separate metrics target for the canary.** Here the mock re-serves `canary.prom` through the
  service's `/metrics`, but in Phase 3 the canary runs in Airflow, not on Cloud Run, and if the
  service is down those series would vanish with it. The canary's metrics should go to their own
  target (a Pushgateway or a textfile collector); `DistillateCanaryMissing` covers the gap until
  then.
- Measure the int8 package's p95 and activate `DistillateLatencyP95High` against it.
- Record `distillate_canary_reference_mean_auc` at promotion and activate
  `DistillateCanaryAUCDropVsPromoted`; switch it to a positives-weighted mean or toxicity AUC
  (see "Rare labels").
- Build the `civil_start` drift reference, export `distillate_drift_psi` and
  `distillate_delayed_label_mean_auc` from `drift_ingest_dag`, and activate the drift rules.
- Route `action: retrain` alerts through Alertmanager to Airflow's REST API (`retrain_dag`).
