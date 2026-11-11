# Model registry and rollback

**Owner:** Vaishnavi · **Code:** `src/pipeline/register_model.py`, `src/pipeline/rollback_model.py`,
`src/pipeline/registry.py` · **Config:** `configs/registry.yaml` · **Tests:** `tests/pipeline/`

A trained student becomes a registered model only after the gate passes. Registration puts it
in two places, as the work plan's section 2.5 describes:

| Where | What it holds |
|---|---|
| MLflow model registry, model `distillate-student` | the version, its lineage tags, its training run, and the aliases `staging` and `production` |
| Artifact Registry, `us-east1/distillate-models`, package `distillate-student` | the package files, with the MLflow version number as the package version |

Registry stages are deprecated, so aliases are used throughout, including for rollback.

## The flow

```text
src.student.train  →  <run>/model/ (weights, tokenizer, meta.json) + predictions
src.eval.gate      →  gate_report.json (passed: true / false), exit 0 or 1
register_model     →  only if passed: MLflow vN + Artifact Registry upload, then @staging → vN
(Phase 3)          →  promotion moves @production
rollback_model     →  moves @production (or @staging) back to an earlier version
```

## The MLflow server

The team server is `http://35.227.25.230:5000`, on the `mlflow-server` VM. It has a password
login, and its artifacts are stored in `gs://distillate-mlops-data/mlflow/`. It runs from 9 AM to
2 AM Eastern to save credits. Set these before running either script:

```bash
export MLFLOW_TRACKING_URI=http://35.227.25.230:5000
export MLFLOW_TRACKING_USERNAME=team
export MLFLOW_TRACKING_PASSWORD=...        # ask Vaishnavi; never commit it
```

Setup and rebuild instructions are in [`infra/mlflow/README.md`](../infra/mlflow/README.md).

## Registering a candidate

```bash
python -m src.pipeline.register_model \
    --package outputs/student/<run>/model \
    --gate-report outputs/student/<run>/gate/gate_report.json \
    --out outputs/student/<run>/registration.json
```

What it checks before registering anything:

* **The package is complete:** `meta.json`, `config.json`, the weights, and a tokenizer.
  `thresholds.json` is uploaded too when it is present.
* **The gate report passed** (`passed: true`) **for this exact model:** its `model_version` must
  equal the one in `meta.json`. A blocked model is never registered.

The order matters. The version is created first, then the package is uploaded to Artifact
Registry, and only after both succeed does `@staging` move to the new version. If the upload
fails, the version is tagged `ar_upload=failed`, keeps no alias, and the command exits 1.

Lineage tags on every version: `model_version`, `model_id`, `alpha`, `temperature`,
`teacher_version`, `prompt_version`, `stack_sha256`, `training_data_sha256`,
`training_data_dvc_hash`, `code_commit`, `max_length`, `gate_passed`, `gate_mean_auc`,
`registered_at`, `ar_upload`, `ar_package`.

`--no-upload` registers in MLflow only, for runs without GCP access.

## Rolling back

```bash
python -m src.pipeline.rollback_model --list                       # versions, aliases, gate
python -m src.pipeline.rollback_model --previous --reason "canary AUC dropped"
python -m src.pipeline.rollback_model --to 3                       # any earlier version
python -m src.pipeline.rollback_model --to 3 --alias staging
```

* `--previous` picks the newest version older than the current one whose gate passed.
* A version without a passed gate on record is refused unless you add `--force`.
* Every move is recorded: `<alias>_previous` on the registered model, and `last_alias_move`
  (from, to, reason, time) on the target version.
* Serving and the gate resolve `models:/distillate-student@production`, so moving the alias
  is the whole rollback in Phase 2. Phase 3 adds the Cloud Run traffic shift. Artifact
  Registry versions are immutable, so nothing there changes.

## Notifications

Both scripts post one Slack message per operation through `SLACK_WEBHOOK_URL`: registered,
not registered (with the reason), rolled back, or rollback refused. Without the webhook, the
messages are only logged. A broken webhook never fails the operation.

## Tests

`tests/pipeline/test_registry.py` (25 tests) runs on a temporary SQLite MLflow store, with a
fake `gcloud` for the upload, so it needs no network and no GCP. What they check:

* **Registration:** staging is set and the upload is called correctly; a second candidate
  moves staging.
* **Refusals:** a blocked gate, a report for another model, a missing report, and an
  incomplete package are each refused.
* **Failed upload:** it leaves no alias.
* **Other registration cases:** `--no-upload`; an unknown training run; a soft-deleted registry
  experiment.
* **Rollback:** `--previous` skips versions that never passed the gate; `--to`; `--force`;
  unknown versions; nothing older; a no-op; the version table; CLI exit codes.
* **Slack:** works with and without a webhook; a broken webhook never raises.

Both scripts were also run against the live server, on a throwaway model name: two
registrations, a rollback, and `--list`.
