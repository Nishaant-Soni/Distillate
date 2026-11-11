# CI-triggered training (`model-ci.yml`)

**Owner:** Vaishnavi · **Workflow:** `.github/workflows/model-ci.yml` · **Helpers:**
`src/pipeline/ci.py`, `register_model.py` · **Tests:** `tests/pipeline/test_ci.py`

This implements the work plan's section 5.1: a push that changes model code trains a candidate on
GCP, gates it automatically, and registers it only if it passes. A Slack message goes out in
every case.

```text
push to main (Model-Pipeline/src, configs, Dockerfile, requirements)   or   run by hand
  1. lint-test       ruff + pytest
  2. train           build Model-Pipeline/Dockerfile → distillate-images/trainer:<sha>
                     Vertex AI custom job (CPU, vertex-trainer identity), wait for it
  3. gate-register   download gs://distillate-mlops-data/ci/<run>/ → check it is complete
                     fetch @production's predictions → src.eval.gate
                     passed: register_model → @staging + Artifact Registry
                     blocked: nothing registered, run marked failed
  4. notify          one Slack message: failed step, gate passed + version, or blocked + why
```

**Rollback:** a blocked candidate never gets an alias. To undo a promotion, use
`python -m src.pipeline.rollback_model --previous` (see `docs/registry.md`).

## The training job's contract

This is what CI needs from the student lane's entry point (work plan 5.1: "one entry point …
inside one Docker image"). CI starts the container with exactly:

```bash
python -m src.pipeline.run --config <config> --out /gcs/distillate-mlops-data/ci/<run>
```

* `<config>` is relative to `Model-Pipeline/`. The default is `configs/ci.yaml`, a small run:
  the gate is expected to block it, which still proves every step.
* `/gcs/<bucket>/…` is Vertex's Cloud Storage mount, so the job writes there as to a local folder.
* `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` and `CODE_COMMIT`
  are set in the environment.
* The job exits 0 only if every file below was written.

The output folder, which `python -m src.pipeline.ci check-outputs` checks before the gate runs:

| Path | Contents |
|---|---|
| `package/` | the section 2.5 package (`src.student.package`): `model.onnx`, `tokenizer/`, `thresholds.json`, `meta.json` with `model_version` and `mlflow_run_id` |
| `predictions/val.parquet`, `gate.parquet`, `bias.parquet` | prediction files (CONTRACTS.md §2.2) on validation, the gate set and the Civil bias set |
| `gold/val.parquet`, `gate.parquet`, `bias.parquet` | `id` and the label columns, plus the identity columns for `bias`; never text |

The file stems come from `configs/gate.yaml` (`splits`). If the gate scores the 20,000-row gate
subset, set `design.gate` there and write predictions for those ids only.

The image (`Model-Pipeline/Dockerfile`, student lane) must have `src/` and `configs/` in its
working directory, so that `python -m src.pipeline.run` resolves.

## One-time setup

All of this is done except the items marked ⏳.

| What | Where | Status |
|---|---|---|
| Artifact Registry `distillate-images` (Docker; keeps the latest 3 images, deletes others after 7 days) and `distillate-models` (generic) | GCP | done |
| Workload Identity: pool `github`, provider `distillate-repo` (only `Nishaant-Soni/Distillate`), identity `github-ci` | GCP | done |
| `github-ci`: Artifact Registry writer on both repositories; Vertex AI user | GCP | done |
| `vertex-trainer` identity for the training job | GCP | created |
| `vertex-trainer` and `github-ci` bucket and log access (commands below) | GCP | ⏳ |
| Variables `GCP_WIF_PROVIDER`, `GCP_CI_SERVICE_ACCOUNT`, `MLFLOW_TRACKING_URI` | GitHub → Settings → Actions → Variables | ⏳ |
| Secrets `MLFLOW_TRACKING_PASSWORD`, `SLACK_WEBHOOK_URL` | GitHub → Settings → Actions → Secrets | ⏳ |
| `Model-Pipeline/Dockerfile` and `src.pipeline.run` | student lane | ⏳ |

```bash
P=distillate-mlops; B=gs://distillate-mlops-data
TRAINER=serviceAccount:vertex-trainer@$P.iam.gserviceaccount.com
CI=serviceAccount:github-ci@$P.iam.gserviceaccount.com
CI_PREFIX='resource.name.startsWith("projects/_/buckets/distillate-mlops-data/objects/ci/")'

# Training job: read the bucket (DVC data, soft labels), write only ci/, write its logs.
gcloud storage buckets add-iam-policy-binding $B --member=$TRAINER --role=roles/storage.objectViewer --condition=None
gcloud storage buckets add-iam-policy-binding $B --member=$TRAINER --role=roles/storage.legacyBucketReader --condition=None
gcloud storage buckets add-iam-policy-binding $B --member=$TRAINER --role=roles/storage.objectUser \
  --condition="expression=$CI_PREFIX,title=ci-prefix-only"
gcloud projects add-iam-policy-binding $P --member=$TRAINER --role=roles/logging.logWriter --condition=None

# CI: start jobs as vertex-trainer, and read and write ci/ only.
gcloud iam service-accounts add-iam-policy-binding vertex-trainer@$P.iam.gserviceaccount.com \
  --project $P --member=$CI --role=roles/iam.serviceAccountUser
gcloud storage buckets add-iam-policy-binding $B --member=$CI --role=roles/storage.legacyBucketReader --condition=None
gcloud storage buckets add-iam-policy-binding $B --member=$CI --role=roles/storage.objectUser \
  --condition="expression=$CI_PREFIX,title=ci-prefix-only"
```

`legacyBucketReader` only lets an identity list the bucket and read its metadata. Object reads
and writes stay limited to the prefixes above.

## Cost

* Each run uses one `n1-standard-4` for the length of the job, about $0.19 an hour.
* The smoke config takes minutes, so a run costs cents.
* Images older than 7 days are deleted (the latest 3 are always kept).
* The `ci/<run>/` outputs are small.
* The workflow runs only when model code changes, or by hand.

## Tests

`tests/pipeline/test_ci.py` (11 tests) covers:

* the output check, with missing files named;
* `register_model --predictions` logging the files, and refusing a missing folder before
  anything is registered;
* `fetch-production`: before any promotion, after one, and for a production version with no
  predictions;
* the GitHub step outputs;
* Slack without a webhook.

The workflow's YAML, the Vertex job spec it renders, and its 15 shell steps were checked by
parsing them.
