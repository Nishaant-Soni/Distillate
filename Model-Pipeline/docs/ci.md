# CI-triggered training (`model-ci.yml`)

**Owner:** Vaishnavi · **Workflow:** `.github/workflows/model-ci.yml` · **Helpers:**
`src/pipeline/ci.py`, `register_model.py` · **Tests:** `tests/pipeline/test_ci.py`

This implements the work plan's section 5.1: a push that changes model code trains a candidate on
GCP, gates it automatically, and registers it only if it passes. A Slack message goes out in
every case.

```text
push to main (Model-Pipeline/src, configs, Dockerfile, requirements)   or   run by hand
  0. settings        the repository variables and secrets are set (fails fast otherwise)
  1. lint-test       ruff + pytest
  2. train           build Model-Pipeline/Dockerfile → distillate-images/trainer:<sha>
                     Vertex AI custom job (CPU, vertex-trainer identity), wait for it
  3. gate-register   download gs://distillate-mlops-data/ci/<run>/ → check it is complete
                     fetch @production's predictions → src.eval.gate
                     passed: register_model → @staging + Artifact Registry
                     blocked: nothing registered; the run fails, except for the
                     smoke candidate, which is expected to be blocked and only warns
  4. notify          one Slack message: failed step, gate passed + version, or blocked + why
```

**Rollback:** a blocked candidate never gets an alias. To undo a promotion, use
`python -m src.pipeline.rollback_model --previous` (see `docs/registry.md`).

**Why the settings check:** without `MLFLOW_TRACKING_URI` and `MLFLOW_TRACKING_PASSWORD`,
nothing would fail. MLflow would fall back to a local `mlruns/` in the container and on the
runner, the regression check would find no production model, and a passing gate would
"register" into a throwaway store. So the first job stops the run, naming what is missing.

## The training job

The job runs the student lane's entry point, `src.pipeline.run` (the `Model-Pipeline/Dockerfile`
image, built from the repository root). The image has the code and the `.dvc` pointers but no
data, so the job first copies the processed tables from the DVC remote:

```bash
python -m src.pipeline.fetch_data --out /tmp/processed --tables train val gate bias
python -m src.pipeline.run --config configs/student.yaml --set <overrides> \
    pipeline.data_dir=/tmp/processed paths.official_dir=/tmp/processed/wikipedia \
    paths.out_dir=/gcs/distillate-mlops-data/ci/<run>
```

* **Smoke overrides** (`SMOKE_OVERRIDES` in the workflow): gold-only MiniLM, maximum length 64,
  2,000 training rows, one epoch, CPU, the package on, and only the gate and bias sets scored.
  This takes minutes, and the gate is expected to block it, so a blocked smoke run is a
  warning (yellow in Slack), not a failure. A manual run can pass its own
  `overrides`, for example the selected recipe, to train a real candidate. A blocked real
  candidate fails the run.
* **`fetch_data`** reads at `HEAD` (the image's commit). Without a revision, DVC only looks at
  the workspace, which has no data in the image. This was checked against the real bucket from
  a data-less copy of the repository: gate 63,978 rows, bias 21,577.
* **Output location:** `/gcs/<bucket>/…` is Vertex's Cloud Storage mount. `run.py` writes one
  `<run_name>-<time>/` folder there, and CI picks it up.
* **Environment:** `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`
  and `CODE_COMMIT` are set.
* **Known limitation, the MLflow password:** it is passed as an environment variable in the
  Vertex job spec, so anyone who can view custom jobs in the project (project owners today) can
  read it, and it stays with every job. Phase 3: store it in Secret Manager, grant
  `vertex-trainer` `roles/secretmanager.secretAccessor` on that one secret, and read it when the
  job starts.
* **Overrides:** every key was checked against `configs/student.yaml` (unknown keys fail the
  run).

The run folder, which `python -m src.pipeline.ci check-outputs` checks before the gate runs:

| Path | Contents |
|---|---|
| `package/` | the section 2.5 package (`src.student.package`): `model.onnx`, `tokenizer/`, `thresholds.json`, `meta.json` with `model_version` and `mlflow_run_id` |
| `predictions/val.parquet`, `gate.parquet`, `bias_eval.parquet` | prediction files (CONTRACTS.md §2.2) on validation, the gate set and the Civil bias set |
| `gold/val.parquet`, `gate.parquet`, `bias_eval.parquet` | `id` and the label columns, plus the identity columns for `bias_eval`; never text |

The file stems come from `configs/gate.yaml` (`splits`). If the gate scores the 20,000-row gate
subset, set `design.gate` there and write predictions for those ids only.

## One-time setup

All of this is done (checked on Nov 15).

| What | Where | Status |
|---|---|---|
| Artifact Registry `distillate-images` (Docker; keeps the latest 3 images, deletes others after 7 days) and `distillate-models` (generic) | GCP | done |
| Workload Identity: pool `github`, provider `distillate-repo` (only `Nishaant-Soni/Distillate`), identity `github-ci` | GCP | done |
| `github-ci`: Artifact Registry writer on both repositories; Vertex AI user | GCP | done |
| `vertex-trainer` identity for the training job | GCP | done |
| `vertex-trainer` and `github-ci` bucket and log access (commands below) | GCP | done |
| Variables `GCP_WIF_PROVIDER`, `GCP_CI_SERVICE_ACCOUNT`, `MLFLOW_TRACKING_URI` | GitHub → Settings → Actions → Variables | done |
| Secrets `MLFLOW_TRACKING_PASSWORD` (done), `SLACK_WEBHOOK_URL` (optional: without it, messages are only logged) | GitHub → Settings → Actions → Secrets | done |
| `Model-Pipeline/Dockerfile` and `src.pipeline.run` | student lane | done (on main) |

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
