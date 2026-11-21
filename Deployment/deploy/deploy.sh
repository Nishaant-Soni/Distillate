#!/usr/bin/env bash
# Deploy a model version to Cloud Run (work plan v5, section 6.1). Owner: Ruchita.
#
#   deploy.sh --version <n|staging|production>     deploy a registered version
#   deploy.sh --package-dir <dir> [--label <name>] deploy a local package for testing; it is
#                                                  smoke-tested but never given traffic or
#                                                  @production (the degraded-candidate test)
#
# Steps:
#   1. resolve the version in the MLflow registry (a number or an alias); it must have passed
#      the gate and be published to Artifact Registry
#   2. download the package from Artifact Registry (distillate-models)
#   3. check every file against its SHA-256 in meta.json
#   4. make sure the package has reference_outputs.json (serving's scorer.py computes it if not)
#   5. build serve.Dockerfile with the package baked in (Cloud Build) and push it; the build
#      itself fails on a broken package (scorer.py --package)
#   6. deploy a Cloud Run revision with no traffic, tagged v<version>-<commit>
#   7. smoke test the tagged URL: /health, then the reference inputs against the package
#   8. pass: shift 100% of traffic to it and move @production; fail: nothing moves
#   Always: a line in deployments.jsonl, the GitHub job summary, and Slack.
#
# Needs gcloud (logged in) and python3 with Deployment/deploy/requirements.txt (and
# Deployment/serving/requirements.txt for packages without reference_outputs.json). MLflow:
# MLFLOW_TRACKING_URI (and MLFLOW_TRACKING_USERNAME) set; the password comes from
# MLFLOW_TRACKING_PASSWORD if set, else from the mlflow-password secret in Secret Manager.
#
# Exit codes: 0 deployed (or a test package passed); 1 smoke test failed (traffic unchanged);
# 2 usage; 3 stopped before any traffic change; 4 traffic moved but @production didn't.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Settings: override any of these from the environment.
PROJECT="${PROJECT:-distillate-mlops}"
REGION="${REGION:-us-east1}"
SERVICE="${SERVICE:-distillate-api}"
IMAGE="${IMAGE:-$REGION-docker.pkg.dev/$PROJECT/distillate-images/distillate-api}"
AR_REPOSITORY="${AR_REPOSITORY:-distillate-models}"
AR_PACKAGE="${AR_PACKAGE:-distillate-student}"
MODEL_NAME="${MODEL_NAME:-distillate-student}"
SERVING_DIR="${SERVING_DIR:-$REPO_ROOT/Deployment/serving}"
# serve.Dockerfile (Deployment/serving/README.md): the build context is Deployment/serving with
# the checked package copied to package/, built with -f serve.Dockerfile and the registry
# version as --build-arg REGISTRY_VERSION. deploy.sh builds a copy, so the repo stays clean.
DOCKERFILE="${DOCKERFILE:-serve.Dockerfile}"
PACKAGE_SUBDIR="${PACKAGE_SUBDIR:-package}"
# Serving matched the pipeline's own ONNX scoring to 6e-8 on a real package. 1e-4 assumes the
# expected scores come from the ONNX model (serving's scorer.py does); reference outputs written
# from PyTorch scores could differ by more (ONNX parity allows up to 1e-3 in logits).
SMOKE_TOLERANCE="${SMOKE_TOLERANCE:-0.0001}"
DEPLOYMENTS_LOG="${DEPLOYMENTS_LOG:-gs://distillate-mlops-data/deployments/deployments.jsonl}"
PYTHON="${PYTHON:-python3}"
TOOLS="$SCRIPT_DIR/deploy_tools.py"

usage() {
  sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
}

SPEC=""
PACKAGE_DIR=""
LABEL="test"
need_value() {
  if [ $# -lt 2 ] || [ -z "$2" ]; then
    echo "deploy.sh: $1 needs a value" >&2
    usage >&2
    exit 2
  fi
}

while [ $# -gt 0 ]; do
  case "$1" in
    --version) need_value "$@"; SPEC="${2:-}"; shift 2 ;;
    --package-dir) need_value "$@"; PACKAGE_DIR="${2:-}"; shift 2 ;;
    --label) need_value "$@"; LABEL="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "deploy.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if { [ -z "$SPEC" ] && [ -z "$PACKAGE_DIR" ]; } || { [ -n "$SPEC" ] && [ -n "$PACKAGE_DIR" ]; }; then
  echo "deploy.sh: give exactly one of --version or --package-dir" >&2
  usage >&2
  exit 2
fi

# rollback.sh --to <n> picks revisions tagged v<n>-...; a test package must never look like one.
if [ -n "$PACKAGE_DIR" ] && printf '%s' "$LABEL" | tr '[:upper:]' '[:lower:]' | grep -Eq '^v[0-9]+'; then
  echo "deploy.sh: --label can't start with v<number> (reserved for registered versions)" >&2
  exit 2
fi

STATUS="error"
STEP="start"
DETAIL=""
VERSION=""
MODEL_VERSION=""
TAG=""
IMAGE_REF=""
FROM_VERSION=""
WORK="$(mktemp -d)"

finish() {
  local code=$?
  # set -e exits with 1 on any unguarded failure; 1 means "smoke test failed" only when that
  # step set it, so an unexpected 1 is reported as 3 (stopped before any traffic change).
  if [ "$STATUS" = "error" ] && [ "$code" -eq 1 ]; then
    code=3
  fi
  if [ "$STATUS" = "error" ] && [ -z "$DETAIL" ]; then
    DETAIL="stopped at step '$STEP' (exit $code); see the log above"
  fi
  "$PYTHON" "$TOOLS" record --log "$DEPLOYMENTS_LOG" --action deploy --status "$STATUS" \
    --step "$STEP" --service "$SERVICE" --version "$VERSION" --model-version "$MODEL_VERSION" \
    --tag "$TAG" --image "$IMAGE_REF" --from-version "$FROM_VERSION" --detail "$DETAIL" \
    --smoke "$WORK/smoke.json" || echo "deploy.sh: warning: the deployment wasn't recorded" >&2
  rm -rf "$WORK"
  exit "$code"
}
trap finish EXIT

fail() {
  DETAIL="$1"
  echo "deploy.sh: $1" >&2
  exit "${2:-3}"
}

field() {
  "$PYTHON" -c 'import json, sys; print(json.loads(sys.argv[1]).get(sys.argv[2]) or "")' "$1" "$2"
}

short_commit() {
  git -C "$REPO_ROOT" rev-parse --short=7 HEAD 2>/dev/null || echo "local"
}

# Cloud Run tags: lowercase letters, digits and hyphens, starting with a letter.
sanitize_tag() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-' '-' | cut -c1-40
}

export PYTHONPATH="$REPO_ROOT/Model-Pipeline${PYTHONPATH:+:$PYTHONPATH}"

if [ -n "$SPEC" ]; then
  # 1. Resolve the version.
  STEP="resolve"
  if [ -z "${MLFLOW_TRACKING_PASSWORD:-}" ]; then
    MLFLOW_TRACKING_PASSWORD="$(gcloud secrets versions access latest --secret=mlflow-password \
      --project="$PROJECT")" || fail "could not read the mlflow-password secret"
    export MLFLOW_TRACKING_PASSWORD
  fi
  RESOLVED="$("$PYTHON" "$TOOLS" resolve --model "$MODEL_NAME" --spec "$SPEC")" \
    || fail "registry unreachable, or v$SPEC can't be deployed (see above)"
  VERSION="$(field "$RESOLVED" version)"
  MODEL_VERSION="$(field "$RESOLVED" model_version)"
  FROM_VERSION="$(field "$RESOLVED" production_version)"
  echo "deploy.sh: deploying $MODEL_NAME v$VERSION ($MODEL_VERSION); production is v${FROM_VERSION:-none}"

  # 2. Download the package.
  STEP="download"
  PACKAGE="$WORK/package"
  gcloud artifacts generic download --project="$PROJECT" --location="$REGION" \
    --repository="$AR_REPOSITORY" --package="$AR_PACKAGE" --version="$VERSION" \
    --destination="$PACKAGE" --quiet \
    || fail "could not download v$VERSION from Artifact Registry ($AR_REPOSITORY)"
  TAG="$(sanitize_tag "v$VERSION-$(short_commit)")"
else
  PACKAGE="$WORK/package"
  cp -R "$PACKAGE_DIR" "$PACKAGE"
  MODEL_VERSION="$(field "$(cat "$PACKAGE/meta.json")" model_version)"
  TAG="$(sanitize_tag "$LABEL-$(short_commit)")"
  echo "deploy.sh: test deploy of $PACKAGE_DIR ($MODEL_VERSION) as $TAG; it will get no traffic"
fi

# 3. Check the package.
STEP="verify"
"$PYTHON" "$TOOLS" verify --package "$PACKAGE" ${MODEL_VERSION:+--expect-model-version "$MODEL_VERSION"} \
  || fail "the package failed its SHA-256 check"

# 4. Reference outputs for the smoke test.
STEP="reference"
"$PYTHON" "$TOOLS" reference --package "$PACKAGE" || fail "no reference outputs for the smoke test"

# 5. Build the serving image with the package baked in.
STEP="build"
[ -f "$SERVING_DIR/$DOCKERFILE" ] \
  || fail "$SERVING_DIR/$DOCKERFILE not found: the serving code isn't on this branch"
CONTEXT="$WORK/context"
cp -R "$SERVING_DIR" "$CONTEXT"
rm -rf "${CONTEXT:?}/$PACKAGE_SUBDIR"
cp -R "$PACKAGE" "$CONTEXT/$PACKAGE_SUBDIR"
IMAGE_REF="$IMAGE:$TAG"
# gcloud builds submit --tag only finds a file named Dockerfile and can't pass build args, so
# the build runs from a one-step config. BuildKit for serve.Dockerfile's syntax line and its
# own serve.Dockerfile.dockerignore. Logs go to Cloud Logging only, so gcloud can stream them
# without read access to the build-log bucket (the caller needs Logs Viewer).
cat > "$WORK/cloudbuild.yaml" <<CONFIG
steps:
  - name: gcr.io/cloud-builders/docker
    env: ["DOCKER_BUILDKIT=1"]
    args: ["build", "-f", "$DOCKERFILE", "--build-arg", "REGISTRY_VERSION=${VERSION:-$TAG}",
           "-t", "$IMAGE_REF", "."]
images: ["$IMAGE_REF"]
options:
  logging: CLOUD_LOGGING_ONLY
CONFIG
gcloud builds submit "$CONTEXT" --project="$PROJECT" --config="$WORK/cloudbuild.yaml" --quiet \
  || fail "the image build failed (a broken package fails here too: see the build log)"

# 6. Deploy a revision with no traffic.
STEP="deploy-revision"
gcloud run services describe "$SERVICE" --region="$REGION" --project="$PROJECT" >/dev/null 2>&1 \
  || fail "Cloud Run service $SERVICE doesn't exist: create it from Deployment/deploy/service.yaml (setup_gcp.sh)"
gcloud run deploy "$SERVICE" --image="$IMAGE_REF" --region="$REGION" --project="$PROJECT" \
  --no-traffic --tag="$TAG" --quiet || fail "the Cloud Run revision didn't deploy"
gcloud run services describe "$SERVICE" --region="$REGION" --project="$PROJECT" \
  --format=json > "$WORK/service.json"
TAG_URL="$("$PYTHON" "$TOOLS" tag-url --service-json "$WORK/service.json" --tag "$TAG")" \
  || fail "no URL for the tagged revision $TAG"
echo "deploy.sh: revision tagged $TAG at $TAG_URL (no traffic)"

# 7. Smoke test the tagged revision.
STEP="smoke-test"
if ! "$PYTHON" "$SCRIPT_DIR/smoke_test.py" --url "$TAG_URL" \
    --reference "$PACKAGE/reference_outputs.json" --thresholds "$PACKAGE/thresholds.json" \
    --model-version "$MODEL_VERSION" --tolerance "$SMOKE_TOLERANCE" --out "$WORK/smoke.json"; then
  STATUS="smoke-failed"
  DETAIL="smoke test failed on $TAG; traffic and @production stay on the previous version"
  echo "deploy.sh: $DETAIL" >&2
  exit 1
fi

if [ -n "$PACKAGE_DIR" ]; then
  STATUS="smoke-passed-not-promoted"
  DETAIL="test package passed the smoke test; it gets no traffic"
  exit 0
fi

# 8. Promote: all traffic to the new revision, then @production.
STEP="shift-traffic"
gcloud run services update-traffic "$SERVICE" --region="$REGION" --project="$PROJECT" \
  --to-tags="$TAG=100" --quiet || fail "could not shift traffic to $TAG; traffic is unchanged"
STEP="move-alias"
if ! (cd "$REPO_ROOT/Model-Pipeline" && "$PYTHON" -m src.pipeline.rollback_model --to "$VERSION" \
    --alias production --reason "deployed by deploy.sh ($TAG)"); then
  STATUS="partial"
  fail "traffic is on $TAG but @production didn't move: run rollback_model --to $VERSION --alias production" 4
fi
STATUS="deployed"
DETAIL="v$VERSION serving 100% of traffic on $TAG"
echo "deploy.sh: $DETAIL"
