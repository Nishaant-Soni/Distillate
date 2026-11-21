#!/usr/bin/env bash
# Roll the live service back to an earlier model version (work plan v5, section 6.1).
# Owner: Ruchita.
#
#   rollback.sh --previous [--reason "<why>"]   back to the version before @production
#   rollback.sh --to <n> [--reason "<why>"]      to version n
#
# Moves Cloud Run traffic to that version's revision (the newest one tagged v<n>-...), then
# moves @production to it with Phase 2's rollback_model.py, which only accepts versions that
# passed the gate and were published. The traffic moves first, so users are back on the old
# model even if the registry step fails; the exit code and Slack say so.
#
# The target version must already have a revision: deploy.sh tags every revision it deploys,
# and Cloud Run keeps them. If it was never deployed, deploy it with deploy.sh --version <n>.
#
# Needs the same setup as deploy.sh. Exit codes: 0 rolled back; 2 usage; 3 nothing moved;
# 4 traffic moved but @production didn't.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PROJECT="${PROJECT:-distillate-mlops}"
REGION="${REGION:-us-east1}"
SERVICE="${SERVICE:-distillate-api}"
MODEL_NAME="${MODEL_NAME:-distillate-student}"
DEPLOYMENTS_LOG="${DEPLOYMENTS_LOG:-gs://distillate-mlops-data/deployments/deployments.jsonl}"
PYTHON="${PYTHON:-python3}"
TOOLS="$SCRIPT_DIR/deploy_tools.py"

usage() {
  sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'
}

MODE=""
TARGET=""
REASON="manual rollback"
need_value() {
  if [ $# -lt 2 ] || [ -z "$2" ]; then
    echo "rollback.sh: $1 needs a value" >&2
    usage >&2
    exit 2
  fi
}

while [ $# -gt 0 ]; do
  case "$1" in
    --previous) MODE="previous"; shift ;;
    --to) need_value "$@"; MODE="to"; TARGET="${2:-}"; shift 2 ;;
    --reason) need_value "$@"; REASON="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "rollback.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [ -z "$MODE" ] || { [ "$MODE" = "to" ] && ! [[ "$TARGET" =~ ^[0-9]+$ ]]; }; then
  echo "rollback.sh: give --previous or --to <version number>" >&2
  usage >&2
  exit 2
fi

STATUS="error"
STEP="start"
DETAIL=""
CURRENT=""
TAG=""
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
  "$PYTHON" "$TOOLS" record --log "$DEPLOYMENTS_LOG" --action rollback --status "$STATUS" \
    --step "$STEP" --service "$SERVICE" --version "$TARGET" --tag "$TAG" \
    --from-version "$CURRENT" --detail "$DETAIL" \
    || echo "rollback.sh: warning: the rollback wasn't recorded" >&2
  rm -rf "$WORK"
  exit "$code"
}
trap finish EXIT

fail() {
  DETAIL="$1"
  echo "rollback.sh: $1" >&2
  exit "${2:-3}"
}

field() {
  "$PYTHON" -c 'import json, sys; print(json.loads(sys.argv[1]).get(sys.argv[2]) or "")' "$1" "$2"
}

export PYTHONPATH="$REPO_ROOT/Model-Pipeline${PYTHONPATH:+:$PYTHONPATH}"
if [ -z "${MLFLOW_TRACKING_PASSWORD:-}" ]; then
  MLFLOW_TRACKING_PASSWORD="$(gcloud secrets versions access latest --secret=mlflow-password \
    --project="$PROJECT")" || fail "could not read the mlflow-password secret"
  export MLFLOW_TRACKING_PASSWORD
fi

# 1. Which version to go back to.
STEP="resolve"
if [ "$MODE" = "previous" ]; then
  PREVIOUS="$("$PYTHON" "$TOOLS" previous --model "$MODEL_NAME")" \
    || fail "no earlier version to roll back to (see above)"
  TARGET="$(field "$PREVIOUS" version)"
  CURRENT="$(field "$PREVIOUS" current)"
else
  CURRENT="$(field "$("$PYTHON" "$TOOLS" resolve --model "$MODEL_NAME" --spec production)" version)" \
    || CURRENT=""
fi
echo "rollback.sh: rolling back from v${CURRENT:-?} to v$TARGET ($REASON)"

# 2. The target must have passed the gate and be published, checked before any traffic moves
#    (rollback_model applies the same rule to the alias, but only after the traffic shift).
STEP="check-target"
"$PYTHON" "$TOOLS" resolve --model "$MODEL_NAME" --spec "$TARGET" >/dev/null \
  || fail "v$TARGET can't be rolled back to: it must have passed the gate and be published"

# 3. Find that version's revision and move the traffic.
STEP="find-revision"
gcloud run services describe "$SERVICE" --region="$REGION" --project="$PROJECT" \
  --format=json > "$WORK/service.json" || fail "could not read the Cloud Run service $SERVICE"
TAG="$("$PYTHON" "$TOOLS" find-tag --service-json "$WORK/service.json" --version "$TARGET")" \
  || fail "v$TARGET has no tagged revision; deploy it first with deploy.sh --version $TARGET"

STEP="shift-traffic"
gcloud run services update-traffic "$SERVICE" --region="$REGION" --project="$PROJECT" \
  --to-tags="$TAG=100" --quiet || fail "could not shift traffic to $TAG; traffic is unchanged"

# 4. Move @production.
STEP="move-alias"
if ! (cd "$REPO_ROOT/Model-Pipeline" && "$PYTHON" -m src.pipeline.rollback_model --to "$TARGET" \
    --alias production --reason "rollback.sh: $REASON"); then
  STATUS="partial"
  fail "traffic is back on $TAG but @production didn't move: run rollback_model --to $TARGET --alias production" 4
fi
STATUS="rolled-back"
DETAIL="v$TARGET serving 100% of traffic on $TAG ($REASON)"
echo "rollback.sh: $DETAIL"
