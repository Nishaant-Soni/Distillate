"""Retraining: the steps of retrain_dag (Phase 2 skeleton, completed in Phase 3).

One run of retrain_dag takes a retraining request through five steps:

1. pull_data: the data the candidate trains on. Records the trigger and the Git revision whose
   DVC pointers name the data, and how many drift batches the stream has delivered. Phase 3
   triggers data_prep_dag first and adds the delivered batches (soft labels, delayed gold labels,
   reviewer overrides) to the retraining config.
2. submit_training: the Vertex AI custom job, the same image and service account as model-ci.
   The skeleton builds and records the job spec but submits nothing, so a run costs nothing.
3. gate: src.eval.gate against @production's predictions, as in CI. A stub until a job runs.
4. register: register_model (@staging + Artifact Registry) only if the gate passed.
5. notify: one Slack message with every step's outcome; the run record is written to
   ``retrain.reports_dir/<run_id>.json``.

Each step is ``step_<name>(cfg, run) -> dict``: ``run`` is the record so far (the trigger, and one
entry per finished step), and the return value is this step's entry. The DAG passes the record
from task to task through XCom; this module never imports Airflow, so it is tested on its own.

Run all five once, without Airflow: python -m scripts.retrain --reason "manual check"
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.alerts import send_alert
from scripts.utils import PIPELINE_ROOT, atomic_write, get_logger, load_config

logger = get_logger(__name__)

# (task id, function), in order. retrain_dag builds one task per entry.
STEPS: list[tuple[str, str]] = [
    ("pull_data", "step_pull_data"),
    ("submit_training", "step_submit_training"),
    ("gate", "step_gate"),
    ("register", "step_register"),
    ("notify", "step_notify"),
]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PIPELINE_ROOT / p).resolve()


def new_run(trigger: dict[str, Any] | None = None, run_id: str | None = None) -> dict[str, Any]:
    """The record a run starts with. ``trigger`` is the DAG run's conf (reason, source, ...)."""
    trigger = dict(trigger or {})
    trigger.setdefault("reason", "manual")
    trigger.setdefault("source", "manual")
    run_id = run_id or "retrain-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return {"run_id": run_id, "trigger": trigger, "steps": {}}


def _git_revision() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PIPELINE_ROOT,
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _delivered_drift_batches(cfg: dict[str, Any]) -> int:
    """How many drift batches drift_ingest_dag has released so far (0 before the first)."""
    state_file = cfg.get("drift_ingest", {}).get("state_file")
    if not state_file or not resolve(state_file).exists():
        return 0
    state = json.loads(resolve(state_file).read_text(encoding="utf-8"))
    return int(state.get("next_batch", 0))


def step_pull_data(cfg: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "stub",
        "data_rev": _git_revision(),
        "drift_batches_delivered": _delivered_drift_batches(cfg),
        "note": "Phase 3: trigger data_prep_dag and wait, then add the delivered drift batches "
                "to the retraining config",
    }


def job_spec(cfg: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """The custom job model-ci submits, with the retraining config and its own output folder.

    The output goes under ``ci/`` because that is the only prefix the training identity
    (vertex-trainer) can write. The image tag is the data revision: model-ci tags every image
    it builds with its commit.
    """
    rcfg = cfg["retrain"]
    out_dir = f"/gcs/{rcfg['bucket']}/ci/retrain/{run['run_id']}"
    revision = run["steps"].get("pull_data", {}).get("data_rev") or "unknown"
    image = f"{rcfg['image']}:{revision}"
    sets = [*rcfg.get("overrides", []), "pipeline.data_dir=/tmp/processed",
            "paths.official_dir=/tmp/processed/wikipedia", f"paths.out_dir={out_dir}"]
    script = ("python -m src.pipeline.fetch_data --out /tmp/processed --tables train val gate bias"
              f" && python -m src.pipeline.run --config {rcfg['config']} --set "
              + " ".join(shlex.quote(s) for s in sets))
    return {
        "project": rcfg["project"], "region": rcfg["region"],
        "display_name": run["run_id"], "service_account": rcfg["service_account"],
        "machine_type": rcfg["machine_type"], "image": image,
        "command": ["bash", "-c", script], "out_dir": out_dir,
    }


def step_submit_training(cfg: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    spec = job_spec(cfg, run)
    logger.info("retrain %s: would submit %s on %s", run["run_id"], spec["image"],
                spec["machine_type"])
    return {"mode": "stub", "submitted": False, "job_spec": spec,
            "note": "Phase 3: gcloud ai custom-jobs create with this spec, then wait for it"}


def step_gate(cfg: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    submitted = run["steps"].get("submit_training", {}).get("submitted", False)
    return {"mode": "stub", "result": "not_run",
            "note": "no job was submitted" if not submitted else
                    "Phase 3: src.eval.gate on the job's predictions against @production"}


def step_register(cfg: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    gate = run["steps"].get("gate", {}).get("result")
    if gate != "passed":
        return {"mode": "stub", "registered": False, "reason": f"gate {gate or 'missing'}"}
    return {"mode": "stub", "registered": False,
            "note": "Phase 3: src.pipeline.register_model --package <out>/package "
                    f"(@{cfg['retrain']['candidate_alias']})"}


def step_notify(cfg: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    steps = run["steps"]
    gate = steps.get("gate", {}).get("result", "missing")
    registered = steps.get("register", {}).get("registered", False)
    record = resolve(cfg["retrain"]["reports_dir"]) / f"{run['run_id']}.json"
    atomic_write(record, json.dumps(run, indent=2, default=str))
    stub = any(s.get("mode") == "stub" for s in steps.values())
    shown = record.relative_to(PIPELINE_ROOT) if record.is_relative_to(PIPELINE_ROOT) else record
    posted = send_alert(
        "ok" if registered else "warn",
        f"retrain_dag{' (skeleton)' if stub else ''}: gate {gate}, "
        f"{'registered' if registered else 'nothing registered'}",
        {"run": run["run_id"], "reason": run["trigger"]["reason"],
         "data_rev": (steps.get("pull_data", {}).get("data_rev") or "")[:7],
         "record": str(shown)},
    )
    return {"mode": "real", "record": str(record), "slack_posted": posted}


def run_all(cfg: dict[str, Any], trigger: dict[str, Any] | None = None) -> dict[str, Any]:
    """All five steps in order, as retrain_dag runs them."""
    run = new_run(trigger)
    for name, func in STEPS:
        run["steps"][name] = getattr(sys.modules[__name__], func)(cfg, run)
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reason", default="manual")
    args = parser.parse_args(argv)
    print(json.dumps(run_all(load_config(), {"reason": args.reason}), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
