"""Retraining: the steps of retrain_dag (work plan v5 6.1, Model-Pipeline/docs/retraining.md).

One run of retrain_dag takes a retraining request through seven steps:

1. pull_data: which drift batches the candidate trains on. ``state.json`` from drift_ingest
   has ``next_batch = n`` (batches 0 .. n-1 released, labels arrived for 0 .. n-2): train on
   0 .. n-3, hold out n-2, leave n-1 (no labels yet). Until three batches are out (n < 3)
   there is nothing to train on, so the run is skipped with a Slack message and every later
   step does nothing. Posts "retraining triggered" (or "skipped") to Slack.
2. submit_training: model-ci's Vertex AI custom job (same image, identity and output layout)
   with the retraining ``--set`` overrides, then waits for it. The image is pinned by digest:
   the newest trainer image in Artifact Registry, unless ``retrain.image_tag`` names one. With
   ``retrain.submit: false`` it records the job spec and submits nothing, so a run is free.
3. gate: downloads the job's outputs and runs src.eval.gate against @production's predictions,
   as model-ci's gate-register job does.
4. promote_check: src.pipeline.promote_check, the candidate against @production on the held-out
   batch (mean AUC over the five Civil labels). Production's scores are the ones drift_ingest
   stored when it scored that batch; if @production has moved since, the batch is rescored
   through the live service first. ``--trained-on`` comes from the package's meta.json.
5. register: register_model (@staging + Artifact Registry), only if both checks passed.
6. dispatch: ``repository_dispatch`` (event ``deploy-model``) to deploy.yml with the version.
   Posts "deployment requested" to Slack.
7. notify: writes the run record to ``retrain.reports_dir/<run_id>.json``, copies it to
   ``retrain.records_uri`` (the demo app's Status page reads it) and posts the result to Slack.

Each step is ``step_<name>(cfg, run) -> dict``: ``run`` is the record so far (the trigger, and one
entry per finished step), and the return value is this step's entry. The DAG passes the record
from task to task through XCom; this module never imports Airflow, so it is tested on its own.

The Model-Pipeline steps (fetch-production, the gate, promote_check, register_model) run as
subprocesses in ``retrain.model_pipeline_dir`` with ``$RETRAIN_MODEL_PYTHON`` (the Airflow image's
/opt/mpenv), which has mlflow; the MLflow settings come from the environment. No secret is put in
the run record: the MLflow password goes only into the job's (deleted) config file.

Run all seven once, without Airflow: python -m scripts.retrain --reason "manual check"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scripts import drift_ingest
from scripts.alerts import send_alert
from scripts.utils import PIPELINE_ROOT, atomic_write, get_logger, load_config

logger = get_logger(__name__)

# (task id, function), in order. retrain_dag builds one task per entry.
STEPS: list[tuple[str, str]] = [
    ("pull_data", "step_pull_data"),
    ("submit_training", "step_submit_training"),
    ("gate", "step_gate"),
    ("promote_check", "step_promote_check"),
    ("register", "step_register"),
    ("dispatch", "step_dispatch"),
    ("notify", "step_notify"),
]

TERMINAL_JOB_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
                       "JOB_STATE_EXPIRED"}
HOLDOUT_STEM = "drift_holdout"     # Model-Pipeline src/student/retrain.py


class RetrainError(RuntimeError):
    """A step can't go on: the task fails and its failure callback posts to Slack."""


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PIPELINE_ROOT / p).resolve()


def safe_run_id(raw: str) -> str:
    """An Airflow run id (``manual__2026-11-21T18:30:00.123+00:00``) made safe for a GCS path
    and a Vertex display name: anything but letters, digits, ``.``, ``_`` and ``-`` becomes
    ``-``."""
    return re.sub(r"[^A-Za-z0-9._-]", "-", raw)


def new_run(trigger: dict[str, Any] | None = None, run_id: str | None = None) -> dict[str, Any]:
    """The record a run starts with. ``trigger`` is the DAG run's conf (reason, source, ...)."""
    trigger = dict(trigger or {})
    trigger.setdefault("reason", "manual")
    trigger.setdefault("source", "manual")
    run_id = safe_run_id(run_id or "retrain-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    return {"run_id": run_id, "trigger": trigger, "steps": {}}


def skipped(run: dict[str, Any]) -> bool:
    return bool(run["steps"].get("pull_data", {}).get("skipped"))


def _run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """subprocess.run, text output captured; tests replace it."""
    return subprocess.run(command, capture_output=True, text=True, **kwargs)


def _check(command: list[str], what: str, **kwargs: Any) -> subprocess.CompletedProcess:
    out = _run(command, **kwargs)
    if out.returncode != 0:
        raise RetrainError(f"{what} failed (exit {out.returncode}): {out.stderr.strip()[-2000:]}")
    return out


def _model_pipeline(cfg: dict[str, Any], *args: str, what: str,
                    ok_codes: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess:
    """``python -m <args>`` in Model-Pipeline/, with the Python that has its dependencies."""
    rcfg = cfg["retrain"]
    python = os.environ.get("RETRAIN_MODEL_PYTHON") or rcfg.get("model_python") or sys.executable
    out = _run([python, "-m", *args], cwd=resolve(rcfg["model_pipeline_dir"]))
    if out.returncode not in ok_codes:
        raise RetrainError(f"{what} failed (exit {out.returncode}): {out.stderr.strip()[-2000:]}")
    return out


def _work_dir(cfg: dict[str, Any], run: dict[str, Any]) -> Path:
    return resolve(cfg["retrain"]["reports_dir"]) / run["run_id"]


# ---------------------------------------------------------------- 1. pull_data


def _git_revision() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PIPELINE_ROOT,
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def batch_plan(next_batch: int, min_batches: int = 3) -> dict[str, Any] | None:
    """Which batches to train on and hold out when ``next_batch = n`` (docs/retraining.md 1);
    None while fewer than ``min_batches`` are out."""
    if next_batch < max(3, min_batches):
        return None
    return {"drift_batches": list(range(next_batch - 2)), "holdout_batch": next_batch - 2,
            "unlabeled_batch": next_batch - 1}


def overrides_for(cfg: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    """The ``--set`` overrides that make src.pipeline.run a retraining run: the production
    recipe, the drift batches to train on and hold out, then any extra overrides."""
    batches = ",".join(str(k) for k in plan["drift_batches"])
    return [*cfg["retrain"].get("recipe", []),
            f"retrain.drift_batches=[{batches}]",
            f"retrain.holdout_batch={plan['holdout_batch']}",
            *cfg["retrain"].get("overrides", [])]


def step_pull_data(cfg: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    dcfg = cfg["drift_ingest"]
    n = int(drift_ingest.load_state(dcfg).get("next_batch", 0))
    min_batches = int(cfg["retrain"].get("min_batches", 3))
    plan = batch_plan(n, min_batches)
    base = {"data_rev": _git_revision(), "next_batch": n,
            "state_file": str(dcfg["state_file"])}
    if plan is None:
        reason = (f"{n} drift batch{'es' if n != 1 else ''} released; retraining starts once "
                  f"{max(3, min_batches)} are out (train on 0..n-3, hold out n-2)")
        send_alert("warn", "retrain_dag: retraining skipped",
                   {"run": run["run_id"], "trigger": run["trigger"]["reason"], "why": reason})
        return {**base, "skipped": True, "reason": reason}
    sets = overrides_for(cfg, plan)
    first, last = plan["drift_batches"][0], plan["drift_batches"][-1]
    send_alert("ok", "retrain_dag: retraining triggered",
               {"run": run["run_id"], "trigger": run["trigger"]["reason"],
                "source": run["trigger"]["source"],
                "train on": f"drift batches {first}-{last}" if last > first
                            else f"drift batch {first}",
                "held out": f"batch {plan['holdout_batch']}"})
    return {**base, "skipped": False, **plan, "overrides": sets}


# ---------------------------------------------------------------- 2. submit_training


def resolve_image(cfg: dict[str, Any]) -> str:
    """The training image to run: ``image:<image_tag>`` if set, else the newest image in
    Artifact Registry by digest (model-ci pushes one per commit, so this one exists)."""
    rcfg = cfg["retrain"]
    if rcfg.get("image_tag"):
        return f"{rcfg['image']}:{rcfg['image_tag']}"
    out = _check(["gcloud", "artifacts", "docker", "images", "list", rcfg["image"],
                  "--sort-by=~UPDATE_TIME", "--limit=1", "--format=value(version)"],
                 "looking up the newest training image")
    digest = out.stdout.strip()
    if not digest.startswith("sha256:"):
        raise RetrainError(f"no training image found in {rcfg['image']}: run model-ci first")
    return f"{rcfg['image']}@{digest}"


def job_spec(cfg: dict[str, Any], run: dict[str, Any], image: str | None = None) -> dict[str, Any]:
    """The custom job model-ci submits, with the retraining overrides and its own output folder.

    The output goes under ``ci/`` because that is the only prefix the training identity
    (vertex-trainer) can write. No secret is in it: the MLflow settings are added when the job's
    config file is written.
    """
    rcfg = cfg["retrain"]
    out_dir = f"/gcs/{rcfg['bucket']}/ci/retrain/{run['run_id']}"
    sets = [*run["steps"].get("pull_data", {}).get("overrides", rcfg.get("overrides", [])),
            "pipeline.data_dir=/tmp/processed", "paths.official_dir=/tmp/processed/wikipedia",
            f"paths.out_dir={out_dir}"]
    tables = " ".join(rcfg.get("fetch_tables", ["train", "val", "gate", "bias"]))
    script = (f"python -m src.pipeline.fetch_data --out /tmp/processed --tables {tables}"
              f" && python -m src.pipeline.run --config {rcfg['config']} --set "
              + " ".join(shlex.quote(s) for s in sets))
    return {
        "project": rcfg["project"], "region": rcfg["region"],
        "display_name": run["run_id"], "service_account": rcfg["service_account"],
        "machine_type": rcfg["machine_type"], "image": image,
        "command": ["bash", "-c", script], "out_dir": out_dir,
    }


def _job_config(spec: dict[str, Any], data_rev: str | None) -> dict[str, Any]:
    missing = [k for k in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_PASSWORD")
               if not os.environ.get(k)]
    if missing:
        raise RetrainError(f"{', '.join(missing)} not set: the job logs to the team MLflow server")
    env = [{"name": "MLFLOW_TRACKING_URI", "value": os.environ["MLFLOW_TRACKING_URI"]},
           {"name": "MLFLOW_TRACKING_USERNAME",
            "value": os.environ.get("MLFLOW_TRACKING_USERNAME", "team")},
           {"name": "MLFLOW_TRACKING_PASSWORD", "value": os.environ["MLFLOW_TRACKING_PASSWORD"]},
           {"name": "CODE_COMMIT", "value": data_rev or "unknown"}]
    return {"workerPoolSpecs": [{
        "machineSpec": {"machineType": spec["machine_type"]}, "replicaCount": 1,
        "containerSpec": {"imageUri": spec["image"], "command": spec["command"], "env": env}}]}


def submit_job(spec: dict[str, Any], data_rev: str | None) -> str:
    """``gcloud ai custom-jobs create``; returns the job's resource name."""
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(_job_config(spec, data_rev), f)      # JSON is valid YAML
        out = _check(["gcloud", "ai", "custom-jobs", "create", f"--project={spec['project']}",
                      f"--region={spec['region']}", f"--display-name={spec['display_name']}",
                      f"--config={path}", f"--service-account={spec['service_account']}",
                      "--format=value(name)"], "submitting the Vertex AI job")
    finally:
        os.unlink(path)
    return out.stdout.strip()


def wait_for_job(name: str, region: str, poll_s: float, timeout_s: float) -> str:
    """Polls until the job ends; returns its final state."""
    deadline = time.monotonic() + timeout_s
    while True:
        out = _check(["gcloud", "ai", "custom-jobs", "describe", name, f"--region={region}",
                      "--format=value(state)"], "checking the Vertex AI job")
        state = out.stdout.strip()
        if state in TERMINAL_JOB_STATES:
            return state
        if time.monotonic() > deadline:
            raise RetrainError(f"{name} still {state} after {timeout_s / 3600:.1f} h")
        logger.info("%s: %s", name, state)
        time.sleep(poll_s)


def step_submit_training(cfg: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    if skipped(run):
        return {"skipped": True}
    rcfg = cfg["retrain"]
    data_rev = run["steps"]["pull_data"].get("data_rev")
    if not rcfg.get("submit"):
        spec = job_spec(cfg, run, image=None)
        logger.info("retrain %s: retrain.submit is false, job not submitted", run["run_id"])
        return {"submitted": False, "job_spec": spec,
                "note": "retrain.submit is false: set it to true to run the Vertex AI job"}
    spec = job_spec(cfg, run, image=resolve_image(cfg))
    name = submit_job(spec, data_rev)
    logger.info("retrain %s: submitted %s (%s)", run["run_id"], name, spec["image"])
    state = wait_for_job(name, spec["region"], float(rcfg.get("poll_s", 60)),
                         float(rcfg.get("job_timeout_h", 5)) * 3600)
    if state != "JOB_STATE_SUCCEEDED":
        raise RetrainError(f"Vertex AI job {name} ended in {state}")
    return {"submitted": True, "job_spec": spec, "job_name": name, "job_state": state}


# ---------------------------------------------------------------- 3. gate


def download_outputs(cfg: dict[str, Any], run: dict[str, Any], dest: Path) -> str:
    """Copies the job's run folder (``ci/retrain/<run_id>/<run name>/``) to ``dest``."""
    prefix = f"gs://{cfg['retrain']['bucket']}/ci/retrain/{run['run_id']}/"
    listing = _check(["gcloud", "storage", "ls", prefix], f"listing {prefix}").stdout.split()
    folders = [line for line in listing if line.endswith("/") and line != prefix]
    if len(folders) != 1:
        raise RetrainError(f"expected one run folder under {prefix}, found {folders}")
    run_dir = folders[0].rstrip("/")
    dest.mkdir(parents=True, exist_ok=True)
    _check(["gcloud", "storage", "cp", "-r", f"{run_dir}/*", str(dest)],
           f"downloading {run_dir}")
    return run_dir


def step_gate(cfg: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    if skipped(run):
        return {"result": "not_run", "reason": "skipped"}
    if not run["steps"].get("submit_training", {}).get("submitted"):
        return {"result": "not_run", "reason": "no job was submitted"}
    work = _work_dir(cfg, run)
    job = work / "job"
    run_dir = download_outputs(cfg, run, job)
    _model_pipeline(cfg, "src.pipeline.ci", "check-outputs", "--dir", str(job),
                    what="checking the job's outputs")
    fetched = _model_pipeline(cfg, "src.pipeline.ci", "fetch-production", "--out",
                              str(job / "production"), what="fetching @production")
    last = (fetched.stdout.strip().splitlines() or ["none"])[-1]
    production = None if last == "none" else json.loads(last)
    args = ["src.eval.gate", "--config", "configs/gate.yaml",
            "--candidate", str(job / "predictions"),
            "--thresholds", str(job / "package" / "thresholds.json"),
            "--gold", str(job / "gold"), "--out", str(job / "gate")]
    if production:
        args += ["--production", str(job / "production")]
    _model_pipeline(cfg, *args, what="the gate", ok_codes=(0, 1))
    report_path = job / "gate" / "gate_report.json"
    if not report_path.exists():
        raise RetrainError(f"the gate wrote no {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _run(["gcloud", "storage", "cp", "-r", str(job / "gate"), f"{run_dir}/retrain-gate/"])
    return {"result": "passed" if report.get("passed") else "blocked",
            "failed_checks": report.get("failed_checks", []),
            "model_version": report.get("model_version"),
            "production": production, "run_dir": run_dir, "job_dir": str(job)}


# ---------------------------------------------------------------- 4. promote_check


def _served_version(path: Path) -> str | None:
    versions = pd.read_parquet(path, columns=["model_version"])["model_version"].astype(str)
    return versions.iloc[0] if versions.nunique() == 1 else None


def production_scores(cfg: dict[str, Any], run: dict[str, Any], k: int,
                      model_version: str) -> tuple[Path, bool]:
    """@production's scores on batch k: drift_ingest's stored file if the live service was
    serving @production then, else a fresh rescore of the batch through the live service."""
    dcfg = cfg["drift_ingest"]
    stored = resolve(dcfg["reports_dir"]) / "predictions" / f"batch_{k:02d}.parquet"
    if stored.exists() and _served_version(stored) == model_version:
        return stored, False
    logger.info("batch %d was scored by %s, not @production (%s): rescoring it", k,
                _served_version(stored) if stored.exists() else "nothing", model_version)
    scores = drift_ingest.score_batch(drift_ingest.release_batch(dcfg, k),
                                      drift_ingest.get_service(dcfg))
    served = scores["model_version"].astype(str).unique()
    if list(served) != [model_version]:
        raise RetrainError(f"the live service serves {sorted(served)}, but @production is "
                           f"{model_version}: wait for the deploy to finish, then rerun")
    path = _work_dir(cfg, run) / f"production_batch_{k:02d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(path, index=False)
    return path, True


def step_promote_check(cfg: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    gate = run["steps"].get("gate", {})
    if gate.get("result") != "passed":
        return {"result": "not_run", "reason": f"gate {gate.get('result', 'missing')}"}
    production = gate.get("production")
    if not production:
        return {"result": "passed", "reason": "no @production to compare with"}
    job = Path(gate["job_dir"])
    meta = json.loads((job / "package" / "meta.json").read_text(encoding="utf-8"))
    trained = meta.get("retrain") or {}
    k = run["steps"]["pull_data"]["holdout_batch"]
    if trained.get("holdout_batch") != k:
        raise RetrainError(f"the package's meta.json holds out batch {trained.get('holdout_batch')}"
                           f", but this run asked for batch {k}: the retrain overrides were lost")
    if not production.get("model_version"):
        raise RetrainError(f"@production (v{production.get('version')}) has no model_version tag"
                           ", so its scores can't be matched to the live service")
    prod_path, rescored = production_scores(cfg, run, k, production["model_version"])
    out = job / "promotion"
    shutil.rmtree(out, ignore_errors=True)   # never read an earlier attempt's verdict
    proc = _model_pipeline(
        cfg, "src.pipeline.promote_check", "--batch", str(k),
        "--candidate", str(job / "predictions" / f"{HOLDOUT_STEM}.parquet"),
        "--production", str(prod_path),
        "--gold", str(drift_ingest.batch_path(cfg["drift_ingest"], k)),
        "--production-model-version", production["model_version"],
        "--trained-on", *[str(b) for b in trained.get("drift_batches", [])],
        "--out", str(out), what="promote_check", ok_codes=(0, 1))
    report_path = out / "promote_check.json"
    if not report_path.exists():   # exit 1 is also how an uncaught error ends
        raise RetrainError(f"promote_check exited {proc.returncode} without writing "
                           f"{report_path}: it stopped on an error, see its log")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _run(["gcloud", "storage", "cp", "-r", str(out), f"{gate['run_dir']}/retrain-promotion/"])
    return {"result": "passed" if proc.returncode == 0 else "failed", "batch": k,
            "gain": report.get("gain"), "trained_on": report.get("trained_on"),
            "production_file": str(prod_path), "rescored": rescored}


# ---------------------------------------------------------------- 5. register


def step_register(cfg: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    steps = run["steps"]
    for check in ("gate", "promote_check"):
        result = steps.get(check, {}).get("result")
        if result != "passed":
            return {"registered": False, "reason": f"{check} {result or 'missing'}"}
    job = Path(steps["gate"]["job_dir"])
    _model_pipeline(cfg, "src.pipeline.register_model", "--package", str(job / "package"),
                    "--gate-report", str(job / "gate" / "gate_report.json"),
                    "--predictions", str(job / "predictions"),
                    "--out", str(job / "registration.json"), what="register_model")
    registration = json.loads((job / "registration.json").read_text(encoding="utf-8"))
    return {"registered": True, "version": str(registration["version"]),
            "alias": cfg["retrain"]["candidate_alias"]}


# ---------------------------------------------------------------- 6. dispatch


def _post(url: str, body: dict[str, Any], token: str) -> int:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}",
                 "X-GitHub-Api-Version": "2022-11-28"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def step_dispatch(cfg: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    register = run["steps"].get("register", {})
    if not register.get("registered"):
        return {"dispatched": False, "reason": "nothing registered"}
    rcfg = cfg["retrain"]
    token = os.environ.get("GITHUB_DISPATCH_TOKEN")
    if not token:
        raise RetrainError(f"v{register['version']} is registered but GITHUB_DISPATCH_TOKEN is "
                           "not set: deploy it by hand (deploy.yml, workflow_dispatch)")
    version = register["version"]
    status = _post(f"https://api.github.com/repos/{rcfg['github_repo']}/dispatches",
                   {"event_type": rcfg["dispatch_event"],
                    "client_payload": {"version": version, "retrain_run": run["run_id"]}}, token)
    if status != 204:
        raise RetrainError(f"repository_dispatch returned HTTP {status} (v{version} is registered;"
                           " deploy it by hand)")
    send_alert("ok", f"retrain_dag: deployment of v{version} requested",
               {"run": run["run_id"], "workflow": "deploy.yml",
                "actions": f"https://github.com/{rcfg['github_repo']}/actions"})
    return {"dispatched": True, "version": version, "event_type": rcfg["dispatch_event"]}


# ---------------------------------------------------------------- 7. notify


def outcome(run: dict[str, Any]) -> tuple[str, str]:
    """The Slack level and title for the whole run."""
    steps = run["steps"]
    if skipped(run):
        return "warn", "retrain_dag: skipped (not enough drift batches yet)"
    if steps.get("dispatch", {}).get("dispatched"):
        return "ok", (f"retrain_dag: v{steps['register']['version']} registered as "
                      f"@{steps['register']['alias']}, deployment requested")
    if not steps.get("submit_training", {}).get("submitted"):
        return "warn", "retrain_dag: job not submitted (retrain.submit is false)"
    gate = steps.get("gate", {}).get("result")
    if gate != "passed":
        return "warn", f"retrain_dag: gate {gate}, production kept"
    if steps.get("promote_check", {}).get("result") != "passed":
        gain = steps["promote_check"].get("gain")
        shown = f" (gain {gain:+.4f})" if isinstance(gain, (int, float)) else ""
        return "warn", f"retrain_dag: candidate didn't beat production{shown}, production kept"
    return "fail", "retrain_dag: registered but not dispatched"


def step_notify(cfg: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    rcfg = cfg["retrain"]
    record = resolve(rcfg["reports_dir"]) / f"{run['run_id']}.json"
    level, title = outcome(run)
    run = {**run, "outcome": {"level": level, "title": title},
           "finished_at": datetime.now(UTC).isoformat()}
    atomic_write(record, json.dumps(run, indent=2, default=str))
    uploaded = None
    if rcfg.get("records_uri"):
        uri = rcfg["records_uri"].rstrip("/")
        copies = [_run(["gcloud", "storage", "cp", str(record), f"{uri}/{name}"])
                  for name in (record.name, "latest.json")]
        uploaded = all(c.returncode == 0 for c in copies)
        if not uploaded:
            logger.warning("couldn't copy %s to %s: %s", record.name, uri,
                           next(c.stderr for c in copies if c.returncode != 0).strip())
    steps = run["steps"]
    shown = record.relative_to(PIPELINE_ROOT) if record.is_relative_to(PIPELINE_ROOT) else record
    details = {"run": run["run_id"], "reason": run["trigger"]["reason"],
               "gate": steps.get("gate", {}).get("result", "missing"),
               "promote_check": steps.get("promote_check", {}).get("result", "missing"),
               "record": str(shown)}
    posted = send_alert(level, title, details)
    return {"record": str(record), "uploaded": uploaded, "slack_posted": posted,
            "level": level, "title": title}


def run_all(cfg: dict[str, Any], trigger: dict[str, Any] | None = None) -> dict[str, Any]:
    """All seven steps in order, as retrain_dag runs them."""
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
