"""Register a candidate that passed the gate: MLflow alias ``staging`` plus Artifact Registry.

Runs after ``src.eval.gate`` in CI (model-ci.yml) and in retrain_dag. In order:

1. Check the package folder (train.py's ``model/``: weights, tokenizer, ``meta.json``, and
   ``thresholds.json`` once tuned) and the gate report. Only a report with ``passed: true``
   for this package's ``model_version`` is accepted, so a blocked model is never registered.
2. Log the package and the gate report to the training run in MLflow (``meta.json`` names
   it), then create a new version of ``distillate-student`` from that package.
3. Upload the package to the ``distillate-models`` generic repository in Artifact Registry,
   with the MLflow version number as the package version.
4. Only after both succeed, point the ``staging`` alias at the new version.

If the upload fails, the version stays registered without the alias, is tagged
``ar_upload=failed``, and the command exits 1, so a half-published model never becomes the
candidate. One Slack message goes out either way (logged only, without SLACK_WEBHOOK_URL).

    python -m src.pipeline.register_model --package outputs/student/<run>/model \\
        --gate-report outputs/student/<run>/gate/gate_report.json
    python -m src.pipeline.register_model ... --no-upload     # MLflow only, no GCP access
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.pipeline.registry import client, load_config, notify
from src.teacher.config import get_logger

logger = get_logger(__name__)
REGISTRY_EXPERIMENT = "distillate-registry"   # for packages whose training run is unknown


class RegistrationError(RuntimeError):
    """The package or gate report can't be registered; the message says why."""


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_package(package: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Return ``meta.json`` after checking the folder holds a complete model package."""
    if not package.is_dir():
        raise RegistrationError(f"package folder {package} not found")
    spec = cfg["package"]
    names = {p.name for p in package.iterdir()}
    missing = [n for n in spec["required"] if n not in names]
    if not names & set(spec["weights_any"]):
        missing.append(f"weights (one of {spec['weights_any']})")
    if not names & set(spec["tokenizer_any"]):
        missing.append(f"tokenizer (one of {spec['tokenizer_any']})")
    if missing:
        raise RegistrationError(f"package {package} is missing: {', '.join(missing)}")
    meta = json.loads((package / "meta.json").read_text(encoding="utf-8"))
    if not meta.get("model_version"):
        raise RegistrationError(f"{package / 'meta.json'} has no model_version")
    return meta


def check_gate(report_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """Return the gate report if it passed this exact model; raise otherwise."""
    if not report_path.exists():
        raise RegistrationError(f"gate report {report_path} not found: run src.eval.gate first")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("model_version") != meta["model_version"]:
        raise RegistrationError(
            f"gate report is for {report.get('model_version')!r}, but the package is "
            f"{meta['model_version']!r}")
    if report.get("passed") is not True:
        failed = ", ".join(report.get("failed_checks") or []) or "unknown"
        raise RegistrationError(f"gate did not pass ({failed}); nothing registered")
    return report


# ---------------------------------------------------------------------------
# MLflow and Artifact Registry
# ---------------------------------------------------------------------------

def _training_run(mlflow_client, meta: dict[str, Any]) -> str:
    """The run that trained this package, or a new registry run when it isn't on the server."""
    from mlflow.exceptions import MlflowException

    run_id = meta.get("mlflow_run_id")
    if run_id:
        try:
            mlflow_client.get_run(run_id)
            return run_id
        except MlflowException:
            logger.warning("training run %s is not on this server; logging to %s", run_id,
                           REGISTRY_EXPERIMENT)
    experiment = mlflow_client.get_experiment_by_name(REGISTRY_EXPERIMENT)
    if experiment is None:
        experiment_id = mlflow_client.create_experiment(REGISTRY_EXPERIMENT)
    else:
        experiment_id = experiment.experiment_id
        # Deleting an experiment only soft-deletes it and keeps its name reserved.
        if experiment.lifecycle_stage == "deleted":
            mlflow_client.restore_experiment(experiment_id)
    run = mlflow_client.create_run(experiment_id, run_name=f"register-{meta['model_version']}",
                                   tags={"model_version": meta["model_version"]})
    mlflow_client.set_terminated(run.info.run_id)
    return run.info.run_id


def _ensure_registered_model(mlflow_client, name: str) -> None:
    from mlflow.exceptions import MlflowException

    try:
        mlflow_client.get_registered_model(name)
    except MlflowException:
        mlflow_client.create_registered_model(
            name, description="Distillate toxicity student (six sigmoid labels).")


def version_tags(meta: dict[str, Any], report: dict[str, Any]) -> dict[str, str]:
    """Lineage tags on the model version: what it is, what it was trained on, that it passed."""
    keys = ("model_version", "model_id", "alpha", "temperature", "teacher_version",
            "prompt_version", "stack_sha256", "training_data_sha256", "training_data_dvc_hash",
            "code_commit", "max_length")
    tags = {k: str(meta[k]) for k in keys if meta.get(k) is not None}
    gate_auc = report.get("metrics", {}).get("gate", {}).get("mean_auc")
    tags.update(gate_passed="true", registered_at=datetime.now(UTC).isoformat(timespec="seconds"))
    if gate_auc is not None:
        tags["gate_mean_auc"] = f"{float(gate_auc):.6f}"
    return tags


def artifact_registry_path(ar: dict[str, Any], version: str) -> str:
    return (f"projects/{ar['project']}/locations/{ar['location']}/repositories/"
            f"{ar['repository']}/packages/{ar['package']}/versions/{version}")


def upload_package(package: Path, version: str, ar: dict[str, Any], runner=subprocess.run) -> str:
    """Upload every file in ``package`` as one generic package version; return its path."""
    command = ["gcloud", "artifacts", "generic", "upload", f"--project={ar['project']}",
               f"--location={ar['location']}", f"--repository={ar['repository']}",
               f"--package={ar['package']}", f"--version={version}",
               f"--source-directory={package}", "--quiet"]
    logger.info("uploading %s to Artifact Registry as %s version %s", package, ar["package"],
                version)
    result = runner(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RegistrationError(f"Artifact Registry upload failed: {result.stderr.strip()[-500:]}")
    return artifact_registry_path(ar, version)


def register(cfg: dict[str, Any], package: Path, gate_report: Path, mlflow_client,
             upload: bool = True, alias: str | None = None, runner=subprocess.run
             ) -> dict[str, Any]:
    """Register ``package``; return a summary. Raises RegistrationError on any refusal."""
    meta = check_package(package, cfg)
    report = check_gate(gate_report, meta)
    name = cfg["mlflow"]["registered_model"]
    alias = alias or cfg["mlflow"]["candidate_alias"]

    run_id = _training_run(mlflow_client, meta)
    mlflow_client.log_artifacts(run_id, str(package), "package")
    mlflow_client.log_artifact(run_id, str(gate_report), "gate")
    if gate_report.with_suffix(".md").exists():
        mlflow_client.log_artifact(run_id, str(gate_report.with_suffix(".md")), "gate")

    _ensure_registered_model(mlflow_client, name)
    source = f"{mlflow_client.get_run(run_id).info.artifact_uri}/package"
    mv = mlflow_client.create_model_version(name, source, run_id=run_id,
                                            tags=version_tags(meta, report),
                                            description=f"{meta['model_version']}, gate passed")
    version = str(mv.version)
    logger.info("registered %s version %s from run %s", name, version, run_id)

    ar = cfg["artifact_registry"]
    ar_path = None
    if upload and ar.get("enabled", True):
        try:
            ar_path = upload_package(package, version, ar, runner)
        except RegistrationError:
            mlflow_client.set_model_version_tag(name, version, "ar_upload", "failed")
            raise
        mlflow_client.set_model_version_tag(name, version, "ar_upload", "ok")
        mlflow_client.set_model_version_tag(name, version, "ar_package", ar_path)
    else:
        mlflow_client.set_model_version_tag(name, version, "ar_upload", "skipped")

    mlflow_client.set_registered_model_alias(name, alias, version)
    return {"registered_model": name, "version": version, "alias": alias,
            "model_version": meta["model_version"], "run_id": run_id, "source": source,
            "artifact_registry": ar_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", required=True, type=Path,
                        help="package folder: train.py's <run>/model (+ thresholds.json)")
    parser.add_argument("--gate-report", required=True, type=Path,
                        help="gate_report.json from src.eval.gate for this package")
    parser.add_argument("--config", default=None)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--alias", default=None, help="default: mlflow.candidate_alias")
    parser.add_argument("--no-upload", action="store_true", help="skip Artifact Registry")
    parser.add_argument("--out", type=Path, default=None, help="write the summary JSON here")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    try:
        summary = register(cfg, args.package, args.gate_report, client(args.tracking_uri),
                           upload=not args.no_upload, alias=args.alias)
    except RegistrationError as exc:
        notify(cfg, "fail", "Model not registered", {"package": str(args.package),
                                                     "reason": str(exc)})
        return 1
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    notify(cfg, "ok", f"Registered {summary['registered_model']} v{summary['version']} "
           f"as {summary['alias']}", {"model_version": summary["model_version"],
                                       "artifact_registry": summary["artifact_registry"]
                                       or "skipped"})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
