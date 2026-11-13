"""Log the Phase 2 student runs to MLflow from their committed records: nothing is rerun.

The ablation session (``kaggle/13_student_ablation_all.ipynb``) logged to per-GPU SQLite files
on Kaggle, which did not come back. Its records did (``reports/student/phase2/``); this turns
them into MLflow runs on the team server, in the notebook's experiment:

* one run per arm (``runs/<arm>/``), named after the training run, with the parameters, the
  per-epoch validation history as stepped metrics, and the metric and tag names that training
  and ``src.pipeline.run`` log (``best_val_mean_auc``, ``gate_mean_auc``, ``latency_p95_ms``,
  ``gate_passed``, ...), plus every committed file of the arm as an artifact;
* ``selection``: the rule, the selected arm, and the results and selection files.

Each arm's run is tagged ``kaggle_run_id`` with the id its Kaggle run had (``meta.json``), so
the two can be matched; if a run with that id is already on the server, the arm is skipped.
Each run is also tagged with a hash of everything it logs, once it has finished: a record
already logged with the same hash is skipped, so the script can be rerun without duplicates.

    python -m src.student.mlflow_log --dry-run      # what would be logged
    python -m src.student.mlflow_log                # MLFLOW_TRACKING_URI, else mlruns/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.student.config import LABELS, PIPELINE_ROOT, load_config
from src.teacher.config import get_logger

logger = get_logger(__name__)

RECORDS_DIR = PIPELINE_ROOT / "reports" / "student" / "phase2"
EXPERIMENT = "distillate-student-ablation"      # the notebook's mlflow.experiment
EVAL_KEYS = ("mean_auc", "mean_pr_auc", "macro_f1", "ece", "brier")
SELECTION_FILES = ("selection.json", "selection.md", "selection_as_run.md", "results.json",
                   "results.md", "run_info.json", "arm_status.json", "eval_tables.json")


@dataclass
class Record:
    """One MLflow run to log: its name doubles as the ``record`` tag."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    steps: dict[str, list[float]] = field(default_factory=dict)   # metric -> value per epoch
    artifacts: list[Path] = field(default_factory=list)
    artifact_dirs: dict[str, Path] = field(default_factory=dict)  # MLflow path -> folder

    def files(self) -> list[Path]:
        return sorted([*self.artifacts, *(p for d in self.artifact_dirs.values()
                                          for p in d.rglob("*") if p.is_file())])

    def content_hash(self) -> str:
        """SHA-256 of everything the run would hold, artifact bytes included."""
        digest = hashlib.sha256(json.dumps(
            [self.name, self.params, self.metrics, self.tags, self.steps], sort_keys=True,
            default=str).encode())
        for path in self.files():
            digest.update(path.name.encode() + hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()


def _numbers(values: Mapping[str, Any]) -> dict[str, float]:
    """Only finite numbers (MLflow metrics are floats; NaN, None and text are left out)."""
    return {k: float(v) for k, v in values.items()
            if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def arm_record(run_dir: Path, tuned: Iterable[str]) -> Record:
    """One arm: its training summary, lineage, evaluation, gate and latency records."""
    summary = read_json(run_dir / "summary.json")
    meta = read_json(run_dir / "meta.json")
    pipeline = read_json(run_dir / "pipeline.json")
    latency = read_json(run_dir / "latency.json")
    evals = read_json(run_dir / "eval_metrics.json")
    history = summary["history"]

    params = {key.split("=", 1)[0]: key.split("=", 1)[1] for key in tuned}
    params |= {"loss.alpha": summary["alpha"], "loss.temperature": summary["temperature"],
               "model_id": summary["model_id"], "model.max_length": meta["max_length"],
               "seed": meta["seed"], "train_rows": summary["data"]["train_rows"],
               "val_rows": summary["data"]["val_rows"], "epochs_run": len(history),
               "soft_label_arm": meta["soft_label_arm"] or "none",
               "teacher_version": meta["teacher_version"] or "none",
               "prompt_version": meta["prompt_version"] or "none",
               "training_data_sha256": meta["training_data_sha256"],
               "latency_format": latency["format"], "latency_threads": latency["threads"]}
    if summary["run_name"].startswith("distilbert-mixed-5ep"):
        params["train.epochs"] = len(history)        # the 5-epoch arm's one override

    metrics = _numbers({
        "best_val_mean_auc": summary["best_val_mean_auc"],
        "best_val_mean_pr_auc": summary["best_val_mean_pr_auc"],
        "best_epoch": summary["best_epoch"], "steps": summary["steps"],
        "truncated_share_train": summary["truncated_share_train"],
        "val_mean_f1_at_thresholds": summary["val_mean_f1_at_thresholds"],
        "duration_s": pipeline["duration_s"],
        **{f"threshold_{k}": v for k, v in summary["thresholds"].items()},
        **{f"latency_{k}": latency[k] for k in ("p50_ms", "p90_ms", "p95_ms", "p99_ms",
                                                 "mean_ms")},
    })
    if (summary.get("package") or {}).get("status") == "ok":
        metrics["onnx_parity"] = float(summary["package"]["onnx_parity"])
    for stem, block in evals.items():
        metrics |= _numbers({f"{stem}_{k}": block.get(k) for k in EVAL_KEYS})
        metrics |= _numbers({f"{stem}_flag_{k}": v for k, v in block.get("flag", {}).items()})
        metrics |= _numbers({f"{stem}_auc_{label}": block["per_label"][label].get("auc")
                             for label in LABELS if label in block.get("per_label", {})})
    steps = {key: [float(epoch[key]) for epoch in history]
             for key in history[0] if key not in ("epoch", "step")
             and all(isinstance(epoch.get(key), int | float) for epoch in history)}
    # The validation block repeats the best epoch's val_mean_auc and val_mean_pr_auc: keep the
    # per-epoch history under those names (best_val_* has the best epoch's).
    metrics = {k: v for k, v in metrics.items() if k not in steps}

    gate = pipeline.get("gate") or {}
    tags = {"record": summary["run_name"], "kind": "student-arm", "lane": "student",
            "model_version": summary["model_version"], "code_commit": meta["code_commit"],
            "data_source": meta["data_source"], "data_hash": pipeline["data"]["data_hash"],
            "soft_label_arm": meta["soft_label_arm"] or "none",
            "gate_passed": str(gate.get("passed")), "bias_passed": str(gate.get("bias_passed")),
            "package_status": (summary.get("package") or {}).get("status", "none"),
            "kaggle_run_id": meta["mlflow_run_id"] or "none"}
    return Record(name=summary["run_name"], params=params, metrics=metrics, tags=tags,
                  steps=steps, artifacts=sorted(p for p in run_dir.iterdir() if p.is_file()))


def selection_record(records_dir: Path) -> Record:
    """The selection: its rule and result, with the results tables and plots."""
    selection = read_json(records_dir / "selection.json")
    run_info = read_json(records_dir / "run_info.json")
    chosen = selection["selected"] or {}
    params = {f"select.{k}": v if v is not None else "null"
              for k, v in selection["rule"].items()}
    params |= {"candidates_seen": len(selection["candidates"]),
               "eligible": sum(bool(c["eligible"]) for c in selection["candidates"]),
               "tied": ",".join(selection.get("tied") or []) or "none",
               "session_hours": run_info["hours"], "session_gpus": run_info["gpus"]}
    metrics = _numbers({"selected_val_mean_auc": chosen.get("val_mean_auc"),
                        "selected_val_mean_pr_auc": chosen.get("val_mean_pr_auc")})
    tags = {"record": "selection", "kind": "selection", "lane": "student", "step": "selection",
            "selected": chosen.get("arm", "none"),
            "selected_model_version": chosen.get("model_version", "none"),
            "code_commit": run_info["code_commit"]}
    return Record(name="selection", params=params, metrics=metrics, tags=tags,
                  artifacts=[records_dir / name for name in SELECTION_FILES
                             if (records_dir / name).exists()],
                  artifact_dirs={"plots": records_dir / "plots"})


def build_records(records_dir: Path = RECORDS_DIR) -> list[Record]:
    """Every arm under ``runs/``, then the selection."""
    tuned = read_json(records_dir / "run_info.json")["tuned"]
    arms = sorted(p for p in (records_dir / "runs").iterdir() if (p / "summary.json").exists())
    if not arms:
        raise ValueError(f"no arm records under {records_dir / 'runs'}")
    return [*(arm_record(p, tuned) for p in arms), selection_record(records_dir)]


def tracking(cfg: Mapping[str, Any], experiment: str) -> str:
    """Point MLflow at ``MLFLOW_TRACKING_URI`` or the local store; return the experiment id."""
    import mlflow

    # Read before set_tracking_uri, which writes MLFLOW_TRACKING_URI into the environment.
    server = os.environ.get("MLFLOW_TRACKING_URI")
    uri = server
    if not uri:
        db = Path(cfg["paths"]["mlflow_db"])
        db.parent.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{db}"
    mlflow.set_tracking_uri(uri)
    if not server:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
    if mlflow.get_experiment_by_name(experiment) is None:
        # A server stores artifacts where it's configured to; the local store next to its db.
        artifacts = None if server else Path(cfg["paths"]["mlflow_artifacts"]).resolve().as_uri()
        mlflow.create_experiment(experiment, artifact_location=artifacts)
    return mlflow.set_experiment(experiment).experiment_id


def run_exists(run_id: str) -> bool:
    """Whether the tracking server already has a run with this id (the Kaggle run's)."""
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    try:
        MlflowClient().get_run(run_id)
    except MlflowException as error:
        if error.error_code == "RESOURCE_DOES_NOT_EXIST":
            return False
        raise
    return True


def _log(record: Record, digest: str) -> str:
    import mlflow

    tags = {**record.tags, "source": "committed records (reports/student/phase2)"}
    with mlflow.start_run(run_name=record.name, tags=tags) as active:
        mlflow.log_params({k: str(v)[:500] for k, v in record.params.items()})
        mlflow.log_metrics(record.metrics)
        for key, values in record.steps.items():
            for epoch, value in enumerate(values):
                mlflow.log_metric(key, value, step=epoch)
        for path in record.artifacts:
            mlflow.log_artifact(str(path))
        for name, folder in record.artifact_dirs.items():
            mlflow.log_artifacts(str(folder), name)
        # Last: a run whose logging failed partway has no hash, so a rerun logs it again.
        mlflow.set_tag("content_sha256", digest)
        return active.info.run_id


def log_records(cfg: Mapping[str, Any], records: Iterable[Record],
                experiment: str = EXPERIMENT) -> list[dict[str, str]]:
    """Log each record unless it, or its Kaggle run, is already in the tracking store."""
    import mlflow

    experiment_id = tracking(cfg, experiment)
    out = []
    for record in records:
        kaggle = record.tags.get("kaggle_run_id", "none")
        if kaggle != "none" and run_exists(kaggle):
            out.append({"record": record.name, "status": "skipped (Kaggle run on the server)",
                        "run_id": kaggle})
            continue
        digest = record.content_hash()
        found = mlflow.search_runs(
            [experiment_id], output_format="list",
            filter_string=f"tags.record = '{record.name}' and tags.content_sha256 = '{digest}'")
        if found:
            out.append({"record": record.name, "status": "skipped (already logged)",
                        "run_id": found[0].info.run_id})
            continue
        out.append({"record": record.name, "status": "logged", "run_id": _log(record, digest)})
        logger.info("logged %s (%d params, %d metrics, %d files)", record.name,
                    len(record.params), len(record.metrics), len(record.files()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Log the Phase 2 student runs to MLflow.")
    parser.add_argument("--config", default=None, help="for the local store's paths")
    parser.add_argument("--records", default=str(RECORDS_DIR))
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--dry-run", action="store_true", help="print, don't log")
    args = parser.parse_args()
    records = build_records(Path(args.records))
    if args.dry_run:
        print(json.dumps([{"record": r.name, "params": len(r.params), "metrics": len(r.metrics),
                           "stepped": sorted(r.steps), "files": len(r.files()),
                           "kaggle_run_id": r.tags.get("kaggle_run_id")} for r in records],
                         indent=2))
        return
    print(json.dumps(log_records(load_config(args.config), records, args.experiment), indent=2))


if __name__ == "__main__":
    main()
