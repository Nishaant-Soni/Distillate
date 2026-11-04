"""Log the teacher lane's runs to MLflow from what they already recorded: nothing is rerun.

Phase 1 kept every result in committed JSON files (``artifacts/``) and in the shard manifests;
this turns them into MLflow runs in one experiment, so the lane sits next to the student's
training runs in the shared tracking server:

* ``pilot``: the pilot's setup and decision, with one child run per configuration (each
  prompt in ``artifacts/pilot/`` and ``artifacts/prompt_v2/``, and the TF-IDF baseline);
* ``labeling-train``, ``labeling-gate``, ``labeling-drift``: the raw teacher's shard runs,
  summed over their manifests (rows, failures, GPU seconds, coverage, code commits);
* ``soft-labels-stack-v1`` and ``soft-labels-calqwen-v1``: the soft labels' checks, with the
  drift stream's per-batch AUCs as stepped metrics.

Each run is tagged with a hash of everything it logs. A record already logged with the same
hash is skipped, so the script can be rerun (on the VM's server, say) without duplicates; a
record whose content changed is logged again as a new run.

    python -m src.teacher.mlflow_log --dry-run      # what would be logged
    python -m src.teacher.mlflow_log                # MLFLOW_TRACKING_URI, else mlruns/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.teacher.config import LABELS, PIPELINE_ROOT, get_logger, load_config
from src.teacher.prompts import prompt_version
from src.teacher.storage import Storage, get_storage

logger = get_logger(__name__)

SUMMARY_KEYS = ("mean_auc", "mean_pr_auc", "ece", "brier")
LABEL_KEYS = ("auc", "pr_auc", "ece", "brier")
COUNT_FIELDS = ("row_count", "failed_rows", "empty_rows", "truncated_rows", "retried_cells",
                "censored_cells")


@dataclass
class Record:
    """One MLflow run to log: its name doubles as the ``record`` tag."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    steps: dict[str, list[float]] = field(default_factory=dict)   # metric -> value per step
    artifacts: list[Path] = field(default_factory=list)
    children: list[Record] = field(default_factory=list)

    def content_hash(self) -> str:
        """SHA-256 of everything the run would hold, children and artifact bytes included."""
        digest = hashlib.sha256(json.dumps(
            [self.name, self.params, self.metrics, self.tags, self.steps], sort_keys=True,
            default=str).encode())
        for path in sorted(self.artifacts):
            digest.update(path.name.encode() + hashlib.sha256(path.read_bytes()).digest())
        for child in self.children:
            digest.update(child.content_hash().encode())
        return digest.hexdigest()


def _numbers(values: Mapping[str, Any]) -> dict[str, float]:
    """Only finite numbers (MLflow metrics are floats; NaN and text are left out)."""
    out = {}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        if math.isfinite(value):
            out[key] = float(value)
    return out


def label_metrics(prefix: str, per_label: Mapping[str, Mapping[str, Any]],
                  keys: Iterable[str] = LABEL_KEYS) -> dict[str, float]:
    """``<prefix>auc_toxicity``, ... and ``<prefix>mean_auc``, ... over the labels present."""
    out: dict[str, Any] = {}
    for key in keys:
        values = [per_label[label][key] for label in LABELS
                  if label in per_label and key in per_label[label]]
        out |= {f"{prefix}{key}_{label}": per_label[label].get(key) for label in LABELS
                if label in per_label}
        finite = [v for v in values if isinstance(v, int | float) and math.isfinite(v)]
        if finite:
            out[f"{prefix}mean_{key}"] = sum(finite) / len(finite)
    return _numbers(out)


def summary_metrics(prefix: str, block: Mapping[str, Any]) -> dict[str, float]:
    """A check file's ``{mean_auc, mean_pr_auc, ece, brier, auc: {label: ...}}`` block."""
    out: dict[str, Any] = {f"{prefix}{k}": block.get(k) for k in SUMMARY_KEYS}
    out |= {f"{prefix}auc_{label}": v for label, v in block.get("auc", {}).items()}
    return _numbers(out)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def config_record(folder: Path, metrics_file: Path) -> Record:
    """One pilot configuration: a prompt's ``metrics_<prompt>.json``, or the TF-IDF baseline's."""
    data = read_json(metrics_file)
    if "prompt_version" in data:
        name = f"{folder.name}/{data['prompt_version']}"
        metrics = _numbers(data["info"]) | _numbers(
            {k: v for k, v in data["saturation"].items() if not isinstance(v, dict)})
        metrics |= label_metrics("raw_", data["raw"])
        if "calibrated_cross_fitted" in data:
            metrics |= label_metrics("calibrated_", data["calibrated_cross_fitted"])
        params = {"prompt_version": data["prompt_version"], "folder": folder.name}
    else:
        name = f"{folder.name}/{data['model_version']}"
        metrics = _numbers({k: v for k, v in data.items() if k != "metrics"})
        metrics |= label_metrics("", data["metrics"])
        params = {"model_version": data["model_version"], "folder": folder.name}
    return Record(name=name, params=params, metrics=metrics, artifacts=[metrics_file],
                  tags={"record": name, "kind": "pilot-configuration"})


def pilot_record(cfg: Mapping[str, Any]) -> Record:
    """The pilot: sample, decision, and one child per configuration it compared."""
    pilot_dir, v2_dir = Path(cfg["paths"]["pilot_dir"]), Path(cfg["paths"]["prompt_v2_dir"])
    results = read_json(pilot_dir / "pilot_results.json")
    sample = read_json(pilot_dir / "pilot_summary.json")
    selection = read_json(v2_dir / "selection.json")
    decision = results["decision"]
    params = {
        "model_version": results["model_version"], "vllm_version": results["vllm_version"],
        "code_commit": results["code_commit"], "seed": cfg["seed"],
        "pilot_rows": sample["rows"], "population": sample["population"],
        "strata": json.dumps(sample["strata"], sort_keys=True),
        "best_zero_shot": decision["best_zero_shot"],
        "lora_contingency": decision["lora_contingency"],
        "prompt_v2_label": selection["label"], "prompt_v2_chosen": selection["chosen"],
        "prompt_v2_code_commit": selection["code_commit"],
        "adopted_prompt": prompt_version(cfg["prompt"]["protocol"], cfg["prompt"]["shots"],
                                         cfg["prompt"].get("text_version", "v1")),
    }
    metrics = _numbers({"vllm_load_s": results.get("vllm_load_s"),
                        "best_zero_shot_mean_auc": decision["best_zero_shot_mean_auc"],
                        "tfidf_mean_auc": decision["tfidf_mean_auc"],
                        "prompt_v2_v1_auc": selection["v1_auc"]})
    metrics |= {f"prompt_v2_candidate_auc_{k}": float(v)
                for k, v in selection["candidate_auc"].items()}
    children = [config_record(folder, path) for folder in (pilot_dir, v2_dir)
                for path in sorted(folder.glob("metrics_*.json"))]
    artifacts = [pilot_dir / "pilot_results.json", pilot_dir / "pilot_summary.json",
                 v2_dir / "selection.json", v2_dir / "v2_results.json"]
    return Record(name="pilot", params=params, metrics=metrics, artifacts=artifacts,
                  children=children, tags={"record": "pilot", "kind": "pilot",
                                           "model_version": results["model_version"],
                                           "code_commit": results["code_commit"]})


def read_manifests(storage: Storage, prefix: str, cache: Path) -> dict[str, dict[str, Any]]:
    """The manifests directly under ``prefix`` (not those of runs nested under it)."""
    manifests = {}
    for path in storage.list(prefix):
        rest = path[len(prefix):]
        if "/" not in rest and rest.endswith(".manifest.json"):
            name = rest.removesuffix(".manifest.json")
            manifests[name] = read_json(Path(storage.download(path, cache)))
    if not manifests:
        raise ValueError(f"no manifests under {prefix}")
    return dict(sorted(manifests.items()))


def labeling_record(run: str, prefix: str, manifests: Mapping[str, Mapping[str, Any]],
                    out_dir: Path) -> Record:
    """One labeling run, summed over its shards; the manifests themselves as an artifact."""
    versions = {(m["prompt_version"], m["model_version"]) for m in manifests.values()}
    if len(versions) != 1:
        raise ValueError(f"{prefix}: shards name several prompt/model versions: {versions}")
    (prompt_version, model_version), = versions
    values = list(manifests.values())
    totals = {f: sum(int(m.get(f, 0)) for m in values) for f in COUNT_FIELDS}
    seconds = sum(float(m.get("seconds", 0.0)) for m in values)
    rows = totals.pop("row_count")
    metrics = _numbers({"rows": rows, "shards": len(values), **totals,
                        "failure_rate": totals["failed_rows"] / rows if rows else None,
                        "gpu_seconds": seconds,
                        "rows_per_gpu_second": rows / seconds if seconds else None,
                        "coverage_min": min((m["coverage_min"] for m in values
                                             if "coverage_min" in m), default=None)})
    commits = sorted({m["code_commit"] for m in values if "code_commit" in m})
    created = sorted(m["created_at"] for m in values if "created_at" in m)
    params = {"prefix": prefix, "prompt_version": prompt_version, "model_version": model_version,
              "vllm_version": ",".join(sorted({m["vllm_version"] for m in values
                                               if "vllm_version" in m})),
              "code_commit": ",".join(commits),
              "first_shard_at": created[0] if created else "", "last_shard_at":
              created[-1] if created else ""}
    if any("layout" in m for m in values):
        params["layout"] = next(m["layout"] for m in values if "layout" in m)
    path = out_dir / f"labeling-{run}-manifests.json"
    path.write_text(json.dumps({"prefix": prefix, "manifests": manifests}, indent=2,
                               sort_keys=True) + "\n")
    return Record(name=f"labeling-{run}", params=params, metrics=metrics, artifacts=[path],
                  tags={"record": f"labeling-{run}", "kind": "labeling",
                        "prompt_version": prompt_version, "model_version": model_version,
                        "code_commit": ",".join(commits)})


def stack_record(cfg: Mapping[str, Any]) -> Record:
    """stack-v1: the adopted stack, its soft-label checks, and the drift stream per batch."""
    stack_dir, oof_dir = Path(cfg["paths"]["stack_dir"]), Path(cfg["paths"]["oof_dir"])
    check = read_json(stack_dir / "stack_check.json")
    drift = read_json(stack_dir / "drift_check.json")
    adopted = check["stacks"][check["decision"]["adopt"]]
    stack_file = stack_dir / f"stack_{check['prompt_version']}.json"
    soft = check["soft_labels"]
    params = {k: adopted[k] for k in ("stack_version", "method", "C", "prompt_version",
                                      "model_version", "baseline_version")}
    params |= {"soft_label_model_version": soft["model_version"],
               "stack_sha256": hashlib.sha256(stack_file.read_bytes()).hexdigest(),
               "adopted": check["decision"]["adopt"], "drift_rows": drift["rows"],
               "drift_batches": drift["batches"]}
    metrics: dict[str, float] = {}
    for key, block in soft.items():
        if isinstance(block, dict) and "mean_auc" in block:
            metrics |= summary_metrics(f"{key}_", block)
    for key, block in drift["overall"].items():
        metrics |= summary_metrics(f"drift_{key}_", block)
    batches = sorted(drift["per_batch"], key=lambda b: b["batch"])
    steps = {f"drift_batch_{k}_mean_auc": [float(b[f"{k}_mean_auc"]) for b in batches]
             for k in ("stack", "teacher_raw", "tfidf")}
    return Record(name="soft-labels-stack-v1", params=params, metrics=metrics, steps=steps,
                  artifacts=[stack_dir / "stack_check.json", stack_dir / "drift_check.json",
                             stack_file, oof_dir / "oof_check.json"],
                  tags={"record": "soft-labels-stack-v1", "kind": "soft-labels",
                        "model_version": soft["model_version"]})


def calqwen_record(cfg: Mapping[str, Any]) -> Record:
    """calqwen-v1: the pilot's frozen Platt calibration on Qwen's logits, and its checks."""
    calqwen_dir = Path(cfg["paths"]["calqwen_dir"])
    check = read_json(calqwen_dir / "calqwen_check.json")
    params = {k: check[k] for k in ("calqwen_version", "model_version", "prompt_version",
                                    "calibration_file", "calibration_sha256")}
    params["calibration_version"] = check["calibration"]["calibration_version"]
    metrics: dict[str, float] = {}
    for key, block in check.items():
        if isinstance(block, dict) and "mean_auc" in block:
            metrics |= summary_metrics(f"{key}_", block)
    calibration = Path(cfg["paths"]["pilot_dir"]) / Path(check["calibration_file"]).name
    return Record(name="soft-labels-calqwen-v1", params=params, metrics=metrics,
                  artifacts=[calqwen_dir / "calqwen_check.json", calibration],
                  tags={"record": "soft-labels-calqwen-v1", "kind": "soft-labels",
                        "model_version": check["model_version"]})


def build_records(cfg: Mapping[str, Any], storage: Storage | None, out_dir: Path,
                  which: Iterable[str] = ("pilot", "labeling", "soft_labels")) -> list[Record]:
    """The records to log; ``labeling`` reads the manifests from ``storage``."""
    which = set(which)
    records = []
    if "pilot" in which:
        records.append(pilot_record(cfg))
    if "labeling" in which:
        if storage is None:
            raise ValueError("labeling records read the shard manifests: pass a storage")
        with tempfile.TemporaryDirectory() as tmp:
            for run, prefix in cfg["mlflow"]["labeling"].items():
                records.append(labeling_record(run, prefix,
                                               read_manifests(storage, prefix, Path(tmp)),
                                               out_dir))
    if "soft_labels" in which:
        records += [stack_record(cfg), calqwen_record(cfg)]
    return records


def tracking(cfg: Mapping[str, Any]) -> str:
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
    name = cfg["mlflow"]["experiment"]
    if mlflow.get_experiment_by_name(name) is None:
        # A server stores artifacts where it's configured to; the local store next to its db.
        artifacts = None if server else Path(cfg["paths"]["mlflow_artifacts"]).resolve().as_uri()
        mlflow.create_experiment(name, artifact_location=artifacts)
    # Active, so the runs land in it (not in MLflow's Default experiment).
    return mlflow.set_experiment(name).experiment_id


def _log(record: Record, digest: str | None, nested: bool = False) -> str:
    """One run, and its children nested under it (``digest`` only on the top-level run)."""
    import mlflow

    tags = {**record.tags, "lane": "teacher", "source": "committed records and shard manifests"}
    if digest:
        tags["content_sha256"] = digest
    with mlflow.start_run(run_name=record.name, nested=nested, tags=tags) as active:
        mlflow.log_params({k: str(v) for k, v in record.params.items()})
        mlflow.log_metrics(record.metrics)
        for key, values in record.steps.items():
            for step, value in enumerate(values):
                mlflow.log_metric(key, value, step=step)
        for path in record.artifacts:
            mlflow.log_artifact(str(path))
        for child in record.children:
            _log(child, None, nested=True)
        return active.info.run_id


def log_records(cfg: Mapping[str, Any], records: Iterable[Record]) -> list[dict[str, str]]:
    """Log each record unless a run with the same name and content hash already exists."""
    import mlflow

    experiment_id = tracking(cfg)
    out = []
    for record in records:
        digest = record.content_hash()
        found = mlflow.search_runs(
            [experiment_id], output_format="list",
            filter_string=f"tags.record = '{record.name}' and tags.content_sha256 = '{digest}'")
        if found:
            out.append({"record": record.name, "status": "skipped (already logged)",
                        "run_id": found[0].info.run_id})
            continue
        out.append({"record": record.name, "status": "logged", "run_id": _log(record, digest)})
        logger.info("logged %s (%d params, %d metrics, %d children)", record.name,
                    len(record.params), len(record.metrics), len(record.children))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Log the teacher lane's runs to MLflow.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--records", nargs="+", default=["pilot", "labeling", "soft_labels"],
                        choices=["pilot", "labeling", "soft_labels"])
    parser.add_argument("--local-root", default=None,
                        help="a local copy of the shard store (default: the configured store)")
    parser.add_argument("--dry-run", action="store_true", help="print, don't log")
    args = parser.parse_args()
    cfg = load_config(args.config)
    storage = get_storage(cfg, args.local_root) if "labeling" in args.records else None
    with tempfile.TemporaryDirectory() as tmp:
        records = build_records(cfg, storage, Path(tmp), args.records)
        if args.dry_run:
            print(json.dumps([{"record": r.name, "params": len(r.params),
                               "metrics": len(r.metrics), "children": [c.name for c in
                                                                        r.children],
                               "artifacts": [str(p.relative_to(PIPELINE_ROOT)) if
                                             p.is_relative_to(PIPELINE_ROOT) else p.name
                                             for p in r.artifacts]}
                              for r in records], indent=2))
            return
        print(json.dumps(log_records(cfg, records), indent=2))


if __name__ == "__main__":
    main()
