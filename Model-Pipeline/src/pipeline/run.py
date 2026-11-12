"""The model pipeline's single entry point (work plan v4 §5.1).

    python -m src.pipeline.run --config configs/student.yaml --set loss.alpha=0.34 ...

In order, for the student the config gives:

1. **Data.** The evaluation tables (gate, H2, bias, obfuscation) through
   ``src.data.load_data``: the files on disk, a DVC revision (``pipeline.data_rev``), or a
   folder in the same layout (``pipeline.data_dir``; on Kaggle, ``src.student.eval_sets``
   builds it from the competition files). Loaded before training, so a missing table fails
   in seconds, not after an hour. Their hash is logged.
2. **Train** (``src.student.train``): the model, ``thresholds.json``, the validation
   prediction file, and the §2.5 package when ``package.enabled``.
3. **Validate.** Prediction files for the gate, H2, bias and obfuscation sets next to
   ``val.parquet`` (``src.student.evaluate``), the gold-label files (``<run>/gold/``, no text),
   and every metric (``src.eval.metrics``) in ``<run>/eval/metrics.json``.
4. **Bias check and gate** (``src.eval.gate``, ``configs/gate.yaml``, no production model):
   ``<run>/eval/gate_report.{json,md}``, the slice FPRs included.
5. **Latency** (``pipeline.latency``): the package's batch-1 CPU latency, ``<run>/latency.json``.
6. **Plots** (``pipeline.plots``): per-label confusion matrices at the tuned thresholds and
   per-label AUC / PR-AUC bars, ``<run>/plots/``.
7. **MLflow**: all of the above into the training run.

``<run>/pipeline.json`` records what each step did. A failed step after training is recorded
there and the run keeps everything before it.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.student.config import load_config
from src.teacher.config import LABELS, PIPELINE_ROOT, get_logger

logger = get_logger(__name__)

TABLE_STEMS = {"gate": "gate", "h2": "h2_eval", "bias": "bias_eval",
               "obfuscation": "obfuscation"}


def _path(value: str | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else PIPELINE_ROOT / p


def labels_for(stem: str) -> tuple[str, ...]:
    from src.eval.metrics import CIVIL_LABELS

    return CIVIL_LABELS if stem in ("h2_eval", "bias_eval") else LABELS


def load_eval_tables(p: Mapping[str, Any]):
    from src.data.load_data import load_data

    return load_data(rev=p.get("data_rev"), data_dir=_path(p.get("data_dir")),
                     with_teacher=False, tables=tuple(p["tables"]))


def eval_metrics(run_dir: Path) -> dict[str, Any]:
    """Every metric for each prediction file against its gold file, at the tuned thresholds."""
    from src.eval.metrics import evaluate, load_thresholds, read_frame, to_jsonable

    out = {}
    for gold_path in sorted((run_dir / "gold").glob("*.parquet")):
        stem = gold_path.stem
        pred_path = run_dir / "predictions" / f"{stem}.parquet"
        if not pred_path.exists():
            continue
        labels = labels_for(stem)
        thresholds = load_thresholds(run_dir / "predictions" / "thresholds.json", labels)
        out[stem] = evaluate(read_frame(pred_path), read_frame(gold_path), labels, thresholds)
    out = to_jsonable(out)
    (run_dir / "eval").mkdir(exist_ok=True)
    (run_dir / "eval" / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def gate_report(run_dir: Path, gate_config: Path) -> dict[str, Any]:
    from src.eval import gate

    cfg = gate.load_config(gate_config)
    report = gate.to_jsonable(gate.run_gate(cfg, run_dir / "predictions", run_dir / "gold"))
    (run_dir / "eval" / "gate_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (run_dir / "eval" / "gate_report.md").write_text(gate.render_markdown(report))
    return report


def bias_passed(report: Mapping[str, Any]) -> bool:
    return all(c["passed"] is not False for c in report["checks"]
               if c["name"].startswith("bias"))


def run(cfg: Mapping[str, Any]) -> dict[str, Any]:
    from src.student import train
    from src.student.data import load_splits
    from src.student.eval_sets import write_gold
    from src.student.evaluate import score_run

    p = cfg["pipeline"]
    record: dict[str, Any] = {"steps": {}, "errors": {}}
    start = time.perf_counter()

    def step(name, fn):
        t0 = time.perf_counter()
        try:
            result = fn()
            record["steps"][name] = {"status": "ok", "seconds": round(time.perf_counter() - t0, 1)}
            return result
        except Exception as error:   # after training: record, keep what's done
            logger.exception("pipeline step %s failed", name)
            record["steps"][name] = {"status": "failed",
                                     "seconds": round(time.perf_counter() - t0, 1)}
            record["errors"][name] = f"{type(error).__name__}: {error}"
            record["errors"][f"{name}_traceback"] = traceback.format_exc(limit=5)
            return None

    # 1. Data first, so a missing table fails before training.
    loaded = load_eval_tables(p)
    record["data"] = {"data_hash": loaded.data_hash, "files": loaded.file_hashes,
                      "rev": loaded.rev, "rows": {k: len(v) for k, v in loaded.tables.items()}}

    # 2. Train.
    summary = train.run(cfg)
    run_dir = Path(summary["run_dir"])
    record.update(run_dir=str(run_dir), run_name=summary["run_name"],
                  model_version=summary["model_version"],
                  best_val_mean_auc=summary["best_val_mean_auc"],
                  best_val_mean_pr_auc=summary["best_val_mean_pr_auc"],
                  package=summary.get("package"))

    def write(name="pipeline.json"):
        record["duration_s"] = round(time.perf_counter() - start, 1)
        (run_dir / name).write_text(json.dumps(record, indent=2, default=str) + "\n")

    write()
    tables = {TABLE_STEMS[k]: v for k, v in loaded.tables.items()}

    # 3. Validate: prediction files, gold files, metrics.
    record["scoring"] = step("score", lambda: score_run(
        run_dir, {stem: t[["id", "text"]] for stem, t in tables.items()},
        int(p["score_batch_size"])))

    def gold():
        _, val_df, _ = load_splits(cfg)
        return write_gold({"val": val_df, **tables}, run_dir / "gold")

    record["gold_rows"] = step("gold", gold)
    metrics = step("metrics", lambda: eval_metrics(run_dir))
    if metrics:
        record["headline"] = {stem: {k: m[k] for k in ("mean_auc", "mean_pr_auc", "macro_f1")}
                              for stem, m in metrics.items()}
    write()

    # 4. Bias check and gate.
    report = step("gate", lambda: gate_report(run_dir, _path(p["gate_config"])))
    if report:
        record["gate"] = {"passed": report["passed"], "failed_checks": report["failed_checks"],
                          "bias_passed": bias_passed(report)}

    # 5. Latency.
    package = summary.get("package") or {}
    if p.get("latency") and package.get("status") == "ok":
        def latency():
            from src.student.evaluate import measure_latency

            _, val_df, _ = load_splits(cfg)
            result = measure_latency(package["dir"], val_df["text"].tolist(),
                                     int(p["latency_threads"]), int(p["latency_requests"]))
            (run_dir / "latency.json").write_text(json.dumps(result, indent=2) + "\n")
            return result

        record["latency"] = step("latency", latency)

    # 6. Plots.
    if p.get("plots") and metrics:
        def plots():
            from src.eval import plots as plots_mod

            return plots_mod.run_plots(run_dir, run_dir / "plots")

        record["plots"] = step("plots", plots)
    write()

    # 7. MLflow: into the training run.
    if cfg["mlflow"]["enabled"] and summary.get("mlflow_run_id"):
        def log():
            import mlflow

            with mlflow.start_run(run_id=summary["mlflow_run_id"]):
                mlflow.set_tag("data_hash", loaded.data_hash)
                for stem, h in (record.get("headline") or {}).items():
                    mlflow.log_metrics({f"{stem}_{k}": v for k, v in h.items()
                                        if isinstance(v, (int, float))})
                if record.get("gate"):
                    mlflow.set_tags({"gate_passed": record["gate"]["passed"],
                                     "bias_passed": record["gate"]["bias_passed"]})
                lat = record.get("latency")
                if lat:
                    mlflow.log_metrics({"latency_p50_ms": lat["p50_ms"],
                                        "latency_p95_ms": lat["p95_ms"],
                                        "latency_p99_ms": lat["p99_ms"]})
                for sub in ("eval", "plots"):
                    if (run_dir / sub).exists():
                        mlflow.log_artifacts(str(run_dir / sub), sub)
                for name in ("scoring.json", "latency.json"):
                    if (run_dir / name).exists():
                        mlflow.log_artifact(str(run_dir / name))
                for stem in tables:
                    path = run_dir / "predictions" / f"{stem}.parquet"
                    if path.exists():
                        mlflow.log_artifact(str(path), "predictions")
                write()
                mlflow.log_artifact(str(run_dir / "pipeline.json"))

        step("mlflow", log)
    write()
    logger.info("pipeline done: %s (%s)", run_dir, "errors: " + ", ".join(
        k for k in record["errors"] if not k.endswith("_traceback")) if record["errors"]
        else "all steps ok")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train, validate and check one student.")
    parser.add_argument("--config", default=None, help="default configs/student.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    record = run(load_config(args.config, args.set))
    print(json.dumps({k: record.get(k) for k in (
        "run_dir", "best_val_mean_auc", "headline", "gate", "errors")}, indent=2, default=str))
    return 1 if any(not k.endswith("_traceback") for k in record["errors"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
