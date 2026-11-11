"""Run the sensitivity analysis on the selected student and write ``reports/sensitivity.md``.

    python -m src.sensitivity.run --model-dir outputs/student/<run>/model
    python -m src.sensitivity.run --model-dir <dir> --rev <commit>   # validation at a DVC version
    python -m src.sensitivity.run --model-dir <dir> --sample 2000 --no-mlflow   # a quick look

Scores validation once at the model's own length, then runs the four analyses (Optuna
importance only when the study is configured) and writes:

* ``reports/sensitivity/results.json`` and the plots next to it, then ``reports/sensitivity.md``
  from them, with the findings file included: numbers, words and masked lines, no comment text;
* ``reports/summaries/sensitivity.json``: the stage summary;
* ``outputs/sensitivity/shap_selection.parquet``: which comments SHAP explained, by id
  (gitignored);
* an MLflow run with the parameters, headline metrics and every file above.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.sensitivity import plots, report
from src.sensitivity.attributions import shap_analysis
from src.sensitivity.obfuscation import substitution_table
from src.sensitivity.scorer import Scorer, load_scorer
from src.sensitivity.sweeps import (
    check_frame,
    load_thresholds,
    obfuscation_sweep,
    truncation_sweep,
)
from src.teacher.config import LABELS, PIPELINE_ROOT, get_logger, load_config

logger = get_logger(__name__)

DEFAULT_CONFIG = PIPELINE_ROOT / "configs" / "sensitivity.yaml"


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows as JSON-safe dicts (NaN becomes null)."""
    return json.loads(frame.to_json(orient="records"))


def prepare_validation(frame: pd.DataFrame, sample: int | None, seed: int) -> pd.DataFrame:
    """``id``, ``text`` and the six labels; a seeded sample of ``sample`` rows if set."""
    frame = frame[["id", "text", *LABELS]].copy()
    frame["id"] = frame["id"].astype(str)
    if sample and sample < len(frame):
        frame = frame.sample(n=sample, random_state=seed)
    return frame.sort_values("id", ignore_index=True)


def load_validation(cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validation and its lineage: through ``load_data`` (at ``data.rev``, or the files on
    disk), or from ``data.val_file`` when set. The sha256 is the validation file's either way."""
    data, seed = cfg["data"], int(cfg["seed"])
    if data.get("val_file"):
        content = Path(data["val_file"]).read_bytes()
        frame = pd.read_parquet(io.BytesIO(content))
        lineage = {"data_source": "file", "data_rev": None,
                   "data_sha256": hashlib.sha256(content).hexdigest()}
    else:
        from src.data.load_data import TABLES, load_data

        loaded = load_data(rev=data.get("rev"), data_dir=data.get("dir"), tables=("val",),
                           with_teacher=False)
        frame = loaded["val"]
        lineage = {"data_source": "load_data", "data_rev": loaded.rev,
                   "data_sha256": loaded.file_hashes[TABLES["val"]]}
    return prepare_validation(frame, data.get("sample"), seed), lineage


def run(cfg: Mapping[str, Any], scorer: Scorer | None = None,
        frame: pd.DataFrame | None = None, code_commit: str = "unknown") -> dict[str, Any]:
    """The whole analysis. ``scorer`` and ``frame`` default to the configured model and file."""
    started = time.time()
    paths, seed = cfg["paths"], int(cfg["seed"])
    model_dir = cfg["model"]["dir"]
    if scorer is None:
        if not model_dir:
            raise ValueError("set model.dir (or --model-dir) to the selected student's folder")
        scorer = load_scorer(model_dir, cfg["model"]["batch_size"], cfg["model"]["device"],
                             cfg["model"]["fp16"], cfg["model"].get("max_length"))
    if frame is None:
        frame, lineage = load_validation(cfg)
    else:
        lineage = {"data_source": "frame", "data_rev": None, "data_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(frame).to_numpy()).hexdigest()}
    check_frame(frame)
    thresholds = load_thresholds(model_dir, cfg["model"]["threshold"])
    has_file = model_dir is not None and (Path(model_dir) / "thresholds.json").exists()
    bins = int(cfg["metrics"]["ece_bins"])
    texts = frame["text"].fillna("").astype(str).tolist()
    logger.info("scoring %d validation rows with %s", len(frame), scorer.model_version)

    clean = scorer.probabilities(texts)
    truncation = truncation_sweep(scorer, frame, cfg["truncation"]["max_lengths"],
                                  cfg["truncation"]["band_edges"], thresholds, bins)
    obfuscation = obfuscation_sweep(scorer, frame, cfg["obfuscation"]["rates"],
                                    substitution_table(paths["data_pipeline"]), seed,
                                    thresholds, bins, clean=clean)
    shap = shap_analysis(scorer, frame, clean, thresholds, cfg["shap"], seed)

    results: dict[str, Any] = {
        "setup": {
            "model_version": scorer.model_version, "max_length": scorer.max_length,
            "thresholds_source": "the model's thresholds.json" if has_file
            else f"{cfg['model']['threshold']} for every label (no thresholds.json)",
            "thresholds": thresholds, "rows": len(frame),
            "positives": {label: int(frame[label].sum()) for label in LABELS},
            **lineage, "code_commit": code_commit, "seed": seed,
        },
        "truncation": {k: records(v) for k, v in truncation.items()},
        "obfuscation": records(obfuscation),
        "shap": {"config": dict(cfg["shap"]), "explained": int(shap["selection"]["id"].nunique()),
                 "counts": records(shap["selection"].groupby(["label", "group"]).size()
                                   .rename("n").reset_index()),
                 "tokens": records(shap["tokens"]), "examples": records(shap["examples"])},
        "hyperparameters": None, "skipped": {},
    }
    optuna_cfg = cfg["optuna"]
    if optuna_cfg.get("storage") and optuna_cfg.get("study_name"):
        from src.sensitivity.hyperparameters import hyperparameter_analysis

        hyper = hyperparameter_analysis(optuna_cfg, seed)
        results["hyperparameters"] = {
            "summary": hyper["summary"],
            "importances": records(hyper["importances"].reset_index()),
            "trials": records(hyper["trials"]),
            "stability": None if hyper["stability"] is None else {
                "seeds": hyper["stability"]["seeds"], "top": hyper["stability"]["top"],
                "ranges": records(hyper["stability"]["ranges"].reset_index())}}
    else:
        results["skipped"]["hyperparameters"] = ("optuna.storage and optuna.study_name are not "
                                                 "set (the shared study isn't available yet)")

    files = write_outputs(cfg, results, shap["selection"])
    summary = {
        "task": "sensitivity", "rows_in": len(frame), "model_version": scorer.model_version,
        **lineage, "code_commit": code_commit,
        "headline": headline(truncation["overall"], obfuscation),
        "skipped": results["skipped"], "outputs": {k: str(v) for k, v in files.items()},
        "duration_s": round(time.time() - started, 1),
    }
    if cfg["mlflow"]["enabled"]:
        summary["mlflow_run_id"] = log_to_mlflow(cfg, results, summary, files)
    summary_path = Path(paths["summary"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def headline(overall: pd.DataFrame, obfuscation: pd.DataFrame) -> dict[str, float]:
    """Mean-over-labels AUC per max_length and per rate, and missed per rate (NaN left out)."""
    out = {}
    for row in overall[overall["label"] == "mean"].itertuples():
        out[f"truncation_{row.max_length}_mean_auc"] = float(row.auc)
    for row in obfuscation[obfuscation["label"] == "mean"].itertuples():
        out[f"obfuscation_{row.rate:g}_mean_auc"] = float(row.auc)
        out[f"obfuscation_{row.rate:g}_missed"] = float(row.missed)
    return {k: round(v, 6) for k, v in out.items() if v == v}


def write_outputs(cfg: Mapping[str, Any], results: Mapping[str, Any],
                  selection: pd.DataFrame) -> dict[str, Path]:
    """results.json, the plots and the report; the SHAP selection to the scratch folder."""
    paths = cfg["paths"]
    report_dir, report_path = Path(paths["report_dir"]), Path(paths["report"])
    report_dir.mkdir(parents=True, exist_ok=True)
    files = {"results": report_dir / "results.json"}
    files["results"].write_text(json.dumps(results, indent=2) + "\n")

    trunc = {k: pd.DataFrame(v) for k, v in results["truncation"].items()}
    tokens = pd.DataFrame(results["shap"]["tokens"],
                          columns=["label", "group", "word", "mean_value"])
    pngs = {"truncation": plots.truncation_plot(trunc["overall"], report_dir / "truncation.png"),
            "bands": plots.bands_plot(trunc["bands"], report_dir / "length_bands.png"),
            "obfuscation": plots.obfuscation_plot(pd.DataFrame(results["obfuscation"]),
                                                  report_dir / "obfuscation.png"),
            "words_caught": plots.words_plot(tokens, "caught", report_dir / "words_caught.png")}
    hyper = results["hyperparameters"]
    if hyper is not None:
        importances = pd.DataFrame(hyper["importances"]).set_index("param")
        pngs["importance"] = plots.importance_plot(importances, report_dir / "importance.png")
        pngs["trials"] = plots.trials_plot(pd.DataFrame(hyper["trials"]),
                                           list(importances.index), report_dir / "trials.png",
                                           hyper["summary"].get("log_params", ()))
    files |= {f"plot_{k}": v for k, v in pngs.items()}

    findings_path = Path(paths["findings"])
    findings = findings_path.read_text() if findings_path.exists() else None
    links = {k: report.relative(v, report_path.parent) for k, v in pngs.items()}
    report_path.write_text(report.render(results, links, findings))
    files["report"] = report_path

    out_dir = Path(paths["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    files["shap_selection"] = out_dir / "shap_selection.parquet"
    selection.to_parquet(files["shap_selection"], index=False)
    logger.info("wrote %s and %d plots", report_path, len(pngs))
    return files


def log_to_mlflow(cfg: Mapping[str, Any], results: Mapping[str, Any],
                  summary: Mapping[str, Any], files: Mapping[str, Path]) -> str:
    """One MLflow run: what was analysed, the headline metrics, and every output file.

    Same store as the student's training: ``MLFLOW_TRACKING_URI`` if set, else local SQLite.
    """
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
    mlflow.set_experiment(name)
    setup = results["setup"]
    with mlflow.start_run(run_name=f"sensitivity-{setup['model_version']}") as active:
        mlflow.set_tags({"model_version": setup["model_version"],
                         "code_commit": setup["code_commit"]})
        mlflow.log_params({
            "max_length": setup["max_length"], "rows": setup["rows"],
            "data_sha256": setup["data_sha256"], "data_source": setup["data_source"],
            "data_rev": setup["data_rev"] or "on disk", "seed": setup["seed"],
            "thresholds_source": setup["thresholds_source"],
            "truncation_max_lengths": cfg["truncation"]["max_lengths"],
            "obfuscation_rates": cfg["obfuscation"]["rates"],
            "shap_per_group": cfg["shap"]["per_group"], "shap_max_evals": cfg["shap"]["max_evals"],
            "optuna_study": cfg["optuna"].get("study_name") or "none",
        })
        mlflow.log_metrics(summary["headline"])
        for key, path in files.items():
            if key != "shap_selection":                  # ids stay in the scratch folder
                mlflow.log_artifact(str(path), "sensitivity")
        return active.info.run_id


def main() -> None:
    from src.teacher.soft_labels import git_state

    parser = argparse.ArgumentParser(description="Sensitivity analysis of the selected student.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-dir", help="the student's folder (overrides model.dir)")
    parser.add_argument("--rev", help="Git revision: validation as DVC versioned it there")
    parser.add_argument("--val", help="a validation Parquet to read instead of load_data")
    parser.add_argument("--sample", type=int, help="score only N validation rows")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.model_dir:
        cfg["model"]["dir"] = args.model_dir
    if args.rev:
        cfg["data"]["rev"] = args.rev
    if args.val:
        cfg["data"]["val_file"] = args.val
    if args.sample:
        cfg["data"]["sample"] = args.sample
    if args.no_mlflow:
        cfg["mlflow"]["enabled"] = False
    commit, dirty = git_state(["src", "configs"])
    summary = run(cfg, code_commit=commit + ("-dirty" if dirty else ""))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
