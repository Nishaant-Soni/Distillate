"""Promotion rule for retraining candidates: beat @production on the held-out drift batch.

Work plan v5 (6.1): a retrained candidate goes through the same gate as CI, and must also beat
production on the most recent labeled drift batch that the retraining did not use, by mean AUC
over the five Civil labels. retrain_dag runs this after the gate; exit code 0 is pass, 1 fail.

Inputs, all for that one batch:

* ``--candidate``: the candidate's prediction file (CONTRACTS.md section 2.2: ``id``,
  ``p_<label>``, ``model_version``), written by the retraining job.
* ``--production``: production's predictions. Either a section 2.2 file, or the file
  drift_ingest stored when the batch was scored through the live service
  (``reports/drift/predictions/batch_<k>.parquet``: ``score_<label>``, no text). With
  ``--production-model-version``, a file from a different model (production moved since the
  batch was scored) is refused rather than compared.
* ``--gold``: the batch's labels: the drift batch file itself (only ``id`` and the label
  columns are read) or an ``id`` + labels file.

Both models are scored on the same rows (``ok`` in both), ranking on ``p_<label>`` (logit
columns are dropped, since the live service returns only scores). ``--trained-on`` lists the
batches the candidate trained on; the check refuses to run if the held-out batch is among them.

Writes promote_check.json and promote_check.md to ``--out``.

    python -m src.pipeline.promote_check --batch 7 \\
        --candidate <job>/predictions/drift_holdout.parquet \\
        --production ../Data-Pipeline/reports/drift/predictions/batch_07.parquet \\
        --gold ../Data-Pipeline/data/processed/civil/drift/batch_07.parquet \\
        --production-model-version <v> --trained-on 0 1 2 3 4 5 --out <job>/promotion
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.eval.metrics import evaluate, paired_bootstrap, read_frame, to_jsonable
from src.teacher.config import PIPELINE_ROOT

DEFAULT_CONFIG = PIPELINE_ROOT / "configs" / "promotion.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with Path(path or DEFAULT_CONFIG).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def as_predictions(frame: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    """``id``, ``p_<label>``, ``model_version`` (and ``status`` if present), from a section 2.2
    file or a drift_ingest file (``score_<label>``). Logit columns are dropped."""
    out = frame.copy()
    for label in labels:
        if f"p_{label}" not in out and f"score_{label}" in out:
            out[f"p_{label}"] = out[f"score_{label}"]
    missing = [f"p_{label}" for label in labels if f"p_{label}" not in out]
    if missing or "id" not in out or "model_version" not in out:
        raise ValueError(f"not a prediction file: missing {missing or ['id or model_version']}")
    keep = ["id", *[f"p_{label}" for label in labels], "model_version"]
    if "status" in out:
        keep.append("status")
    out = out[keep]
    out["id"] = out["id"].astype(str)
    return out


def as_gold(frame: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    """``id`` and the label columns only; text and everything else is dropped on read."""
    missing = [c for c in ("id", *labels) if c not in frame]
    if missing:
        raise ValueError(f"gold file is missing {missing}")
    gold = frame[["id", *labels]].copy()
    gold["id"] = gold["id"].astype(str)
    return gold


def _model_version(pred: pd.DataFrame) -> str:
    versions = pred["model_version"].astype(str).unique()
    if len(versions) != 1:
        raise ValueError(f"one prediction file holds several model versions: {sorted(versions)}")
    return str(versions[0])


def eligible_labels(gold: pd.DataFrame, labels: Sequence[str],
                    min_positives: int | None) -> tuple[list[str], dict[str, int]]:
    """Labels with both classes in the batch (and at least ``min_positives`` positives)."""
    positives = {label: int(gold[label].sum()) for label in labels}
    floor = max(1, int(min_positives or 1))
    used = [label for label in labels if floor <= positives[label] < len(gold)]
    return used, positives


def compare(candidate: pd.DataFrame, production: pd.DataFrame, gold: pd.DataFrame,
            cfg: dict[str, Any]) -> dict[str, Any]:
    """The rule on one batch: candidate mean AUC minus production's, above ``min_gain``."""
    labels = list(cfg["labels"])
    cand, prod = as_predictions(candidate, labels), as_predictions(production, labels)
    gold = as_gold(gold, labels)
    for name, pred in (("candidate", cand), ("production", prod)):
        extra = set(pred["id"]) - set(gold["id"])
        if extra:
            raise ValueError(f"{name} has {len(extra)} ids that aren't in the batch")
    ok = set(gold["id"])
    for pred in (cand, prod):
        rows = pred if "status" not in pred else pred[pred["status"] == "ok"]
        ok &= set(rows["id"])
    gold = gold[gold["id"].isin(ok)].reset_index(drop=True)
    if gold.empty:
        raise ValueError("no row is scored by both models")

    used, positives = eligible_labels(gold, labels, cfg.get("min_positives"))
    if not used:
        raise ValueError(f"no label has enough positives in the batch: {positives}")
    used_t = tuple(used)
    cand_m = evaluate(cand, gold, used_t)
    prod_m = evaluate(prod, gold, used_t)
    gain = cand_m["mean_auc"] - prod_m["mean_auc"]
    passed = bool(not math.isnan(gain) and gain > cfg["min_gain"])
    boot = None
    if cfg.get("bootstrap"):
        boot = paired_bootstrap(cand, prod, gold, used_t, n_boot=int(cfg["bootstrap"]),
                                seed=int(cfg.get("seed", 42)))["metrics"]["mean_auc"]
    return {
        "passed": passed,
        "rule": f"candidate mean AUC - production mean AUC > {cfg['min_gain']}",
        "gain": gain,
        "candidate": {"model_version": _model_version(cand), "mean_auc": cand_m["mean_auc"],
                      "auc": {k: v["auc"] for k, v in cand_m["per_label"].items()}},
        "production": {"model_version": _model_version(prod), "mean_auc": prod_m["mean_auc"],
                       "auc": {k: v["auc"] for k, v in prod_m["per_label"].items()}},
        "rows": len(gold),
        "positives": positives,
        "labels_used": used,
        "labels_left_out": [label for label in labels if label not in used],
        "min_positives": cfg.get("min_positives"),
        "bootstrap_gain": boot,
    }


def run(cfg: dict[str, Any], batch: int, candidate: str | Path, production: str | Path,
        gold: str | Path, production_model_version: str | None = None,
        trained_on: Sequence[int] = ()) -> dict[str, Any]:
    """Load the three files, check them against the batch, and apply the rule."""
    if batch in set(trained_on):
        raise ValueError(f"batch {batch} was in the candidate's training data; it can't be the "
                         "held-out batch")
    prod = read_frame(production)
    if production_model_version is not None:
        found = _model_version(prod)
        if found != production_model_version:
            raise ValueError(f"the production predictions are from {found}, but @production is "
                             f"{production_model_version}: rescore the batch with @production")
    report = compare(read_frame(candidate), prod, read_frame(gold), cfg)
    return {"batch": batch, "trained_on": sorted(set(trained_on)), **report}


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def render_markdown(report: dict[str, Any]) -> str:
    verdict = "PASSED" if report["passed"] else "FAILED"
    cand, prod = report["candidate"], report["production"]
    lines = [f"# Promotion check: {verdict}", "",
             f"Drift batch {report['batch']} ({report['rows']} rows), held out from training "
             f"(trained on batches {report['trained_on'] or 'none'}).", "",
             f"Rule: {report['rule']}. Gain: **{_fmt(report['gain'])}**.", ""]
    boot = report.get("bootstrap_gain")
    if boot:
        lines += [f"Paired bootstrap ({boot['replicates']} replicates): 95% interval "
                  f"[{_fmt(boot['ci95'][0])}, {_fmt(boot['ci95'][1])}], share above 0 "
                  f"{_fmt(boot['p_gt_0'])}.", ""]
    lines += ["| Label | Positives | Candidate AUC | Production AUC |", "|---|---|---|---|"]
    for label, n in report["positives"].items():
        used = label in report["labels_used"]
        lines.append(f"| {label}{'' if used else ' (left out)'} | {n} "
                     f"| {_fmt(cand['auc'].get(label))} | {_fmt(prod['auc'].get(label))} |")
    lines += [f"| **mean** | | **{_fmt(cand['mean_auc'])}** | **{_fmt(prod['mean_auc'])}** |",
              "", f"Candidate `{cand['model_version']}`; production `{prod['model_version']}`."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--batch", type=int, required=True, help="the held-out drift batch")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--production", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--production-model-version", default=None)
    parser.add_argument("--trained-on", type=int, nargs="*", default=[])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    report = run(load_config(args.config), args.batch, args.candidate, args.production,
                 args.gold, args.production_model_version, args.trained_on)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "promote_check.json").write_text(
        json.dumps(to_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "promote_check.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"Promotion check {'PASSED' if report['passed'] else 'FAILED'}: gain "
          f"{_fmt(report['gain'])} on batch {args.batch}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
