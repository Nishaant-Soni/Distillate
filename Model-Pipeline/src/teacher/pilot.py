"""Pilot evaluation: per-run metrics, calibration, and the memo's comparison table.

The GPU notebook (``kaggle/02_pilot.ipynb``) only scores; everything here runs on CPU, so
the results can be recomputed locally from the saved ``pilot_<prompt_version>.parquet``.

Decision rule: the best zero-shot configuration's mean AUC against TF-IDF's.
If it is lower, the LoRA contingency runs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.teacher.config import LABELS
from src.teacher.prompts import FewShotExample


def fewshot_examples(train, fewshot_ids: Mapping[str, Sequence[str]]) -> list[FewShotExample]:
    """Few-shot examples from ``fewshot_ids.json``'s ``ids``, in ``LABELS`` order."""
    rows = train.set_index("id")
    ids = [i for label in LABELS for i in fewshot_ids[label]]
    missing = [i for i in ids if i not in rows.index]
    if missing:
        raise ValueError(f"few-shot ids not in train: {missing}")
    return [FewShotExample(rows.at[i, "text"], {label: int(rows.at[i, label]) for label in LABELS})
            for i in ids]


def pilot_truth(train, pilot_ids):
    """Gold labels of the pilot rows (``id`` and the six labels)."""
    truth = train.loc[train["id"].isin(set(pilot_ids["id"])), ["id", *LABELS]]
    if len(truth) != len(pilot_ids):
        raise ValueError(f"{len(pilot_ids) - len(truth)} pilot ids are not in train")
    return truth.reset_index(drop=True)


def saturation(pred) -> dict[str, Any]:
    """Share of ``ok`` cells whose float32 ``p`` is exactly 1.0 or 0.0."""
    ok = pred[pred["status"] == "ok"]
    p = ok[[f"p_{label}" for label in LABELS]].to_numpy()
    return {
        "share_p_eq_1": round(float((p == 1.0).mean()), 4),
        "share_p_eq_0": round(float((p == 0.0).mean()), 4),
        "share_p_eq_1_per_label": {label: round(float(v), 4)
                                   for label, v in zip(LABELS, (p == 1.0).mean(axis=0),
                                                       strict=True)},
    }


def summarize_run(pred, truth, weights, cfg: Mapping[str, Any], prompt_version: str,
                  model_version: str) -> dict[str, Any]:
    """Raw weighted metrics, cross-fitted calibrated metrics, and the full-pilot calibration."""
    from src.teacher.calibrate import cross_fit, fit_calibration
    from src.teacher.metrics import evaluate

    bins, c = cfg["metrics"]["ece_bins"], cfg["calibration"]
    calibrated = cross_fit(pred, truth, weights, folds=c["folds"], seed=cfg["seed"])
    return {
        "prompt_version": prompt_version,
        "raw": evaluate(pred, truth, weights, bins),
        "calibrated": evaluate(calibrated, truth, weights, bins),
        "calibration": fit_calibration(pred, truth, weights, c["version"], prompt_version,
                                       model_version),
        "saturation": saturation(pred),
    }


def results_table(runs: Mapping[str, Mapping[str, Any]], tfidf, tfidf_name: str,
                  rows_per_s: Mapping[str, float] | None = None):
    """One row per configuration plus TF-IDF: the memo table.

    ``runs``: ``summarize_run`` results keyed by prompt version. ``tfidf``: TF-IDF's
    ``evaluate`` table. Calibrated ECE and Brier are cross-fitted; TF-IDF is not calibrated.
    """
    import math

    import pandas as pd

    def row(raw, calibrated=None, speed=None):
        out = {f"auc_{label}": raw.loc[label, "auc"] for label in LABELS}
        out.update({f"pr_auc_{label}": raw.loc[label, "pr_auc"] for label in LABELS})
        out.update({
            "mean_auc": raw.loc["mean", "auc"],
            "mean_pr_auc": raw.loc["mean", "pr_auc"],
            "ece_raw": raw.loc["mean", "ece"],
            "ece_calibrated": math.nan if calibrated is None else calibrated.loc["mean", "ece"],
            "brier_raw": raw.loc["mean", "brier"],
            "brier_calibrated": (math.nan if calibrated is None
                                 else calibrated.loc["mean", "brier"]),
            "failure_rate": raw.loc["mean", "failure_rate"],
            "coverage": raw.loc["mean", "coverage"] if "coverage" in raw else math.nan,
            "rows_per_s": math.nan if speed is None else speed,
        })
        return out

    speeds = rows_per_s or {}
    table = {name: row(r["raw"], r["calibrated"], speeds.get(name)) for name, r in runs.items()}
    table[tfidf_name] = row(tfidf)
    return pd.DataFrame.from_dict(table, orient="index")


def decision(table, tfidf_name: str) -> dict[str, Any]:
    """The memo rule: best zero-shot mean AUC compared with TF-IDF's."""
    zero_shot = [name for name in table.index if "-zs-" in name]
    if not zero_shot:
        raise ValueError("no zero-shot run in the table")
    best = max(zero_shot, key=lambda name: table.loc[name, "mean_auc"])
    best_auc, tfidf_auc = float(table.loc[best, "mean_auc"]), float(table.loc[tfidf_name,
                                                                              "mean_auc"])
    return {
        "best_zero_shot": best,
        "best_zero_shot_mean_auc": round(best_auc, 4),
        "tfidf_mean_auc": round(tfidf_auc, 4),
        "lora_contingency": best_auc < tfidf_auc,
        "rule": "LoRA contingency runs if the best zero-shot mean AUC is below TF-IDF's",
    }


def save_run(out_dir: str | Path, pred, summary: Mapping[str, Any],
             info: Mapping[str, Any]) -> None:
    """``pilot_<pv>.parquet``, ``metrics_<pv>.json``, and ``calibration_<pv>.json``."""
    from src.teacher.calibrate import save_calibration

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pv = summary["prompt_version"]
    pred.to_parquet(out_dir / f"pilot_{pv}.parquet", index=False)
    metrics = {
        "prompt_version": pv,
        "info": dict(info),
        "saturation": summary["saturation"],
        "raw": json.loads(summary["raw"].round(5).to_json(orient="index")),
        "calibrated_cross_fitted": json.loads(summary["calibrated"].round(5)
                                              .to_json(orient="index")),
    }
    (out_dir / f"metrics_{pv}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_calibration(summary["calibration"], out_dir / f"calibration_{pv}.json")
