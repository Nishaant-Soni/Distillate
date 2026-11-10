"""Per-label decision thresholds, chosen on validation (v4 §5: "Thresholds: per label, on
validation; saved as thresholds.json").

Each label's threshold maximizes F1 on the validation prediction file against the gold
labels, searched over the distinct predicted scores inside ``[min, max]`` and the two bounds
(so a rare label can't end at an extreme, and a best cut above ``max`` is capped at ``max``).
Ties go to the higher threshold (fewer false positives). Thresholds are stored unrounded.

Two files:

* ``thresholds.json``: ``{"toxicity": 0.41, ...}`` and nothing else, the format the gate
  (``src/eval/metrics.load_thresholds``), the sensitivity analysis, and serving (§2.4) read;
* ``thresholds_report.json``: the method and, per label, F1, precision, and recall at the
  chosen threshold and at 0.5, with the number of positives.

    python -m src.student.thresholds --pred <run>/predictions/val.parquet \
        --gold <val parquet with id and labels> --out <run>/model
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.teacher.config import LABELS, get_logger

logger = get_logger(__name__)

DEFAULTS = {"method": "max_f1", "min": 0.01, "max": 0.99}


def _counts(y, p, threshold: float) -> dict[str, float]:
    import numpy as np

    flag = p >= threshold
    tp = int(np.sum(flag & (y == 1)))
    fp = int(np.sum(flag & (y == 0)))
    fn = int(np.sum(~flag & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall,
            "flag_rate": float(flag.mean()) if len(flag) else 0.0}


def best_f1_threshold(y, p, low: float = 0.01, high: float = 0.99) -> float:
    """The cut-off ``t`` in ``[low, high]`` with the highest F1 (``p >= t`` flags).

    Candidates are every distinct score inside the bounds plus the bounds themselves, so a
    label whose best cut lies above ``high`` gets ``high`` (flagging every score at or above
    it), not a lower score. The value returned is the exact candidate, unrounded: rounding
    could move it past the row it was chosen at.
    """
    import numpy as np

    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype="float64")
    if y.sum() == 0:
        return float(high)               # no positives: flag as little as possible
    order = np.argsort(p, kind="stable")
    p_asc, y_asc = p[order], y[order]
    positives_from = np.r_[np.cumsum(y_asc[::-1])[::-1], 0]    # positives in p_asc[i:]
    inside = np.unique(p[(p >= low) & (p <= high)])
    cuts = np.unique(np.r_[inside, float(low), float(high)])
    start = np.searchsorted(p_asc, cuts, side="left")          # first row flagged by each cut
    tp = positives_from[start]
    flagged = len(p) - start
    f1 = 2 * tp / (flagged + y.sum())
    best = np.flatnonzero(f1 == f1.max())
    return float(cuts[best].max())       # ties: the higher threshold


def choose(pred, gold, settings: Mapping[str, Any] | None = None) -> tuple[dict, dict]:
    """``(thresholds, report)`` from a prediction file and gold labels joined by id."""
    s = {**DEFAULTS, **(settings or {})}
    if s["method"] != "max_f1":
        raise ValueError(f"thresholds.method {s['method']!r}: only max_f1 is implemented")
    merged = pred[["id", *[f"p_{k}" for k in LABELS]]].merge(
        gold[["id", *LABELS]], on="id", how="inner", validate="one_to_one")
    if len(merged) != len(pred):
        raise ValueError(f"{len(pred) - len(merged)} predictions have no gold label")
    thresholds, per_label = {}, {}
    for label in LABELS:
        y, p = merged[label].to_numpy(), merged[f"p_{label}"].to_numpy()
        t = best_f1_threshold(y, p, float(s["min"]), float(s["max"]))
        thresholds[label] = t            # exact: JSON keeps every digit of a float
        per_label[label] = {"threshold": t, "positives": int(y.sum()),
                            "at_threshold": _counts(y, p, t), "at_0.5": _counts(y, p, 0.5)}
    report = {"method": s["method"], "bounds": [s["min"], s["max"]], "rows": len(merged),
              "split": "validation", "labels": per_label,
              "mean_f1": sum(v["at_threshold"]["f1"] for v in per_label.values()) / len(LABELS),
              "mean_f1_at_0.5": sum(v["at_0.5"]["f1"] for v in per_label.values()) / len(LABELS)}
    return thresholds, report


def write(thresholds: Mapping[str, float], report: Mapping[str, Any], folder: str | Path) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "thresholds.json").write_text(json.dumps(dict(thresholds), indent=2) + "\n")
    (folder / "thresholds_report.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description="Per-label thresholds on validation.")
    parser.add_argument("--pred", required=True, help="validation prediction file")
    parser.add_argument("--gold", required=True, help="Parquet with id and the six labels")
    parser.add_argument("--out", required=True, help="folder for thresholds.json")
    args = parser.parse_args()
    gold = pd.read_parquet(args.gold, columns=["id", *LABELS])
    gold["id"] = gold["id"].astype(str)
    pred = pd.read_parquet(args.pred)
    pred["id"] = pred["id"].astype(str)
    thresholds, report = choose(pred, gold)
    write(thresholds, report, args.out)
    logger.info("thresholds: %s (mean F1 %.4f, at 0.5 %.4f)", json.dumps(thresholds),
                report["mean_f1"], report["mean_f1_at_0.5"])


if __name__ == "__main__":
    main()
