"""Weighted pilot metrics: ROC AUC, PR-AUC (average precision), ECE, and Brier score.

Weights are the pilot's inclusion weights, so every metric estimates its value on the full
training population rather than on the enriched sample. Rows with ``status == "failed"``
are left out of the metrics and counted in ``failure_rate`` instead.
"""

from __future__ import annotations

from typing import Any

from src.teacher.config import LABELS


def weighted_ece(y, p, w, bins: int = 10) -> float:
    """Expected calibration error over equal-width bins of ``p`` in [0, 1].

    ``sum_b (W_b / W) * |weighted mean(y in b) - weighted mean(p in b)|``.
    """
    import numpy as np

    y, p, w = (np.asarray(a, dtype="float64") for a in (y, p, w))
    which = np.minimum((p * bins).astype(int), bins - 1)   # p == 1.0 goes in the last bin
    total, ece = w.sum(), 0.0
    for b in range(bins):
        in_bin = which == b
        wb = w[in_bin].sum()
        if wb > 0:
            observed = np.average(y[in_bin], weights=w[in_bin])
            predicted = np.average(p[in_bin], weights=w[in_bin])
            ece += wb / total * abs(observed - predicted)
    return float(ece)


def label_metrics(y, p, w, bins: int = 10) -> dict[str, float]:
    """AUC, PR-AUC, ECE, and Brier for one label. AUC and PR-AUC are NaN if only one class."""
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    y, p, w = (np.asarray(a, dtype="float64") for a in (y, p, w))
    one_class = len(np.unique(y)) < 2
    return {
        "auc": float("nan") if one_class else float(roc_auc_score(y, p, sample_weight=w)),
        "pr_auc": float("nan") if one_class else float(average_precision_score(y, p,
                                                                               sample_weight=w)),
        "ece": weighted_ece(y, p, w, bins),
        "brier": float(np.average((p - y) ** 2, weights=w)),
    }


def evaluate(pred, truth, weights=None, bins: int = 10):
    """Per-label metrics for a prediction file against gold labels.

    ``pred``: ``id``, ``p_<label>``, optionally ``status`` and ``cov_<label>``.
    ``truth``: ``id`` and the six label columns. ``weights``: frame with ``id`` and ``weight``
    (pilot inclusion weights); every row weighs 1 if omitted.

    Returns one row per label plus a ``mean`` row, with ``n``, ``n_pos``, ``failure_rate``,
    ``auc``, ``pr_auc``, ``ece``, ``brier``, and ``coverage`` when ``cov_<label>`` is present.
    """
    import numpy as np
    import pandas as pd

    pred = pred.drop(columns=[c for c in LABELS if c in pred])
    frame = truth[["id", *LABELS]].merge(pred, on="id", how="left", validate="one_to_one",
                                         indicator=True)
    unscored = frame.pop("_merge") != "both"
    if unscored.any():
        raise ValueError(f"{int(unscored.sum())} truth ids have no prediction")
    if weights is not None:
        frame = frame.merge(weights[["id", "weight"]], on="id", how="left", validate="one_to_one")
        if frame["weight"].isna().any():
            raise ValueError("some ids have no weight")
    else:
        frame["weight"] = 1.0

    ok = (frame["status"] == "ok") if "status" in frame else pd.Series(True, index=frame.index)
    rows: dict[str, dict[str, Any]] = {}
    for label in LABELS:
        sub = frame[ok]
        metrics = label_metrics(sub[label], sub[f"p_{label}"], sub["weight"], bins)
        row: dict[str, Any] = {"n": int(ok.sum()), "n_pos": int(sub[label].sum()),
                               "failure_rate": float(1 - ok.mean()), **metrics}
        if f"cov_{label}" in frame:
            row["coverage"] = float(frame.loc[ok, f"cov_{label}"].mean())
        rows[label] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    numeric = [c for c in ("auc", "pr_auc", "ece", "brier", "coverage") if c in table]
    table.loc["mean"] = {**{c: np.nan for c in table.columns},
                         **table[numeric].mean().to_dict(),
                         "n": table["n"].iloc[0], "failure_rate": table["failure_rate"].iloc[0]}
    return table
