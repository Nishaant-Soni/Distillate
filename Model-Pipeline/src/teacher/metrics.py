"""Weighted pilot metrics: ROC AUC, PR-AUC (average precision), ECE, and Brier score.

Weights are the pilot's inclusion weights, so every metric estimates its value on the full
training population rather than on the enriched sample. Rows with ``status == "failed"``
are left out of the metrics and counted in ``failure_rate`` instead.

AUC and PR-AUC only depend on the ranking. When a prediction file has ``logit_<label>``
(teacher runs), they rank on it, because float32 ``p`` ties at exactly 1.0 for confident
answers; ECE and Brier always use ``p``.
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


def label_metrics(y, p, w, bins: int = 10, score=None) -> dict[str, float]:
    """AUC, PR-AUC, ECE, and Brier for one label. AUC and PR-AUC are NaN if only one class.

    ``score`` (default ``p``) is what AUC and PR-AUC rank on; it must order rows like ``p``.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    y, p, w = (np.asarray(a, dtype="float64") for a in (y, p, w))
    s = p if score is None else np.asarray(score, dtype="float64")
    one_class = len(np.unique(y)) < 2
    return {
        "auc": float("nan") if one_class else float(roc_auc_score(y, s, sample_weight=w)),
        "pr_auc": float("nan") if one_class else float(average_precision_score(y, s,
                                                                               sample_weight=w)),
        "ece": weighted_ece(y, p, w, bins),
        "brier": float(np.average((p - y) ** 2, weights=w)),
    }


def evaluate(pred, truth, weights=None, bins: int = 10):
    """Per-label metrics for a prediction file against gold labels.

    ``pred``: ``id``, ``p_<label>``, optionally ``logit_<label>``, ``status``, ``cov_<label>``.
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
        score = sub[f"logit_{label}"] if f"logit_{label}" in sub else None
        metrics = label_metrics(sub[label], sub[f"p_{label}"], sub["weight"], bins, score)
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


def paired_bootstrap(preds, truth, weights, pairs, strata=None, n_boot: int = 1000,
                     seed: int = 42) -> dict[str, dict[str, float]]:
    """Bootstrap intervals for differences in mean AUC and PR-AUC between models.

    ``preds``: prediction frames keyed by model name, all scoring the ``truth`` ids.
    ``pairs``: ``(a, b)`` names; each result is ``a - b``. Rows are resampled with
    replacement within ``strata`` (a frame with ``id`` and ``stratum``, e.g. the pilot's), so
    every replicate keeps the sample design; the inclusion weights ride along. Only rows
    ``ok`` in every model are used, and only labels with both classes in the sample.
    Means are over those labels, so pairs are compared on identical rows and labels.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    frame = truth[["id", *LABELS]].merge(weights[["id", "weight"]], on="id",
                                         validate="one_to_one")
    ok = np.ones(len(frame), dtype=bool)
    scores: dict[str, dict[str, Any]] = {}
    for name, pred in preds.items():
        joined = frame[["id"]].merge(pred, on="id", how="left", validate="one_to_one")
        if "status" in joined:
            ok &= (joined["status"] == "ok").to_numpy()
        scores[name] = {label: joined[f"logit_{label}" if f"logit_{label}" in joined
                                      else f"p_{label}"].to_numpy(dtype="float64")
                        for label in LABELS}
    labels = [label for label in LABELS if len(np.unique(frame.loc[ok, label])) == 2]
    y = {label: frame[label].to_numpy()[ok] for label in labels}
    w = frame["weight"].to_numpy()[ok]
    s = {name: {label: v[label][ok] for label in labels} for name, v in scores.items()}
    groups = [np.arange(int(ok.sum()))]
    if strata is not None:
        stratum = frame[["id"]].merge(strata[["id", "stratum"]], on="id", how="left")
        stratum = stratum["stratum"].to_numpy()[ok]
        groups = [np.flatnonzero(stratum == g) for g in np.unique(stratum)]

    def means(idx, name):
        auc, pr = [], []
        for label in labels:
            yl, sl, wl = y[label][idx], s[name][label][idx], w[idx]
            if len(np.unique(yl)) < 2:
                return np.nan, np.nan
            auc.append(roc_auc_score(yl, sl, sample_weight=wl))
            pr.append(average_precision_score(yl, sl, sample_weight=wl))
        return float(np.mean(auc)), float(np.mean(pr))

    rng = np.random.default_rng(seed)
    full = np.concatenate(groups)
    draws: dict[tuple[str, str], list[tuple[float, float]]] = {pair: [] for pair in pairs}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g)) for g in groups])
        m = {name: means(idx, name) for name in preds}
        for a, b in pairs:
            draws[(a, b)].append((m[a][0] - m[b][0], m[a][1] - m[b][1]))
    out = {}
    for a, b in pairs:
        d = np.array(draws[(a, b)])
        d = d[~np.isnan(d).any(axis=1)]
        point_a, point_b = means(full, a), means(full, b)
        result: dict[str, Any] = {"labels": labels, "rows": int(ok.sum()), "replicates": len(d)}
        for k, metric in enumerate(("mean_auc", "mean_pr_auc")):
            result[f"{metric}_diff"] = round(point_a[k] - point_b[k], 5)
            result[f"{metric}_ci95"] = [round(float(np.percentile(d[:, k], 2.5)), 5),
                                        round(float(np.percentile(d[:, k], 97.5)), 5)]
            result[f"{metric}_p_gt_0"] = round(float(np.mean(d[:, k] > 0)), 4)
        out[f"{a} - {b}"] = result
    return out
