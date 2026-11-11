"""Metrics for any prediction file against gold labels (Ruchita).

Inputs follow CONTRACTS.md section 2.2: a prediction file has ``id``, ``p_<label>`` and
``model_version``, and may have ``logit_<label>`` and ``status``. Extra columns are ignored,
so teacher shards, the TF-IDF baseline and every student go in the same way.

Per label: ROC-AUC, PR-AUC (average precision), F1, precision, recall and false positive
rate at the label's threshold, ECE (10 equal-width bins) and Brier score. Over labels: mean
column-wise ROC-AUC, mean PR-AUC, macro-F1, mean ECE and mean Brier. The flag decision (any
label at or above its threshold) is scored against gold "any label": precision, recall, F1
and false positive rate.

Definitions match src/teacher/metrics.py, so teacher and student numbers are comparable:

* weights (the gate subset's ``weight``) are sample weights in every metric;
* AUC and PR-AUC rank on ``logit_<label>`` when the file has it, because float32 ``p`` ties
  at exactly 1.0 for confident answers; ECE, Brier and the thresholded metrics use ``p``;
* rows whose ``status`` is not ``ok`` are dropped and reported as the failure rate.

Civil Comments metrics use ``CIVIL_LABELS`` (no severe_toxicity: almost no Civil rows reach
0.5 on it). Bootstrap intervals resample within ``stratum`` when the design frame has it.

    python -m src.eval.metrics --pred gate.parquet --gold gate_gold.parquet \\
        --design artifacts/gate_subset_ids.parquet --bootstrap 1000 --out metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

LABELS = ("toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack")
CIVIL_LABELS = tuple(label for label in LABELS if label != "severe_toxicity")
DEFAULT_THRESHOLD = 0.5
ECE_BINS = 10
HEADLINE = ("mean_auc", "mean_pr_auc", "macro_f1",
            "flag_precision", "flag_recall", "flag_f1", "flag_fpr")


# ---------------------------------------------------------------------------
# Loading and alignment
# ---------------------------------------------------------------------------

def read_frame(path: str | Path) -> pd.DataFrame:
    """Read one Parquet file, or every ``*.parquet`` in a folder (e.g. teacher shards)."""
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files in {path}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return pd.read_parquet(path)


def load_thresholds(path: str | Path | None, labels: tuple[str, ...]) -> dict[str, float]:
    """Per-label thresholds from thresholds.json; 0.5 for every label when no file is given.

    Accepts ``{"toxicity": 0.4, ...}`` or the same mapping under a ``thresholds`` key.
    """
    if path is None:
        return {label: DEFAULT_THRESHOLD for label in labels}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data = data.get("thresholds", data)
    missing = [label for label in labels if label not in data]
    if missing:
        raise ValueError(f"{path} has no threshold for {missing}")
    return {label: float(data[label]) for label in labels}


@dataclass
class Aligned:
    """Gold labels, scores and weights for the rows that count, as arrays."""

    labels: tuple[str, ...]
    y: dict[str, np.ndarray]
    p: dict[str, np.ndarray]
    score: dict[str, np.ndarray]
    w: np.ndarray
    strata: np.ndarray | None
    n_total: int
    n_failed: int
    model_version: str | None
    weighted: bool

    @property
    def n(self) -> int:
        return len(self.w)


def align(
    pred: pd.DataFrame,
    gold: pd.DataFrame,
    labels: tuple[str, ...] = LABELS,
    design: pd.DataFrame | None = None,
) -> Aligned:
    """Join a prediction file to gold labels by id, drop non-ok rows, attach weights.

    ``design`` is an optional frame with ``id`` and ``weight`` (and ``stratum`` for the
    bootstrap), such as ``artifacts/gate_subset_ids.parquet``. Every gold id must have a
    prediction and, when a design is given, a weight.
    """
    labels = tuple(labels)
    missing = [f"p_{label}" for label in labels if f"p_{label}" not in pred.columns]
    if missing:
        raise ValueError(f"Prediction file is missing {missing}")
    missing = [label for label in labels if label not in gold.columns]
    if missing:
        raise ValueError(f"Gold labels are missing {missing}")

    keep = ["id", *[c for label in labels for c in (f"p_{label}", f"logit_{label}")
                    if c in pred.columns]]
    if "status" in pred.columns:
        keep.append("status")
    scores = pred[keep].assign(id=pred["id"].astype(str))
    frame = gold[["id", *labels]].assign(id=gold["id"].astype(str)).merge(
        scores, on="id", how="left", validate="one_to_one", indicator=True)
    unscored = int((frame.pop("_merge") != "both").sum())
    if unscored:
        raise ValueError(f"{unscored} of {len(frame)} gold ids have no prediction")

    weighted = design is not None
    if weighted:
        cols = ["id", "weight", *(["stratum"] if "stratum" in design.columns else [])]
        frame = frame.merge(design[cols].assign(id=design["id"].astype(str)), on="id",
                            how="left", validate="one_to_one")
        if frame["weight"].isna().any():
            raise ValueError(f"{int(frame['weight'].isna().sum())} gold ids have no weight")
    else:
        frame["weight"] = 1.0

    ok = (frame["status"] == "ok") if "status" in frame else pd.Series(True, index=frame.index)
    rows = frame[ok.to_numpy()]
    p_cols = [f"p_{label}" for label in labels]
    if rows[p_cols].isna().any().any():
        raise ValueError("Some ok rows have missing p_<label> values")

    model_version = None
    if "model_version" in pred.columns and pred["model_version"].notna().any():
        model_version = str(pred["model_version"].dropna().iloc[0])
    return Aligned(
        labels=labels,
        y={label: rows[label].to_numpy(dtype="int8") for label in labels},
        p={label: rows[f"p_{label}"].to_numpy(dtype="float64") for label in labels},
        score={label: rows[f"logit_{label}" if f"logit_{label}" in rows else f"p_{label}"]
               .to_numpy(dtype="float64") for label in labels},
        w=rows["weight"].to_numpy(dtype="float64"),
        strata=rows["stratum"].astype(str).to_numpy() if "stratum" in rows else None,
        n_total=len(frame),
        n_failed=int((~ok).sum()),
        model_version=model_version,
        weighted=weighted,
    )


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def weighted_ece(y: np.ndarray, p: np.ndarray, w: np.ndarray, bins: int = ECE_BINS) -> float:
    """Expected calibration error over equal-width bins of ``p`` (p == 1.0 in the last bin)."""
    y, p, w = (np.asarray(a, dtype="float64") for a in (y, p, w))
    which = np.minimum((p * bins).astype(int), bins - 1)
    total, ece = w.sum(), 0.0
    for b in range(bins):
        in_bin = which == b
        wb = w[in_bin].sum()
        if wb > 0:
            observed = np.average(y[in_bin], weights=w[in_bin])
            predicted = np.average(p[in_bin], weights=w[in_bin])
            ece += wb / total * abs(observed - predicted)
    return float(ece)


def confusion_rates(y: np.ndarray, decision: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """Weighted precision, recall, F1 and FPR. Undefined precision/recall/F1 count as 0."""
    y, decision = np.asarray(y).astype(bool), np.asarray(decision).astype(bool)
    tp, fp = w[y & decision].sum(), w[~y & decision].sum()
    fn, tn = w[y & ~decision].sum(), w[~y & ~decision].sum()
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    fpr = fp / (fp + tn) if fp + tn > 0 else math.nan
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1),
            "fpr": float(fpr)}


def _rank_metrics(y: np.ndarray, score: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return math.nan, math.nan
    return (float(roc_auc_score(y, score, sample_weight=w)),
            float(average_precision_score(y, score, sample_weight=w)))


def label_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray, threshold: float,
                  score: np.ndarray | None = None) -> dict[str, float]:
    """All per-label metrics. ``score`` (default ``p``) is what AUC and PR-AUC rank on."""
    y, p, w = np.asarray(y), np.asarray(p, dtype="float64"), np.asarray(w, dtype="float64")
    auc, pr_auc = _rank_metrics(y, p if score is None else np.asarray(score), w)
    return {
        "n_pos": int(np.asarray(y).sum()),
        "auc": auc,
        "pr_auc": pr_auc,
        **confusion_rates(y, p >= threshold, w),
        "ece": weighted_ece(y, p, w),
        "brier": float(np.average((p - y) ** 2, weights=w)),
        "threshold": float(threshold),
    }


def _flag(data: Aligned, thresholds: dict[str, float], idx: np.ndarray | None = None):
    sel = slice(None) if idx is None else idx
    gold_any = np.zeros(len(data.w), dtype=bool)[sel]
    flag = np.zeros(len(data.w), dtype=bool)[sel]
    for label in data.labels:
        gold_any |= data.y[label][sel] == 1
        flag |= data.p[label][sel] >= thresholds[label]
    return gold_any, flag


def _headline(data: Aligned, thresholds: dict[str, float], idx: np.ndarray) -> dict[str, float]:
    """The bootstrapped metrics, on rows ``idx``."""
    w = data.w[idx]
    aucs, prs, f1s = [], [], []
    for label in data.labels:
        y = data.y[label][idx]
        auc, pr = _rank_metrics(y, data.score[label][idx], w)
        aucs.append(auc)
        prs.append(pr)
        f1s.append(confusion_rates(y, data.p[label][idx] >= thresholds[label], w)["f1"])
    gold_any, flag = _flag(data, thresholds, idx)
    flag_rates = confusion_rates(gold_any, flag, w)
    return {"mean_auc": _nanmean(aucs), "mean_pr_auc": _nanmean(prs),
            "macro_f1": float(np.mean(f1s)),
            **{f"flag_{k}": v for k, v in flag_rates.items()}}


def _nanmean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return float(np.mean(finite)) if finite else math.nan


def evaluate(
    pred: pd.DataFrame,
    gold: pd.DataFrame,
    labels: tuple[str, ...] = LABELS,
    thresholds: dict[str, float] | None = None,
    design: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Every metric for one prediction file. Means skip labels with only one class."""
    data = align(pred, gold, labels, design)
    thresholds = thresholds or {label: DEFAULT_THRESHOLD for label in labels}
    per_label = {label: label_metrics(data.y[label], data.p[label], data.w,
                                      thresholds[label], data.score[label])
                 for label in data.labels}
    gold_any, flag = _flag(data, thresholds)
    return {
        "model_version": data.model_version,
        "labels": list(data.labels),
        "weighted": data.weighted,
        "n": data.n,
        "n_failed": data.n_failed,
        "failure_rate": data.n_failed / data.n_total if data.n_total else 0.0,
        "mean_auc": _nanmean([m["auc"] for m in per_label.values()]),
        "mean_pr_auc": _nanmean([m["pr_auc"] for m in per_label.values()]),
        "macro_f1": float(np.mean([m["f1"] for m in per_label.values()])),
        "ece": float(np.mean([m["ece"] for m in per_label.values()])),
        "brier": float(np.mean([m["brier"] for m in per_label.values()])),
        "flag": {**confusion_rates(gold_any, flag, data.w),
                 "n_flagged": int(flag.sum()), "n_gold_any": int(gold_any.sum())},
        "per_label": per_label,
    }


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _groups(strata: np.ndarray | None, n: int) -> list[np.ndarray]:
    if strata is None:
        return [np.arange(n)]
    return [np.flatnonzero(strata == g) for g in np.unique(strata)]


def _resample(groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """One bootstrap replicate: rows drawn with replacement within each stratum."""
    return np.concatenate([rng.choice(g, size=len(g)) for g in groups])


def bootstrap_ci(
    pred: pd.DataFrame,
    gold: pd.DataFrame,
    labels: tuple[str, ...] = LABELS,
    thresholds: dict[str, float] | None = None,
    design: pd.DataFrame | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Point estimate and 95% percentile interval for each headline metric."""
    data = align(pred, gold, labels, design)
    thresholds = thresholds or {label: DEFAULT_THRESHOLD for label in labels}
    groups = _groups(data.strata, data.n)
    point = _headline(data, thresholds, np.concatenate(groups))
    rng = np.random.default_rng(seed)
    draws = [_headline(data, thresholds, _resample(groups, rng)) for _ in range(n_boot)]
    out = {}
    for metric in HEADLINE:
        values = np.array([d[metric] for d in draws], dtype="float64")
        values = values[~np.isnan(values)]
        ci = [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] \
            if len(values) else [math.nan, math.nan]
        out[metric] = {"point": point[metric], "ci95": ci, "replicates": int(len(values))}
    return out


def paired_bootstrap(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    gold: pd.DataFrame,
    labels: tuple[str, ...] = LABELS,
    thresholds_a: dict[str, float] | None = None,
    thresholds_b: dict[str, float] | None = None,
    design: pd.DataFrame | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Differences a - b in each headline metric, on the same resampled rows each time.

    Only rows ``ok`` in both files are used, so the two models are compared on identical
    rows. Returns the point difference, its 95% interval and the share of replicates > 0.
    """
    ok_ids = set(gold["id"].astype(str))
    for pred in (pred_a, pred_b):
        if "status" in pred.columns:
            ok_ids &= set(pred.loc[pred["status"] == "ok", "id"].astype(str))
    gold_ok = gold[gold["id"].astype(str).isin(ok_ids)]
    a = align(pred_a, gold_ok, labels, design)
    b = align(pred_b, gold_ok, labels, design)
    ta = thresholds_a or {label: DEFAULT_THRESHOLD for label in labels}
    tb = thresholds_b or {label: DEFAULT_THRESHOLD for label in labels}
    groups = _groups(a.strata, a.n)
    full = np.concatenate(groups)
    point_a, point_b = _headline(a, ta, full), _headline(b, tb, full)
    rng = np.random.default_rng(seed)
    diffs: dict[str, list[float]] = {m: [] for m in HEADLINE}
    for _ in range(n_boot):
        idx = _resample(groups, rng)
        ha, hb = _headline(a, ta, idx), _headline(b, tb, idx)
        for m in HEADLINE:
            diffs[m].append(ha[m] - hb[m])
    out = {}
    for m in HEADLINE:
        d = np.array(diffs[m], dtype="float64")
        d = d[~np.isnan(d)]
        out[m] = {
            "diff": point_a[m] - point_b[m],
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            if len(d) else [math.nan, math.nan],
            "p_gt_0": float(np.mean(d > 0)) if len(d) else math.nan,
            "replicates": int(len(d)),
        }
    return {"rows": a.n, "labels": list(labels), "metrics": out}


# ---------------------------------------------------------------------------
# JSON and CLI
# ---------------------------------------------------------------------------

def to_jsonable(value: Any) -> Any:
    """NaN -> None and numpy scalars -> Python, so the output is valid JSON."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", required=True, help="prediction file, or a folder of shards")
    parser.add_argument("--gold", required=True, help="gold labels (id plus label columns)")
    parser.add_argument("--labels", choices=["wikipedia", "civil"], default="wikipedia",
                        help="civil drops severe_toxicity")
    parser.add_argument("--thresholds", default=None, help="thresholds.json (default 0.5)")
    parser.add_argument("--design", default=None,
                        help="frame with id, weight[, stratum], e.g. gate_subset_ids.parquet")
    parser.add_argument("--bootstrap", type=int, default=0, help="replicates (0 = none)")
    parser.add_argument("--out", default=None, help="write the result as JSON here")
    args = parser.parse_args(argv)

    labels = CIVIL_LABELS if args.labels == "civil" else LABELS
    pred, gold = read_frame(args.pred), read_frame(args.gold)
    design = read_frame(args.design) if args.design else None
    thresholds = load_thresholds(args.thresholds, labels)
    result = evaluate(pred, gold, labels, thresholds, design)
    if args.bootstrap:
        result["bootstrap"] = bootstrap_ci(pred, gold, labels, thresholds, design,
                                           n_boot=args.bootstrap)
    text = json.dumps(to_jsonable(result), indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
