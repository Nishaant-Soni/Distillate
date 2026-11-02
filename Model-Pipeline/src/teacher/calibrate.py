"""Weighted per-label Platt scaling on the teacher's ``logit_<label>`` columns.

``p_calibrated = sigmoid(slope * logit + intercept)``, one (slope, intercept) per label,
fitted by weighted logistic regression with the pilot inclusion weights, so the fit targets
the training population rather than the enriched pilot.

The input is always ``logit_<label>`` (log P(yes) - log P(no)), never ``logit(p)``: float32
``p`` is exactly 1.0 for confident answers, where ``logit(p)`` is infinite and every
saturated row would collapse to one value. ``apply_calibration`` refuses frames without the
logit columns.

* ``cross_fit``: five-fold out-of-fold calibrated probabilities, for an honest ECE after
  calibration (each row is calibrated by a fit that did not see it).
* ``fit_calibration``: one fit on all pilot rows, saved as ``calibration.json``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.teacher.config import LABELS

INPUT = "logit"   # recorded in calibration.json; the only input apply_calibration accepts
RIDGE = 1e-6      # keeps the fit finite if a label is perfectly separated


def fit_platt(z, y, w) -> tuple[float, float]:
    """Weighted logistic regression of ``y`` on ``z``. Returns ``(slope, intercept)``."""
    import numpy as np
    from scipy.optimize import minimize

    z, y, w = (np.asarray(a, dtype="float64") for a in (z, y, w))
    if not (np.isfinite(z).all() and np.isfinite(w).all()):
        raise ValueError("logits and weights must be finite")
    if len(np.unique(y)) < 2:
        raise ValueError("Platt scaling needs both classes")
    w = w / w.mean()   # same optimum; keeps the loss on a stable scale

    def loss(theta):
        a, b = theta
        s = a * z + b
        # -[y log sigmoid(s) + (1 - y) log(1 - sigmoid(s))], computed without overflow
        nll = np.sum(w * (np.logaddexp(0.0, s) - y * s)) / len(z)
        r = 1.0 / (1.0 + np.exp(-s)) - y
        grad = np.array([np.sum(w * r * z), np.sum(w * r)]) / len(z)
        return nll + RIDGE * a * a, grad + np.array([2 * RIDGE * a, 0.0])

    result = minimize(loss, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Platt fit did not converge: {result.message}")
    return float(result.x[0]), float(result.x[1])


def platt(z, slope: float, intercept: float):
    """``sigmoid(slope * z + intercept)`` as float64."""
    import numpy as np
    from scipy.special import expit

    return expit(slope * np.asarray(z, dtype="float64") + intercept)


def _labeled_rows(pred, truth, weights):
    """``pred`` joined to gold labels and weights, ``ok`` rows only."""
    pred = pred.drop(columns=[c for c in LABELS if c in pred])
    frame = truth[["id", *LABELS]].merge(pred, on="id", how="inner", validate="one_to_one")
    if len(frame) != len(truth):
        raise ValueError(f"{len(truth) - len(frame)} truth ids have no prediction")
    frame = frame.merge(weights[["id", "weight"]], on="id", how="left", validate="one_to_one")
    if frame["weight"].isna().any():
        raise ValueError("some ids have no weight")
    missing = [f"logit_{label}" for label in LABELS if f"logit_{label}" not in frame]
    if missing:
        raise ValueError(f"calibration needs the logit columns, missing {missing}")
    if "status" in frame:
        frame = frame[frame["status"] == "ok"]
    return frame.reset_index(drop=True)


def cross_fit(pred, truth, weights, folds: int = 5, seed: int = 42):
    """Out-of-fold calibrated ``p_<label>`` for every ``pred`` row (NaN for failed rows).

    Folds are stratified per label, so every fold has positives of the rare labels.
    The result has ``id``, ``p_<label>``, and ``status``, ready for ``metrics.evaluate``.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    rows = _labeled_rows(pred, truth, weights)
    out = pred[["id"]].copy()
    out["status"] = pred["status"] if "status" in pred else "ok"
    position = {i: k for k, i in enumerate(out["id"])}
    where = np.array([position[i] for i in rows["id"]])
    for label in LABELS:
        z, y, w = rows[f"logit_{label}"], rows[label].to_numpy(), rows["weight"]
        calibrated = np.full(len(rows), np.nan)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fit_idx, test_idx in splitter.split(np.zeros(len(y)), y):
            slope, intercept = fit_platt(z.iloc[fit_idx], y[fit_idx], w.iloc[fit_idx])
            calibrated[test_idx] = platt(z.iloc[test_idx], slope, intercept)
        column = np.full(len(out), np.nan)
        column[where] = calibrated
        out[f"p_{label}"] = column.astype("float32")
    return out


def fit_calibration(pred, truth, weights, version: str, prompt_version: str,
                    model_version: str) -> dict[str, Any]:
    """One Platt fit per label on all ``ok`` rows; the content of ``calibration.json``."""
    rows = _labeled_rows(pred, truth, weights)
    labels = {}
    for label in LABELS:
        slope, intercept = fit_platt(rows[f"logit_{label}"], rows[label], rows["weight"])
        labels[label] = {"slope": slope, "intercept": intercept,
                         "n": len(rows), "n_pos": int(rows[label].sum())}
    return {
        "calibration_version": version,
        "method": "weighted Platt scaling",
        "input": INPUT,
        "formula": "p = sigmoid(slope * logit_<label> + intercept)",
        "prompt_version": prompt_version,
        "model_version": model_version,
        "labels": labels,
    }


def apply_calibration(pred, calibration: Mapping[str, Any]):
    """Replace ``p_<label>`` with calibrated probabilities computed from ``logit_<label>``.

    Raises if the frame has no logit columns (``p``-only frames cannot be calibrated
    correctly) or if its prompt or model version differs from the calibration's.
    """
    if calibration.get("input") != INPUT:
        raise ValueError(f"calibration input must be {INPUT!r}, got {calibration.get('input')!r}")
    missing = [f"logit_{label}" for label in LABELS if f"logit_{label}" not in pred]
    if missing:
        raise ValueError(f"calibration applies to logit_<label> columns, missing {missing}")
    for key in ("prompt_version", "model_version"):
        if key in pred and (pred[key] != calibration[key]).any():
            raise ValueError(f"{key} of the predictions differs from the calibration's")
    out = pred.copy()
    for label in LABELS:
        c = calibration["labels"][label]
        out[f"p_{label}"] = platt(pred[f"logit_{label}"], c["slope"],
                                  c["intercept"]).astype("float32")
    return out


def save_calibration(calibration: Mapping[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(calibration, indent=2) + "\n")


def load_calibration(path: str | Path) -> dict[str, Any]:
    calibration = json.loads(Path(path).read_text())
    if calibration.get("input") != INPUT or set(calibration["labels"]) != set(LABELS):
        raise ValueError(f"{path} is not a logit-based calibration for {list(LABELS)}")
    return calibration
