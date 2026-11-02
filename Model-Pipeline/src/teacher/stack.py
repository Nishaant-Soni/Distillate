"""Stacked teacher: per-label logistic regression on the teacher's logit and TF-IDF's score.

``logit_stack = w_teacher * logit_<label> + w_baseline * logit(p_baseline) + intercept`` and
``p = sigmoid(logit_stack)``, one set of weights per label, fitted by weighted logistic
regression with the pilot inclusion weights, so the probabilities target the training
population (the stack is its own calibration).

The teacher input is ``logit_<label>``, never ``logit(p)``, for the same reason as in
``calibrate.py``. The baseline only has ``p_<label>``; it is clipped to
``[BASELINE_CLIP, 1 - BASELINE_CLIP]`` before ``logit``.

The baseline's scores must come from a model that never trained on the rows being scored
(the pilot is held out of TF-IDF's training rows); otherwise they are close to the gold
labels and the stack learns to copy them.

* ``cross_fit_stack``: five-fold out-of-fold stacked predictions, for honest pilot metrics.
* ``fit_stack``: one fit on all pilot rows, saved as ``stack_<prompt_version>.json``.
* ``apply_stack``: the frozen fit applied to new rows (the shift check, later the shards).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.teacher.config import LABELS

BASELINE_CLIP = 1e-7
INPUTS = ["teacher logit_<label>", "baseline logit(p_<label>)"]


def features(frame, label: str):
    """The two stack inputs for ``label`` from a joined frame (``t_logit_*``, ``b_p_*``)."""
    import numpy as np
    from scipy.special import logit

    z_teacher = frame[f"t_logit_{label}"].to_numpy(dtype="float64")
    p_baseline = frame[f"b_p_{label}"].to_numpy(dtype="float64")
    x = np.column_stack([z_teacher, logit(np.clip(p_baseline, BASELINE_CLIP,
                                                  1 - BASELINE_CLIP))])
    if not np.isfinite(x).all():
        raise ValueError(f"non-finite stack inputs for {label}")
    return x


def join(teacher, baseline):
    """Teacher and baseline predictions side by side, one row per teacher row.

    Columns: ``id``, ``status`` (the teacher's), ``t_logit_<label>``, ``b_p_<label>``.
    """
    missing = [f"logit_{label}" for label in LABELS if f"logit_{label}" not in teacher]
    if missing:
        raise ValueError(f"the stack needs the teacher's logit columns, missing {missing}")
    t = teacher[["id", *[f"logit_{label}" for label in LABELS]]].rename(
        columns={f"logit_{label}": f"t_logit_{label}" for label in LABELS})
    t["status"] = teacher["status"].to_numpy() if "status" in teacher else "ok"
    b = baseline[["id", *[f"p_{label}" for label in LABELS]]].rename(
        columns={f"p_{label}": f"b_p_{label}" for label in LABELS})
    frame = t.merge(b, on="id", how="left", validate="one_to_one", indicator=True)
    if (frame.pop("_merge") != "both").any():
        raise ValueError("some teacher ids have no baseline prediction")
    return frame


def _labeled(teacher, baseline, truth, weights):
    """Joined ``ok`` rows with gold labels and weights."""
    frame = join(teacher, baseline)
    frame = frame[frame["status"] == "ok"]
    frame = frame.merge(truth[["id", *LABELS]], on="id", how="inner", validate="one_to_one")
    frame = frame.merge(weights[["id", "weight"]], on="id", how="left", validate="one_to_one")
    if frame["weight"].isna().any():
        raise ValueError("some ids have no weight")
    return frame.reset_index(drop=True)


def _fit(x, y, w, C: float):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(x, y, sample_weight=w / w.mean())
    return model


def _output(ids, status, scores: Mapping[str, Any]):
    """``id``, ``p_<label>``, ``logit_<label>``, ``status`` from per-label stacked logits."""
    import pandas as pd
    from scipy.special import expit

    out = pd.DataFrame({"id": ids})
    for label in LABELS:
        out[f"p_{label}"] = expit(scores[label]).astype("float32")
    for label in LABELS:
        out[f"logit_{label}"] = scores[label].astype("float32")
    out["status"] = status
    return out


def cross_fit_stack(teacher, baseline, truth, weights, folds: int = 5, seed: int = 42,
                    C: float = 1.0):
    """Out-of-fold stacked predictions for every teacher row (NaN for failed rows)."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    rows = _labeled(teacher, baseline, truth, weights)
    frame = join(teacher, baseline)
    position = {i: k for k, i in enumerate(frame["id"])}
    where = np.array([position[i] for i in rows["id"]])
    w = rows["weight"].to_numpy()
    scores = {}
    for label in LABELS:
        x, y = features(rows, label), rows[label].to_numpy()
        oof = np.full(len(rows), np.nan)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fit_idx, test_idx in splitter.split(x, y):
            oof[test_idx] = _fit(x[fit_idx], y[fit_idx], w[fit_idx], C).decision_function(
                x[test_idx])
        column = np.full(len(frame), np.nan)
        column[where] = oof
        scores[label] = column
    return _output(frame["id"].to_numpy(), frame["status"].to_numpy(), scores)


def fit_stack(teacher, baseline, truth, weights, version: str, prompt_version: str,
              model_version: str, baseline_version: str, C: float = 1.0) -> dict[str, Any]:
    """One fit per label on all ``ok`` rows; the content of ``stack_<prompt_version>.json``."""
    rows = _labeled(teacher, baseline, truth, weights)
    w = rows["weight"].to_numpy()
    labels = {}
    for label in LABELS:
        model = _fit(features(rows, label), rows[label].to_numpy(), w, C)
        labels[label] = {"w_teacher": float(model.coef_[0][0]),
                         "w_baseline": float(model.coef_[0][1]),
                         "intercept": float(model.intercept_[0]),
                         "n": len(rows), "n_pos": int(rows[label].sum())}
    return {
        "stack_version": version,
        "method": "weighted logistic regression",
        "inputs": INPUTS,
        "formula": ("p = sigmoid(w_teacher * logit_<label> + w_baseline * "
                    f"logit(clip(p_baseline, {BASELINE_CLIP})) + intercept)"),
        "C": C,
        "prompt_version": prompt_version,
        "model_version": model_version,
        "baseline_version": baseline_version,
        "labels": labels,
    }


def apply_stack(teacher, baseline, stack: Mapping[str, Any]):
    """Stacked ``p_<label>`` and ``logit_<label>`` for every teacher row (NaN if failed).

    Raises if the teacher's prompt or model version, or the baseline's version, differs from
    the stack's.
    """
    import numpy as np

    for frame, key in ((teacher, "prompt_version"), (teacher, "model_version"),
                       (baseline, "baseline_version")):
        column = "model_version" if key == "baseline_version" else key
        if column in frame and (frame[column] != stack[key]).any():
            raise ValueError(f"{key} of the predictions differs from the stack's")
    joined = join(teacher, baseline)
    ok = (joined["status"] == "ok").to_numpy()
    scores = {}
    for label in LABELS:
        c = stack["labels"][label]
        column = np.full(len(joined), np.nan)
        x = features(joined[ok], label)
        column[ok] = c["w_teacher"] * x[:, 0] + c["w_baseline"] * x[:, 1] + c["intercept"]
        scores[label] = column
    return _output(joined["id"].to_numpy(), joined["status"].to_numpy(), scores)


def save_stack(stack: Mapping[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(stack, indent=2) + "\n")


def load_stack(path: str | Path) -> dict[str, Any]:
    stack = json.loads(Path(path).read_text())
    if stack.get("inputs") != INPUTS or set(stack["labels"]) != set(LABELS):
        raise ValueError(f"{path} is not a stack over {INPUTS} for {list(LABELS)}")
    return stack
