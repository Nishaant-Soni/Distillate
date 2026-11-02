import math

import numpy as np
import pandas as pd
import pytest

from src.teacher.calibrate import (
    apply_calibration,
    cross_fit,
    fit_calibration,
    fit_platt,
    load_calibration,
    platt,
    save_calibration,
)
from src.teacher.config import LABELS
from src.teacher.metrics import evaluate, weighted_ece


def overconfident(n=3000, seed=0):
    """Gold labels from true log-odds t; the "teacher" reports 4 * t + 3, so it is far too
    sure and biased towards yes. Platt scaling should recover slope 0.25, intercept -0.75."""
    rng = np.random.default_rng(seed)
    ids = [f"r{i:05d}" for i in range(n)]
    truth = pd.DataFrame({"id": ids})
    pred = pd.DataFrame({"id": ids, "status": "ok"})
    for label in LABELS:
        t = rng.normal(-2.0, 2.0, n)
        truth[label] = (rng.random(n) < 1 / (1 + np.exp(-t))).astype("int8")
        logit = 4 * t + 3
        pred[f"logit_{label}"] = logit.astype("float32")
        pred[f"p_{label}"] = (1 / (1 + np.exp(-logit))).astype("float32")
    weights = pd.DataFrame({"id": ids, "weight": 1.0})
    return pred, truth, weights


def test_fit_recovers_known_scaling():
    pred, truth, _ = overconfident(n=20000)
    slope, intercept = fit_platt(pred["logit_toxicity"], truth["toxicity"], np.ones(len(pred)))
    assert slope == pytest.approx(0.25, abs=0.02)
    assert intercept == pytest.approx(-0.75, abs=0.08)


def test_integer_weights_equal_duplicated_rows():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 3, 400)
    y = (rng.random(400) < 1 / (1 + np.exp(-z / 2))).astype(int)
    w = rng.integers(1, 4, 400)
    weighted = fit_platt(z, y, w)
    duplicated = fit_platt(np.repeat(z, w), np.repeat(y, w), np.ones(w.sum()))
    assert weighted == pytest.approx(duplicated, abs=1e-4)


def test_saturated_logits_stay_distinct():
    # p is exactly 1.0 in float32 for all three, logit(p) is infinite; calibrated p differs.
    z = np.array([18.0, 25.0, 31.0])
    assert (np.float32(1 / (1 + np.exp(-z))) == 1.0).all()
    calibrated = platt(z, 0.2, -1.0)
    assert np.all(np.diff(calibrated) > 0) and calibrated.max() < 1


def test_cross_fit_lowers_ece():
    pred, truth, weights = overconfident()
    raw = evaluate(pred, truth, weights)
    calibrated = evaluate(cross_fit(pred, truth, weights, folds=5, seed=42), truth, weights)
    for label in LABELS:
        assert calibrated.loc[label, "ece"] < raw.loc[label, "ece"] / 3, label
        assert calibrated.loc[label, "brier"] < raw.loc[label, "brier"], label


def test_cross_fit_is_out_of_fold_and_keeps_failed_rows():
    pred, truth, weights = overconfident(n=500)
    pred.loc[3, "status"] = "failed"
    pred.loc[3, [f"logit_{label}" for label in LABELS]] = np.nan
    out = cross_fit(pred, truth, weights, folds=5, seed=42)
    assert list(out["id"]) == list(pred["id"])
    assert out.loc[3, [f"p_{label}" for label in LABELS]].isna().all()
    assert out.drop(index=3)[[f"p_{label}" for label in LABELS]].notna().all(axis=None)
    # Out of fold: not the same as one fit on everything.
    full = apply_calibration(pred.drop(index=3), fit_calibration(
        pred, truth, weights, "test", "p2-zs-v1", "m@r"))
    assert not np.allclose(out.drop(index=3)["p_toxicity"], full["p_toxicity"])


def test_calibration_round_trip_and_apply(tmp_path):
    pred, truth, weights = overconfident()
    cal = fit_calibration(pred, truth, weights, "platt-v1", "p2-zs-v1", "m@r")
    assert cal["input"] == "logit"
    save_calibration(cal, tmp_path / "calibration.json")
    loaded = load_calibration(tmp_path / "calibration.json")
    assert loaded == cal
    out = apply_calibration(pred, loaded)
    c = cal["labels"]["insult"]
    want = 1 / (1 + math.exp(-(c["slope"] * pred.loc[0, "logit_insult"] + c["intercept"])))
    assert out.loc[0, "p_insult"] == pytest.approx(want, rel=1e-5)
    assert out["logit_insult"].equals(pred["logit_insult"])    # logits are left as they are
    ece_before = weighted_ece(truth["insult"], pred["p_insult"], np.ones(len(pred)))
    ece_after = weighted_ece(truth["insult"], out["p_insult"], np.ones(len(pred)))
    assert ece_after < ece_before


def test_refuses_p_only_frames():
    # Never calibrate on logit(p): without logit columns, fitting and applying both fail.
    pred, truth, weights = overconfident(n=200)
    cal = fit_calibration(pred, truth, weights, "platt-v1", "p2-zs-v1", "m@r")
    p_only = pred.drop(columns=[f"logit_{label}" for label in LABELS])
    with pytest.raises(ValueError, match="logit"):
        apply_calibration(p_only, cal)
    with pytest.raises(ValueError, match="logit"):
        fit_calibration(p_only, truth, weights, "platt-v1", "p2-zs-v1", "m@r")
    with pytest.raises(ValueError, match="logit"):
        cross_fit(p_only, truth, weights)


def test_refuses_mismatched_versions():
    pred, truth, weights = overconfident(n=200)
    cal = fit_calibration(pred, truth, weights, "platt-v1", "p2-zs-v1", "m@r")
    pred["prompt_version"] = "p2-fs-v1"
    with pytest.raises(ValueError, match="prompt_version"):
        apply_calibration(pred, cal)
