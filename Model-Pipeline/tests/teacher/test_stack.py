import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.teacher.config import LABELS
from src.teacher.stack import (
    apply_stack,
    cross_fit_stack,
    fit_stack,
    load_stack,
    save_stack,
)


def synthetic(n=3000, seed=0):
    """A teacher and a baseline that each see part of a hidden score; truth from the sum."""
    rng = np.random.default_rng(seed)
    ids = [f"{i:06x}" for i in range(n)]
    truth = pd.DataFrame({"id": ids})
    teacher = pd.DataFrame({"id": ids, "status": "ok", "prompt_version": "p2-zs-v1",
                            "model_version": "m@r"})
    baseline = pd.DataFrame({"id": ids, "model_version": "tfidf-lr-v1"})
    for label in LABELS:
        a, b = rng.normal(0, 1.5, n), rng.normal(0, 1.5, n)
        truth[label] = (rng.random(n) < 1 / (1 + np.exp(-(a + b - 2)))).astype("int8")
        teacher[f"logit_{label}"] = (6 * a).astype("float32")              # overconfident
        teacher[f"p_{label}"] = (1 / (1 + np.exp(-6 * a))).astype("float32")
        baseline[f"p_{label}"] = (1 / (1 + np.exp(-(b - 1)))).astype("float32")
    weights = pd.DataFrame({"id": ids, "weight": rng.uniform(1, 5, n)})
    return teacher, baseline, truth, weights


def auc(truth, frame, label, column):
    return roc_auc_score(truth[label], frame[column])


def test_cross_fit_beats_either_input():
    teacher, baseline, truth, weights = synthetic()
    oof = cross_fit_stack(teacher, baseline, truth, weights)
    for label in LABELS:
        stacked = auc(truth, oof, label, f"logit_{label}")
        assert stacked > auc(truth, teacher, label, f"logit_{label}") + 0.02
        assert stacked > auc(truth, baseline, label, f"p_{label}") + 0.02
    assert oof["p_toxicity"].between(0, 1).all()


def test_fit_weights_and_apply_round_trip(tmp_path):
    teacher, baseline, truth, weights = synthetic()
    stack = fit_stack(teacher, baseline, truth, weights, "stack-v1", "p2-zs-v1", "m@r",
                      "tfidf-lr-v1")
    c = stack["labels"]["threat"]
    assert c["w_teacher"] == pytest.approx(1 / 6, rel=0.25)   # undoes the overconfidence
    assert c["w_baseline"] == pytest.approx(1.0, rel=0.25)
    save_stack(stack, tmp_path / "stack.json")
    loaded = load_stack(tmp_path / "stack.json")
    out = apply_stack(teacher, baseline, loaded)
    expected = (c["w_teacher"] * teacher["logit_threat"].astype("float64")
                + c["w_baseline"] * np.log(baseline["p_threat"] / (1 - baseline["p_threat"]))
                + c["intercept"])
    np.testing.assert_allclose(out["logit_threat"], expected, rtol=1e-4, atol=1e-4)


def test_failed_rows_stay_failed():
    teacher, baseline, truth, weights = synthetic(600)
    teacher.loc[:9, "status"] = "failed"
    teacher.loc[:9, [f"logit_{label}" for label in LABELS]] = np.nan
    oof = cross_fit_stack(teacher, baseline, truth, weights)
    assert oof.loc[:9, "p_toxicity"].isna().all() and oof.loc[10:, "p_toxicity"].notna().all()
    stack = fit_stack(teacher, baseline, truth, weights, "s", "p2-zs-v1", "m@r", "tfidf-lr-v1")
    out = apply_stack(teacher, baseline, stack)
    assert (out.loc[:9, "status"] == "failed").all() and out.loc[:9, "p_insult"].isna().all()


def test_saturated_baseline_is_clipped():
    teacher, baseline, truth, weights = synthetic(600)
    baseline.loc[:4, "p_obscene"] = np.float32(1.0)
    baseline.loc[5:9, "p_obscene"] = np.float32(0.0)
    oof = cross_fit_stack(teacher, baseline, truth, weights)
    assert np.isfinite(oof["logit_obscene"]).all()


def test_refuses_p_only_teacher_and_version_mismatch():
    teacher, baseline, truth, weights = synthetic(600)
    with pytest.raises(ValueError, match="logit"):
        cross_fit_stack(teacher.drop(columns=[f"logit_{label}" for label in LABELS]),
                        baseline, truth, weights)
    stack = fit_stack(teacher, baseline, truth, weights, "s", "p2-zs-v1", "m@r", "tfidf-lr-v1")
    with pytest.raises(ValueError, match="prompt_version"):
        apply_stack(teacher.assign(prompt_version="p2-fs-v1"), baseline, stack)
    with pytest.raises(ValueError, match="baseline_version"):
        apply_stack(teacher, baseline.assign(model_version="tfidf-lr-v2"), stack)


def test_missing_baseline_rows_raise():
    teacher, baseline, truth, weights = synthetic(600)
    with pytest.raises(ValueError, match="baseline"):
        cross_fit_stack(teacher, baseline.iloc[1:], truth, weights)
