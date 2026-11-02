import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.teacher.config import LABELS
from src.teacher.metrics import evaluate, label_metrics, weighted_ece


def test_integer_weights_equal_duplicated_rows():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1)
    w = rng.integers(1, 4, 200)
    weighted = label_metrics(y, p, w)
    dup = label_metrics(np.repeat(y, w), np.repeat(p, w), np.ones(w.sum()))
    for key in ("auc", "pr_auc", "ece", "brier"):
        assert weighted[key] == pytest.approx(dup[key]), key
    assert weighted["auc"] == pytest.approx(roc_auc_score(np.repeat(y, w), np.repeat(p, w)))


def test_ece():
    # Perfectly calibrated bins give 0; constant 0.9 on a 50% base rate gives 0.4.
    y = np.array([0, 1] * 50)
    assert weighted_ece(y, np.full(100, 0.5), np.ones(100)) == pytest.approx(0.0)
    assert weighted_ece(y, np.full(100, 0.9), np.ones(100)) == pytest.approx(0.4)
    assert weighted_ece([1], [1.0], [1]) == pytest.approx(0.0)    # p == 1 lands in the last bin


def test_single_class_gives_nan_auc():
    m = label_metrics([0, 0, 0], [0.1, 0.2, 0.3], [1, 1, 1])
    assert np.isnan(m["auc"]) and np.isnan(m["pr_auc"]) and m["brier"] > 0


def frames(n=40):
    rng = np.random.default_rng(1)
    truth = pd.DataFrame({"id": [f"r{i}" for i in range(n)]})
    pred = pd.DataFrame({"id": truth["id"][::-1].to_numpy()})   # different order on purpose
    for label in LABELS:
        truth[label] = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype("int8")
        pred[f"p_{label}"] = rng.random(n)
    return truth, pred


def test_evaluate_excludes_failed_rows():
    truth, pred = frames()
    pred["status"] = "ok"
    pred.loc[0, "status"] = "failed"
    pred.loc[0, [f"p_{label}" for label in LABELS]] = np.nan
    table = evaluate(pred, truth)
    assert list(table.index) == [*LABELS, "mean"]
    assert table.loc["toxicity", "n"] == 39
    assert table.loc["toxicity", "failure_rate"] == pytest.approx(1 / 40)
    assert table.loc["mean", "auc"] == pytest.approx(table.loc[list(LABELS), "auc"].mean())


def test_evaluate_uses_weights():
    truth, pred = frames()
    weights = pd.DataFrame({"id": truth["id"], "weight": np.linspace(1, 5, 40)})
    weighted = evaluate(pred, truth, weights)
    unweighted = evaluate(pred, truth)
    assert weighted.loc["toxicity", "brier"] != pytest.approx(unweighted.loc["toxicity", "brier"])


def test_evaluate_rejects_missing_predictions():
    truth, pred = frames()
    with pytest.raises(ValueError, match="no prediction"):
        evaluate(pred.iloc[1:], truth)


def test_ranking_metrics_use_logits_when_present():
    # Three positives and two negatives all have p == 1.0; only the logits rank them.
    truth = pd.DataFrame({"id": list("abcdefgh")})
    pred = pd.DataFrame({"id": list("abcdefgh")})
    for label in LABELS:
        truth[label] = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype="int8")
        pred[f"p_{label}"] = np.array([1, 1, 1, 1, 1, 0.1, 0.1, 0.1], dtype="float32")
    tied = evaluate(pred, truth)
    for label in LABELS:
        pred[f"logit_{label}"] = np.array([30, 29, 28, 20, 19, -2, -2, -2], dtype="float32")
    ranked = evaluate(pred, truth)
    assert tied.loc["toxicity", "auc"] < 1.0
    assert ranked.loc["toxicity", "auc"] == pytest.approx(1.0)
    assert ranked.loc["toxicity", "ece"] == pytest.approx(tied.loc["toxicity", "ece"])


def test_paired_bootstrap():
    from src.teacher.metrics import paired_bootstrap

    rng = np.random.default_rng(0)
    n = 1500
    truth = pd.DataFrame({"id": [str(i) for i in range(n)]})
    good, noisy = truth[["id"]].copy(), truth[["id"]].copy()
    for label in LABELS:
        signal = rng.normal(0, 1, n)
        truth[label] = (signal + rng.normal(0, 0.5, n) > 1).astype("int8")
        good[f"p_{label}"] = 1 / (1 + np.exp(-signal))
        noisy[f"p_{label}"] = 1 / (1 + np.exp(-(signal + rng.normal(0, 2, n))))
    truth.loc[:, "threat"] = 0                                   # one class: left out
    weights = pd.DataFrame({"id": truth["id"], "weight": rng.uniform(1, 3, n)})
    strata = pd.DataFrame({"id": truth["id"], "stratum": np.where(np.arange(n) < 300, "a", "b")})
    out = paired_bootstrap({"good": good, "noisy": noisy}, truth, weights,
                           [("good", "noisy"), ("good", "good")], strata=strata, n_boot=200)
    gn, gg = out["good - noisy"], out["good - good"]
    assert "threat" not in gn["labels"] and len(gn["labels"]) == 5
    assert gn["mean_auc_diff"] > 0 and gn["mean_auc_ci95"][0] > 0 and gn["mean_auc_p_gt_0"] == 1
    assert gg["mean_auc_diff"] == 0 and gg["mean_auc_ci95"] == [0, 0]

    only = paired_bootstrap({"good": good, "noisy": noisy}, truth, weights, [("good", "noisy")],
                            n_boot=50, labels=["obscene"])["good - noisy"]
    assert only["labels"] == ["obscene"]
