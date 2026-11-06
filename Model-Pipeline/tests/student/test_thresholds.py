"""Per-label thresholds: the F1 search, its edge cases, and the file the gate reads."""

import json

import numpy as np
import pandas as pd
import pytest

from src.eval.metrics import load_thresholds
from src.student import thresholds
from src.teacher.config import LABELS


def brute_force(y, p, low, high):
    best, best_t = -1.0, None
    for t in sorted(set(p)):
        if not low <= t <= high:
            continue
        flag = p >= t
        tp, fp, fn = (flag & (y == 1)).sum(), (flag & (y == 0)).sum(), (~flag & (y == 1)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn)
        if f1 >= best:
            best, best_t = f1, t
    return best_t


@pytest.mark.parametrize("seed", range(5))
def test_search_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(400) < 0.15).astype(int)
    p = np.clip(0.3 * y + rng.random(400) * 0.7, 0, 1).round(3)   # rounding makes ties
    assert thresholds.best_f1_threshold(y, p, 0.01, 0.99) == brute_force(y, p, 0.01, 0.99)


def test_ties_take_the_higher_threshold_and_bounds_hold():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    assert thresholds.best_f1_threshold(y, p, 0.01, 0.99) == 0.8     # 0.3 and lower are worse
    assert thresholds.best_f1_threshold(y, np.array([0.999, 0.998, 0.3, 0.2, 0.1]),
                                        0.01, 0.99) == pytest.approx(0.3)
    assert thresholds.best_f1_threshold(np.zeros(5, int), p, 0.01, 0.99) == 0.99


def _frames(n=300, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"{i:016x}" for i in range(n)]
    gold = pd.DataFrame({"id": ids, **{k: (rng.random(n) < 0.2).astype("int8") for k in LABELS}})
    pred = pd.DataFrame({"id": ids, **{f"p_{k}": np.clip(gold[k] * 0.4 + rng.random(n) * 0.6,
                                                          0, 1) for k in LABELS}})
    return pred, gold


def test_file_is_the_flat_mapping_the_gate_reads(tmp_path):
    pred, gold = _frames()
    chosen, report = thresholds.choose(pred, gold)
    thresholds.write(chosen, report, tmp_path)
    on_disk = json.loads((tmp_path / "thresholds.json").read_text())
    assert sorted(on_disk) == sorted(LABELS)                 # nothing but the six labels
    assert load_thresholds(tmp_path / "thresholds.json", LABELS) == chosen
    assert report["mean_f1"] >= report["mean_f1_at_0.5"] - 1e-12
    for label in LABELS:
        assert 0.01 <= chosen[label] <= 0.99


def test_predictions_without_gold_labels_fail():
    pred, gold = _frames()
    with pytest.raises(ValueError, match="no gold label"):
        thresholds.choose(pred, gold.iloc[10:])
