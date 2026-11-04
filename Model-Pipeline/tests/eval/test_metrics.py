"""Tests for src/eval/metrics.py. No network; synthetic data only.

The last test checks the teacher's real gate-subset numbers against Nishaant's reference
files. It needs the private shards and gold labels, so it runs only when the paths are set:

    DISTILLATE_GATE_GOLD=<gate gold parquet> \\
    DISTILLATE_STACK_GATE=<soft_labels/stack-v1/p2-zs-v2b/gate folder> \\
    DISTILLATE_RAW_GATE=<teacher/p2-zs-v2b/gate folder> \\
    DISTILLATE_CALQWEN_GATE=<calibrated-Qwen gate folder> \\
    pytest tests/eval/test_metrics.py -k reference
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.eval import metrics
from src.eval.metrics import CIVIL_LABELS, LABELS

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


def make_data(n: int = 400, seed: int = 0, noise: float = 0.6, logits: bool = False):
    """Gold labels and a prediction file that is informative but not perfect."""
    rng = np.random.default_rng(seed)
    gold = pd.DataFrame({"id": [f"r{i:04d}" for i in range(n)]})
    pred = pd.DataFrame({"id": gold["id"], "model_version": "test-model"})
    for k, label in enumerate(LABELS):
        y = (rng.random(n) < 0.15 + 0.03 * k).astype("int8")
        z = (y * 2 - 1) * 1.5 + rng.normal(0, 1 + noise, n)
        gold[label] = y
        pred[f"p_{label}"] = (1 / (1 + np.exp(-z))).astype("float32")
        if logits:
            pred[f"logit_{label}"] = z.astype("float32")
    pred["status"] = "ok"
    return pred, gold


def test_matches_teacher_metric_definitions():
    """Same AUC, PR-AUC, ECE and Brier as src/teacher/metrics.py on weighted data."""
    teacher = pytest.importorskip("src.teacher.metrics")
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1)
    w = rng.choice([1.25, 3.2], 500)
    ours = metrics.label_metrics(y, p, w, threshold=0.5)
    theirs = teacher.label_metrics(y, p, w)
    for key in ("auc", "pr_auc", "ece", "brier"):
        assert ours[key] == pytest.approx(theirs[key]), key


def test_integer_weights_equal_duplicated_rows():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.3 + rng.random(300) * 0.7, 0, 1)
    w = rng.integers(1, 4, 300)
    weighted = metrics.label_metrics(y, p, w, threshold=0.5)
    dup = metrics.label_metrics(np.repeat(y, w), np.repeat(p, w), np.ones(w.sum()), 0.5)
    for key in ("auc", "pr_auc", "precision", "recall", "f1", "fpr", "ece", "brier"):
        assert weighted[key] == pytest.approx(dup[key]), key


def test_auc_ranks_on_logit_when_present():
    # p ties at 1.0 for the three confident rows; the logits still order them.
    gold = pd.DataFrame({"id": list("abcdef"), **{lab: [1, 0, 1, 0, 1, 0] for lab in LABELS}})
    pred = pd.DataFrame({"id": list("abcdef"), "status": "ok"})
    logit = np.array([8.0, 9.0, 7.0, -1.0, -2.0, -3.0])
    for lab in LABELS:
        pred[f"p_{lab}"] = np.array([1.0, 1.0, 1.0, 0.27, 0.12, 0.05], dtype="float32")
        pred[f"logit_{lab}"] = logit
    with_logit = metrics.evaluate(pred, gold)
    p_only = metrics.evaluate(pred.drop(columns=[f"logit_{lab}" for lab in LABELS]), gold)
    y = gold["toxicity"].to_numpy()
    assert with_logit["per_label"]["toxicity"]["auc"] == pytest.approx(roc_auc_score(y, logit))
    assert p_only["per_label"]["toxicity"]["auc"] == pytest.approx(
        roc_auc_score(y, pred["p_toxicity"]))
    assert with_logit["per_label"]["toxicity"]["auc"] != p_only["per_label"]["toxicity"]["auc"]


def test_failed_rows_are_dropped_and_counted():
    pred, gold = make_data(200)
    pred.loc[:9, "status"] = "failed"
    pred.loc[:9, [f"p_{lab}" for lab in LABELS]] = np.nan
    result = metrics.evaluate(pred, gold)
    assert result["n"] == 190
    assert result["n_failed"] == 10
    assert result["failure_rate"] == pytest.approx(0.05)


def test_every_gold_id_needs_a_prediction():
    pred, gold = make_data(50)
    with pytest.raises(ValueError, match="1 of 50 gold ids have no prediction"):
        metrics.evaluate(pred.iloc[1:], gold)


def test_extra_columns_are_ignored():
    pred, gold = make_data(100)
    pred["cov_toxicity"] = 1.0
    pred["prompt_version"] = "p2-zs-v2b"
    assert metrics.evaluate(pred, gold)["n"] == 100


def test_civil_metrics_skip_severe_toxicity():
    pred, gold = make_data(100)
    result = metrics.evaluate(pred.drop(columns=["p_severe_toxicity"]), gold, CIVIL_LABELS)
    assert result["labels"] == list(CIVIL_LABELS)
    assert "severe_toxicity" not in result["per_label"]


def test_flag_decision_by_hand():
    # Thresholds 0.5. Flag = any label >= 0.5; gold = any label positive.
    gold = pd.DataFrame({"id": list("abcd"), **{lab: [0, 0, 0, 0] for lab in LABELS}})
    gold.loc[0, "threat"] = 1          # a: positive, flagged          -> TP
    gold.loc[1, "insult"] = 1          # b: positive, not flagged      -> FN
    pred = pd.DataFrame({"id": list("abcd"), **{f"p_{lab}": 0.1 for lab in LABELS}})
    pred.loc[0, "p_threat"] = 0.9
    pred.loc[2, "p_obscene"] = 0.7     # c: negative, flagged          -> FP
    flag = metrics.evaluate(pred, gold)["flag"]                       # d: TN
    assert flag["precision"] == pytest.approx(0.5)
    assert flag["recall"] == pytest.approx(0.5)
    assert flag["f1"] == pytest.approx(0.5)
    assert flag["fpr"] == pytest.approx(0.5)


def test_thresholds_change_f1_not_auc(tmp_path):
    pred, gold = make_data(300)
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"thresholds": {lab: 0.9 for lab in LABELS}}))
    high = metrics.evaluate(pred, gold, thresholds=metrics.load_thresholds(path, LABELS))
    base = metrics.evaluate(pred, gold)
    assert high["mean_auc"] == pytest.approx(base["mean_auc"])
    assert high["per_label"]["toxicity"]["recall"] < base["per_label"]["toxicity"]["recall"]


def test_bootstrap_ci_contains_point_and_is_seeded():
    pred, gold = make_data(300)
    design = pd.DataFrame({"id": gold["id"], "weight": 1.0,
                           "stratum": np.where(np.arange(300) < 50, "rare", "rest")})
    a = metrics.bootstrap_ci(pred, gold, design=design, n_boot=50, seed=7)
    b = metrics.bootstrap_ci(pred, gold, design=design, n_boot=50, seed=7)
    assert a == b
    for metric, r in a.items():
        lo, hi = r["ci95"]
        assert lo <= r["point"] <= hi, metric


def test_resampling_keeps_stratum_sizes():
    strata = np.array(["rest"] * 90 + ["threat"] * 10)
    groups = metrics._groups(strata, len(strata))
    idx = metrics._resample(groups, np.random.default_rng(0))
    assert len(idx) == 100
    assert (strata[idx] == "threat").sum() == 10


def test_paired_bootstrap_identical_files_differ_by_zero():
    pred, gold = make_data(200)
    out = metrics.paired_bootstrap(pred, pred, gold, n_boot=30)
    for r in out["metrics"].values():
        assert r["diff"] == pytest.approx(0.0)
        assert r["ci95"] == pytest.approx([0.0, 0.0])


def test_paired_bootstrap_detects_a_better_model():
    good, gold = make_data(400, noise=0.0)
    bad, _ = make_data(400, noise=3.0)
    out = metrics.paired_bootstrap(good, bad, gold, n_boot=100)
    assert out["metrics"]["mean_auc"]["diff"] > 0
    assert out["metrics"]["mean_auc"]["p_gt_0"] > 0.95


def test_cli_writes_json(tmp_path):
    pred, gold = make_data(100)
    pred.to_parquet(tmp_path / "pred.parquet")
    gold.to_parquet(tmp_path / "gold.parquet")
    out = tmp_path / "metrics.json"
    assert metrics.main(["--pred", str(tmp_path / "pred.parquet"),
                         "--gold", str(tmp_path / "gold.parquet"), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["n"] == 100


# ---------------------------------------------------------------------------
# Reference check against Nishaant's gate-subset numbers (needs private data)
# ---------------------------------------------------------------------------

REFERENCES = [
    ("DISTILLATE_STACK_GATE", "stack/stack_check.json", ("soft_labels", "gate_subset")),
    ("DISTILLATE_RAW_GATE", "calqwen/calqwen_check.json", ("gate_subset_teacher_raw",)),
    ("DISTILLATE_CALQWEN_GATE", "calqwen/calqwen_check.json", ("gate_subset",)),
]


@pytest.mark.parametrize("env, ref_file, keys", REFERENCES, ids=["stack", "raw", "calqwen"])
def test_reference_gate_subset_numbers(env, ref_file, keys):
    gold_path, pred_path = os.environ.get("DISTILLATE_GATE_GOLD"), os.environ.get(env)
    if not gold_path or not pred_path:
        pytest.skip(f"set DISTILLATE_GATE_GOLD and {env} to run")
    reference = json.loads((ARTIFACTS / ref_file).read_text())
    for key in keys:
        reference = reference[key]
    design = pd.read_parquet(ARTIFACTS / "gate_subset_ids.parquet")
    gold = metrics.read_frame(gold_path)
    gold = gold[gold["id"].astype(str).isin(set(design["id"].astype(str)))]
    result = metrics.evaluate(metrics.read_frame(pred_path), gold, design=design)
    # The reference files round to 5 decimals.
    for key in ("mean_auc", "mean_pr_auc", "ece", "brier"):
        assert result[key] == pytest.approx(reference[key], abs=2e-5), key
    for label, auc in reference["auc"].items():
        assert result["per_label"][label]["auc"] == pytest.approx(auc, abs=2e-5), label


def test_reference_files_are_present():
    """The committed references the check above compares against."""
    assert (ARTIFACTS / "gate_subset_ids.parquet").exists()
    stack = json.loads((ARTIFACTS / "stack" / "stack_check.json").read_text())
    assert stack["soft_labels"]["gate_subset"]["mean_auc"] == pytest.approx(0.982, abs=5e-4)


def test_average_precision_is_pr_auc():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.4, 0.35, 0.8])
    result = metrics.label_metrics(y, p, np.ones(5), 0.5)
    assert result["pr_auc"] == pytest.approx(average_precision_score(y, p))
