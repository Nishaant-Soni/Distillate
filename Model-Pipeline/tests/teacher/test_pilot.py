import json

import numpy as np
import pandas as pd
import pytest

from src.teacher.config import LABELS, load_config
from src.teacher.metrics import evaluate
from src.teacher.pilot import (
    decision,
    fewshot_examples,
    pilot_truth,
    results_table,
    saturation,
    save_run,
    summarize_run,
)


def synthetic(n=1500, seed=0):
    """Train rows, pilot ids with weights, and a fake teacher run with logits."""
    rng = np.random.default_rng(seed)
    ids = [f"{i:06x}" for i in range(n)]
    train = pd.DataFrame({"id": ids, "text": [f"comment {i}" for i in range(n)]})
    pred = pd.DataFrame({"id": ids, "status": "ok"})
    for label in LABELS:
        t = rng.normal(-1.5, 2.0, n)
        train[label] = (rng.random(n) < 1 / (1 + np.exp(-t))).astype("int8")
        logit = 5 * t + 4
        pred[f"logit_{label}"] = logit.astype("float32")
        pred[f"p_{label}"] = (1 / (1 + np.exp(-logit))).astype("float32")
        pred[f"cov_{label}"] = np.float32(1.0)
    pilot_ids = pd.DataFrame({"id": ids, "weight": rng.uniform(1, 10, n)})
    return train, pilot_ids, pred


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_fewshot_examples_order_and_labels():
    train, _, _ = synthetic(50)
    ids = {label: [train["id"][2 * j], train["id"][2 * j + 1]] for j, label in enumerate(LABELS)}
    examples = fewshot_examples(train, ids)
    assert [e.text for e in examples] == [f"comment {i}" for i in range(12)]
    assert examples[3].labels == {label: int(train.loc[3, label]) for label in LABELS}
    with pytest.raises(ValueError, match="not in train"):
        fewshot_examples(train, {**ids, "threat": ["nope", "nope2"]})


def test_pilot_truth_checks_ids():
    train, pilot_ids, _ = synthetic(100)
    assert len(pilot_truth(train, pilot_ids.iloc[:40])) == 40
    with pytest.raises(ValueError):
        pilot_truth(train.iloc[:30], pilot_ids.iloc[:40])


def test_saturation():
    pred = pd.DataFrame({"status": ["ok", "ok", "failed"]})
    for label in LABELS:
        pred[f"p_{label}"] = np.array([1.0, 0.5, np.nan], dtype="float32")
    s = saturation(pred)
    assert s["share_p_eq_1"] == 0.5 and s["share_p_eq_0"] == 0.0


def test_summarize_table_and_decision(cfg, tmp_path):
    train, pilot_ids, pred = synthetic()
    truth = pilot_truth(train, pilot_ids)
    runs = {pv: summarize_run(pred, truth, pilot_ids, cfg, pv, "m@r")
            for pv in ("p2-zs-v1", "p2-fs-v1")}
    run = runs["p2-zs-v1"]
    assert run["calibrated"].loc["mean", "ece"] < run["raw"].loc["mean", "ece"]
    assert run["calibration"]["input"] == "logit"

    tfidf_pred = pred[["id"]].copy()
    rng = np.random.default_rng(5)
    for label in LABELS:
        tfidf_pred[f"p_{label}"] = rng.random(len(pred))     # an uninformative baseline
    tfidf = evaluate(tfidf_pred, truth, pilot_ids)
    table = results_table(runs, tfidf, "tfidf-lr-v1", {"p2-zs-v1": 6.3})
    assert list(table.index) == ["p2-zs-v1", "p2-fs-v1", "tfidf-lr-v1"]
    assert table.loc["p2-zs-v1", "rows_per_s"] == 6.3
    assert np.isnan(table.loc["tfidf-lr-v1", "ece_calibrated"])
    d = decision(table, "tfidf-lr-v1")
    assert d["best_zero_shot"] == "p2-zs-v1" and d["lora_contingency"] is False

    save_run(tmp_path, pred, run, {"seconds": 1.0})
    assert (tmp_path / "pilot_p2-zs-v1.parquet").exists()
    metrics = json.loads((tmp_path / "metrics_p2-zs-v1.json").read_text())
    assert set(metrics["raw"]) == {*LABELS, "mean"}
    cal = json.loads((tmp_path / "calibration_p2-zs-v1.json").read_text())
    assert cal["calibration_version"] == cfg["calibration"]["version"]


def test_decision_triggers_lora():
    table = pd.DataFrame({"mean_auc": [0.95, 0.99, 0.985]},
                         index=["p2-zs-v1", "p2-fs-v1", "tfidf-lr-v1"])
    d = decision(table, "tfidf-lr-v1")
    assert d["lora_contingency"] is True    # few-shot beating TF-IDF does not count
