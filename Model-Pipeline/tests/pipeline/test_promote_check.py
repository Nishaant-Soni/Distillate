import json

import numpy as np
import pandas as pd
import pytest

from src.pipeline.promote_check import (
    as_predictions,
    compare,
    eligible_labels,
    load_config,
    main,
    run,
)

LABELS = ["toxicity", "obscene", "threat", "insult", "identity_attack"]


def cfg(**changes):
    return {**load_config(), "bootstrap": 50, **changes}


def batch(n=400, seed=0):
    """A drift batch: id, text (must never be needed), and gold labels (all six)."""
    rng = np.random.default_rng(seed)
    gold = pd.DataFrame({"id": [f"c{i}" for i in range(n)], "text": ["some comment"] * n})
    for label, rate in zip([*LABELS, "severe_toxicity"], [0.3, 0.1, 0.05, 0.2, 0.1, 0.0],
                           strict=True):
        gold[label] = (rng.random(n) < rate).astype("int8")
    return gold


def scores(gold, signal, version, column="p", seed=1):
    """Predictions whose ranking follows the gold labels with strength ``signal`` (below 1,
    so the classes overlap and AUC stays under 1)."""
    rng = np.random.default_rng(seed)
    out = pd.DataFrame({"id": gold["id"], "model_version": version})
    for label in [*LABELS, "severe_toxicity"]:
        noise = rng.random(len(gold))
        out[f"{column}_{label}"] = np.clip(signal * gold[label] + noise, 0, None) / (1 + signal)
    return out


def test_a_better_candidate_passes_and_a_worse_one_fails():
    gold = batch()
    good, prod, bad = scores(gold, 0.8, "cand"), scores(gold, 0.4, "prod", seed=2), \
        scores(gold, 0.05, "worse", seed=3)
    report = compare(good, prod, gold, cfg())
    assert report["passed"] and report["gain"] > 0
    assert report["labels_used"] == LABELS and report["rows"] == len(gold)
    assert report["bootstrap_gain"]["ci95"][0] > 0
    assert not compare(bad, prod, gold, cfg())["passed"]


def test_equal_models_do_not_pass():
    gold = batch()
    same = scores(gold, 0.4, "m")
    assert compare(same, same.assign(model_version="p"), gold, cfg())["gain"] == 0
    assert not compare(same, same.assign(model_version="p"), gold, cfg())["passed"]


def test_min_gain():
    gold = batch()
    report = compare(scores(gold, 0.8, "c"), scores(gold, 0.4, "p", seed=2), gold,
                     cfg(min_gain=0.5))
    assert not report["passed"] and "> 0.5" in report["rule"]


def test_drift_ingest_files_are_read_as_production():
    gold = batch()
    live = scores(gold, 0.4, "prod", column="score", seed=2)   # score_<label>, no p_<label>
    live["flagged_toxicity"] = True
    report = compare(scores(gold, 0.8, "cand"), live, gold, cfg())
    assert report["production"]["model_version"] == "prod" and report["passed"]


def test_logits_are_ignored_so_both_models_rank_on_p():
    gold = batch()
    cand = scores(gold, 0.05, "cand")
    for label in LABELS:                    # perfect logits, useless p: p is what counts
        cand[f"logit_{label}"] = gold[label] * 10.0
    assert "logit_toxicity" not in as_predictions(cand, LABELS)
    assert not compare(cand, scores(gold, 0.4, "prod", seed=2), gold, cfg())["passed"]


def test_rows_failed_by_either_model_are_left_out_of_both():
    gold = batch()
    cand = scores(gold, 0.8, "cand").assign(status="ok")
    cand.loc[:9, "status"] = "failed"
    assert compare(cand, scores(gold, 0.4, "prod", seed=2), gold, cfg())["rows"] == len(gold) - 10


def test_min_positives_leaves_rare_labels_out():
    gold = batch()
    used, positives = eligible_labels(gold, LABELS, 30)
    assert "threat" not in used and positives["threat"] < 30
    report = compare(scores(gold, 0.8, "c"), scores(gold, 0.4, "p", seed=2), gold,
                     cfg(min_positives=30))
    assert "threat" in report["labels_left_out"] and report["labels_used"] == used


def test_a_label_without_positives_is_never_used():
    gold = batch()
    gold["threat"] = 0
    used, _ = eligible_labels(gold, LABELS, None)
    assert "threat" not in used


def test_ids_outside_the_batch_are_an_error():
    gold = batch()
    cand = scores(gold, 0.8, "c")
    cand.loc[0, "id"] = "elsewhere"
    with pytest.raises(ValueError, match="aren't in the batch"):
        compare(cand, scores(gold, 0.4, "p", seed=2), gold, cfg())


def _files(tmp_path, cand_version="cand", prod_version="prod"):
    gold = batch()
    paths = {name: tmp_path / f"{name}.parquet" for name in ("cand", "prod", "gold")}
    scores(gold, 0.8, cand_version).to_parquet(paths["cand"])
    scores(gold, 0.4, prod_version, column="score", seed=2).to_parquet(paths["prod"])
    gold.to_parquet(paths["gold"])
    return paths


def test_the_held_out_batch_cannot_be_a_training_batch(tmp_path):
    p = _files(tmp_path)
    with pytest.raises(ValueError, match="training data"):
        run(cfg(), 5, p["cand"], p["prod"], p["gold"], trained_on=[3, 4, 5])


def test_production_predictions_from_another_model_are_refused(tmp_path):
    p = _files(tmp_path, prod_version="v1-old")
    with pytest.raises(ValueError, match="rescore"):
        run(cfg(), 5, p["cand"], p["prod"], p["gold"], production_model_version="v2")


def test_cli_writes_the_report_without_text_and_sets_the_exit_code(tmp_path):
    p = _files(tmp_path)
    out = tmp_path / "out"
    code = main(["--batch", "5", "--candidate", str(p["cand"]), "--production", str(p["prod"]),
                 "--gold", str(p["gold"]), "--production-model-version", "prod",
                 "--trained-on", "0", "1", "2", "3", "--out", str(out)])
    assert code == 0
    report = json.loads((out / "promote_check.json").read_text())
    assert report["passed"] and report["batch"] == 5 and report["trained_on"] == [0, 1, 2, 3]
    text = (out / "promote_check.json").read_text() + (out / "promote_check.md").read_text()
    assert "some comment" not in text and "PASSED" in text

    worse = tmp_path / "worse.parquet"
    scores(batch(), 0.02, "worse").to_parquet(worse)
    assert main(["--batch", "5", "--candidate", str(worse), "--production", str(p["prod"]),
                 "--gold", str(p["gold"]), "--out", str(out)]) == 1
