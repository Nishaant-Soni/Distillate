"""Tests for src/eval/gate.py on synthetic prediction and gold files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval import gate
from src.eval.metrics import CIVIL_LABELS, LABELS

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "gate.yaml"


def _gold(n: int, seed: int, identities: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gold = pd.DataFrame({"id": [f"g{seed}_{i:05d}" for i in range(n)]})
    for k, label in enumerate(LABELS):
        gold[label] = (rng.random(n) < 0.12 + 0.03 * k).astype("int8")
    if identities:
        gold["female"] = np.where(rng.random(n) < 0.3, 1.0, 0.0)
        gold["muslim"] = np.where(rng.random(n) < 0.3, 1.0, 0.0)
    return gold


def _pred(gold: pd.DataFrame, seed: int, noise: float, version: str,
          biased_column: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pred = pd.DataFrame({"id": gold["id"], "model_version": version})
    for label in LABELS:
        z = (gold[label].to_numpy() * 2 - 1) * 2.0 + rng.normal(0, 1 + noise, len(gold))
        if biased_column and label == "toxicity":
            z = z + 3.0 * (gold[biased_column].to_numpy() >= 0.5)   # false positives on a slice
        pred[f"p_{label}"] = 1 / (1 + np.exp(-z))
    return pred


def write_run(folder: Path, noise: float, version: str, seed: int = 1,
              biased_column: str | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    golds = {"val": _gold(1500, 1), "gate": _gold(1500, 2), "bias": _gold(1500, 3, True)}
    for stem, gold in golds.items():
        bias_col = biased_column if stem == "bias" else None
        _pred(gold, seed + len(stem), noise, version, bias_col).to_parquet(
            folder / f"{stem}.parquet")
    return folder


def write_gold(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    _gold(1500, 1).to_parquet(folder / "val.parquet")
    _gold(1500, 2).to_parquet(folder / "gate.parquet")
    _gold(1500, 3, True).to_parquet(folder / "bias.parquet")
    return folder


@pytest.fixture
def cfg():
    return gate.load_config(CONFIG)


def test_good_candidate_passes_without_production(tmp_path, cfg):
    gold = write_gold(tmp_path / "gold")
    cand = write_run(tmp_path / "cand", noise=0.0, version="cand")
    report = gate.run_gate(cfg, cand, gold)
    assert report["passed"], report["failed_checks"]
    skipped = [c for c in report["checks"] if c["passed"] is None]
    assert [c["name"] for c in skipped] == ["gate set: no regression"]


def test_degraded_candidate_is_blocked_by_validation(tmp_path, cfg):
    gold = write_gold(tmp_path / "gold")
    cand = write_run(tmp_path / "cand", noise=40.0, version="degraded")   # close to random
    report = gate.run_gate(cfg, cand, gold)
    assert not report["passed"]
    assert "validation: mean AUC" in report["failed_checks"]


def test_regression_against_production_blocks(tmp_path, cfg):
    gold = write_gold(tmp_path / "gold")
    prod = write_run(tmp_path / "prod", noise=0.0, version="prod")
    cand = write_run(tmp_path / "cand", noise=1.5, version="cand", seed=5)
    cfg["validation"]["min_mean_auc"] = 0.5      # isolate the regression check
    cfg["validation"]["min_pr_auc"] = 0.0
    report = gate.run_gate(cfg, cand, gold, production=prod)
    assert "gate set: mean AUC vs production" in report["failed_checks"]


def test_same_model_passes_regression(tmp_path, cfg):
    gold = write_gold(tmp_path / "gold")
    prod = write_run(tmp_path / "prod", noise=0.0, version="prod")
    report = gate.run_gate(cfg, prod, gold, production=prod)
    regression = [c for c in report["checks"] if c["name"].startswith("gate set")]
    assert regression and all(c["passed"] for c in regression)


def test_biased_slice_is_blocked(tmp_path, cfg):
    gold = write_gold(tmp_path / "gold")
    cand = write_run(tmp_path / "cand", noise=0.0, version="cand", biased_column="muslim")
    report = gate.run_gate(cfg, cand, gold)
    assert "bias: religion:muslim" in report["failed_checks"]
    assert "bias: gender:female" not in report["failed_checks"]


def test_bias_rise_against_production(tmp_path, cfg):
    gold = write_gold(tmp_path / "gold")
    prod = write_run(tmp_path / "prod", noise=0.0, version="prod")
    cand = write_run(tmp_path / "cand", noise=0.0, version="cand", biased_column="muslim")
    cfg["bias"]["max_fpr_gap_pts"] = None        # only the rise check is on
    report = gate.run_gate(cfg, cand, gold, production=prod)
    assert "bias: religion:muslim" in report["failed_checks"]


def test_thresholds_file_next_to_predictions_is_used(tmp_path, cfg):
    gold = write_gold(tmp_path / "gold")
    cand = write_run(tmp_path / "cand", noise=0.0, version="cand")
    (cand / "thresholds.json").write_text(json.dumps({lab: 0.3 for lab in LABELS}))
    report = gate.run_gate(cfg, cand, gold)
    assert report["metrics"]["validation"]["per_label"]["toxicity"]["threshold"] == 0.3


def test_bias_label_must_be_a_civil_label(tmp_path, cfg):
    gold = write_gold(tmp_path / "gold")
    cand = write_run(tmp_path / "cand", noise=0.0, version="cand")
    cfg["bias"]["label"] = "severe_toxicity"
    assert "severe_toxicity" not in CIVIL_LABELS
    with pytest.raises(ValueError, match="bias.label"):
        gate.run_gate(cfg, cand, gold)


def test_cli_exit_codes_and_reports(tmp_path):
    gold = write_gold(tmp_path / "gold")
    good = write_run(tmp_path / "good", noise=0.0, version="good")
    bad = write_run(tmp_path / "bad", noise=40.0, version="bad")
    common = ["--config", str(CONFIG), "--gold", str(gold)]
    assert gate.main([*common, "--candidate", str(good), "--out", str(tmp_path / "g")]) == 0
    assert gate.main([*common, "--candidate", str(bad), "--out", str(tmp_path / "b")]) == 1
    report = json.loads((tmp_path / "b" / "gate_report.json").read_text())
    assert report["passed"] is False
    markdown = (tmp_path / "b" / "gate_report.md").read_text()
    assert markdown.startswith("# Gate report: BLOCKED")


def test_missing_split_file_is_an_error(tmp_path, cfg):
    gold = write_gold(tmp_path / "gold")
    cand = write_run(tmp_path / "cand", noise=0.0, version="cand")
    (cand / "bias.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="bias.parquet"):
        gate.run_gate(cfg, cand, gold)
