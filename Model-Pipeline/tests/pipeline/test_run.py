"""The pipeline entry point end to end on the tiny offline model, then selection and results."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.metrics import LABELS
from src.student.eval_sets import IDENTITY_COLUMNS
from src.student.synthetic import make_rows
from tests.student.conftest import smoke_cfg

pytest.importorskip("onnxruntime")
pytest.importorskip("matplotlib")

FAST = ("mlflow.enabled=false", "train.max_steps=6", "train.epochs=1", "package.enabled=true",
        "pipeline.latency_requests=5")


def write_eval_tables(root: Path, n: int = 240) -> Path:
    """Synthetic tables in the data pipeline's layout (ids distinct from training)."""
    data = root / "processed"
    rng = np.random.default_rng(7)
    gate = make_rows(n, seed=101)
    civil = make_rows(2 * n, seed=202)
    for column in IDENTITY_COLUMNS:
        civil[column] = rng.choice([0.0, 0.0, 0.0, 0.6], size=len(civil)).astype("float32")
    obf = gate[gate["toxicity"] == 1].head(30).assign(
        id=lambda d: d["id"] + "_obf", text=lambda d: d["text"].str.replace("e", "3"))
    tables = {"wikipedia/gate.parquet": gate, "civil/h2_eval.parquet": civil.iloc[:n].drop(
        columns=IDENTITY_COLUMNS), "civil/bias_eval.parquet": civil.iloc[n:],
        "obfuscation.parquet": obf}
    for rel, frame in tables.items():
        (data / rel).parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(data / rel, index=False)
    return data


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    from src.pipeline.run import run
    from src.student.synthetic import build

    root = tmp_path_factory.mktemp("pipeline")
    build(root, n=400)
    data = write_eval_tables(root)
    records = {}
    for name, extra in (("distilbert-gold", ["loss.alpha=1.0"]),
                        ("distilbert-mixed", ["loss.alpha=0.5"])):
        cfg = smoke_cfg(root, *FAST, f"pipeline.data_dir={data}", f"run_name={name}", *extra)
        records[name] = run(cfg)
    return root, records


def test_every_step_runs_and_writes_its_files(two_runs):
    _, records = two_runs
    for record in records.values():
        errors = {k: v for k, v in record["errors"].items() if not k.endswith("_traceback")}
        assert errors == {}, errors
        run_dir = Path(record["run_dir"])
        for stem in ("val", "gate", "h2_eval", "bias_eval", "obfuscation"):
            assert (run_dir / "predictions" / f"{stem}.parquet").exists(), stem
            gold = pd.read_parquet(run_dir / "gold" / f"{stem}.parquet")
            assert "text" not in gold.columns and set(LABELS) <= set(gold.columns)
        assert (run_dir / "predictions" / "thresholds.json").exists()
        assert set(IDENTITY_COLUMNS) <= set(pd.read_parquet(run_dir / "gold" / "bias_eval.parquet"))
        metrics = json.loads((run_dir / "eval" / "metrics.json").read_text())
        assert metrics["h2_eval"]["labels"] == [x for x in LABELS if x != "severe_toxicity"]
        report = json.loads((run_dir / "eval" / "gate_report.json").read_text())
        assert report["bias"]["slices"] and "passed" in report
        latency = json.loads((run_dir / "latency.json").read_text())
        assert latency["batch_size"] == 1 and latency["p95_ms"] > 0
        for name in ("confusion_gate.png", "confusion_h2_eval.png", "per_label_auc.png"):
            assert (run_dir / "plots" / name).stat().st_size > 1000, name
        pipeline = json.loads((run_dir / "pipeline.json").read_text())
        assert pipeline["data"]["rows"]["gate"] == 240


def test_prediction_files_keep_the_id_order_and_have_no_text(two_runs):
    root, records = two_runs
    run_dir = Path(records["distilbert-mixed"]["run_dir"])
    pred = pd.read_parquet(run_dir / "predictions" / "gate.parquet")
    gate = pd.read_parquet(root / "processed" / "wikipedia" / "gate.parquet")
    assert pred["id"].tolist() == gate["id"].tolist()
    assert list(pred.columns) == ["id", *[f"p_{k}" for k in LABELS], "model_version"]


def test_selection_follows_the_rule(two_runs, tmp_path):
    from src.student import select_best
    from src.student.config import load_config

    _, records = two_runs
    runs = [r["run_dir"] for r in records.values()]
    cfg = load_config("configs/smoke.yaml", ["select.latency_budget_ms=100000"])
    result = select_best.run(runs, tmp_path, cfg)
    eligible = [c for c in result["candidates"] if c["eligible"]]
    if eligible:
        best = max(c["val_mean_auc"] for c in eligible)
        assert result["selected"]["val_mean_auc"] >= best - cfg["select"]["auc_tie"]
    assert (tmp_path / "selection.md").read_text().startswith("# Student selection")
    # A budget nobody meets: no candidate, and each says why.
    none = select_best.run(runs, tmp_path / "strict",
                           load_config("configs/smoke.yaml", ["select.latency_budget_ms=0"]))
    assert none["selected"] is None
    assert all(any("latency" in r for r in c["reasons"]) for c in none["candidates"])


def test_select_breaks_ties_by_pr_auc():
    from src.student.select_best import select

    def cand(arm, auc, pr, ms=10.0, bias=True):
        return {"arm": arm, "val_mean_auc": auc, "val_mean_pr_auc": pr, "bias_passed": bias,
                "package_ok": True, "latency": {"p95_ms": ms}}

    rule = {"latency_stat": "p95_ms", "latency_budget_ms": 50, "auc_tie": 0.0002}
    result = select([cand("a", 0.99360, 0.760), cand("b", 0.99350, 0.770),
                     cand("c", 0.99400, 0.700, bias=False), cand("d", 0.99390, 0.790, ms=80)],
                    rule)
    assert result["selected"]["arm"] == "b"            # a and b tie; c and d are out
    assert sorted(result["tied"]) == ["a", "b"]


def test_results_across_arms(two_runs, tmp_path):
    from src.eval.results import build

    _, records = two_runs
    result = build([r["run_dir"] for r in records.values()], tmp_path, n_boot=20, workers=2,
                   selected="distilbert-mixed")
    assert list(result["arms"]) == ["distilbert-gold", "distilbert-mixed"]
    mixed = result["arms"]["distilbert-mixed"]
    assert mixed["ci"]["gate"]["mean_auc"]["replicates"] == 20
    assert "distilbert-mixed - distilbert-gold" in result["pairs"]
    assert result["leak_check"] is not None and mixed["obfuscation"]["rows"] == 30
    text = (tmp_path / "results.md").read_text()
    assert "## H1: in-domain" in text and "## H2: cross-domain" in text
    for name in ("h1_h2.png", "per_label_auc_gate.png", "per_label_pr_auc_h2_eval.png",
                 "confusion_distilbert-mixed_gate.png"):
        assert (tmp_path / "plots" / name).stat().st_size > 1000, name


def test_truncation_sweep_writes_one_file_per_length(two_runs):
    from src.eval.results import truncation
    from src.student.evaluate import truncation_sweep

    root, records = two_runs
    run_dir = Path(records["distilbert-mixed"]["run_dir"])
    gate = pd.read_parquet(root / "processed" / "wikipedia" / "gate.parquet")
    written = truncation_sweep(run_dir, gate, lengths=(8, 16), batch_size=32)
    assert sorted(written) == ["16", "8"]
    table = truncation(run_dir)
    assert list(table) == ["8", "16"] and 0 <= table["8"]["mean_auc"] <= 1
