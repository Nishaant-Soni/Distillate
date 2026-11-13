"""Tests for src/eval/bias.py (Phase 2 bias analysis) on synthetic files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval import bias
from src.eval.metrics import LABELS

SETTINGS = {"identity_threshold": 0.5, "min_slice_size": 100, "report_flag_pts": 5.0,
            "length_bands": [100, 500]}
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "gate.yaml"


def test_bias_aucs_by_hand():
    # Slice = rows 0-3. Its negatives (0, 1) score high, so BPSN is low.
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    score = np.array([0.9, 0.8, 0.95, 0.85, 0.1, 0.2, 0.7, 0.6])
    in_slice = np.array([True] * 4 + [False] * 4)
    aucs = bias.bias_aucs(y, score, in_slice)
    assert aucs["subgroup_auc"] == pytest.approx(0.75)   # 3 of 4 pairs ordered
    # BPSN: slice negatives {0.9, 0.8} vs rest positives {0.7, 0.6} -> all misordered.
    assert aucs["bpsn_auc"] == pytest.approx(0.0)
    # BNSP: slice positives {0.95, 0.85} vs rest negatives {0.1, 0.2} -> all ordered.
    assert aucs["bnsp_auc"] == pytest.approx(1.0)


def test_slice_table_rates_and_flag():
    y = np.array([0] * 200 + [1] * 50)
    decision = np.zeros(250, dtype=bool)
    in_slice = np.zeros(250, dtype=bool)
    in_slice[:120] = True                  # 120 negatives in the slice
    decision[:24] = True                   # slice FPR 24/120 = 0.20; overall 24/200 = 0.12
    decision[200:240] = True               # 40 of 50 positives caught
    score = decision.astype(float)
    table, overall = bias.slice_table(y, score, decision, {"s": in_slice}, 100, 5.0)
    row = table.iloc[0]
    assert overall["fpr"] == pytest.approx(0.12)
    assert row["fpr"] == pytest.approx(0.20)
    assert row["fpr_gap_pts"] == pytest.approx(8.0)
    assert row["judged"] and row["flagged"]
    small, _ = bias.slice_table(y, score, decision, {"s": in_slice}, 500, 5.0)
    assert not small.iloc[0]["judged"] and not small.iloc[0]["flagged"]


def test_wikipedia_slices_cover_every_row_in_characters():
    gold = pd.DataFrame({"char_len": [5, 99, 100, 499, 500, 2000],
                         "has_identity_term": [True, False] * 3,
                         "has_obfuscation": [False] * 5 + [True]})
    slices = bias.wikipedia_slices(gold, [100, 500])
    bands = [k for k in slices if k.startswith("chars_")]
    assert bands == ["chars_0_99", "chars_100_499", "chars_500+"]
    assert sum(int(slices[k].sum()) for k in bands) == len(gold)
    assert slices["obfuscation_flag"].tolist() == [False] * 5 + [True]


def test_wikipedia_slices_skip_missing_columns():
    assert bias.wikipedia_slices(pd.DataFrame({"id": ["a"]}), [100, 500]) == {}


# ---------------------------------------------------------------------------
# End to end on synthetic arms
# ---------------------------------------------------------------------------

def _civil_gold(n: int = 3000, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gold = pd.DataFrame({"id": [str(100000 + i) for i in range(n)]})
    for k, label in enumerate(LABELS):
        gold[label] = (rng.random(n) < 0.10 + 0.02 * k).astype("int8")
    gold["muslim"] = np.where(rng.random(n) < 0.15, 1.0, 0.0)
    gold["female"] = np.where(rng.random(n) < 0.20, 1.0, 0.0)
    gold.loc[gold.index % 13 == 0, "female"] = np.nan        # unannotated
    return gold


def _val_gold(n: int = 2000, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gold = pd.DataFrame({"id": [f"w{i:05d}" for i in range(n)]})
    for k, label in enumerate(LABELS):
        gold[label] = (rng.random(n) < 0.10 + 0.02 * k).astype("int8")
    gold["char_len"] = rng.integers(10, 1500, n).astype("int32")
    gold["has_identity_term"] = rng.random(n) < 0.2
    gold["has_obfuscation"] = rng.random(n) < 0.05
    return gold


def _pred(gold: pd.DataFrame, seed: int, version: str, biased: str | None = None):
    rng = np.random.default_rng(seed)
    pred = pd.DataFrame({"id": gold["id"], "model_version": version})
    for label in LABELS:
        z = (gold[label].to_numpy() * 2 - 1) * 2.0 + rng.normal(0, 1, len(gold))
        if biased and label == "toxicity" and biased in gold:
            z = z + 3.0 * (gold[biased].fillna(0).to_numpy() >= 0.5)
        pred[f"p_{label}"] = 1 / (1 + np.exp(-z))
    return pred


def write_arm(folder: Path, gold_dir: Path, version: str, biased: str | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    civil, val = pd.read_parquet(gold_dir / "bias_eval.parquet"), \
        pd.read_parquet(gold_dir / "val.parquet")
    _pred(civil, 1, version, biased).to_parquet(folder / "bias_eval.parquet")
    _pred(val, 2, version).to_parquet(folder / "val.parquet")
    (folder / "thresholds.json").write_text(json.dumps({lab: 0.5 for lab in LABELS}))
    return folder


@pytest.fixture
def gold_dir(tmp_path):
    folder = tmp_path / "gold"
    folder.mkdir()
    _civil_gold().to_parquet(folder / "bias_eval.parquet")
    _val_gold().to_parquet(folder / "val.parquet")
    return folder


def test_biased_arm_is_flagged_and_fair_arm_is_not(tmp_path, gold_dir):
    fair = bias.analyze_arm(write_arm(tmp_path / "fair", gold_dir, "fair"), gold_dir, SETTINGS)
    skewed = bias.analyze_arm(write_arm(tmp_path / "skew", gold_dir, "skew", "muslim"),
                              gold_dir, SETTINGS)
    fair_t = fair["civil"]["toxicity"]["table"].set_index("slice")
    skew_t = skewed["civil"]["toxicity"]["table"].set_index("slice")
    assert not fair_t.loc["religion:muslim", "flagged"]
    assert skew_t.loc["religion:muslim", "flagged"]
    assert skew_t.loc["religion:muslim", "bpsn_auc"] < fair_t.loc["religion:muslim", "bpsn_auc"]
    assert skewed["model_version"] == "skew"


def test_wikipedia_tables_use_the_lexicon_slices(tmp_path, gold_dir):
    res = bias.analyze_arm(write_arm(tmp_path / "a", gold_dir, "a"), gold_dir, SETTINGS)
    slices = set(res["wikipedia"]["toxicity"]["table"]["slice"])
    assert {"identity_mention", "chars_0_99", "obfuscation_flag"} <= slices


def test_several_civil_labels(tmp_path, gold_dir):
    res = bias.analyze_arm(write_arm(tmp_path / "a", gold_dir, "a"), gold_dir, SETTINGS,
                           labels=("toxicity", "insult"))
    assert set(res["civil"]) == {"toxicity", "insult"}


def test_cli_writes_tables_json_and_plot(tmp_path, gold_dir):
    pytest.importorskip("matplotlib")
    before = write_arm(tmp_path / "before", gold_dir, "before", "muslim")
    after = write_arm(tmp_path / "after", gold_dir, "after")
    out = tmp_path / "out"
    assert bias.main(["--gold", str(gold_dir), "--arm", f"mixed={before}",
                      "--arm", f"weighted={after}", "--before", "mixed", "--after", "weighted",
                      "--config", str(CONFIG), "--plots", "--out", str(out)]) == 0
    tables = (out / "bias_tables.md").read_text()
    assert "## Arms compared" in tables and "religion:muslim" in tables
    data = json.loads((out / "bias_phase2.json").read_text())
    assert set(data) == {"mixed", "weighted"}
    assert list((out / "plots").glob("fpr_by_slice_toxicity_*.png"))


def test_cli_rejects_severe_toxicity(tmp_path, gold_dir):
    arm = write_arm(tmp_path / "a", gold_dir, "a")
    with pytest.raises(SystemExit):
        bias.main(["--gold", str(gold_dir), "--arm", f"a={arm}",
                   "--labels", "severe_toxicity", "--out", str(tmp_path / "o")])


# ---------------------------------------------------------------------------
# Review follow-ups: FPR ratio, warnings, matched FPR, paired BPSN
# ---------------------------------------------------------------------------

def test_fpr_ratio_column():
    y = np.array([0] * 200 + [1] * 50)
    decision = np.zeros(250, dtype=bool)
    in_slice = np.zeros(250, dtype=bool)
    in_slice[:120] = True
    decision[:24] = True                   # slice FPR 0.20, overall 0.12
    table, _ = bias.slice_table(y, decision.astype(float), decision, {"s": in_slice}, 100, 5.0)
    assert table.iloc[0]["fpr_ratio"] == pytest.approx(0.20 / 0.12)


def test_rank_auc_matches_sklearn_with_ties():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    score = np.round(rng.random(300), 1)   # many ties
    assert bias._rank_auc(y, score) == pytest.approx(roc_auc_score(y, score))


def test_missing_wikipedia_columns_warn(tmp_path, gold_dir):
    arm = write_arm(tmp_path / "a", gold_dir, "a")
    _val_gold().drop(columns=["char_len", "has_obfuscation"]).to_parquet(gold_dir / "val.parquet")
    res = bias.analyze_arm(arm, gold_dir, SETTINGS)
    assert any("char_len" in w and "has_obfuscation" in w for w in res["warnings"])
    assert "identity_mention" in set(res["wikipedia"]["toxicity"]["table"]["slice"])


def test_missing_civil_file_warns(tmp_path, gold_dir):
    arm = write_arm(tmp_path / "a", gold_dir, "a")
    (arm / "bias_eval.parquet").unlink()
    res = bias.analyze_arm(arm, gold_dir, SETTINGS)
    assert res["civil"] == {}
    assert any(w.startswith("civil section skipped") for w in res["warnings"])


def test_matched_fpr_hits_the_target(tmp_path, gold_dir):
    res = bias.analyze_arm(write_arm(tmp_path / "a", gold_dir, "a"), gold_dir, SETTINGS,
                           matched_fpr=0.05)
    matched = res["civil_matched"]["toxicity"]
    assert matched["overall"]["fpr"] == pytest.approx(0.05, abs=0.005)
    assert set(matched["table"]["slice"]) == set(res["civil"]["toxicity"]["table"]["slice"])


def test_paired_bpsn_same_arm_is_zero(tmp_path, gold_dir):
    arm = write_arm(tmp_path / "a", gold_dir, "a")
    out = bias.paired_bpsn(arm, arm, gold_dir, SETTINGS, n_boot=20)
    assert out["slices"]
    for r in out["slices"].values():
        assert r["diff"] == pytest.approx(0.0)
        assert not r["excludes_zero"]


def test_paired_bpsn_detects_a_worse_arm(tmp_path, gold_dir):
    fair = write_arm(tmp_path / "fair", gold_dir, "fair")
    skew = write_arm(tmp_path / "skew", gold_dir, "skew", "muslim")
    out = bias.paired_bpsn(skew, fair, gold_dir, SETTINGS, n_boot=100, seed=1)
    muslim = out["slices"]["religion:muslim"]
    assert muslim["diff"] < 0 and muslim["excludes_zero"]


def test_cli_matched_and_paired_sections(tmp_path, gold_dir):
    fair = write_arm(tmp_path / "fair", gold_dir, "fair")
    skew = write_arm(tmp_path / "skew", gold_dir, "skew", "muslim")
    out = tmp_path / "out"
    assert bias.main(["--gold", str(gold_dir), "--arm", f"fair={fair}", "--arm", f"skew={skew}",
                      "--config", str(CONFIG), "--matched-fpr", "0.05",
                      "--compare", "skew", "fair", "--n-boot", "30", "--out", str(out)]) == 0
    tables = (out / "bias_tables.md").read_text()
    assert "## Arms compared at matched overall FPR (5%)" in tables
    assert "## Paired bootstrap: BPSN AUC, skew minus fair" in tables
    assert "FPR ratio" in tables
    assert "skew-fair" in json.loads((out / "bias_phase2.json").read_text())["paired_bpsn"]


def test_cli_rejects_unknown_compare_arm(tmp_path, gold_dir):
    arm = write_arm(tmp_path / "a", gold_dir, "a")
    with pytest.raises(SystemExit):
        bias.main(["--gold", str(gold_dir), "--arm", f"a={arm}", "--compare", "a", "zzz",
                   "--out", str(tmp_path / "o")])
