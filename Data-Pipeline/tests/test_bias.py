"""Tests for scripts/bias.py.

Covers the plan's three required checks (word-boundary lexicon matching, per-group
metrics, reweighting touches only non-toxic identity rows) plus factor selection and an
end-to-end run on synthetic, contract-shaped data with a planted bias. No network.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts import bias
from scripts.columns import IDENTITY_COLUMNS, LABELS

# ---------------------------------------------------------------------------
# Synthetic contract-shaped data (CONTRACTS.md section 2.2)
# ---------------------------------------------------------------------------

NEUTRAL = ["thanks", "edit", "article", "citation", "please", "discuss", "source", "page",
           "good", "point", "agree", "history", "section", "reference", "update"]
TOXIC = ["idiot", "stupid", "moron", "trash", "pathetic", "loser"]
# Term in the text -> Civil identity column it sets.
IDENTITY = {"muslim": "muslim", "gay": "homosexual_gay_or_lesbian", "black": "black",
            "women": "female", "jewish": "jewish"}


def _comment(rng: np.random.Generator, toxic: bool, identity: str | None) -> str:
    words = list(rng.choice(NEUTRAL, size=8))
    if toxic and rng.random() < 0.7:  # some toxic rows lack a toxic word, like real data
        words += list(rng.choice(TOXIC, size=2))
    if identity:
        words.append(identity)
    rng.shuffle(words)
    return " ".join(words)


def _rows(n: int, seed: int, identity_toxic_rate: float):
    """Identity rows are toxic at ``identity_toxic_rate``, others at 0.15 (the planted bias)."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        identity = str(rng.choice(list(IDENTITY))) if rng.random() < 0.3 else None
        toxic = rng.random() < (identity_toxic_rate if identity else 0.15)
        yield i, identity, toxic, _comment(rng, toxic, identity), rng


def make_wikipedia(n: int, seed: int, identity_toxic_rate: float) -> pd.DataFrame:
    records = []
    for i, identity, toxic, text, rng in _rows(n, seed, identity_toxic_rate):
        labels = {label: 0 for label in LABELS}
        labels["toxicity"] = labels["insult"] = int(toxic)
        records.append({"id": f"w{seed}_{i:05d}", "text": text, **labels,
                        "char_len": len(text), "has_identity_term": identity is not None,
                        "has_obfuscation": bool(rng.random() < 0.1)})
    df = pd.DataFrame(records)
    for label in LABELS:
        df[label] = df[label].astype("int8")
    df["char_len"] = df["char_len"].astype("int32")
    df["sample_weight"] = np.float32(1.0)
    return df


def make_civil(n: int, seed: int) -> pd.DataFrame:
    records = []
    for i, identity, toxic, text, _ in _rows(n, seed, identity_toxic_rate=0.15):
        labels = {label: 0 for label in LABELS}
        labels["toxicity"] = int(toxic)
        identities = {c: 0.0 for c in IDENTITY_COLUMNS}
        if identity:
            identities[IDENTITY[identity]] = 1.0
        records.append({"id": str(100000 + i), "text": text, **labels,
                        **{f"{k}_frac": float(v) for k, v in labels.items()}, **identities})
    df = pd.DataFrame(records)
    for label in LABELS:
        df[label] = df[label].astype("int8")
    df.loc[df.index % 17 == 0, "muslim"] = np.nan  # unannotated values must not count
    return df


def make_cfg(tmp_path, lexicon) -> dict:
    return {
        "seed": 42,
        "labels": {"names": LABELS},
        "paths": {"processed_dir": str(tmp_path / "processed"),
                  "reports_dir": str(tmp_path / "reports"),
                  "summaries_dir": str(tmp_path / "reports" / "summaries"),
                  "identity_terms": str(lexicon)},
        "bias": {"label": "toxicity", "min_slice_size": 30, "fpr_flag_margin_pts": 5.0,
                 "reweight_factors": [1, 2, 3, 5], "max_auc_drop": 0.005,
                 "identity_threshold": 0.5, "length_bands": [60, 75],
                 "tfidf": {"max_features": 5000, "ngram_max": 2, "min_df": 1},
                 "logreg": {"C": 4.0, "max_iter": 1000}},
    }


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------

def test_load_lexicon_skips_comments_and_blanks(tmp_path):
    path = tmp_path / "terms.txt"
    path.write_text("# gender\nWomen\n\n  muslim \n# religion\nmentally ill\n", encoding="utf-8")
    assert sorted(bias.load_lexicon(path)) == ["mentally ill", "muslim", "women"]


def test_word_boundary_matching():
    pattern = bias.compile_lexicon(["muslim", "trans", "gay", "mentally ill"])
    texts = pd.Series(["Muslim people live here", "public transport is late", "muslimx",
                       "she is TRANS", "he seems mentally   ill", "a gayety of colors",
                       "gay.", None])
    expected = [True, False, False, True, True, False, True, False]
    assert bias.mentions_identity(texts, pattern).tolist() == expected


# ---------------------------------------------------------------------------
# Per-group metrics
# ---------------------------------------------------------------------------

def _hand_example():
    gold = pd.DataFrame({"id": [str(i) for i in range(10)],
                         "toxicity": [0, 0, 0, 0, 1, 0, 0, 0, 1, 1]})
    preds = bias.predictions_frame(
        gold["id"], np.array([0.9, 0.8, 0.1, 0.2, 0.9, 0.1, 0.1, 0.2, 0.9, 0.3]),
        "toxicity", LABELS)
    slices = {"A": pd.Series([True] * 5 + [False] * 5, index=gold["id"])}
    return gold, preds, slices


def test_per_group_metrics_match_hand_computation():
    gold, preds, slices = _hand_example()
    table, overall = bias.slice_report(preds, gold, slices, threshold=0.5, label="toxicity",
                                       min_size=1, margin_pts=5.0)
    row = table.iloc[0]
    # Slice A: negatives 0-3 predicted [1, 1, 0, 0] -> FPR 0.5; its one positive is caught.
    assert row["n"] == 5
    assert row["toxicity_rate"] == pytest.approx(0.2)
    assert row["fpr"] == pytest.approx(0.5)
    assert row["fnr"] == pytest.approx(0.0)
    assert row["auc"] == pytest.approx(3.5 / 4)  # one tie at 0.9 counts half
    # Overall: 2 false positives of 7 negatives, 1 miss of 3 positives.
    assert overall["fpr"] == pytest.approx(2 / 7)
    assert overall["fnr"] == pytest.approx(1 / 3)
    assert row["fpr_gap_pts"] == pytest.approx((0.5 - 2 / 7) * 100)
    assert row["judged"] and row["flagged"]


def test_small_slices_are_not_judged():
    gold, preds, slices = _hand_example()
    table, _ = bias.slice_report(preds, gold, slices, threshold=0.5, label="toxicity",
                                 min_size=100, margin_pts=5.0)
    assert not table.iloc[0]["judged"]
    assert not table.iloc[0]["flagged"]


def test_prediction_file_must_cover_every_gold_id():
    gold, preds, slices = _hand_example()
    with pytest.raises(ValueError, match="covers 9 of 10"):
        bias.slice_report(preds.iloc[1:], gold, slices, 0.5, "toxicity", 1, 5.0)


def test_civil_slices_use_the_identity_threshold():
    df = pd.DataFrame({"id": ["1", "2", "3", "4"], "muslim": [0.5, 0.4999, np.nan, 0.0],
                       "female": [0.0, 0.0, 0.0, 1.0]})
    slices = bias.civil_slices(df, threshold=0.5)
    assert slices["religion:muslim"].tolist() == [True, False, False, False]
    assert slices["religion:any"].tolist() == [True, False, False, False]
    assert slices["gender:any"].tolist() == [False, False, False, True]


def test_wikipedia_length_bands_cover_every_row():
    df = make_wikipedia(200, seed=1, identity_toxic_rate=0.15)
    slices = bias.wikipedia_slices(df, length_bands=[60, 75])
    bands = [s for name, s in slices.items() if name.startswith("length_")]
    assert sum(int(s.sum()) for s in bands) == len(df)


# ---------------------------------------------------------------------------
# Mitigation
# ---------------------------------------------------------------------------

def test_reweighting_touches_only_nontoxic_identity_rows():
    train = pd.DataFrame({"has_identity_term": [True, True, False, False],
                          "toxicity": [0, 1, 0, 1]})
    weights = bias.reweight(train, 3, "toxicity")
    assert weights.dtype == np.float32
    assert weights.tolist() == [3.0, 1.0, 1.0, 1.0]


def test_select_factor_picks_smallest_passing():
    trials = [{"factor": 1, "val_auc": 0.950, "identity_fpr_gap_pts": 12.0},
              {"factor": 2, "val_auc": 0.949, "identity_fpr_gap_pts": 7.0},
              {"factor": 3, "val_auc": 0.948, "identity_fpr_gap_pts": 3.0},
              {"factor": 5, "val_auc": 0.947, "identity_fpr_gap_pts": 1.0}]
    assert bias.select_factor(trials, margin_pts=5.0, max_auc_drop=0.005) == (3, True)


def test_select_factor_respects_the_auc_drop():
    trials = [{"factor": 1, "val_auc": 0.950, "identity_fpr_gap_pts": 12.0},
              {"factor": 3, "val_auc": 0.940, "identity_fpr_gap_pts": 3.0},
              {"factor": 5, "val_auc": 0.946, "identity_fpr_gap_pts": 2.0}]
    assert bias.select_factor(trials, margin_pts=5.0, max_auc_drop=0.005) == (5, True)


def test_select_factor_falls_back_to_no_reweighting():
    trials = [{"factor": 1, "val_auc": 0.95, "identity_fpr_gap_pts": 12.0},
              {"factor": 5, "val_auc": 0.90, "identity_fpr_gap_pts": 1.0}]
    assert bias.select_factor(trials, margin_pts=5.0, max_auc_drop=0.005) == (1, False)


def test_select_factor_keeps_one_when_there_is_no_bias():
    trials = [{"factor": 1, "val_auc": 0.95, "identity_fpr_gap_pts": 1.0},
              {"factor": 2, "val_auc": 0.95, "identity_fpr_gap_pts": 0.5}]
    assert bias.select_factor(trials, margin_pts=5.0, max_auc_drop=0.005) == (1, True)


def test_choose_threshold_separable_scores():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    threshold = bias.choose_threshold(y, scores)
    assert 0.3 < threshold <= 0.8


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_run_end_to_end(tmp_path):
    wiki = tmp_path / "processed" / "wikipedia"
    civil = tmp_path / "processed" / "civil"
    wiki.mkdir(parents=True)
    civil.mkdir(parents=True)
    # Planted bias: identity mentions are mostly toxic in train, not in validation.
    train = make_wikipedia(1500, seed=1, identity_toxic_rate=0.7)
    make_wikipedia(600, seed=2, identity_toxic_rate=0.15).to_parquet(wiki / "val.parquet")
    train.to_parquet(wiki / "train.parquet")
    make_civil(600, seed=3).to_parquet(civil / "bias_eval.parquet")
    lexicon = tmp_path / "identity_terms.txt"
    lexicon.write_text("\n".join(IDENTITY), encoding="utf-8")

    summary = bias.run(make_cfg(tmp_path, lexicon))

    assert summary["chosen_factor"] in (1, 2, 3, 5)
    assert summary["rows_out"] == len(train)
    json.dumps(summary)  # the summary must be valid JSON for run_summary.json

    written = pd.read_parquet(wiki / "train.parquet")
    assert written["id"].tolist() == train["id"].tolist()
    assert written["sample_weight"].dtype == np.float32
    untouched = ~(written["has_identity_term"] & (written["toxicity"] == 0))
    assert (written.loc[untouched, "sample_weight"] == 1.0).all()
    assert set(written["sample_weight"].unique()) <= {1.0, float(summary["chosen_factor"])}

    report = (tmp_path / "reports" / "bias" / "bias_report.md").read_text(encoding="utf-8")
    assert "## 6. Limits" in report
    metrics = json.loads((tmp_path / "reports" / "bias" / "bias_metrics.json").read_text())
    assert {"civil_before", "civil_after", "trials"} <= metrics.keys()
