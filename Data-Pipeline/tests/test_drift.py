"""Tests for scripts/drift.py: PSI, the hashed baseline, input and flag-rate drift."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts import drift

WORDS = ["the", "article", "edit", "source", "page", "thanks", "please", "revert", "talk",
         "section", "citation", "policy", "editor", "change", "fixed", "review"]


def _comments(n, seed, n_words=(5, 30)):
    """Made-up comments drawn from a small fixed vocabulary."""
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(WORDS, size=rng.integers(*n_words))) for _ in range(n)]


def _frame(texts):
    return pd.DataFrame({"text": texts, "char_len": [len(t) for t in texts]})


@pytest.fixture
def cfg(tmp_path):
    source = tmp_path / "train.parquet"
    _frame(_comments(2000, seed=1)).to_parquet(source, index=False)
    return {
        "paths": {"summaries_dir": str(tmp_path / "summaries")},
        "drift": {
            "baseline_source": str(source),
            "baseline_file": str(tmp_path / "monitoring" / "drift_baseline.json"),
            "psi_bins": 10, "vocab_min_count": 2,
            "psi_warn": 0.1, "psi_alert": 0.25, "psi_epsilon": 1e-4,
        },
    }


def test_psi_is_zero_for_identical_distributions():
    shares = [0.1] * 10
    assert drift.psi(shares, shares, 1e-4) == pytest.approx(0.0)


def test_psi_grows_with_the_shift_and_stays_finite_with_empty_bins():
    baseline = [0.5, 0.5, 0.0]
    small = drift.psi(baseline, [0.45, 0.55, 0.0], 1e-4)
    large = drift.psi(baseline, [0.0, 0.2, 0.8], 1e-4)
    assert 0 < small < large and np.isfinite(large)


def test_status_uses_the_thresholds():
    assert drift.status(0.05, 0.1, 0.25) == "ok"
    assert drift.status(0.1, 0.1, 0.25) == "warn"
    assert drift.status(0.3, 0.1, 0.25) == "alert"
    assert drift.status(None, 0.1, 0.25) == "unknown"


def test_features_on_known_text():
    texts = pd.Series(["you are 1d10t", "hello world", "!!!", ""])
    assert drift.words(texts).tolist() == [["you", "are", "d", "t"], ["hello", "world"], [], []]
    share = drift.nonalpha_share(texts)
    assert share[1] == 0.0 and share[2] == 1.0 and share[3] == 0.0
    assert share[0] == pytest.approx(3 / 11)  # "1", "1", "0" out of 11 non-space characters


def test_oov_rate_counts_unknown_words():
    tokens = drift.words(pd.Series(["edit the page", "edit zorblax quuxle", "123"]))
    rate = drift.oov_rate(tokens, {"edit", "the", "page"}, hashed=False)
    assert rate.tolist() == pytest.approx([0.0, 2 / 3, 0.0])


def _string_values(obj):
    """Every string value in a JSON-like object (keys excluded)."""
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _string_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _string_values(value)
    elif isinstance(obj, str):
        yield obj


def test_baseline_stores_no_words(cfg):
    baseline = drift.build_baseline(cfg)
    assert baseline["vocab_size"] == len(WORDS)
    stored = set(_string_values(baseline))
    assert not stored & set(WORDS)  # only hashes, never the words
    assert len(baseline["vocab_hashes"]) == len(WORDS)
    assert set(baseline["features"]) == set(drift.INPUT_FEATURES)
    for feature in baseline["features"].values():
        assert sum(feature["shares"]) == pytest.approx(1.0)


def test_baseline_is_deterministic(cfg):
    assert drift.build_baseline(cfg) == drift.build_baseline(cfg)


def test_load_baseline_builds_it_when_missing(cfg):
    baseline = drift.load_baseline(cfg)
    assert drift._resolve(cfg["drift"]["baseline_file"]).exists()
    assert baseline["_vocab"] == set(baseline["vocab_hashes"])


def test_load_baseline_without_building_gives_a_clear_error(cfg):
    with pytest.raises(FileNotFoundError, match="python -m scripts.drift"):
        drift.load_baseline(cfg, build_if_missing=False)


def test_batch_like_training_data_has_no_drift(cfg):
    baseline = drift.load_baseline(cfg)
    result = drift.input_drift(_frame(_comments(2000, seed=2)), baseline, cfg)
    assert result["status"] == "ok", result
    assert result["max_psi"] < cfg["drift"]["psi_warn"]


def test_shifted_batch_raises_alerts_on_each_feature(cfg):
    baseline = drift.load_baseline(cfg)
    rng = np.random.default_rng(3)
    unknown = [f"zz{i}q" for i in range(200)]
    texts = [" ".join(rng.choice(unknown, size=60)) + " 1d10t!!! 4ss" for _ in range(2000)]
    result = drift.input_drift(_frame(texts), baseline, cfg)
    assert result["status"] == "alert"
    for name in drift.INPUT_FEATURES:
        assert result["features"][name]["status"] == "alert", (name, result)


def test_input_drift_works_without_a_char_len_column(cfg):
    baseline = drift.load_baseline(cfg)
    batch = pd.DataFrame({"text": _comments(500, seed=4)})
    assert drift.input_drift(batch, baseline, cfg)["rows"] == 500


def test_flag_rate_drift(cfg):
    same = drift.flag_rate_drift(pd.DataFrame({"flag": [True] * 10 + [False] * 90}), 0.1, cfg)
    assert same["psi"] == pytest.approx(0.0) and same["status"] == "ok"
    shifted = drift.flag_rate_drift(pd.DataFrame({"flag": [True] * 50 + [False] * 50}), 0.1, cfg)
    assert shifted["status"] == "alert" and shifted["flag_rate"] == 0.5
    unknown = drift.flag_rate_drift(pd.DataFrame({"flag": [True, False]}), None, cfg)
    assert unknown["psi"] is None and unknown["status"] == "unknown"


def test_results_are_json_ready(cfg):
    baseline = drift.load_baseline(cfg)
    json.dumps(drift.input_drift(_frame(_comments(100, seed=5)), baseline, cfg))
    json.dumps(drift.flag_rate_drift(pd.DataFrame({"flag": [True, False]}), 0.3, cfg))


def test_run_writes_the_baseline_and_a_summary(cfg):
    summary = drift.run(cfg)
    assert summary["rows_in"] == 2000 and summary["vocab_size"] == len(WORDS)
    assert drift._resolve(cfg["drift"]["baseline_file"]).exists()
