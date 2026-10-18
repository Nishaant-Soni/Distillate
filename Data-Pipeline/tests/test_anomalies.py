"""Tests for scripts/anomalies.py: each check fires on bad input and passes on clean input."""

import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from scripts import anomalies  # noqa: E402
from scripts.stubs import civil_frame, wikipedia_frame  # noqa: E402
from scripts.utils import load_config  # noqa: E402

N = 300
RAW_ROWS = 50


def _wiki(prefix, n=N):
    return wikipedia_frame(n, 42, prefix)


def _clean_sets():
    """Contract-shaped outputs with distinct ids and identical train/val prevalence."""
    train = _wiki("wiki_train")
    val = train.assign(id=train["id"].str.replace("wiki_train", "wiki_val"))
    train = train.assign(sample_weight=pd.Series(1.0, index=train.index, dtype="float32"))
    obf = _wiki("wiki_gate", 40)
    return {
        "wikipedia/train": train,
        "wikipedia/val": val,
        "wikipedia/gate": _wiki("wiki_gate"),
        "wikipedia/canary_wikipedia": _wiki("wiki_canary", 50),
        "obfuscation": obf.assign(id=obf["id"] + "_obf"),
        "civil/h2_eval": civil_frame(N, 42, "civil_test"),
        "civil/canary_civil": civil_frame(N, 42, "civil_validation"),
        # Identities test is a subset of Civil test, so it shares ids with h2_eval.
        "civil/bias_eval": civil_frame(80, 42, "civil_test", with_identities=True),
        **{f"civil/drift/batch_{b:02d}": civil_frame(10, 42, f"civil_train_{b:02d}")
           for b in range(20)},
    }


def _write(cfg, rel, df):
    path = Path(cfg["paths"]["processed_dir"]) / f"{rel}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config()
    for key in ("raw_dir", "processed_dir", "summaries_dir"):
        cfg["paths"][key] = str(tmp_path / key)
    for dataset in cfg["datasets"].values():
        dataset["expected_rows"] = {split: RAW_ROWS for split in dataset["expected_rows"]}
    raw = Path(cfg["paths"]["raw_dir"])
    for rel in anomalies.RAW_FILES.values():
        (raw / rel).parent.mkdir(parents=True, exist_ok=True)
        _wiki("raw", RAW_ROWS).to_parquet(raw / rel, index=False)
    for rel, df in _clean_sets().items():
        _write(cfg, rel, df)
    return cfg


def _result(summary, check, target=None):
    rows = [r for r in summary["results"] if r["check"] == check
            and (target is None or r["target"] == target)]
    assert rows, f"no result for {check} {target}"
    return rows


def _failed_run(cfg):
    with pytest.raises(anomalies.AnomalyError):
        anomalies.run(cfg)
    path = Path(cfg["paths"]["summaries_dir"]) / "anomalies.json"
    return json.loads(path.read_text())


def test_clean_data_passes_every_check(cfg):
    summary = anomalies.run(cfg)
    assert summary["checks"]["fail"] == 0
    assert summary["checks"]["warn"] == 0
    assert all(r["passed"] for r in summary["results"])


def test_raw_row_count_mismatch_fails(cfg):
    cfg["datasets"]["wikipedia"]["expected_rows"]["train"] = RAW_ROWS + 1
    summary = _failed_run(cfg)
    assert not _result(summary, "raw_row_counts", "wikipedia/train.parquet")[0]["passed"]


def test_raw_counts_are_skipped_while_acquire_is_a_stub(cfg):
    summaries = Path(cfg["paths"]["summaries_dir"])
    summaries.mkdir(parents=True, exist_ok=True)
    (summaries / "acquire.json").write_text(json.dumps({"mode": "stub"}))
    cfg["datasets"]["wikipedia"]["expected_rows"]["train"] = 159_571
    summary = anomalies.run(cfg)
    assert "skipped" in _result(summary, "raw_row_counts")[0]["detail"]


def test_id_overlap_between_sets_fails(cfg):
    leaked = _wiki("wiki_gate", 5)  # gate ids inside train: test data leaking into training
    _write(cfg, "wikipedia/train", pd.concat([_clean_sets()["wikipedia/train"],
                                              leaked.assign(sample_weight=1.0)]))
    summary = _failed_run(cfg)
    assert not _result(summary, "id_overlap", "gate & train")[0]["passed"]


def test_allowed_overlap_does_not_fail(cfg):
    summary = anomalies.run(cfg)
    row = _result(summary, "id_overlap", "bias_eval & h2_eval")[0]
    assert row["passed"] and "allowed" in row["detail"]
    assert "80 shared" in row["detail"]


def test_missing_text_above_limit_fails(cfg):
    df = _clean_sets()["civil/h2_eval"].copy()
    df.loc[:9, "text"] = ""  # 10 of 300 rows, far above 0.1%
    _write(cfg, "civil/h2_eval", df)
    summary = _failed_run(cfg)
    assert not _result(summary, "missing_text_or_labels", "h2_eval")[0]["passed"]


def test_missing_text_below_limit_warns_without_failing(cfg):
    df = civil_frame(2000, 42, "civil_test")
    df.loc[0, "text"] = ""  # 1 of 2,000 rows = 0.05%, under the 0.1% limit
    _write(cfg, "civil/h2_eval", df)
    summary = anomalies.run(cfg)
    assert summary["checks"]["fail"] == 0
    row = _result(summary, "missing_text_or_labels", "h2_eval")[0]
    assert row["severity"] == "warn" and not row["passed"]


def test_missing_label_fails(cfg):
    df = _clean_sets()["wikipedia/gate"].copy()
    df["threat"] = df["threat"].astype("float32")
    df.loc[:4, "threat"] = None
    _write(cfg, "wikipedia/gate", df)
    summary = _failed_run(cfg)
    assert not _result(summary, "missing_text_or_labels", "gate")[0]["passed"]


def test_out_of_domain_label_fails(cfg):
    df = _clean_sets()["wikipedia/val"].copy()
    df.loc[0, "insult"] = 2
    _write(cfg, "wikipedia/val", df)
    summary = _failed_run(cfg)
    assert not _result(summary, "label_domain", "val")[0]["passed"]


def test_fraction_outside_zero_one_fails(cfg):
    df = _clean_sets()["civil/canary_civil"].copy()
    df.loc[0, "toxicity_frac"] = 1.5
    _write(cfg, "civil/canary_civil", df)
    summary = _failed_run(cfg)
    assert not _result(summary, "label_domain", "canary_civil")[0]["passed"]


def test_duplicate_ids_fail(cfg):
    df = _clean_sets()["wikipedia/canary_wikipedia"]
    _write(cfg, "wikipedia/canary_wikipedia", pd.concat([df, df.head(3)]))
    summary = _failed_run(cfg)
    assert not _result(summary, "duplicate_ids", "canary_wikipedia")[0]["passed"]


def test_long_text_warns_without_failing(cfg):
    df = _clean_sets()["wikipedia/gate"].copy()
    df.loc[0, "text"] = "x" * (cfg["preprocess"]["max_char_len_flag"] + 1)
    _write(cfg, "wikipedia/gate", df)
    summary = anomalies.run(cfg)
    assert summary["checks"]["warn"] == 1
    assert not _result(summary, "long_text", "gate")[0]["passed"]


def test_prevalence_gap_warns_without_failing(cfg):
    df = _clean_sets()["wikipedia/val"].copy()
    df["toxicity"] = pd.Series(1, index=df.index, dtype="int8")  # 100% vs about 10%
    _write(cfg, "wikipedia/val", df)
    summary = anomalies.run(cfg)
    assert not _result(summary, "prevalence_gap")[0]["passed"]
    assert summary["checks"]["fail"] == 0


def test_unparseable_created_date_fails(cfg):
    df = _clean_sets()["civil/bias_eval"].copy()
    df["created_date"] = df["created_date"].astype(str)
    df.loc[0, "created_date"] = "not a date"
    _write(cfg, "civil/bias_eval", df)
    summary = _failed_run(cfg)
    assert not _result(summary, "created_date", "bias_eval")[0]["passed"]


def test_timestamps_without_fractional_seconds_still_parse(cfg):
    df = _clean_sets()["civil/h2_eval"].copy()
    df["created_date"] = "2017-01-14 06:43:20.123456+00"
    df.loc[0, "created_date"] = "2017-01-14 06:43:20+00"  # the real Kaggle quirk
    _write(cfg, "civil/h2_eval", df)
    summary = anomalies.run(cfg)
    assert _result(summary, "created_date", "h2_eval")[0]["passed"]


def test_missing_output_file_fails(cfg):
    (Path(cfg["paths"]["processed_dir"]) / "obfuscation.parquet").unlink()
    summary = _failed_run(cfg)
    assert not _result(summary, "file_present", "obfuscation.parquet")[0]["passed"]


def test_every_check_runs_even_after_a_failure(cfg):
    df = _clean_sets()["wikipedia/val"].copy()
    df.loc[0, "insult"] = 2
    _write(cfg, "wikipedia/val", df)
    summary = _failed_run(cfg)
    checks = {r["check"] for r in summary["results"]}
    assert {"raw_row_counts", "id_overlap", "missing_text_or_labels", "label_domain",
            "duplicate_ids", "long_text", "prevalence_gap", "created_date"} <= checks
