"""Tests for scripts/subsets.py: Civil H2 eval, canary, bias set, and drift batches."""

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("sklearn")

from scripts import subsets  # noqa: E402
from scripts.preprocess import civil_paths  # noqa: E402
from scripts.stubs import IDENTITY_COLUMNS, LABELS, civil_frame  # noqa: E402
from scripts.utils import load_config  # noqa: E402

N_ROWS = 1000
H2, CANARY, DRIFT_TOTAL, BATCHES = 300, 100, 400, 20


def _write_inputs(cfg, train=None):
    civil = Path(cfg["paths"]["processed_dir"]) / "civil"
    civil.mkdir(parents=True, exist_ok=True)
    frames = {
        "validation_clean": civil_frame(N_ROWS, 42, "civil_validation"),
        "test_clean": civil_frame(N_ROWS, 42, "civil_test"),
        "identities_test_clean": civil_frame(200, 42, "civil_identities", with_identities=True),
    }
    for name, df in frames.items():
        df.to_parquet(civil / f"{name}.parquet", index=False)
    # Train goes wherever preprocess puts it, which is the cache, not processed.
    train_out = civil_paths(cfg, "train")[1]
    train_out.parent.mkdir(parents=True, exist_ok=True)
    (train if train is not None else civil_frame(N_ROWS, 42, "civil_train")).to_parquet(
        train_out, index=False)


@pytest.fixture
def cfg(tmp_path):
    """Config pointed at a temp dir, with small subset sizes and contract-shaped inputs."""
    cfg = load_config()
    # cache_dir included: preprocess keeps the 1.8M-row Civil train there, and
    # without this the tests read the real 391 MB file instead of the fixture.
    for key in ("processed_dir", "splits_dir", "summaries_dir", "cache_dir"):
        cfg["paths"][key] = str(tmp_path / key)
    cfg["paths"]["civil_exclude_ids"] = str(tmp_path / "exclude_ids.txt")
    # Required file; an empty list (comment only) excludes nothing.
    Path(cfg["paths"]["civil_exclude_ids"]).write_text("# no ids\n", encoding="utf-8")
    cfg["splits"]["civil"].update(
        {"h2_size": H2, "canary_size": CANARY, "drift_total": DRIFT_TOTAL,
         "drift_batches": BATCHES}
    )
    _write_inputs(cfg)
    return cfg


def _civil(cfg):
    return Path(cfg["paths"]["processed_dir"]) / "civil"


def _read(cfg, name):
    return pd.read_parquet(_civil(cfg) / f"{name}.parquet")


def _batches(cfg):
    return [_read(cfg, f"drift/batch_{b:02d}") for b in range(BATCHES)]


def test_subsets_have_the_configured_sizes(cfg):
    subsets.run(cfg)
    assert len(_read(cfg, "h2_eval")) == H2
    assert len(_read(cfg, "canary_civil")) == CANARY
    assert [len(b) for b in _batches(cfg)] == [DRIFT_TOTAL // BATCHES] * BATCHES


def test_bias_set_is_the_full_identities_split(cfg):
    subsets.run(cfg)
    bias = _read(cfg, "bias_eval")
    assert len(bias) == 200
    assert all(col in bias.columns for col in IDENTITY_COLUMNS)


def test_subsets_do_not_overlap(cfg):
    subsets.run(cfg)
    h2 = set(_read(cfg, "h2_eval")["id"])
    canary = set(_read(cfg, "canary_civil")["id"])
    drift = set(pd.concat(_batches(cfg))["id"])
    assert not (h2 & canary) and not (h2 & drift) and not (canary & drift)
    assert len(drift) == DRIFT_TOTAL


def test_drift_batches_are_sorted_by_created_date(cfg):
    subsets.run(cfg)
    stream = pd.concat(_batches(cfg), ignore_index=True)
    assert stream["created_date"].is_monotonic_increasing
    for batch in _batches(cfg):
        assert batch["created_date"].is_monotonic_increasing


def test_drift_is_drawn_across_the_whole_train_period(cfg):
    """Stratified sampling, then sorting, not just the oldest rows of train."""
    shuffled = civil_frame(N_ROWS, 42, "civil_train").sample(frac=1, random_state=3)
    _write_inputs(cfg, train=shuffled)
    subsets.run(cfg)
    stream = pd.concat(_batches(cfg), ignore_index=True)
    train = civil_frame(N_ROWS, 42, "civil_train")
    assert stream["created_date"].max() > train["created_date"].quantile(0.9)


def test_same_seed_gives_the_same_subsets(cfg):
    subsets.run(cfg)
    first = (Path(cfg["paths"]["splits_dir"]) / "civil_drift_ids.csv").read_text()
    h2 = _read(cfg, "h2_eval")["id"].tolist()
    subsets.run(cfg)
    assert (Path(cfg["paths"]["splits_dir"]) / "civil_drift_ids.csv").read_text() == first
    assert _read(cfg, "h2_eval")["id"].tolist() == h2


def test_input_row_order_does_not_change_the_subsets(cfg):
    subsets.run(cfg)
    first = (Path(cfg["paths"]["splits_dir"]) / "civil_drift_ids.csv").read_text()
    _write_inputs(cfg, train=civil_frame(N_ROWS, 42, "civil_train").sample(frac=1,
                                                                           random_state=5))
    subsets.run(cfg)
    assert (Path(cfg["paths"]["splits_dir"]) / "civil_drift_ids.csv").read_text() == first


def test_toxicity_rate_is_preserved_by_stratification(cfg):
    summary = subsets.run(cfg)
    rates = summary["toxicity_pts"]
    assert abs(rates["h2_eval"] - rates["civil_test"]) <= 1.0
    assert abs(rates["canary_civil"] - rates["civil_validation"]) <= 1.0
    assert abs(rates["drift"] - rates["civil_train"]) <= 1.0


def test_subsets_keep_the_contract_columns(cfg):
    subsets.run(cfg)
    for name in ("h2_eval", "canary_civil", "bias_eval", "drift/batch_00"):
        df = _read(cfg, name)
        assert all(str(df[label].dtype) == "int8" for label in LABELS), name
        assert all(f"{label}_frac" in df.columns for label in LABELS), name
        assert str(df["created_date"].dt.tz) == "UTC", name


def test_manifests_match_the_parquet_files(cfg):
    subsets.run(cfg)
    splits = Path(cfg["paths"]["splits_dir"])
    for name in ("h2_eval", "canary_civil", "bias_eval"):
        assert (splits / f"{name}_ids.txt").read_text().split() == _read(cfg, name)["id"].tolist()
    manifest = pd.read_csv(splits / "civil_drift_ids.csv")
    assert list(manifest.columns) == ["id", "batch", "created_date"]
    assert manifest["batch"].tolist() == [b for b in range(BATCHES)
                                          for _ in range(DRIFT_TOTAL // BATCHES)]
    assert manifest["id"].tolist() == pd.concat(_batches(cfg))["id"].tolist()


def test_small_inputs_use_every_row_instead_of_failing(cfg):
    cfg["splits"]["civil"]["h2_size"] = N_ROWS * 2
    summary = subsets.run(cfg)
    assert summary["files"]["h2_eval"] == N_ROWS


def test_summary_records_counts(cfg):
    summary = subsets.run(cfg)
    assert summary["rows_in"]["identities_test"] == 200
    assert summary["files"]["h2_eval"] == H2
    assert summary["rows_out"] == sum(summary["files"].values())
    assert (Path(cfg["paths"]["summaries_dir"]) / "subsets.json").exists()


def test_missing_identities_file_gives_a_clear_error(cfg):
    (_civil(cfg) / "identities_test_clean.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="identities_test_clean.parquet"):
        subsets.run(cfg)


def test_missing_created_date_is_rejected(cfg):
    civil = _civil(cfg)
    civil_frame(N_ROWS, 42, "civil_test").drop(columns="created_date").to_parquet(
        civil / "test_clean.parquet", index=False)
    with pytest.raises(ValueError, match="created_date"):
        subsets.run(cfg)


def test_excluded_ids_never_reach_the_canary_or_drift(cfg):
    validation = civil_frame(N_ROWS, 42, "civil_validation")
    train = civil_frame(N_ROWS, 42, "civil_train")
    excluded = set(validation["id"].head(300)) | set(train["id"].head(50))
    Path(cfg["paths"]["civil_exclude_ids"]).write_text(
        "# teacher shift-check rows\n" + "\n".join(sorted(excluded)) + "\n", encoding="utf-8")
    summary = subsets.run(cfg)
    assert not set(_read(cfg, "canary_civil")["id"]) & excluded
    assert not set(pd.concat(_batches(cfg))["id"]) & excluded
    assert summary["excluded"] == {"ids_listed": 350,
                                   "removed": {"validation": 300, "train": 50}}
    assert summary["dropped"]["teacher_shift_rows"] == 350


def test_empty_exclusion_file_excludes_nothing(cfg):
    summary = subsets.run(cfg)
    assert summary["excluded"] == {"ids_listed": 0, "removed": {"validation": 0, "train": 0}}
    assert "teacher_shift_rows" not in summary["dropped"]


def test_missing_exclusion_file_stops_the_stage(cfg):
    """Running without the list would put teacher prompt-selection rows into evaluation data."""
    Path(cfg["paths"]["civil_exclude_ids"]).unlink()
    with pytest.raises(FileNotFoundError, match="Exclusion list"):
        subsets.run(cfg)


def test_civil_train_is_read_from_the_cache_not_processed(cfg):
    """preprocess writes the 1.8M-row Civil train to cache_dir; subsets must follow it.

    Regression: subsets looked under processed_dir. That path happened to hold a
    200-row stub left over from an earlier run, so the stage silently built the
    40,000-row drift stream from 200 rows of synthetic text and the DAG still
    went green.
    """
    cached = civil_paths(cfg, "train")[1]
    assert cached.exists(), "fixture should place Civil train in the cache"
    assert not (_civil(cfg) / "train_clean.parquet").exists()

    summary = subsets.run(cfg)
    assert summary["rows_in"]["train"] == N_ROWS
    assert len(pd.concat(_batches(cfg))) == DRIFT_TOTAL
