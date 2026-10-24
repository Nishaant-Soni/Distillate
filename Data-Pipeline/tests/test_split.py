"""Tests for scripts/split.py: Wikipedia train / val / canary / gate."""

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("iterstrat")

from scripts import split  # noqa: E402
from scripts.stubs import LABELS, wikipedia_frame  # noqa: E402
from scripts.utils import load_config  # noqa: E402

N_TRAIN = 4000
N_TEST = 500
CANARY = 200


def _write_inputs(cfg, train_df, test_df):
    wiki = Path(cfg["paths"]["processed_dir"]) / "wikipedia"
    wiki.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(wiki / "train_dedup.parquet", index=False)
    test_df.to_parquet(wiki / "test_clean.parquet", index=False)


@pytest.fixture
def cfg(tmp_path):
    """Config pointed at a temp dir, with contract-shaped inputs already written."""
    cfg = load_config()
    for key in ("processed_dir", "splits_dir", "summaries_dir"):
        cfg["paths"][key] = str(tmp_path / key)
    cfg["splits"]["canary_wikipedia_size"] = CANARY
    _write_inputs(cfg, wikipedia_frame(N_TRAIN, 42, "wiki_train"),
                  wikipedia_frame(N_TEST, 42, "wiki_test"))
    return cfg


def _read(cfg, name):
    return pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "wikipedia" / f"{name}.parquet")


def _ids(cfg, name):
    return (Path(cfg["paths"]["splits_dir"]) / f"{name}_ids.txt").read_text().split()


def test_writes_every_output_with_contract_columns(cfg):
    split.run(cfg)
    for name in split.OUTPUTS:
        df = _read(cfg, name)
        assert df["id"].is_unique, name
        assert all(str(df[c].dtype) == "int8" for c in LABELS), name
    train = _read(cfg, "train")
    assert str(train["sample_weight"].dtype) == "float32"
    assert (train["sample_weight"] == 1.0).all()
    for name in ("val", "gate", "canary_wikipedia"):
        assert "sample_weight" not in _read(cfg, name).columns, name


def test_same_seed_gives_the_same_ids(cfg):
    split.run(cfg)
    first = {name: _ids(cfg, name) for name in split.OUTPUTS}
    split.run(cfg)
    assert {name: _ids(cfg, name) for name in split.OUTPUTS} == first


def test_a_different_seed_gives_a_different_split(cfg):
    split.run(cfg)
    first = _ids(cfg, "val")
    cfg["seed"] = 7
    split.run(cfg)
    assert _ids(cfg, "val") != first


def test_input_row_order_does_not_change_the_split(cfg):
    split.run(cfg)
    first = _ids(cfg, "val")
    shuffled = wikipedia_frame(N_TRAIN, 42, "wiki_train").sample(frac=1, random_state=1)
    _write_inputs(cfg, shuffled, wikipedia_frame(N_TEST, 42, "wiki_test"))
    split.run(cfg)
    assert _ids(cfg, "val") == first


def test_sets_are_disjoint_and_cover_train_dedup(cfg):
    split.run(cfg)
    train, val, canary = (set(_ids(cfg, n)) for n in ("train", "val", "canary_wikipedia"))
    assert not (train & val) and not (train & canary) and not (val & canary)
    assert train | val | canary == set(wikipedia_frame(N_TRAIN, 42, "wiki_train")["id"])
    assert not (set(_ids(cfg, "gate")) & (train | val | canary))


def test_validation_share_is_within_one_point_of_config(cfg):
    split.run(cfg)
    held_out = len(_ids(cfg, "val")) + len(_ids(cfg, "canary_wikipedia"))
    assert abs(held_out / N_TRAIN - cfg["splits"]["val_fraction"]) <= 0.01


def test_canary_has_exactly_the_configured_size(cfg):
    split.run(cfg)
    assert len(_ids(cfg, "canary_wikipedia")) == CANARY


def test_every_label_appears_in_train_val_and_canary(cfg):
    split.run(cfg)
    for name in ("train", "val", "canary_wikipedia"):
        df = _read(cfg, name)
        for label in LABELS:
            assert df[label].sum() > 0, f"{label} missing from {name}"


def test_label_prevalence_matches_between_train_and_val(cfg):
    summary = split.run(cfg)
    prev = summary["prevalence_pts"]
    for label in LABELS:
        gap = abs(prev["train"][label] - prev["val"][label])
        assert gap <= cfg["validation"]["prevalence_tolerance_pts"], label


def test_gate_is_the_test_split_unchanged(cfg):
    split.run(cfg)
    expected = wikipedia_frame(N_TEST, 42, "wiki_test").sort_values("id").reset_index(drop=True)
    pd.testing.assert_frame_equal(_read(cfg, "gate"), expected)


def test_id_files_match_the_parquet_files(cfg):
    split.run(cfg)
    for name in split.OUTPUTS:
        assert _ids(cfg, name) == _read(cfg, name)["id"].tolist(), name


def test_summary_records_counts(cfg):
    summary = split.run(cfg)
    assert summary["rows_in"] == {"train_dedup": N_TRAIN, "test_clean": N_TEST}
    assert summary["files"]["canary_wikipedia"] == CANARY
    assert summary["rows_out"] == sum(summary["files"].values())
    assert (Path(cfg["paths"]["summaries_dir"]) / "split.json").exists()


def test_missing_input_gives_a_clear_error(cfg):
    (Path(cfg["paths"]["processed_dir"]) / "wikipedia" / "train_dedup.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="train_dedup.parquet"):
        split.run(cfg)


def test_missing_label_column_is_rejected(cfg):
    _write_inputs(cfg, wikipedia_frame(N_TRAIN, 42, "wiki_train").drop(columns="threat"),
                  wikipedia_frame(N_TEST, 42, "wiki_test"))
    with pytest.raises(ValueError, match="threat"):
        split.run(cfg)
