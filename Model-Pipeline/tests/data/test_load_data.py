"""Tests for src/data/load_data.py, on tiny made-up tables (no real data, no network)."""

import pandas as pd
import pytest

from src.data import load_data as ld
from src.teacher.config import LABELS

TARGETS = [f"p_{label}" for label in LABELS]


def write(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def split(ids):
    frame = pd.DataFrame({"id": ids, "text": [f"t{i}" for i in ids]})
    for label in LABELS:
        frame[label] = 0
    return frame


def teacher(ids):
    frame = pd.DataFrame({"id": ids})
    for label in LABELS:
        for suffix in ("", "_raw", "_calqwen"):
            frame[f"p_{label}{suffix}"] = 0.5
        frame[f"tfidf_oof_{label}"] = 0.4
    frame["prompt_version"] = "p2-zs-v2b"
    frame["model_version"] = "stack-v1+test"
    frame["teacher_version"] = "stack-v1"
    return frame


def build(tmp_path, teacher_ids=None):
    """A processed-data folder with every table; train/val ids a1..a4 and b1..b2."""
    train_ids, val_ids = ["a1", "a2", "a3", "a4"], ["b1", "b2"]
    write(tmp_path / "wikipedia" / "train.parquet", split(train_ids))
    write(tmp_path / "wikipedia" / "val.parquet", split(val_ids))
    write(tmp_path / "wikipedia" / "gate.parquet", split(["g1", "g2"]))
    write(tmp_path / "civil" / "h2_eval.parquet", split(["h1"]))
    write(tmp_path / "civil" / "bias_eval.parquet", split(["h1", "x1"]))
    write(tmp_path / "obfuscation.parquet", split(["g1_obf"]))
    ids = teacher_ids if teacher_ids is not None else train_ids + val_ids + ["dropped"]
    write(tmp_path / "teacher_labels.parquet", teacher(ids))
    return tmp_path


def test_returns_every_table_with_teacher_labels_on_train_and_val(tmp_path):
    data = ld.load_data(data_dir=build(tmp_path))
    assert set(data.tables) == {"train", "val", "gate", "h2", "bias", "obfuscation"}
    assert len(data["train"]) == 4 and len(data["val"]) == 2  # "dropped" doesn't sneak in
    for name in ("train", "val"):
        assert set(TARGETS) <= set(data[name].columns)
        assert "tfidf_oof_toxicity" in data[name].columns
        assert "teacher_version" not in data[name].columns
    assert "p_toxicity" not in data["gate"].columns
    assert data.teacher_versions["teacher_version"] == "stack-v1"


def test_missing_teacher_label_stops_loading(tmp_path):
    folder = build(tmp_path, teacher_ids=["a1", "a2", "a3", "b1", "b2"])  # a4 has no label
    with pytest.raises(ValueError, match="no teacher labels"):
        ld.load_data(data_dir=folder)


def test_without_teacher_skips_the_join(tmp_path):
    folder = build(tmp_path)
    (folder / "teacher_labels.parquet").unlink()
    data = ld.load_data(data_dir=folder, with_teacher=False)
    assert "p_toxicity" not in data["train"].columns


def test_data_hash_is_stable_and_follows_the_content(tmp_path):
    folder = build(tmp_path)
    first = ld.load_data(data_dir=folder).data_hash
    assert ld.load_data(data_dir=folder).data_hash == first
    write(folder / "wikipedia" / "gate.parquet", split(["g1", "g2", "g3"]))
    assert ld.load_data(data_dir=folder).data_hash != first


def test_missing_file_names_it(tmp_path):
    folder = build(tmp_path)
    (folder / "civil" / "h2_eval.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="h2_eval.parquet"):
        ld.load_data(data_dir=folder)


def test_rev_reads_through_dvc(tmp_path, monkeypatch):
    folder = build(tmp_path / "repo" / "data" / "processed")
    (tmp_path / "repo" / ".dvc").mkdir()
    calls = []

    def fake_dvc_read(path, repo, rev):
        calls.append((path, rev))
        return (tmp_path / "repo" / path).read_bytes()

    monkeypatch.setattr(ld, "_dvc_read", fake_dvc_read)
    data = ld.load_data("abc123", data_dir=folder)
    assert data.rev == "abc123" and len(data["train"]) == 4
    assert ("data/processed/wikipedia/train.parquet", "abc123") in calls
    assert data.data_hash == ld.load_data(data_dir=folder).data_hash  # same data, same hash


def test_missing_ablation_score_stops_loading(tmp_path):
    """A missing value in any teacher column, not only the targets, stops loading."""
    folder = build(tmp_path)
    path = folder / "teacher_labels.parquet"
    frame = pd.read_parquet(path)
    frame.loc[frame["id"] == "a2", "p_toxicity_calqwen"] = float("nan")
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="p_toxicity_calqwen: 1"):
        ld.load_data(data_dir=folder)


def test_tables_option_reads_only_the_requested_tables(tmp_path):
    folder = build(tmp_path)
    (folder / "wikipedia" / "gate.parquet").unlink()  # not needed, so its absence is fine
    data = ld.load_data(data_dir=folder, tables=("train", "val"))
    assert set(data.tables) == {"train", "val"}
    assert "p_toxicity" in data["train"].columns


def test_unknown_table_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown tables"):
        ld.load_data(data_dir=build(tmp_path), tables=("train", "test"))


def test_train_data_hash_ignores_evaluation_tables(tmp_path):
    folder = build(tmp_path)
    before = ld.load_data(data_dir=folder)
    write(folder / "wikipedia" / "gate.parquet", split(["g1", "g2", "g3"]))
    after = ld.load_data(data_dir=folder)
    assert after.data_hash != before.data_hash
    assert after.train_data_hash == before.train_data_hash
    assert ld.load_data(data_dir=folder, tables=("train", "val")).train_data_hash == (
        before.train_data_hash
    )
