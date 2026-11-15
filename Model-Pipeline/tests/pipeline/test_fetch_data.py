"""Tests for src/pipeline/fetch_data.py with a fake DVC opener (no DVC, no network)."""

from __future__ import annotations

import io

import pytest

from src.pipeline import fetch_data


class FakeDvc:
    def __init__(self):
        self.calls = []

    def __call__(self, path, repo, rev):
        self.calls.append((path, rev))
        return io.BytesIO(f"bytes of {path}".encode())


def test_files_land_in_the_load_data_layout(tmp_path):
    opener = FakeDvc()
    written = fetch_data.fetch(["train", "val", "gate", "bias"], tmp_path, opener=opener)
    assert set(written) == {"train", "val", "gate", "bias"}
    assert (tmp_path / "wikipedia" / "train.parquet").read_bytes() == \
        b"bytes of Data-Pipeline/data/processed/wikipedia/train.parquet"
    assert (tmp_path / "civil" / "bias_eval.parquet").exists()
    last_path = opener.calls[-1][0]
    assert last_path == "Data-Pipeline/data/processed/civil/bias_eval.parquet"


def test_revision_is_passed_through(tmp_path):
    opener = FakeDvc()
    fetch_data.fetch(["gate"], tmp_path, rev="abc123", opener=opener)
    assert opener.calls == [("Data-Pipeline/data/processed/wikipedia/gate.parquet", "abc123")]


def test_unknown_table_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown tables"):
        fetch_data.fetch(["nope"], tmp_path, opener=FakeDvc())
