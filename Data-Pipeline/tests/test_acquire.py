"""Tests for scripts/acquire.py. No network: tiny fake Kaggle files in tmp_path."""

import zipfile

import pandas as pd
import pytest

from scripts import acquire
from scripts.acquire import (
    build_identities_test,
    build_wiki_test,
    build_wiki_train,
    check_rows,
    civil_frame,
    csv_name_of,
    expected_rows,
    extract_zip,
    extract_zips,
    fetch_competition,
    raw_location,
    read_csv_as_text,
)
from scripts.columns import CIVIL_RAW_COLUMNS, IDENTITY_COLUMNS, LABELS, WIKI_RAW_COLUMNS
from scripts.errors import (
    DataUnavailableError,
    FileCorruptionError,
    LabelError,
    ManifestMismatchError,
)

KAGGLE_WIKI_LABELS = "toxic,severe_toxic,obscene,threat,insult,identity_hate"


def civil_row(**overrides):
    """One Kaggle-shaped Civil Comments row, all values as text."""
    row = {col: "0" for col in CIVIL_RAW_COLUMNS if col not in ("text", "toxicity")}
    row.update({"comment_text": "hello", "target": "0.0", "created_date": "2017-01-01"})
    row.update(overrides)
    return row


def test_literal_na_and_null_comments_stay_text(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text('id,comment_text\na,NA\nb,null\nc,""\n')
    assert read_csv_as_text(path)["comment_text"].tolist() == ["NA", "null", ""]


def test_wiki_train_renamed_and_labels_numeric(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text(f"id,comment_text,{KAGGLE_WIKI_LABELS}\na,NA,0,0,0,0,0,1\n")
    df = build_wiki_train(path)
    assert list(df.columns) == WIKI_RAW_COLUMNS
    assert df.loc[0, "text"] == "NA"
    assert df.loc[0, "identity_attack"] == 1.0
    assert df["toxicity"].dtype == "float32"


def test_wiki_test_drops_unscored_rows(tmp_path):
    (tmp_path / "test.csv").write_text("id,comment_text\na,hello\nb,bye\n")
    (tmp_path / "test_labels.csv").write_text(
        f"id,{KAGGLE_WIKI_LABELS}\na,0,0,0,0,0,0\nb,-1,-1,-1,-1,-1,-1\n"
    )
    df = build_wiki_test(tmp_path / "test.csv", tmp_path / "test_labels.csv")
    assert df["id"].tolist() == ["a"]
    assert list(df.columns) == WIKI_RAW_COLUMNS


def test_missing_wiki_label_raises(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text(f"id,comment_text,{KAGGLE_WIKI_LABELS}\na,hi,0,,0,0,0,0\n")
    with pytest.raises(LabelError):
        build_wiki_train(path)


def test_civil_train_target_becomes_toxicity_and_extras_dropped():
    chunk = pd.DataFrame([civil_row(target="0.7", rating="approved", parent_id="5")])
    out = civil_frame(chunk, toxicity_column="target", source="civil/train")
    assert list(out.columns) == CIVIL_RAW_COLUMNS
    assert out.loc[0, "toxicity"] == pytest.approx(0.7)
    assert "rating" not in out.columns and "parent_id" not in out.columns


def test_civil_blank_identity_becomes_nan_but_blank_label_raises():
    ok = civil_frame(pd.DataFrame([civil_row(muslim="")]), "target", "civil/train")
    assert ok["muslim"].isna().all()
    with pytest.raises(LabelError):
        civil_frame(pd.DataFrame([civil_row(insult="")]), "target", "civil/train")


def test_identities_test_keeps_only_annotated_rows(tmp_path):
    df = pd.DataFrame(
        {"id": ["a", "b", "c"], "identity_annotator_count": [0.0, 3.0, float("nan")]}
    )
    path = tmp_path / "test.parquet"
    df.to_parquet(path, index=False)
    assert build_identities_test(path)["id"].tolist() == ["b"]


def test_csv_name_of():
    assert csv_name_of("train.csv.zip") == "train.csv"
    assert csv_name_of("train.csv") == "train.csv"


def test_nested_zips_are_extracted(tmp_path):
    inner = tmp_path / "build"
    inner.mkdir()
    with zipfile.ZipFile(inner / "train.csv.zip", "w") as archive:
        archive.writestr("train.csv", "id,comment_text\na,hi\n")
    with zipfile.ZipFile(tmp_path / "train.csv.zip.zip", "w") as archive:
        archive.write(inner / "train.csv.zip", "train.csv.zip")
    extract_zips(tmp_path)
    assert read_csv_as_text(tmp_path / "train.csv")["comment_text"].tolist() == ["hi"]


def test_corrupted_zip_is_deleted_and_raises(tmp_path):
    bad = tmp_path / "broken.zip"
    bad.write_bytes(b"this is not a zip file")
    with pytest.raises(FileCorruptionError):
        extract_zip(bad, tmp_path)
    assert not bad.exists()


def test_existing_csv_is_not_downloaded_again(tmp_path, monkeypatch):
    (tmp_path / "train.csv").write_text("id,comment_text\na,hi\n")

    def fail(*args, **kwargs):
        raise AssertionError("should not download")

    monkeypatch.setattr(acquire, "kaggle_download", fail)
    paths = fetch_competition("comp", {"train": "train.csv.zip"}, tmp_path)
    assert paths["train"] == tmp_path / "train.csv"


def test_missing_file_after_download_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire, "kaggle_download", lambda *args, **kwargs: None)
    with pytest.raises(DataUnavailableError):
        fetch_competition("comp", {"train": "train.csv"}, tmp_path)


def test_wrong_row_count_raises():
    with pytest.raises(ManifestMismatchError):
        check_rows("wikipedia/train.parquet", 10, 159571)
    check_rows("wikipedia/train.parquet", 159571, 159571)


def test_expected_rows_read_from_config():
    cfg = {
        "datasets": {
            "wikipedia": {"expected_rows": {"train": 1, "test": 2}},
            "civil_comments": {"expected_rows": {"train": 3, "validation": 4, "test": 5}},
            "civil_identities": {"expected_rows": {"test": 6}},
        }
    }
    assert expected_rows(cfg) == {
        "wikipedia/train.parquet": 1,
        "wikipedia/test.parquet": 2,
        "civil/train.parquet": 3,
        "civil/validation.parquet": 4,
        "civil/test.parquet": 5,
        "civil/identities_test.parquet": 6,
    }


def test_column_lists_have_expected_sizes():
    assert len(LABELS) == 6
    assert len(IDENTITY_COLUMNS) == 24


def test_full_civil_train_goes_to_cache_others_to_raw():
    cfg = {"paths": {"raw_dir": "/data/raw", "cache_dir": "/data/cache"}}
    assert raw_location(cfg, "civil/train.parquet").as_posix() == "/data/cache/civil/train.parquet"
    assert raw_location(cfg, "civil/test.parquet").as_posix() == "/data/raw/civil/test.parquet"
    assert raw_location(cfg, "wikipedia/train.parquet").as_posix() == (
        "/data/raw/wikipedia/train.parquet"
    )
