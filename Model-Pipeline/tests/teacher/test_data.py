import json

import pandas as pd
import pytest

from src.teacher.config import LABELS, load_config, model_version
from src.teacher.data import (
    KAGGLE_LABELS,
    build_data_manifest,
    check_frame,
    clean_text,
    find_competition_dir,
    find_file,
    read_competition_split,
    to_raw_frame,
)


def test_clean_text():
    assert clean_text("ｆｕｌｌ　width") == "full width"             # NFKC
    assert clean_text("a &amp; b &lt;3") == "a & b <3"            # HTML entities
    assert clean_text("  line one\n\n\tline   two  ") == "line one line two"
    assert clean_text("x&nbsp;y") == "x y"                         # unescaped nbsp collapses


def write_competition(folder, zipped=True):
    """A tiny competition folder laid out like Kaggle's, optionally with zipped CSVs."""
    suffix = ".zip" if zipped else ""
    label_cols = list(KAGGLE_LABELS)
    train = pd.DataFrame({"id": ["a1", "a2", "a3"],
                          "comment_text": ["hi &amp; bye", "NA", "line\none"]})
    for i, col in enumerate(label_cols):
        train[col] = [0, int(i == 4), 0]                           # a2 is an insult
    train.to_csv(folder / f"train.csv{suffix}", index=False)

    pd.DataFrame({"id": ["t1", "t2", "t3"], "comment_text": ["scored", "unscored", "null"]}
                 ).to_csv(folder / f"test.csv{suffix}", index=False)
    labels = pd.DataFrame({"id": ["t1", "t2", "t3"]})
    for col in label_cols:
        labels[col] = [1 if col == "toxic" else 0, -1, 0]
    labels.to_csv(folder / f"test_labels.csv{suffix}", index=False)
    return folder


@pytest.mark.parametrize("zipped", [True, False])
def test_train_split(tmp_path, zipped):
    df = to_raw_frame(read_competition_split(write_competition(tmp_path, zipped), "train"))
    assert list(df.columns) == ["id", "text", *LABELS]
    assert list(df["id"]) == ["a1", "a2", "a3"]
    assert list(df["text"]) == ["hi & bye", "NA", "line one"]      # "NA" stays text
    assert df["insult"].tolist() == [0, 1, 0] and str(df["insult"].dtype) == "int8"
    assert str(df["id"].dtype) == "string" and str(df["text"].dtype) == "string"


def test_test_split_drops_unscored_rows(tmp_path):
    df = to_raw_frame(read_competition_split(write_competition(tmp_path), "test"))
    assert list(df["id"]) == ["t1", "t3"]
    assert list(df["text"]) == ["scored", "null"]
    assert df["toxicity"].tolist() == [1, 0]


def test_partial_unscored_row_rejected(tmp_path):
    write_competition(tmp_path, zipped=False)
    labels = pd.read_csv(tmp_path / "test_labels.csv")
    labels.loc[0, "threat"] = -1
    labels.to_csv(tmp_path / "test_labels.csv", index=False)
    with pytest.raises(ValueError, match="part of their labels"):
        read_competition_split(tmp_path, "test")


def test_non_binary_label_rejected(tmp_path):
    write_competition(tmp_path, zipped=False)
    train = pd.read_csv(tmp_path / "train.csv")
    train.loc[0, "toxic"] = 2
    train.to_csv(tmp_path / "train.csv", index=False)
    with pytest.raises(ValueError, match="toxic"):
        to_raw_frame(read_competition_split(tmp_path, "train"))


def test_find_competition_dir(tmp_path):
    folder = tmp_path / "input" / "jigsaw-toxic-comment-classification-challenge"
    folder.mkdir(parents=True)
    write_competition(folder)
    assert find_competition_dir(tmp_path / "input") == folder
    with pytest.raises(FileNotFoundError):
        find_competition_dir(tmp_path / "input" / "missing")


def test_find_file(tmp_path):
    (tmp_path / "nb-output" / "wiki").mkdir(parents=True)
    target = tmp_path / "nb-output" / "wiki" / "wiki_train_raw.parquet"
    target.write_bytes(b"")
    assert find_file(tmp_path, "wiki_train_raw.parquet") == target
    with pytest.raises(FileNotFoundError):
        find_file(tmp_path, "missing.parquet")


def test_check_frame():
    df = pd.DataFrame({"id": pd.Series(["a", "b"], dtype="string"),
                       "text": pd.Series(["x", ""], dtype="string")})
    assert check_frame(df, 2) == {"rows": 2, "empty_text_rows": 1}
    with pytest.raises(ValueError, match="expected 3"):
        check_frame(df, 3)
    with pytest.raises(ValueError, match="duplicate"):
        check_frame(df.assign(id=["a", "a"]), 2)


def test_manifest(tmp_path):
    path = tmp_path / "wiki_train_raw.parquet"
    path.write_bytes(b"abc")
    manifest = json.loads(build_data_manifest({path: {"rows": 2}}, {"kaggle_competition": "x"},
                                              "deadbeef"))
    entry = manifest["files"]["wiki_train_raw.parquet"]
    assert entry["rows"] == 2
    assert entry["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert manifest["source"] == {"kaggle_competition": "x"}
    assert manifest["code_commit"] == "deadbeef"


def test_config_loads():
    cfg = load_config()
    assert model_version(cfg) == (
        "Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554")
    assert cfg["data"]["expected_rows"] == {"train": 159571, "test": 63978}
