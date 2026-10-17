import json

import pytest

from src.teacher.config import LABELS, load_config, model_version
from src.teacher.data import build_data_manifest, check_frame, clean_text, records_to_frame


def test_clean_text():
    assert clean_text("ｆｕｌｌ　width") == "full width"             # NFKC
    assert clean_text("a &amp; b &lt;3") == "a & b <3"            # HTML entities
    assert clean_text("  line one\n\n\tline   two  ") == "line one line two"
    assert clean_text("x&nbsp;y") == "x y"                         # unescaped nbsp collapses


def record(i, text, **labels):
    rec = {"id": f"id{i}".encode(), "text": text.encode(), "language": b"en"}
    rec.update({label: float(labels.get(label, 0)) for label in LABELS})
    return rec


def test_records_to_frame():
    df = records_to_frame([record(1, "hi &amp; bye", insult=1), record(2, "   ")])
    assert list(df.columns) == ["id", "text", *LABELS]
    assert list(df["id"]) == ["id1", "id2"]
    assert list(df["text"]) == ["hi & bye", ""]
    assert df["insult"].tolist() == [1, 0] and str(df["insult"].dtype) == "int8"


def test_non_binary_label_rejected():
    with pytest.raises(ValueError, match="toxicity"):
        records_to_frame([record(1, "x", toxicity=0.4)])


def test_check_frame():
    df = records_to_frame([record(1, "a"), record(2, "")])
    assert check_frame(df, 2) == {"rows": 2, "empty_text_rows": 1}
    with pytest.raises(ValueError, match="expected 3"):
        check_frame(df, 3)
    with pytest.raises(ValueError, match="duplicate"):
        check_frame(records_to_frame([record(1, "a"), record(1, "b")]), 2)


def test_manifest(tmp_path):
    path = tmp_path / "wiki_train_raw.parquet"
    path.write_bytes(b"abc")
    manifest = json.loads(build_data_manifest({path: {"rows": 2}}, "tfds/x:1", "deadbeef"))
    entry = manifest["files"]["wiki_train_raw.parquet"]
    assert entry["rows"] == 2
    assert entry["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert manifest["code_commit"] == "deadbeef"


def test_config_loads():
    cfg = load_config()
    assert model_version(cfg) == (
        "Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554")
    assert cfg["data"]["expected_rows"] == {"train": 159571, "test": 63978}
