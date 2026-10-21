import json

import pandas as pd
import pytest

from src.student import data
from src.student.synthetic import make_rows, make_soft_labels, write_store
from src.teacher.config import LABELS
from tests.student.conftest import smoke_cfg


def test_read_rows_drops_empty_text_and_adds_weight(tmp_path):
    rows = make_rows(50)
    rows.loc[3, "text"] = "   "
    rows.to_parquet(tmp_path / "raw.parquet", index=False)
    out = data.read_rows(tmp_path / "raw.parquet")
    assert len(out) == 49 and rows.loc[3, "id"] not in set(out["id"])
    assert (out["sample_weight"] == 1.0).all()
    assert all(out[k].dtype == "int8" for k in LABELS)


def test_read_rows_keeps_existing_sample_weight(tmp_path):
    rows = make_rows(20)
    rows["sample_weight"] = 2.0
    rows.to_parquet(tmp_path / "raw.parquet", index=False)
    assert (data.read_rows(tmp_path / "raw.parquet")["sample_weight"] == 2.0).all()


@pytest.mark.parametrize("corrupt, message", [
    (lambda df: df.assign(threat=df["threat"].replace(1, 2)), "threat"),
    (lambda df: pd.concat([df, df.head(1)]), "duplicate"),
    (lambda df: df.drop(columns=["insult"]), "missing"),
])
def test_read_rows_rejects_bad_input(tmp_path, corrupt, message):
    corrupt(make_rows(30)).to_parquet(tmp_path / "raw.parquet", index=False)
    with pytest.raises(ValueError, match=message):
        data.read_rows(tmp_path / "raw.parquet")


def test_temp_split_is_seeded_disjoint_and_stratified():
    rows = make_rows(2000)
    train, val = data.temp_split(rows, 0.15, seed=42)
    again, _ = data.temp_split(rows.sample(frac=1, random_state=7), 0.15, seed=42)
    assert list(train["id"]) == list(again["id"])          # row order does not matter
    assert not set(train["id"]) & set(val["id"])
    assert len(train) + len(val) == len(rows)
    assert abs(len(val) / len(rows) - 0.15) < 0.01
    for label in LABELS:                                    # rare labels land in both
        assert val[label].sum() > 0 and train[label].sum() > 0
        assert abs(train[label].mean() - val[label].mean()) < 0.01


def test_gate_rows_are_refused(tmp_path):
    rows = make_rows(30)
    pd.DataFrame({"id": [rows["id"].iloc[4]]}).to_parquet(tmp_path / "gate.parquet")
    with pytest.raises(ValueError, match="gate ids"):
        data.check_no_gate([rows], tmp_path / "gate.parquet")
    data.check_no_gate([rows.drop(index=4)], tmp_path / "gate.parquet")


def test_load_splits_reports_sizes_and_passes_the_gate_check(synthetic_root):
    cfg = smoke_cfg(synthetic_root)
    train, val, info = data.load_splits(cfg)
    assert info["train_rows"] == len(train) and info["val_rows"] == len(val)


def test_soft_labels_pass_full_validation(synthetic_root):
    cfg = smoke_cfg(synthetic_root, "loss.alpha=0.5")
    ids = pd.read_parquet(cfg["paths"]["raw_train"])["id"].tolist()
    soft, info = data.load_soft_labels(cfg, ids)
    assert len(soft) == 600 and info["full_validation"] and info["failed_rows"] == 0
    assert list(soft.columns) == ["id", *data.LOGIT_COLUMNS, "status"]


def test_stack_hash_mismatch_is_rejected(tmp_path):
    rows = make_rows(100)
    write_store(make_soft_labels(rows), tmp_path / "store", stack_sha256="0" * 64)
    rows.to_parquet(tmp_path / "wiki_train_raw.parquet", index=False)
    with pytest.raises(ValueError, match="stack_sha256"):
        data.load_soft_labels(smoke_cfg(tmp_path), None)


def test_wrong_model_version_is_rejected(synthetic_root):
    cfg = smoke_cfg(synthetic_root, "soft_labels.model_version=stack-v2+x")
    with pytest.raises(ValueError, match="model_version"):
        data.load_soft_labels(cfg, None)


def test_missing_ids_fail_full_validation(synthetic_root):
    cfg = smoke_cfg(synthetic_root)
    ids = pd.read_parquet(cfg["paths"]["raw_train"])["id"].tolist() + ["ffffffffffffffff"]
    with pytest.raises(ValueError, match="no label"):
        data.load_soft_labels(cfg, ids)


def test_failed_rows_get_soft_ok_false(tmp_path):
    rows = make_rows(400)
    write_store(make_soft_labels(rows, failed=3), tmp_path / "store")
    rows.to_parquet(tmp_path / "wiki_train_raw.parquet", index=False)
    soft, info = data.load_soft_labels(smoke_cfg(tmp_path), rows["id"].tolist())
    merged = data.attach_soft_labels(rows, soft)
    assert info["failed_rows"] == 3 and int((~merged["soft_ok"]).sum()) == 3


def test_attach_requires_a_label_for_every_row():
    rows = make_rows(20)
    soft = make_soft_labels(rows).iloc[1:]
    with pytest.raises(ValueError, match="no soft label"):
        data.attach_soft_labels(rows, soft[["id", *data.LOGIT_COLUMNS, "status"]])


def test_synthetic_manifest_carries_committed_stack_hash(synthetic_root):
    manifest = json.loads(next((synthetic_root / "store").rglob("*.manifest.json")).read_text())
    assert manifest["stack_sha256"] == data.file_sha256(smoke_cfg(synthetic_root)["paths"]
                                                        ["stack_file"])
