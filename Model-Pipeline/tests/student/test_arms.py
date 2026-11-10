"""The ablation's attribution arms and the official-ids source, end to end on synthetic data."""

import json

import numpy as np
import pandas as pd
import pytest

from src.student import data
from src.student.config import load_config
from src.student.train import run
from tests.student.conftest import smoke_cfg

FAST = ("mlflow.enabled=false", "train.max_steps=8", "train.epochs=1")


@pytest.mark.parametrize("arm, check_field", [("calqwen", "calibration_sha256"),
                                              ("tfidf", "sha256")])
def test_each_arm_trains_and_records_its_lineage(synthetic_root, arm, check_field):
    summary = run(smoke_cfg(synthetic_root, *FAST, f"soft_labels.arm={arm}"))
    meta = json.loads((synthetic_root / "runs" / summary["run_name"] / "model" / "meta.json")
                      .read_text())
    assert meta["soft_label_arm"] == arm and meta["stack_sha256"] is None
    assert list(meta["soft_label_check"]) == [check_field]
    assert summary["soft_labels"]["full_validation"]


def test_calqwen_shards_must_carry_the_committed_calibration_hash(synthetic_root):
    for path in (synthetic_root / "store" / "soft_labels" / "calqwen-v1").rglob("*.manifest.json"):
        manifest = json.loads(path.read_text())
        manifest["calibration_sha256"] = "0" * 64
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="calibration_sha256"):
        data.load_soft_labels(smoke_cfg(synthetic_root, "soft_labels.arm=calqwen"))


def test_tfidf_file_must_match_its_manifest(synthetic_root):
    manifest = synthetic_root / "tfidf_oof.manifest.json"
    record = json.loads(manifest.read_text())
    record["sha256"] = "0" * 64
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="sha256"):
        data.load_soft_labels(smoke_cfg(synthetic_root, "soft_labels.arm=tfidf"))


def test_tfidf_probabilities_become_finite_logits(synthetic_root):
    soft, info = data.load_soft_labels(smoke_cfg(synthetic_root, "soft_labels.arm=tfidf"))
    oof = pd.read_parquet(synthetic_root / "store" / "tfidf" / "tfidf-lr-v1-oof" /
                          "tfidf_oof.parquet")
    merged = soft.merge(oof, on="id")
    expected = np.log(merged["p_toxicity"] / (1 - merged["p_toxicity"]))
    assert np.allclose(merged["logit_toxicity"], expected, atol=1e-4)
    assert np.isfinite(soft[data.LOGIT_COLUMNS].to_numpy()).all()
    assert info["arm"] == "tfidf" and info["prompt_version"] is None


def test_unknown_arm_and_unknown_arm_keys_fail():
    with pytest.raises(ValueError, match="arm"):
        load_config(None, ["soft_labels.arm=lora"])
    with pytest.raises(KeyError, match="unknown keys"):
        load_config(None, ["soft_labels.arms.calqwen={prefix: x, typo: 1}"])


def test_arm_values_replace_the_stack_defaults():
    cfg = load_config(None, ["soft_labels.arm=calqwen"])
    assert cfg["soft_labels"]["prefix"] == "soft_labels/calqwen-v1/p2-zs-v2b/"
    assert cfg["soft_labels"]["check_file"].endswith("calibration_p2-zs-v2b.json")
    assert cfg["soft_labels"]["prompt_version"] == "p2-zs-v2b"     # kept from the stack


def _write_ids(root, train_ids, val_ids):
    folder = root / "splits"
    folder.mkdir()
    (folder / "train_ids.txt").write_text("\n".join(train_ids) + "\n")
    (folder / "val_ids.txt").write_text("\n".join(val_ids) + "\n")
    return folder


def test_ids_source_uses_exactly_the_official_lists(synthetic_root):
    ids = sorted(pd.read_parquet(synthetic_root / "wiki_train_raw.parquet")["id"])
    folder = _write_ids(synthetic_root, ids[:400], ids[450:])        # 50 rows in neither
    cfg = smoke_cfg(synthetic_root, "data.source=ids", f"paths.split_ids_dir={folder}")
    train, val, info = data.load_splits(cfg)
    assert list(train["id"]) == ids[:400] and list(val["id"]) == ids[450:]
    assert info["source"] == "ids" and info["missing_split_ids"] == {"train": 0, "val": 0}

    summary = run(smoke_cfg(synthetic_root, *FAST, "data.source=ids",
                            f"paths.split_ids_dir={folder}"))
    assert summary["data"]["train_rows"] == 400 and summary["soft_labels"]["full_validation"]


def test_ids_source_fails_when_many_official_ids_are_missing(synthetic_root):
    ids = sorted(pd.read_parquet(synthetic_root / "wiki_train_raw.parquet")["id"])
    folder = _write_ids(synthetic_root, [*ids[:400], "ffffffffffffffff"], ids[450:])
    cfg = smoke_cfg(synthetic_root, "data.source=ids", f"paths.split_ids_dir={folder}")
    with pytest.raises(ValueError, match="not in raw train"):
        data.load_splits(cfg)
