"""Tests for scripts/merge_teacher_labels.py.

Builds a tiny shard store from the teacher lane's synthetic fixtures (25 made-up ids): the same
valid/ shards serve as the soft labels, raw teacher and calibrated Qwen, plus a small TF-IDF file
with its manifest. No network is used.
"""

import hashlib
import json
import shutil

import pandas as pd
import pytest

from scripts import merge_teacher_labels as mtl
from scripts import validate_teacher_shards as vts
from scripts.errors import TeacherShardError
from scripts.utils import PIPELINE_ROOT

MODEL_PIPELINE = PIPELINE_ROOT.parent / "Model-Pipeline"
FIXTURES = MODEL_PIPELINE / "tests" / "fixtures" / "teacher"
LABELS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]
PREFIXES = {"soft": "soft/", "raw": "raw/", "calqwen": "calqwen/"}

pytestmark = pytest.mark.skipif(
    not FIXTURES.exists(), reason="Model-Pipeline teacher fixtures not available"
)


def build_store(tmp_path, drop_tfidf_id=False):
    """A local shard store: three copies of the valid fixtures, plus a TF-IDF file."""
    store = tmp_path / "store"
    for prefix in PREFIXES.values():
        shutil.copytree(FIXTURES / "valid", store / prefix)
    ids = sorted(pd.read_csv(FIXTURES / "source_ids.csv", dtype=str)["id"])
    if drop_tfidf_id:
        ids = ids[:-1]
    tfidf = pd.DataFrame({"id": pd.array(ids, dtype="string"), "fold": 0})
    for i, label in enumerate(LABELS):
        tfidf[f"p_{label}"] = pd.Series([0.1 * (i + 1)] * len(ids), dtype="float32")
    tfidf["model_version"] = pd.array(["tfidf-lr-v1"] * len(ids), dtype="string")
    (store / "tfidf").mkdir()
    path = store / "tfidf" / "tfidf_oof.parquet"
    tfidf.to_parquet(path, index=False)
    shards = vts.teacher_modules(MODEL_PIPELINE)["shards"]
    manifest = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "row_count": len(ids),
        "ids_sha256": shards.ids_sha256(ids),
        "baseline_version": "tfidf-lr-v1",
    }
    manifest_path = tmp_path / "tfidf_oof.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return store, manifest_path


def write_report(tmp_path, errors=0, skipped=False):
    report = {"skipped": skipped, "runs": {p: {"errors": errors} for p in PREFIXES.values()}}
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def merge_cfg(tmp_path, store, manifest_path, report_path):
    return {
        "teacher": {
            "model_pipeline_dir": str(MODEL_PIPELINE),
            "backend": None,
            "local_root": str(store),
            "report": str(report_path),
            "merge": {
                "output": str(tmp_path / "teacher_labels.parquet"),
                "teacher_version": "stack-v1",
                "soft": PREFIXES["soft"],
                "raw": PREFIXES["raw"],
                "calqwen": PREFIXES["calqwen"],
                "tfidf": "tfidf/tfidf_oof.parquet",
                "tfidf_manifest": str(manifest_path),
            },
        }
    }


def test_merge_builds_the_contract_table(tmp_path):
    store, manifest = build_store(tmp_path)
    summary = mtl.run(merge_cfg(tmp_path, store, manifest, write_report(tmp_path)))
    table = pd.read_parquet(tmp_path / "teacher_labels.parquet")
    p = [f"p_{label}" for label in LABELS]
    expected = (
        ["id"] + p + [f"{c}_raw" for c in p] + [f"tfidf_oof_{label}" for label in LABELS]
        + [f"{c}_calqwen" for c in p] + ["prompt_version", "model_version", "teacher_version"]
    )
    assert summary["rows_out"] == 25 and list(table.columns) == expected
    assert table["id"].is_unique and table["id"].is_monotonic_increasing
    assert (table["teacher_version"] == "stack-v1").all()
    assert table["tfidf_oof_toxicity"].dtype == "float32"
    assert table["tfidf_oof_toxicity"].round(4).eq(0.1).all()
    assert (table["p_toxicity"] == table["p_toxicity_raw"]).all()  # same fixture shards


def test_tfidf_with_a_missing_id_stops_the_merge(tmp_path):
    store, manifest = build_store(tmp_path, drop_tfidf_id=True)
    with pytest.raises(TeacherShardError, match="tfidf"):
        mtl.run(merge_cfg(tmp_path, store, manifest, write_report(tmp_path)))


def test_tfidf_that_differs_from_its_manifest_stops_the_merge(tmp_path):
    store, manifest = build_store(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["sha256"] = "0" * 64
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(TeacherShardError, match="sha256"):
        mtl.run(merge_cfg(tmp_path, store, manifest, write_report(tmp_path)))


def test_failed_validation_stops_the_merge(tmp_path):
    store, manifest = build_store(tmp_path)
    with pytest.raises(TeacherShardError, match="failed validation"):
        mtl.run(merge_cfg(tmp_path, store, manifest, write_report(tmp_path, errors=2)))


def test_skipped_validation_skips_the_merge(tmp_path):
    store, manifest = build_store(tmp_path)
    summary = mtl.run(merge_cfg(tmp_path, store, manifest, write_report(tmp_path, skipped=True)))
    assert summary["skipped"] is True
    assert not (tmp_path / "teacher_labels.parquet").exists()


def test_missing_validation_report_stops_the_merge(tmp_path):
    store, manifest = build_store(tmp_path)
    with pytest.raises(TeacherShardError, match="run validate_teacher_shards first"):
        mtl.run(merge_cfg(tmp_path, store, manifest, tmp_path / "missing.json"))
