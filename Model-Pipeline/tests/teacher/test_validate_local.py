import json

import numpy as np
import pandas as pd
import pytest

from src.teacher.config import LABELS
from src.teacher.shards import build_manifest, plan_record, plan_shards, shard_frame, shard_name
from src.teacher.storage import LocalStorage, shard_paths, write_shard
from src.teacher.validate_local import check_run, load_run

PV, MV, PREFIX = "p2-zs-v2b", "Qwen/Qwen3-4B-Instruct-2507@abc", "teacher/p2-zs-v2b/"
SOURCE = [f"{i:04x}" for i in range(250)]
SIZE = 100


def scored(ids, failed=()):
    pred = pd.DataFrame({"id": ids, "status": ["failed" if i in failed else "ok" for i in ids]})
    bad = pred["status"] == "failed"
    for label in LABELS:
        pred[f"p_{label}"] = np.where(bad, np.nan, 0.3)
        pred[f"logit_{label}"] = np.where(bad, np.nan, -0.85)
    return pred


def write_run(root, edit=None, skip=()):
    """A valid run of 3 shards; ``edit(k, frame, manifest)`` may corrupt shard k."""
    storage = LocalStorage(root)
    for k, ids in enumerate(plan_shards(SOURCE, SIZE)):
        if k in skip:
            continue
        frame = shard_frame(scored(ids), PV, MV)
        manifest = build_manifest(frame, PV, MV)
        if edit:
            frame, manifest = edit(k, frame, manifest)
        write_shard(storage, PREFIX, shard_name(k), frame, manifest)
    return storage


def validate(tmp_path, storage, **kwargs):
    frames, manifests, schemas, errors = load_run(storage, PREFIX, tmp_path / "cache")
    return errors + check_run(frames, manifests, schemas, SOURCE, SIZE, PV, MV, **kwargs)


def test_valid_run_passes(tmp_path):
    storage = write_run(tmp_path / "store")
    record = plan_record(plan_shards(SOURCE, SIZE), "src", SIZE)
    assert validate(tmp_path, storage, plan_record=json.loads(json.dumps(record))) == []


def test_missing_shard_fails_unless_partial(tmp_path):
    storage = write_run(tmp_path / "store", skip=(2,))
    errors = validate(tmp_path, storage)
    assert any("planned shards missing" in e for e in errors)
    assert any("source ids have no label" in e for e in errors)
    assert validate(tmp_path, storage, partial=True) == []


def test_parquet_without_manifest(tmp_path):
    storage = write_run(tmp_path / "store")
    (tmp_path / "store" / shard_paths(PREFIX, "shard_0001")[1]).unlink()
    errors = validate(tmp_path, storage, partial=True)
    assert errors == ["shard_0001: Parquet without a manifest"]


def corrupt(k, change):
    def edit(i, frame, manifest):
        return change(frame.copy(), dict(manifest)) if i == k else (frame, manifest)
    return edit


def drop_row(f, m):
    return f.iloc[1:].reset_index(drop=True), m


def bad_p(f, m):
    f.loc[0, "p_threat"] = np.float32(1.5)
    return f, m


def failed_with_scores(f, m):
    f.loc[0, "status"] = "failed"
    return f, m


def odd_status(f, m):
    f.loc[0, "status"] = "maybe"
    return f, m


def many_failed(f, m):
    f.loc[:4, "status"] = "failed"
    f.loc[:4, [c for c in f.columns if c.startswith(("p_", "logit_"))]] = np.nan
    return f, m


def wrong_version(f, m):
    f["prompt_version"] = pd.Series(["p2-zs-v1"] * len(f), dtype="string")
    return f, m


def float64(f, m):
    f["logit_insult"] = f["logit_insult"].astype("float64")
    return f, m


def foreign_id(f, m):
    f.loc[0, "id"] = "zzzz"
    return f, {**m, "ids_sha256": "x"}


@pytest.mark.parametrize("change, message", [
    (drop_row, "rows, manifest says"),
    (lambda f, m: (f, {**m, "ids_sha256": "0" * 64}), "id hash differs"),
    (bad_p, "p outside [0, 1]"),
    (failed_with_scores, "failed rows with non-NaN scores"),
    (odd_status, "status values ['maybe']"),
    (many_failed, "failure rate 5.00% above 1%"),
    (wrong_version, "prompt_version ['p2-zs-v1', 'p2-zs-v2b']"),
    (lambda f, m: (f, {**m, "model_version": "other"}), "model_version"),
    (float64, "logit_insult is double, not float32"),
    (foreign_id, "ids not in the source list"),
])
def test_each_check_names_the_shard(tmp_path, change, message):
    storage = write_run(tmp_path / "store", edit=corrupt(1, change))
    errors = validate(tmp_path, storage)
    assert any(e.startswith("shard_0001:") and message in e for e in errors), errors
    assert not any(e.startswith(("shard_0000:", "shard_0002:")) for e in errors)


def test_duplicate_across_shards(tmp_path):
    def edit(k, frame, manifest):
        if k == 1:
            frame = frame.copy()
            frame.loc[0, "id"] = SOURCE[0]                       # also in shard_0000
            manifest = build_manifest(frame, PV, MV)
        return frame, manifest
    errors = validate(tmp_path, write_run(tmp_path / "store", edit=edit))
    assert any("also in shard_0000" in e for e in errors)
    assert any("ids differ from the planned shard" in e for e in errors)
