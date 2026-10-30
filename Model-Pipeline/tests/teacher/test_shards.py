import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.teacher.config import LABELS
from src.teacher.shards import (
    SHARD_COLUMNS,
    build_manifest,
    check_plan,
    ids_sha256,
    plan_record,
    plan_shards,
    shard_frame,
    shard_name,
    worker_shards,
)
from src.teacher.storage import LocalStorage, shard_done, shard_paths, write_shard


def test_plan_is_deterministic_and_complete():
    ids = [f"{i:05x}" for i in range(23)]
    plan = plan_shards(list(reversed(ids)), shard_size=10)
    assert [len(s) for s in plan] == [10, 10, 3]
    assert plan == plan_shards(ids, 10)                       # input order does not matter
    assert [i for s in plan for i in s] == sorted(ids)
    with pytest.raises(ValueError, match="duplicates"):
        plan_shards(ids + ids[:1], 10)


def test_ids_sha256_matches_spec():
    expected = hashlib.sha256(b"a\nb\nc").hexdigest()        # sorted, joined by newlines
    assert ids_sha256(["c", "a", "b"]) == expected


def test_plan_record_and_check():
    plan = plan_shards([str(i) for i in range(25)], 10)
    record = plan_record(plan, "train", 10)
    assert record["num_shards"] == 3 and record["total_rows"] == 25
    assert record["shards"][0]["name"] == "shard_0000" == shard_name(0)
    check_plan(plan, json.loads(json.dumps(record)))
    with pytest.raises(ValueError, match="differs"):
        check_plan(plan_shards([str(i) for i in range(26)], 10), record)


def test_worker_split():
    assert worker_shards(5, 0, 2) == [0, 2, 4] and worker_shards(5, 1, 2) == [1, 3]
    assert worker_shards(3, 0, 1) == [0, 1, 2]
    with pytest.raises(ValueError):
        worker_shards(3, 2, 2)


def scored(ids, failed=()):
    pred = pd.DataFrame({"id": ids, "status": ["failed" if i in failed else "ok" for i in ids]})
    for label in LABELS:
        bad = pred["status"] == "failed"
        pred[f"p_{label}"] = np.where(bad, np.nan, 0.25)
        pred[f"logit_{label}"] = np.where(bad, np.nan, -1.1)
        pred[f"cov_{label}"] = 1.0
    pred["coverage"] = 1.0
    return pred


def test_shard_frame_has_spec_columns_and_types():
    frame = shard_frame(scored(["b", "a"]), "p2-zs-v2b", "m@r")
    assert list(frame.columns) == SHARD_COLUMNS
    assert frame["id"].tolist() == ["a", "b"]                 # sorted by id
    assert frame["id"].dtype == "string" and frame["status"].dtype == "string"
    assert all(frame[c].dtype == np.float32 for c in SHARD_COLUMNS[1:13])
    assert set(frame["prompt_version"]) == {"p2-zs-v2b"}


def test_manifest_fields():
    frame = shard_frame(scored(["x", "y"]), "p2-zs-v2b", "m@r")
    manifest = build_manifest(frame, "p2-zs-v2b", "m@r", code_commit="abc", vllm_version=None)
    assert manifest["row_count"] == 2 and manifest["ids_sha256"] == ids_sha256(["x", "y"])
    assert manifest["code_commit"] == "abc" and "vllm_version" not in manifest


def test_write_shard_parquet_first_then_manifest(tmp_path, monkeypatch):
    storage = LocalStorage(tmp_path)
    frame = shard_frame(scored(["x", "y"]), "p2-zs-v2b", "m@r")
    order = []
    upload = storage.upload
    monkeypatch.setattr(storage, "upload", lambda local, path: (order.append(path),
                                                                upload(local, path)))
    assert not shard_done(storage, "teacher/v/", "shard_0000")
    write_shard(storage, "teacher/v", "shard_0000", frame,
                build_manifest(frame, "p2-zs-v2b", "m@r"))
    parquet, manifest = shard_paths("teacher/v/", "shard_0000")
    assert order == [parquet, manifest]
    assert shard_done(storage, "teacher/v/", "shard_0000")
    assert pd.read_parquet(tmp_path / parquet).equals(frame)
    assert storage.list("teacher/v/") == sorted([parquet, manifest])
    storage.delete(manifest)
    assert not shard_done(storage, "teacher/v/", "shard_0000")
