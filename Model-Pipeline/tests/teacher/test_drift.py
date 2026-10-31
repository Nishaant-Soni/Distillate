import copy
import json

import numpy as np
import pandas as pd
import pytest

from src.teacher.config import LABELS, load_config
from src.teacher.label_worker import run_worker, select_drift_rows
from src.teacher.prompts import PromptBuilder
from src.teacher.shards import (
    build_manifest,
    plan_batches,
    plan_shards,
    read_batches,
    shard_frame,
    shard_name,
)
from src.teacher.soft_labels import run_drift, soft_prefix
from src.teacher.stack import save_stack
from src.teacher.storage import LocalStorage, shard_paths, write_shard
from src.teacher.validate_local import check_run, load_run
from tests.teacher.test_label_worker import Clock, FakeScorer
from tests.teacher.test_shards import scored

PV, MV = "p2-zs-v2b", "Qwen/Qwen3-4B-Instruct-2507@abc"
PREFIX = f"teacher/{PV}/drift/"
WORDS = ["you", "idiot", "kill", "nice", "thanks", "stupid", "hate", "people", "great", "day"]


def drift_data(root, n_batches=3, per_batch=8, seed=0):
    """Batch files and civil_drift_ids.csv shaped like the data lane's.

    Ids are numeric strings in time order, so a batch's ids are not a run of the sorted ids
    (batch 1 holds the largest ones): a plan cut from the sorted ids would differ.
    """
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    order = [1, 0, 2][:n_batches] if n_batches <= 3 else list(range(n_batches))
    rows = []
    for b in range(n_batches):
        for j in range(per_batch):
            text = " ".join(rng.choice(WORDS, 6))
            row = {"id": str(100 + order[b] * per_batch + j), "text": text, "batch": b,
                   "created_date": pd.Timestamp("2016-01-01", tz="UTC") + pd.Timedelta(days=b)}
            for label in LABELS:
                row[label] = np.int8("idiot" in text or "kill" in text or j == 0)
            rows.append(row)
    frame = pd.DataFrame(rows)
    for b, part in frame.groupby("batch"):
        part.drop(columns="batch").to_parquet(root / f"batch_{b:02d}.parquet", index=False)
    frame[["id", "batch", "created_date"]].to_csv(root / "civil_drift_ids.csv", index=False)
    return frame


def test_plan_batches_follows_batches_not_sorted_ids():
    batches = {"9": 0, "1": 0, "5": 1, "3": 2, "7": 1}
    assert plan_batches(batches) == [["1", "9"], ["5", "7"], ["3"]]
    assert plan_shards(list(batches), 2) != plan_batches(batches)
    with pytest.raises(ValueError, match="no gap"):
        plan_batches({"a": 0, "b": 2})


def test_read_batches(tmp_path):
    pd.DataFrame({"id": ["007", "12"], "batch": [1, 0]}).to_csv(tmp_path / "ids.csv",
                                                               index=False)
    assert read_batches(tmp_path / "ids.csv") == {"007": 1, "12": 0}   # ids kept as strings
    pd.DataFrame({"id": ["1", "1"], "batch": [0, 1]}).to_csv(tmp_path / "dup.csv", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        read_batches(tmp_path / "dup.csv")


def test_select_drift_rows(tmp_path):
    frame = drift_data(tmp_path / "drift")
    rows = select_drift_rows(tmp_path / "drift", tmp_path / "drift" / "civil_drift_ids.csv")
    assert list(rows.columns) == ["id", "text", "batch"] and len(rows) == 24
    assert rows["batch"].is_monotonic_increasing
    # Text is taken as is, never cleaned again.
    got, want = rows.set_index("id")["text"], frame.set_index("id")["text"]
    assert got.sort_index().equals(want.sort_index())


@pytest.mark.parametrize("damage, message", [
    (lambda d: (d / "batch_02.parquet").unlink(), "batch files"),
    (lambda d: pd.read_parquet(d / "batch_01.parquet").iloc[:-1]
     .to_parquet(d / "batch_01.parquet"), "batch_01.parquet: ids differ"),
    (lambda d: pd.read_parquet(d / "batch_00.parquet").assign(text=" ")
     .to_parquet(d / "batch_00.parquet"), "empty text"),
])
def test_select_drift_rows_stops_on_a_mismatch(tmp_path, damage, message):
    drift_data(tmp_path / "drift")
    damage(tmp_path / "drift")
    with pytest.raises(ValueError, match=message):
        select_drift_rows(tmp_path / "drift", tmp_path / "drift" / "civil_drift_ids.csv")


def test_select_drift_rows_catches_a_row_in_the_wrong_batch(tmp_path):
    drift_data(tmp_path / "drift")
    ids = pd.read_csv(tmp_path / "drift" / "civil_drift_ids.csv", dtype={"id": str})
    ids.loc[0, "batch"], ids.loc[len(ids) - 1, "batch"] = 2, 0
    ids.to_csv(tmp_path / "drift" / "civil_drift_ids.csv", index=False)
    with pytest.raises(ValueError, match="ids differ"):
        select_drift_rows(tmp_path / "drift", tmp_path / "drift" / "civil_drift_ids.csv")


def test_worker_writes_one_shard_per_batch_and_validates(tmp_path, tokenizer):
    drift_data(tmp_path / "drift")
    csv = tmp_path / "drift" / "civil_drift_ids.csv"
    rows = select_drift_rows(tmp_path / "drift", csv)
    batches = read_batches(csv)
    storage = LocalStorage(tmp_path / "store")
    builder = PromptBuilder(tokenizer, "p2", "zs", text_version="v2b")
    for worker in (0, 1):
        run_worker(rows, plan_batches(batches), worker, 2, storage, PREFIX, builder,
                   FakeScorer(), MV, deadline=1e9, clock=Clock(), log=lambda line: None,
                   extra_manifest={"layout": "one shard per batch: shard_<k> is batch k"})
    frames, manifests, schemas, errors = load_run(storage, PREFIX, tmp_path / "cache")
    assert errors == [] and sorted(frames) == ["shard_0000", "shard_0001", "shard_0002"]
    for k in range(3):
        assert set(frames[shard_name(k)]["id"]) == {i for i, b in batches.items() if b == k}
    assert manifests["shard_0001"]["layout"].startswith("one shard per batch")
    assert check_run(frames, manifests, schemas, sorted(batches), 5000, PV, MV,
                     batches=batches) == []
    # The default plan (sorted ids cut into shards) rejects the same shards.
    assert any("differ from the planned shard" in e
               for e in check_run(frames, manifests, schemas, sorted(batches), 8, PV, MV))


def test_validator_rejects_a_shard_holding_another_batch(tmp_path):
    batches = {"1": 0, "2": 0, "3": 1, "4": 1}
    storage = LocalStorage(tmp_path / "store")
    for k, ids in enumerate([["1", "3"], ["2", "4"]]):      # shards mixed across batches
        frame = shard_frame(scored(ids), PV, MV)
        write_shard(storage, PREFIX, shard_name(k), frame, build_manifest(frame, PV, MV))
    loaded = load_run(storage, PREFIX, tmp_path / "cache")
    errors = check_run(*loaded[:3], sorted(batches), 5000, PV, MV, batches=batches)
    assert "shard_0000: ids differ from the planned shard" in errors
    with pytest.raises(ValueError, match="different ids"):
        check_run(*loaded[:3], ["1", "2"], 5000, PV, MV, batches=batches)


def drift_setup(tmp_path, stack_sha_override=None):
    """Config, store, and files for ``run_drift`` on a small synthetic drift stream."""
    cfg = copy.deepcopy(load_config())
    for key, sub in (("data_dir", "data"), ("oof_dir", "oof"), ("stack_dir", "stack"),
                     ("drift_dir", "drift")):
        cfg["paths"][key] = str(tmp_path / sub)
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    cfg["baseline"].update(max_features=500)
    frame = drift_data(tmp_path / "drift")

    rng = np.random.default_rng(1)
    train = pd.DataFrame({"id": [f"t{i}" for i in range(120)],
                          "text": [" ".join(rng.choice(WORDS, 6)) for _ in range(120)]})
    for label in LABELS:
        train[label] = train["text"].str.contains("idiot|kill").astype("int8")
    train.to_parquet(tmp_path / "data" / cfg["data"]["files"]["train"], index=False)

    stack = {"stack_version": "stack-v1", "method": "weighted logistic regression",
             "inputs": ["teacher logit_<label>", "baseline logit(p_<label>)"], "C": 1.0,
             "prompt_version": PV, "model_version": MV,
             "baseline_version": cfg["baseline"]["model_version"],
             "labels": {label: {"w_teacher": 0.1, "w_baseline": 1.0, "intercept": -0.5}
                        for label in LABELS}}
    stack_path = tmp_path / "stack" / f"stack_{PV}.json"
    save_stack(stack, stack_path)

    storage = LocalStorage(tmp_path / "store")
    from src.teacher.soft_labels import sha256_file
    trained = soft_prefix("stack-v1", PV)
    (tmp_path / "store" / trained).mkdir(parents=True)
    (tmp_path / "store" / shard_paths(trained, "shard_0000")[1]).write_text(json.dumps(
        {"stack_sha256": stack_sha_override or sha256_file(stack_path)}))
    batches = dict(zip(frame["id"], frame["batch"], strict=True))
    for k, ids in enumerate(plan_batches(batches)):
        pred = scored(ids)
        for label in LABELS:
            pred[f"logit_{label}"] = rng.normal(0, 2, len(ids))
        teacher = shard_frame(pred, PV, MV)
        write_shard(storage, PREFIX, shard_name(k), teacher, build_manifest(teacher, PV, MV))
    return cfg, storage, batches


def test_run_drift_writes_valid_batch_shards_and_a_report(tmp_path, monkeypatch):
    cfg, storage, batches = drift_setup(tmp_path)
    monkeypatch.setattr("src.teacher.soft_labels.PIPELINE_ROOT", tmp_path)
    out = LocalStorage(tmp_path / "out")
    report = run_drift(cfg, storage, out, code_commit="abc")

    prefix = f"{soft_prefix('stack-v1', PV)}drift/"
    assert report["upload"]["drift"]["written"] == ["shard_0000", "shard_0001", "shard_0002"]
    mv = f"stack-v1+{MV}+{cfg['baseline']['model_version']}"
    loaded = load_run(out, prefix, tmp_path / "cache")
    assert loaded[3] == [] and check_run(*loaded[:3], sorted(batches), 5000, PV, mv,
                                         batches=batches) == []
    manifest = loaded[1]["shard_0002"]
    assert manifest["stack_sha256"] == report["stack_sha256"]
    assert manifest["cross_fitted"] == "no (held-out rows)" and manifest["code_commit"] == "abc"
    assert manifest["stack_file"] == f"stack/stack_{PV}.json"

    saved = json.loads((tmp_path / "stack" / "drift_check.json").read_text())
    assert saved["rows"] == 24 and saved["batches"] == 3 and "upload" not in saved
    assert set(saved["overall"]) == {"stack", "teacher_raw", "tfidf"}
    assert [b["batch"] for b in saved["per_batch"]] == [0, 1, 2]
    assert (tmp_path / "oof" / "drift_tfidf_full.parquet").exists()

    # A rerun skips every shard; nothing is rewritten.
    again = run_drift(cfg, storage, out, code_commit="abc")
    assert again["upload"]["drift"]["written"] == []


def test_run_drift_refuses_a_stack_other_than_the_training_one(tmp_path):
    cfg, storage, _ = drift_setup(tmp_path, stack_sha_override="0" * 64)
    with pytest.raises(ValueError, match="not the stack that wrote"):
        run_drift(cfg, storage)
