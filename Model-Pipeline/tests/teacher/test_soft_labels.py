import json

import numpy as np
import pandas as pd
import pytest

from src.teacher.config import LABELS
from src.teacher.shards import plan_shards
from src.teacher.soft_labels import (
    choose,
    cross_fit_by_folds,
    shard_frames,
    soft_model_version,
    soft_prefix,
    unit_weights,
    upload_shards,
)
from src.teacher.stack import apply_stack, fit_stack
from src.teacher.storage import LocalStorage, shard_paths
from src.teacher.validate_local import check_run, load_run

PV, MV = "p2-zs-v2b", "Qwen/Qwen3-4B-Instruct-2507@abc"


def synthetic(n=2500, folds=5, seed=0):
    """Teacher shards, out-of-fold TF-IDF (with folds) and gold labels."""
    rng = np.random.default_rng(seed)
    ids = [f"{i:06x}" for i in range(n)]
    truth = pd.DataFrame({"id": ids})
    teacher = pd.DataFrame({"id": ids, "status": "ok", "prompt_version": PV,
                            "model_version": MV})
    oof = pd.DataFrame({"id": ids, "fold": rng.permutation(np.arange(n) % folds),
                        "model_version": "tfidf-lr-v1"})
    for label in LABELS:
        a, b = rng.normal(0, 1.5, n), rng.normal(0, 1.5, n)
        truth[label] = (rng.random(n) < 1 / (1 + np.exp(-(a + b - 2)))).astype("int8")
        teacher[f"logit_{label}"] = (6 * a).astype("float32")
        teacher[f"p_{label}"] = (1 / (1 + np.exp(-6 * a))).astype("float32")
        oof[f"p_{label}"] = (1 / (1 + np.exp(-(b - 1)))).astype("float32")
    return teacher, oof, truth


def test_cross_fit_scores_each_fold_with_a_stack_that_never_saw_it():
    teacher, oof, truth = synthetic()
    out = cross_fit_by_folds(teacher, oof, truth, "stack-v2", 1.0)
    assert out["id"].tolist() == sorted(teacher["id"]) and out["status"].eq("ok").all()
    held = set(oof.loc[oof["fold"] == 2, "id"])
    others = teacher[~teacher["id"].isin(held)]
    stack = fit_stack(others, oof, truth, unit_weights(others["id"]), "stack-v2", PV, MV,
                      "tfidf-lr-v1")
    expected = apply_stack(teacher[teacher["id"].isin(held)], oof, stack).set_index("id")
    got = out.set_index("id").loc[expected.index]
    np.testing.assert_allclose(got["logit_toxicity"], expected["logit_toxicity"], rtol=1e-6)
    full = fit_stack(teacher, oof, truth, unit_weights(teacher["id"]), "stack-v2", PV, MV,
                     "tfidf-lr-v1")
    in_sample = apply_stack(teacher, oof, full).set_index("id").loc[expected.index]
    assert not np.allclose(got["logit_toxicity"], in_sample["logit_toxicity"])


def boot(low, high):
    return {"mean_auc_ci95": [low, high]}


def test_choose():
    gate_ece = {"A": 0.010, "B": 0.009}
    assert choose({"gate_subset": boot(-0.001, 0.002), "civil_shift": boot(-0.002, 0.001)},
                  gate_ece, 0.005) == {"adopt": "B", "reasons": []}
    worse = choose({"gate_subset": boot(0.0, 0.002), "civil_shift": boot(-0.004, -0.001)},
                   gate_ece, 0.005)
    assert worse["adopt"] == "A" and "civil_shift" in worse["reasons"][0]
    assert choose({"gate_subset": boot(0.0, 0.002)}, {"A": 0.010, "B": 0.016},
                  0.005)["adopt"] == "A"


def test_upload_writes_valid_shards_and_resumes(tmp_path):
    teacher, oof, truth = synthetic(n=230)
    stack = fit_stack(teacher, oof, truth, unit_weights(teacher["id"]), "stack-v2", PV, MV,
                      "tfidf-lr-v1")
    soft = cross_fit_by_folds(teacher, oof, truth, "stack-v2", 1.0)
    mv = soft_model_version(stack)
    assert mv == f"stack-v2+{MV}+tfidf-lr-v1"
    plan = plan_shards(teacher["id"].tolist(), 100)
    frames = shard_frames(soft.sample(frac=1, random_state=0), plan, PV, mv)
    assert [len(f) for f in frames] == [100, 100, 30]

    storage, prefix = LocalStorage(tmp_path / "store"), soft_prefix("stack-v2", PV)
    extra = {"stack_version": "stack-v2", "code_commit": "abc"}
    first = upload_shards(storage, prefix, frames, PV, mv, "sha-1", extra, log=lambda m: None)
    assert first == {"written": ["shard_0000", "shard_0001", "shard_0002"], "skipped": []}
    loaded = load_run(storage, prefix, tmp_path / "cache")
    assert loaded[3] == []
    assert check_run(*loaded[:3], teacher["id"].tolist(), 100, PV, mv) == []
    manifest = json.loads((tmp_path / "store" / shard_paths(prefix, "shard_0001")[1])
                          .read_text())
    assert manifest["stack_sha256"] == "sha-1" and manifest["code_commit"] == "abc"

    again = upload_shards(storage, prefix, frames, PV, mv, "sha-1", extra, log=lambda m: None)
    assert again == {"written": [], "skipped": ["shard_0000", "shard_0001", "shard_0002"]}
    with pytest.raises(ValueError, match="another stack"):
        upload_shards(storage, prefix, frames, PV, mv, "sha-2", extra, log=lambda m: None)
