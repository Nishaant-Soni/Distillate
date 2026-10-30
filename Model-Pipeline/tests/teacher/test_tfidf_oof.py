import numpy as np
import pandas as pd
import pytest

from src.teacher.config import LABELS, load_config
from src.teacher.tfidf_oof import assign_folds, oof_predict


def test_folds_are_balanced_and_deterministic():
    ids = [f"{i:05x}" for i in range(1003)]
    folds = assign_folds(list(reversed(ids)), 5, seed=42)
    assert folds.equals(assign_folds(ids, 5, seed=42))
    assert folds["id"].tolist() == sorted(ids)
    assert folds["fold"].value_counts().sort_index().tolist() == [201, 201, 201, 200, 200]
    assert not folds.equals(assign_folds(ids, 5, seed=7))
    with pytest.raises(ValueError, match="duplicates"):
        assign_folds(ids + ids[:1], 5, seed=42)


def test_every_row_scored_by_a_model_that_never_saw_it():
    rng = np.random.default_rng(0)
    n = 60
    train = pd.DataFrame({"id": [f"r{i:03d}" for i in range(n)], "text": "x"})
    for label in LABELS:
        train[label] = rng.integers(0, 2, n)
    seen = {}

    def fake_train_predict(fit, predict, params, seed):
        out = pd.DataFrame({"id": predict["id"].to_numpy()})
        for i in out["id"]:
            seen[i] = set(fit["id"])
        for label in LABELS:
            out[f"p_{label}"] = np.float32(0.5)
        out["model_version"] = params["model_version"]
        return out

    folds = assign_folds(train["id"].tolist(), 5, seed=42)
    oof = oof_predict(train, folds, {"model_version": "tfidf-lr-v1"}, 42, fake_train_predict,
                      log=lambda line: None)
    assert oof["id"].tolist() == sorted(train["id"]) and len(oof) == n
    assert list(oof.columns) == ["id", "fold", *[f"p_{k}" for k in LABELS], "model_version"]
    fold = dict(zip(folds["id"], folds["fold"], strict=True))
    for i in train["id"]:
        assert i not in seen[i]                                   # never trained on itself
        assert len(seen[i]) == n - (folds["fold"] == fold[i]).sum()   # trained on the rest
    assert (oof.set_index("id")["fold"] == folds.set_index("id")["fold"]).all()


def test_rows_without_a_fold_raise():
    train = pd.DataFrame({"id": ["a", "b"], "text": "x"})
    with pytest.raises(ValueError, match="fold"):
        oof_predict(train, assign_folds(["a"], 2, 42), {"model_version": "v"}, 42,
                    lambda *a: None, log=lambda line: None)


def test_config():
    cfg = load_config()
    assert cfg["oof"]["folds"] == 5 and cfg["paths"]["oof_dir"].endswith("artifacts/oof")
