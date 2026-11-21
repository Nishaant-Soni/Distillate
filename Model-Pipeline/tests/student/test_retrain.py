"""Retraining rows (docs/retraining.md): drift batches with their soft labels, obfuscated train
rows, and the settings' checks. Off by default."""

import pandas as pd
import pytest

from src.student import retrain
from src.student.data import attach_soft_labels, load_soft_labels, read_rows
from src.student.synthetic import write_drift
from tests.student.conftest import smoke_cfg

LABELS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]


def cfg_with(root, *overrides):
    return smoke_cfg(root, f"retrain.drift_dir={root}/drift", *overrides)


def soft_train(root, cfg):
    train = read_rows(root / "wiki_train_raw.parquet")
    labels, _ = load_soft_labels(cfg)
    return attach_soft_labels(train, labels)


def test_off_by_default(synthetic_root):
    cfg = smoke_cfg(synthetic_root)
    assert not retrain.enabled(cfg) and retrain.holdout_table(cfg) is None


@pytest.mark.parametrize("overrides, message", [
    (["retrain.drift_batches=[1,2]"], "0, 1, ..., m-1"),
    (["retrain.drift_batches=[0,1]", "retrain.holdout_batch=1"], "also a training batch"),
    (["retrain.drift_batches=[0,1]", "retrain.holdout_batch=5"], "right after the training"),
    (["retrain.holdout_batch=2"], "right after the training"),
    (["retrain.drift_weight=0"], "positive"),
    (["retrain.obfuscation_augment.rows=-1"], "negative"),
])
def test_settings_are_checked(synthetic_root, overrides, message):
    with pytest.raises(ValueError, match=message):
        retrain.settings(smoke_cfg(synthetic_root, *overrides))


def test_drift_batches_join_with_their_own_soft_labels(synthetic_root):
    ids = write_drift(synthetic_root / "drift", synthetic_root / "store", batches=3, n=50)
    cfg = cfg_with(synthetic_root, "retrain.drift_batches=[0,1]", "retrain.holdout_batch=2",
                   "retrain.drift_weight=2.0")
    train = soft_train(synthetic_root, cfg)
    out, info = retrain.extend_training(cfg, train, soft=True)

    drift_ids = set(ids[0]) | set(ids[1])
    assert set(out["id"]) == set(train["id"]) | drift_ids
    assert not set(ids[2]) & set(out["id"])                     # the held-out batch: never
    assert info["drift_rows"] == 100 and info["holdout_batch"] == 2
    drift = out[out["id"].isin(drift_ids)]
    assert (drift["sample_weight"] == 2.0).all() and drift["soft_ok"].all()
    assert (out.loc[out["id"].isin(train["id"]), "sample_weight"] == 1.0).all()
    # each drift row's teacher logits are its own shard's
    shard = pd.read_parquet(next((synthetic_root / "store").rglob("drift/shard_0001.parquet")))
    row = drift.set_index("id").loc[shard["id"].iloc[0]]
    assert row["logit_toxicity"] == pytest.approx(float(shard["logit_toxicity"].iloc[0]))
    assert list(out.columns) == list(train.columns) and out["id"].is_monotonic_increasing


def test_gold_only_runs_need_no_drift_shards(synthetic_root):
    write_drift(synthetic_root / "drift", synthetic_root / "nostore", batches=2, n=30)
    cfg = cfg_with(synthetic_root, "retrain.drift_batches=[0,1]")
    train = read_rows(synthetic_root / "wiki_train_raw.parquet")
    out, info = retrain.extend_training(cfg, train, soft=False)
    assert info["drift_rows"] == 60 and len(out) == len(train) + 60


def test_a_drift_shard_from_another_stack_is_refused(synthetic_root):
    write_drift(synthetic_root / "drift", synthetic_root / "store", batches=2, n=30,
                stack_sha256="0" * 64)
    cfg = cfg_with(synthetic_root, "retrain.drift_batches=[0,1]")
    with pytest.raises(ValueError, match="stack_sha256"):
        retrain.extend_training(cfg, soft_train(synthetic_root, cfg), soft=True)


def test_a_missing_batch_file_or_shard_fails_clearly(synthetic_root):
    write_drift(synthetic_root / "drift", synthetic_root / "store", batches=1, n=30)
    cfg = cfg_with(synthetic_root, "retrain.drift_batches=[0,1]")
    with pytest.raises(FileNotFoundError, match="batch_01"):
        retrain.extend_training(cfg, soft_train(synthetic_root, cfg), soft=True)


def test_obfuscated_rows_keep_their_labels_and_soft_labels(synthetic_root):
    cfg = cfg_with(synthetic_root, "retrain.obfuscation_augment.rows=40")
    train = soft_train(synthetic_root, cfg)
    out, info = retrain.extend_training(cfg, train, soft=True)
    added = out[out["id"].str.endswith("_aug")]
    assert 0 < len(added) <= 40 and info["obfuscation_augment"]["rows"] == len(added)
    assert sum(info["obfuscation_augment"]["techniques"].values()) == len(added)
    originals = train.set_index("id").loc[added["id"].str.removesuffix("_aug")]
    assert (originals["toxicity"] == 1).all()
    for column in [*LABELS, "logit_toxicity", "soft_ok", "sample_weight"]:
        assert list(added[column]) == list(originals[column]), column
    assert (added["text"].to_numpy() != originals["text"].to_numpy()).all()


def test_augmentation_is_seeded_and_ignores_row_order(synthetic_root):
    train = read_rows(synthetic_root / "wiki_train_raw.parquet")
    a, _ = retrain.augment(train, 25, seed=42)
    b, _ = retrain.augment(train.sample(frac=1, random_state=3), 25, seed=42)
    assert a.equals(b)
    c, _ = retrain.augment(train, 25, seed=7)
    assert set(c["id"]) != set(a["id"])


def test_the_held_out_batch_is_read_whole(synthetic_root):
    ids = write_drift(synthetic_root / "drift", synthetic_root / "store", batches=3, n=40)
    cfg = cfg_with(synthetic_root, "retrain.drift_batches=[0,1]", "retrain.holdout_batch=2")
    holdout = retrain.holdout_table(cfg)
    assert sorted(holdout["id"]) == sorted(ids[2]) and set(LABELS) <= set(holdout.columns)
