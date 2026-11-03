import copy
import json

import numpy as np
import pandas as pd
import pytest

from src.teacher.calibrate import (
    apply_calibration,
    cross_fit_log_odds,
    fit_calibration,
    save_calibration,
)
from src.teacher.calqwen import calqwen_model_version, run
from src.teacher.config import LABELS, load_config
from src.teacher.metrics import evaluate
from src.teacher.shards import build_manifest, plan_record, plan_shards, shard_frame, shard_name
from src.teacher.soft_labels import sha256_file, soft_prefix
from src.teacher.storage import LocalStorage, shard_paths, write_shard
from src.teacher.validate_local import check_run, load_run

PV, MV = "p2-zs-v2b", "Qwen/Qwen3-4B-Instruct-2507@abc"
TEACHER = f"teacher/{PV}/"


def overconfident(ids, rng):
    """Gold labels from true log-odds t; the teacher reports 4 * t + 3 (too sure, biased)."""
    truth = pd.DataFrame({"id": ids, "text": "x"})
    pred = pd.DataFrame({"id": ids, "status": "ok"})
    for label in LABELS:
        t = rng.normal(-2.0, 2.0, len(ids))
        truth[label] = (rng.random(len(ids)) < 1 / (1 + np.exp(-t))).astype("int8")
        pred[f"logit_{label}"] = (4 * t + 3).astype("float32")
        pred[f"p_{label}"] = (1 / (1 + np.exp(-(4 * t + 3)))).astype("float32")
    return truth, pred


def write_run(storage, prefix, pred, shard_size):
    for k, ids in enumerate(plan_shards(pred["id"].tolist(), shard_size)):
        frame = shard_frame(pred[pred["id"].isin(ids)], PV, MV)
        write_shard(storage, prefix, shard_name(k), frame, build_manifest(frame, PV, MV))


def setup(tmp_path, n_train=900, n_pilot=300, n_test=400, n_gate=200, seed=0):
    """Config, store, and committed-style files for ``calqwen.run`` on synthetic data.

    One pilot row and one other training row failed, so the shards must keep them as NaN.
    """
    rng = np.random.default_rng(seed)
    cfg = copy.deepcopy(load_config())
    for key, sub in (("data_dir", "data"), ("pilot_dir", "pilot"), ("calqwen_dir", "calqwen")):
        cfg["paths"][key] = str(tmp_path / sub)
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    cfg["paths"]["gate_ids"] = str(tmp_path / "data" / "gate_ids.parquet")
    cfg["paths"]["shard_plan"] = str(tmp_path / "data" / "teacher_shards.json")
    cfg["labeling"]["shard_size"] = 250

    train_ids = [f"{i:06x}" for i in range(n_train)]
    test_ids = [f"t{i:05d}" for i in range(n_test)]
    train, teacher = overconfident(train_ids, rng)
    test, test_teacher = overconfident(test_ids, rng)
    pilot_ids = pd.DataFrame({"id": rng.choice(train_ids, n_pilot, replace=False)})
    pilot_ids["weight"] = rng.uniform(1, 50, n_pilot)
    gate_ids = pd.DataFrame({"id": sorted(rng.choice(test_ids, n_gate, replace=False)),
                             "stratum": "rest"})
    gate_ids["weight"] = rng.uniform(1, 3, n_gate)
    failed = [pilot_ids["id"].iloc[0], next(i for i in train_ids if i not in
                                            set(pilot_ids["id"]))]
    teacher.loc[teacher["id"].isin(failed), "status"] = "failed"
    teacher.loc[teacher["id"].isin(failed),
                [f"{c}_{label}" for c in ("p", "logit") for label in LABELS]] = np.nan

    files = cfg["data"]["files"]
    train.to_parquet(tmp_path / "data" / files["train"], index=False)
    test.to_parquet(tmp_path / "data" / files["test"], index=False)
    gate_ids.to_parquet(cfg["paths"]["gate_ids"], index=False)
    pilot_ids.to_parquet(tmp_path / "pilot" / "pilot_ids.parquet", index=False)
    plan = plan_shards(train_ids, cfg["labeling"]["shard_size"])
    (tmp_path / "data" / "teacher_shards.json").write_text(
        json.dumps(plan_record(plan, "train", cfg["labeling"]["shard_size"])))

    # The pilot run's calibration: fitted on the pilot rows with their weights.
    pilot = teacher[teacher["id"].isin(set(pilot_ids["id"]))].assign(prompt_version=PV,
                                                                     model_version=MV)
    calibration = fit_calibration(pilot, train[train["id"].isin(set(pilot_ids["id"]))],
                                  pilot_ids, "platt-v1", PV, MV)
    save_calibration(calibration, tmp_path / "pilot" / f"calibration_{PV}.json")

    storage = LocalStorage(tmp_path / "store")
    write_run(storage, TEACHER, teacher, cfg["labeling"]["shard_size"])
    write_run(storage, f"{TEACHER}gate/", test_teacher[test_teacher["id"].isin(
        set(gate_ids["id"]))], cfg["labeling"]["shard_size"])
    return cfg, storage, {"train": train, "teacher": teacher, "pilot_ids": pilot_ids,
                          "gate_ids": gate_ids, "test": test, "test_teacher": test_teacher,
                          "calibration": calibration, "failed": failed}


@pytest.fixture
def pipeline_root(tmp_path, monkeypatch):
    monkeypatch.setattr("src.teacher.calqwen.PIPELINE_ROOT", tmp_path)


def test_shards_validate_and_record_the_calibration(tmp_path, pipeline_root):
    cfg, storage, d = setup(tmp_path)
    out = LocalStorage(tmp_path / "out")
    report = run(cfg, storage, out, code_commit="abc")

    mv = calqwen_model_version("calqwen-v1", d["calibration"])
    assert mv == f"calqwen-v1+{MV}"
    prefix = soft_prefix("calqwen-v1", PV)
    assert report["upload"]["prefix"] == "soft_labels/calqwen-v1/p2-zs-v2b/"
    for sub, ids in (("", d["train"]["id"]), ("gate/", d["gate_ids"]["id"])):
        loaded = load_run(out, f"{prefix}{sub}", tmp_path / "cache")
        assert loaded[3] == []
        assert check_run(*loaded[:3], ids.tolist(), 250, PV, mv) == []
    manifest = json.loads((tmp_path / "out" / shard_paths(prefix, "shard_0001")[1]).read_text())
    cal_sha = sha256_file(tmp_path / "pilot" / f"calibration_{PV}.json")
    assert manifest["calibration_sha256"] == cal_sha and "stack_sha256" not in manifest
    assert manifest["calibration_file"] == f"pilot/calibration_{PV}.json"
    assert manifest["code_commit"] == "abc" and manifest["cross_fitted"].startswith("pilot rows")
    gate_manifest = json.loads((tmp_path / "out" / shard_paths(f"{prefix}gate/",
                                                               "shard_0000")[1]).read_text())
    assert gate_manifest["cross_fitted"] == "no (held-out rows)"
    assert gate_manifest["teacher_prefix"] == f"{TEACHER}gate/"

    saved = json.loads((tmp_path / "calqwen" / "calqwen_check.json").read_text())
    assert saved["calibration_sha256"] == cal_sha and "upload" not in saved
    # Same ranking as raw Qwen on the frozen rows; much better calibrated.
    assert saved["train"]["ece"] < saved["train_teacher_raw"]["ece"] / 3
    assert saved["gate_subset"]["mean_auc"] == pytest.approx(
        saved["gate_subset_teacher_raw"]["mean_auc"], abs=1e-6)

    again = run(cfg, storage, out, code_commit="abc")
    assert again["upload"]["train"]["written"] == []
    assert again["upload"]["gate"]["written"] == []


def shards(tmp_path, prefix):
    frames, _, _, errors = load_run(LocalStorage(tmp_path / "out"), prefix, tmp_path / "cache")
    assert errors == []
    return pd.concat(frames.values()).set_index("id")


def test_frozen_rows_get_the_pilot_calibration_and_pilot_rows_are_cross_fitted(
        tmp_path, pipeline_root):
    cfg, storage, d = setup(tmp_path)
    run(cfg, storage, LocalStorage(tmp_path / "out"))
    got = shards(tmp_path, soft_prefix("calqwen-v1", PV))
    cal = d["calibration"]
    pilot = set(d["pilot_ids"]["id"])
    teacher = d["teacher"].set_index("id")

    # Every other training row: the frozen calibration, p and log-odds together.
    rest = teacher[~teacher.index.isin(pilot) & (teacher["status"] == "ok")]
    frozen = apply_calibration(rest.reset_index(), cal).set_index("id")
    for label in LABELS:
        np.testing.assert_allclose(got.loc[rest.index, f"p_{label}"], frozen[f"p_{label}"],
                                   rtol=1e-6)
        c = cal["labels"][label]
        np.testing.assert_allclose(got.loc[rest.index, f"logit_{label}"],
                                   c["slope"] * rest[f"logit_{label}"] + c["intercept"],
                                   rtol=1e-5)

    # Pilot rows: out of fold, so not the in-sample calibration's values.
    pilot_rows = teacher[teacher.index.isin(pilot)].reset_index()
    oof = cross_fit_log_odds(pilot_rows, d["train"][d["train"]["id"].isin(pilot)],
                             d["pilot_ids"], 5, 42)
    np.testing.assert_allclose(got.loc[pilot_rows["id"], "logit_toxicity"],
                               oof["toxicity"].astype("float32"), rtol=1e-6)
    ok = pilot_rows[pilot_rows["status"] == "ok"]
    in_sample = apply_calibration(ok, cal).set_index("id")
    assert not np.allclose(got.loc[ok["id"], "p_toxicity"], in_sample["p_toxicity"])

    # Failed rows stay failed, with NaN scores.
    assert got.loc[d["failed"], "status"].eq("failed").all()
    assert got.loc[d["failed"], [f"p_{label}" for label in LABELS]].isna().all(axis=None)


def test_ranking_is_raw_qwens_on_frozen_rows(tmp_path, pipeline_root):
    cfg, storage, d = setup(tmp_path)
    run(cfg, storage, LocalStorage(tmp_path / "out"))
    got = shards(tmp_path, soft_prefix("calqwen-v1", PV)).reset_index()
    rest = d["train"][~d["train"]["id"].isin(set(d["pilot_ids"]["id"]))]
    calibrated, raw = evaluate(got, rest), evaluate(d["teacher"], rest)
    np.testing.assert_allclose(calibrated["auc"], raw["auc"], atol=1e-9)


def test_refuses_a_calibration_for_another_teacher(tmp_path, pipeline_root):
    cfg, storage, d = setup(tmp_path)
    cal = dict(d["calibration"], model_version="Qwen/other@xyz")
    save_calibration(cal, tmp_path / "pilot" / f"calibration_{PV}.json")
    with pytest.raises(ValueError, match="model_version"):
        run(cfg, storage)


def test_refuses_shards_written_by_another_calibration(tmp_path, pipeline_root):
    cfg, storage, d = setup(tmp_path)
    out = LocalStorage(tmp_path / "out")
    run(cfg, storage, out)
    cal = dict(d["calibration"])
    cal["labels"] = {**cal["labels"], "insult": {**cal["labels"]["insult"], "slope": 0.3}}
    save_calibration(cal, tmp_path / "pilot" / f"calibration_{PV}.json")
    with pytest.raises(ValueError, match="another calibration"):
        run(cfg, storage, out)
