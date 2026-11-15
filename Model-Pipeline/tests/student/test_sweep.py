"""The Optuna sweep on the tiny offline model: trials, model pruning, the setup guard, merge."""

import json
from pathlib import Path

import pytest

from src.student import sweep
from tests.student.conftest import smoke_cfg

optuna = pytest.importorskip("optuna")

FAST = ("mlflow.enabled=false", "train.max_steps=6")


def _cfg(root, *extra):
    return smoke_cfg(root, *FAST, f"sweep.storage=sqlite:///{root}/optuna.db", *extra)


def test_two_trials_complete_and_keep_only_the_best_weights(synthetic_root):
    summary = sweep.run(_cfg(synthetic_root))

    assert summary["trials"] == 2 and summary["states"] == {"COMPLETE": 2}
    best = summary["best"]
    assert 0 < best["value"] <= 1 and set(best["params"]) == set(sweep.PARAMS)
    assert any(o.startswith("loss.alpha=") for o in best["overrides"])
    runs = [Path(t["run_dir"]) for t in (best,)] + [
        Path(p) for p in (synthetic_root / "runs").iterdir()]
    kept = [r for r in set(runs) if (r / "model" / "model.safetensors").exists()]
    assert kept == [Path(best["run_dir"])]           # keep_models: best
    for run in set(runs):
        assert (run / "model" / "meta.json").exists()   # lineage is always kept
    assert json.loads((synthetic_root / "reports" / "sweep.json").read_text())["trials"] == 2


def test_sampled_values_reach_the_training_run(synthetic_root):
    seen = []

    def fake_run(cfg):
        seen.append(cfg)
        return {"best_val_mean_auc": 0.5 + cfg["loss"]["alpha"] / 10, "run_dir": None,
                "run_name": cfg["run_name"]}

    sweep.run(_cfg(synthetic_root, "sweep.n_trials=3"), train_run=fake_run)
    study = optuna.load_study(study_name="smoke", storage=f"sqlite:///{synthetic_root}/optuna.db")
    for cfg, trial in zip(seen, study.trials, strict=True):
        assert cfg["loss"]["alpha"] == trial.params["alpha"]
        assert cfg["train"]["epochs"] == trial.params["epochs"]
        assert cfg["model"]["max_length"] in (16, 32)
        assert cfg["mlflow"]["tags"]["study"] == "smoke"
        assert cfg["run_name"].endswith(f"-t{trial.number:03d}")


def test_a_study_refuses_trials_from_another_setup(synthetic_root):
    def fake_run(cfg):
        return {"best_val_mean_auc": 0.6, "run_dir": None, "run_name": cfg["run_name"]}

    sweep.run(_cfg(synthetic_root, "sweep.n_trials=1"), train_run=fake_run)
    with pytest.raises(ValueError, match="different setup"):
        sweep.run(_cfg(synthetic_root, "sweep.n_trials=1", "train.batch_size=8"),
                  train_run=fake_run)
    # A second worker with the same setup joins the same study.
    sweep.run(_cfg(synthetic_root, "sweep.n_trials=1", "sweep.worker=1"), train_run=fake_run)
    study = optuna.load_study(study_name="smoke", storage=f"sqlite:///{synthetic_root}/optuna.db")
    assert sorted(t.user_attrs["worker"] for t in study.trials) == [0, 1]


def test_gold_only_base_config_is_rejected(synthetic_root):
    with pytest.raises(ValueError, match="soft labels"):
        sweep.run(_cfg(synthetic_root, "loss.alpha=1.0"))


def test_merge_combines_two_halves_once(tmp_path, synthetic_root):
    def fake_run(cfg):
        return {"best_val_mean_auc": cfg["loss"]["alpha"], "run_dir": None,
                "run_name": cfg["run_name"]}

    halves = []
    for worker in (0, 1):
        url = f"sqlite:///{tmp_path}/half{worker}.db"
        sweep.run(smoke_cfg(synthetic_root, *FAST, f"sweep.storage={url}", "sweep.n_trials=2",
                            f"sweep.worker={worker}"), train_run=fake_run)
        halves.append(url)
    into = f"sqlite:///{tmp_path}/study.db"
    first = sweep.merge(into, halves, "smoke")
    again = sweep.merge(into, halves, "smoke")
    assert first["copied"] == 4 and again["copied"] == 0 and again["trials"] == 4
    assert first["best"]["value"] == max(t["value"] for t in first["trials_table"])
    assert first["setup"]["data_source"] == "temp"


def test_redact_hides_the_password():
    assert sweep.redact("postgresql://user:secret@10.0.0.2:5432/optuna") == \
        "postgresql://user:***@10.0.0.2:5432/optuna"
    assert sweep.redact("sqlite:////tmp/a.db") == "sqlite:////tmp/a.db"


def _flaky(cfg):
    """Fails every trial with alpha above 0.5 (as a CUDA out-of-memory error would)."""
    if cfg["loss"]["alpha"] > 0.5:
        raise RuntimeError("CUDA out of memory (simulated)")
    return {"best_val_mean_auc": cfg["loss"]["alpha"], "run_dir": None,
            "run_name": cfg["run_name"]}


def _half(root, tmp_path, worker, *extra, name=None):
    url = f"sqlite:///{tmp_path}/{name or f'half{worker}'}.db"
    sweep.run(smoke_cfg(root, *FAST, f"sweep.storage={url}", "sweep.n_trials=6",
                        f"sweep.worker={worker}", *extra), train_run=_flaky)
    return url


def test_merging_twice_copies_failed_trials_once(tmp_path, synthetic_root):
    halves = [_half(synthetic_root, tmp_path, w) for w in (0, 1)]
    into = f"sqlite:///{tmp_path}/study.db"
    first = sweep.merge(into, halves, "smoke")
    again = sweep.merge(into, halves, "smoke")
    assert first["states"].get("FAIL", 0) > 0                  # the case that used to double
    assert again["copied"] == 0 and again["trials"] == first["trials"] == 12
    keys = [t["trial_key"] for t in again["trials_table"]]
    assert len(set(keys)) == len(keys)


def test_failed_trials_carry_their_key_worker_seed_and_commit(tmp_path, synthetic_root):
    url = _half(synthetic_root, tmp_path, 1, "sweep.session=2")
    study = optuna.load_study(study_name="smoke", storage=url)
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    assert failed
    for t in study.trials:
        attrs = t.user_attrs
        assert attrs["trial_key"].startswith("w1-s2-") and attrs["worker"] == 1
        assert attrs["session"] == 2 and attrs["sampler_seed"] == 42 + 1 + 2000
        assert "code_commit" in attrs


def test_trials_without_keys_still_merge_once(tmp_path):
    """Studies written before trial_key existed (worker 1's half) dedupe by number and start."""
    old = f"sqlite:///{tmp_path}/old.db"
    study = optuna.create_study(study_name="smoke", storage=old, direction="maximize")
    study.set_user_attr("model", "tiny")
    study.optimize(lambda t: t.suggest_float("alpha", 0.1, 0.9), n_trials=3)
    into = f"sqlite:///{tmp_path}/study.db"
    assert sweep.merge(into, [old], "smoke")["copied"] == 3
    assert sweep.merge(into, [old], "smoke")["copied"] == 0


@pytest.mark.skipif(not hasattr(__import__("time"), "tzset"), reason="needs time.tzset")
def test_legacy_keys_do_not_depend_on_the_machine_time_zone(tmp_path, monkeypatch):
    """A warm start on Kaggle (UTC) stamps legacy keys; merging on a laptop in another time
    zone must find the same keys, or every warm-started trial is copied twice."""
    import time

    def keys_in(tz):
        monkeypatch.setenv("TZ", tz)
        time.tzset()
        study = optuna.load_study(study_name="smoke", storage=old)
        return [sweep.trial_key(t) for t in study.trials]

    old = f"sqlite:///{tmp_path}/old.db"
    study = optuna.create_study(study_name="smoke", storage=old, direction="maximize")
    study.set_user_attr("model", "tiny")
    study.optimize(lambda t: t.suggest_float("alpha", 0.1, 0.9), n_trials=2)
    try:
        on_kaggle = keys_in("UTC")
        # The other half warm-started on Kaggle: its copies carry the UTC keys.
        warm = f"sqlite:///{tmp_path}/warm.db"
        assert sweep.merge(warm, [old], "smoke")["copied"] == 2
        assert keys_in("America/New_York") == on_kaggle
        into = f"sqlite:///{tmp_path}/study.db"
        assert sweep.merge(into, [warm, old], "smoke")["copied"] == 2
    finally:
        monkeypatch.undo()
        time.tzset()


def test_a_new_session_samples_new_random_trials(tmp_path, synthetic_root):
    first = _half(synthetic_root, tmp_path, 0, name="a")
    rerun = _half(synthetic_root, tmp_path, 0, name="b")
    fresh = _half(synthetic_root, tmp_path, 0, "sweep.session=1", name="c")

    def params(url):
        return [t.params for t in optuna.load_study(study_name="smoke", storage=url).trials]

    assert params(first) == params(rerun)            # same session: same draws, by design
    assert params(first)[0] != params(fresh)[0]


def test_warm_start_copies_the_other_half_and_the_merge_counts_each_trial_once(
        tmp_path, synthetic_root):
    other = _half(synthetic_root, tmp_path, 1)
    url = f"sqlite:///{tmp_path}/half0.db"
    summary = sweep.run(smoke_cfg(synthetic_root, *FAST, f"sweep.storage={url}",
                                  "sweep.n_trials=3", "sweep.n_startup_trials=4",
                                  f"sweep.warm_start=[{other}]"), train_run=_flaky)
    assert summary["warm_start_copied"] == 6 and summary["trials"] == 9
    study = optuna.load_study(study_name="smoke", storage=url)
    own = [t for t in study.trials if t.user_attrs.get("worker") == 0]
    assert len(own) == 3            # 6 finished trials already, so TPE from the first one
    merged = sweep.merge(f"sqlite:///{tmp_path}/study.db", [url, other], "smoke")
    assert merged["trials"] == 9


def test_warm_start_from_another_setup_is_refused(tmp_path, synthetic_root):
    other = _half(synthetic_root, tmp_path, 1, "train.batch_size=8")
    with pytest.raises(ValueError, match="setup differs"):
        sweep.run(smoke_cfg(synthetic_root, *FAST, f"sweep.storage=sqlite:///{tmp_path}/w0.db",
                            f"sweep.warm_start={other}"), train_run=_flaky)


def test_the_budget_stops_before_a_trial_that_would_overrun():
    from datetime import datetime, timedelta
    from types import SimpleNamespace

    class Study:
        stopped = False

        def stop(self):
            self.stopped = True

    now = datetime(2026, 11, 10)
    hour_long = SimpleNamespace(datetime_start=now, datetime_complete=now + timedelta(hours=1))
    started = __import__("time").perf_counter()
    roomy, tight = Study(), Study()
    sweep.budget_callback(3 * 3600, started)(roomy, hour_long)        # 0 h used + 1 h < 3 h
    sweep.budget_callback(3 * 3600, started - 2.5 * 3600)(tight, hour_long)   # 2.5 + 1 > 3
    assert not roomy.stopped and tight.stopped
