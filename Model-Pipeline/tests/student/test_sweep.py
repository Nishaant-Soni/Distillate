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
