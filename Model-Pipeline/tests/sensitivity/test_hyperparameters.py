import pytest

optuna = pytest.importorskip("optuna")

from src.sensitivity.hyperparameters import (  # noqa: E402
    hyperparameter_analysis,
    importances,
    stability,
    summary,
    trials_table,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial):
    """A made-up sweep where alpha matters most, temperature some, and epochs barely."""
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    temperature = trial.suggest_float("temperature", 1.0, 4.0)
    epochs = trial.suggest_int("epochs", 1, 4)
    return 0.99 - 0.1 * (alpha - 0.3) ** 2 - 0.004 * (temperature - 2.0) ** 2 + 1e-5 * epochs


def make_study(storage=None, name="sweep", n_trials=40, prune=0):
    study = optuna.create_study(study_name=name, storage=storage, direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials)
    for _ in range(prune):
        trial = study.ask()
        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    return study


def test_importances_rank_the_parameter_that_matters_first():
    table = importances(make_study(prune=3), ["fanova", "mdi"], seed=42)
    assert list(table.columns) == ["fanova", "mdi"]
    assert table.index[0] == "alpha" and table.index[-1] == "epochs"
    assert table.sum().round(6).tolist() == [1.0, 1.0]
    assert table.loc["alpha", "mdi"] > table.loc["epochs", "mdi"]
    # Seeded: the same numbers again.
    assert table.equals(importances(make_study(prune=3), ["fanova", "mdi"], seed=42))


def test_params_and_checks():
    study = make_study()
    assert list(importances(study, ["fanova"], 42, params=["alpha", "epochs"]).index) == \
        ["alpha", "epochs"]
    with pytest.raises(ValueError, match="evaluator"):
        importances(study, ["shap"], 42)
    with pytest.raises(ValueError, match="at least 50"):
        importances(study, ["fanova"], 42, min_trials=50)


def test_multi_objective_needs_the_objective():
    study = optuna.create_study(directions=["maximize", "minimize"],
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(lambda t: (objective(t), t.params["epochs"]), n_trials=20)
    with pytest.raises(ValueError, match="2 objectives"):
        importances(study, ["fanova"], 42)
    table = importances(study, ["fanova"], 42, objective=1)
    assert table.index[0] == "epochs"                     # the second objective is epochs
    assert trials_table(study, objective=1)["value"].equals(
        trials_table(study, objective=1)["epochs"].astype(float))


def test_trials_table_and_summary():
    study = make_study(prune=2)
    table = trials_table(study)
    assert len(table) == 40 and {"number", "value", "alpha", "temperature", "epochs"} <= \
        set(table.columns)
    info = summary(study, importances(study, ["fanova"], 42))
    assert info["completed"] == 40 and info["trials"] == 42
    assert info["direction"] == "maximize"
    assert info["best_value"] == table["value"].max()
    assert info["ranking"][0] == "alpha"


def test_log_params_and_best_params_in_the_report():
    from src.sensitivity.report import hyperparameter_section

    def with_lr(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
        return objective(trial) + lr

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(with_lr, n_trials=12)
    table = importances(study, ["fanova"], 42)
    info = summary(study, table)
    assert info["log_params"] == ["lr"]
    results = {"hyperparameters": {"summary": info,
                                   "importances": table.reset_index().to_dict("records")}}
    text = hyperparameter_section(results, {"importance": "i.png", "trials": "t.png"})
    # Significant digits: a learning rate such as 7.9e-05 isn't rounded to 0.0001.
    assert f"lr = {info['best_params']['lr']:.4g}" in text and "e-05" in text


def test_stability_over_seeds_in_the_report():
    from src.sensitivity.report import hyperparameter_section
    from src.sensitivity.run import records

    study = make_study()
    table = importances(study, ["fanova", "mdi"], seed=42)
    spread = stability(study, ["fanova", "mdi"], seeds=range(42, 45))
    assert spread["seeds"] == [42, 43, 44]
    ranges = spread["ranges"]
    assert list(ranges.columns) == ["fanova_min", "fanova_max", "mdi_min", "mdi_max"]
    # The seed-42 values are part of the range, so each sits inside its own.
    assert ((ranges["fanova_min"] <= table["fanova"])
            & (table["fanova"] <= ranges["fanova_max"])).all()
    # alpha dominates the made-up objective, so every seed ranks it first.
    assert spread["top"] == {"fanova": {"alpha": 3}, "mdi": {"alpha": 3}}

    results = {"hyperparameters": {
        "summary": summary(study, table), "importances": records(table.reset_index()),
        "stability": {**spread, "ranges": records(ranges.reset_index())}}}
    text = hyperparameter_section(results, {"importance": "i.png", "trials": "t.png"})
    assert "fanova (range over seeds)" in text
    low, high = ranges.loc["alpha", "fanova_min"], ranges.loc["alpha", "fanova_max"]
    assert f"({low:.3f} to {high:.3f})" in text
    assert ("under 3 seeds (42 to 44, the first being the one above). `alpha` is ranked "
            "first in 3 of 3 (fanova)") in text


def test_reads_the_study_from_storage(tmp_path):
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    make_study(storage=storage, name="distilbert-mixed")
    cfg = {"storage": storage, "study_name": "distilbert-mixed", "objective": None,
           "params": None, "evaluators": ["fanova", "mdi"], "min_trials": 10}
    out = hyperparameter_analysis(cfg, seed=42)
    assert out["summary"]["study_name"] == "distilbert-mixed"
    assert out["importances"].index[0] == "alpha" and len(out["trials"]) == 40
