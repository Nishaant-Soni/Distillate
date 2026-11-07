"""Hyperparameter importance from the shared Optuna study (α, T, learning rate, epochs, length).

Reads the study the sweep wrote (any Optuna storage URL plus the study name) and reports how
much of the objective's variance each hyperparameter explains, with two evaluators so the
ranking can be checked against itself:

* ``fanova``: Optuna's default, functional ANOVA on a random forest fitted to the trials;
* ``mdi``: mean decrease in impurity of a random forest, a cruder but independent estimate.

Both use completed trials only, and only parameters present in every one of them (a parameter
that some trials don't sample can't be compared). Seeded, so a rerun gives the same numbers.
``trials`` is the per-trial table behind the plots of the objective against each parameter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from src.teacher.config import get_logger

logger = get_logger(__name__)

EVALUATORS = {"fanova": "FanovaImportanceEvaluator",
              "mdi": "MeanDecreaseImpurityImportanceEvaluator"}


def load_study(storage: str, study_name: str):
    """The study, read-only in practice: nothing here adds or changes trials."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.load_study(study_name=study_name, storage=storage)


def completed_trials(study) -> list:
    import optuna

    return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]


def objective_target(study, objective: int | None):
    """``None`` for a single-objective study; a multi-objective one needs ``objective``."""
    if len(study.directions) == 1:
        return None
    if objective is None:
        raise ValueError(f"the study has {len(study.directions)} objectives: set "
                         "optuna.objective to the index of the one to explain")
    return lambda trial: trial.values[objective]


def importances(study, evaluators: Sequence[str], seed: int, params: Sequence[str] | None = None,
                objective: int | None = None, min_trials: int = 10) -> pd.DataFrame:
    """One row per parameter, one column per evaluator (normalized to sum to 1), best first."""
    from optuna import importance

    trials = completed_trials(study)
    if len(trials) < min_trials:
        raise ValueError(f"{len(trials)} completed trials; importance needs at least "
                         f"{min_trials}")
    target = objective_target(study, objective)
    columns = {}
    for name in evaluators:
        if name not in EVALUATORS:
            raise ValueError(f"evaluator {name!r}: choose from {sorted(EVALUATORS)}")
        evaluator = getattr(importance, EVALUATORS[name])(seed=seed)
        columns[name] = importance.get_param_importances(
            study, evaluator=evaluator, params=list(params) if params else None, target=target)
    table = pd.DataFrame(columns).astype("float64")
    table.index.name = "param"
    return table.sort_values(evaluators[0], ascending=False)


def trials_table(study, objective: int | None = None) -> pd.DataFrame:
    """``number``, ``value`` (the explained objective) and one ``param`` column per parameter."""
    trials = completed_trials(study)
    rows = [{"number": t.number,
             "value": t.value if objective is None else t.values[objective],
             **t.params} for t in trials]
    return pd.DataFrame(rows).sort_values("number", ignore_index=True)


def summary(study, table: pd.DataFrame, objective: int | None = None) -> dict[str, Any]:
    """What the report states about the study: size, direction, best trial."""
    trials = completed_trials(study)
    index = 0 if objective is None else objective
    direction = study.directions[index].name.lower()
    values = [t.values[index] for t in trials]
    best = trials[values.index(max(values) if direction == "maximize" else min(values))]
    # Parameters sampled on a log scale (the learning rate), so plots can use a log axis.
    log_params = sorted({name for t in trials for name, dist in t.distributions.items()
                         if getattr(dist, "log", False)})
    return {"study_name": study.study_name, "trials": len(study.trials),
            "completed": len(trials), "direction": direction,
            "best_trial": best.number, "best_value": best.values[index],
            "best_params": best.params, "ranking": list(table.index),
            "log_params": log_params}


def hyperparameter_analysis(cfg: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Load the configured study and return ``importances``, ``trials`` and ``summary``."""
    study = load_study(cfg["storage"], cfg["study_name"])
    objective = cfg.get("objective")
    table = importances(study, cfg["evaluators"], seed, cfg.get("params"), objective,
                        int(cfg["min_trials"]))
    info = summary(study, table, objective)
    logger.info("study %s: %d completed trials, ranking %s", info["study_name"],
                info["completed"], info["ranking"])
    return {"importances": table, "trials": trials_table(study, objective), "summary": info}
