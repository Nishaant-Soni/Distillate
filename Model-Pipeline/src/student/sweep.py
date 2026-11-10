"""Optuna sweep over the student's distillation recipe (v4 §5, docs/tuning.md).

TPE over ``alpha``, ``temperature``, learning rate, epochs, and maximum length, on the base
config (DistilBERT, mixed loss, the stack's soft labels). Each trial is one ``train.run()``
with the sampled values, logged to MLflow like any run (tags ``study`` and ``trial``); the
objective is that run's best validation mean AUC. Validation chooses: the gate set is never
read.

Several people share one study through the storage URL (``sweep.storage``): each runs its
own ``sweep.worker`` (sampler seeded ``seed + worker``) and its own ``n_trials``. Every
worker checks the study's recorded search space, base recipe, and data source, so trials
from different setups never mix. Without shared storage, each half runs on its own SQLite
file and ``merge`` copies the completed trials into one study for the importance analysis.

    python -m src.student.sweep --set data.source=ids                  # worker 0
    python -m src.student.sweep --worker 1 --storage postgresql://...  # the second half
    python -m src.student.sweep merge --into sqlite:///study.db --from sqlite:///a.db sqlite:///b.db
    python -m src.student.sweep best --storage sqlite:///study.db      # best values as --set flags
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.student.config import load_config, uses_soft_labels
from src.teacher.config import get_logger

logger = get_logger(__name__)
TASK_ID = "sweep"

# Search-space name -> config key it sets.
PARAMS = {"alpha": ("loss", "alpha"), "temperature": ("loss", "temperature"),
          "lr": ("train", "lr"), "epochs": ("train", "epochs"),
          "max_length": ("model", "max_length")}
INT_PARAMS = ("epochs",)
# The Phase 1 recipe with a middle alpha and T: worker 0's first trial, so the study always
# has one known-reasonable point.
DEFAULT_TRIAL = {"alpha": 0.5, "temperature": 2.0, "lr": 5e-5, "epochs": 3, "max_length": 128}
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def storage_url(cfg: Mapping[str, Any]) -> str:
    """``sweep.storage``, or a SQLite file in ``paths.out_dir``."""
    if cfg["sweep"].get("storage"):
        return str(cfg["sweep"]["storage"])
    out = Path(cfg["paths"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{out / 'optuna.db'}"


def redact(url: str) -> str:
    """The storage URL without a password, for logs and summaries."""
    return re.sub(r"(://[^:/@]+):[^@]*@", r"\1:***@", url)


def check_space(space: Mapping[str, Any]) -> None:
    unknown, missing = sorted(set(space) - set(PARAMS)), sorted(set(PARAMS) - set(space))
    if unknown or missing:
        raise ValueError(f"sweep.space: unknown {unknown}, missing {missing}")
    for name, spec in space.items():
        if isinstance(spec, list):
            if not spec:
                raise ValueError(f"sweep.space.{name}: empty choice list")
        elif not (isinstance(spec, Mapping) and "low" in spec and "high" in spec
                  and spec["low"] <= spec["high"]):
            raise ValueError(f"sweep.space.{name} must be a list or {{low, high[, log]}}")


def suggest(trial, space: Mapping[str, Any]) -> dict[str, Any]:
    """The trial's sampled values, one per search-space entry."""
    params: dict[str, Any] = {}
    for name in PARAMS:
        spec = space[name]
        if isinstance(spec, list):
            params[name] = trial.suggest_categorical(name, spec)
        elif name in INT_PARAMS:
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        else:
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]),
                                               log=bool(spec.get("log", False)))
    return params


def in_space(params: Mapping[str, Any], space: Mapping[str, Any]) -> bool:
    for name, value in params.items():
        spec = space[name]
        if isinstance(spec, list):
            if value not in spec:
                return False
        elif not spec["low"] <= value <= spec["high"]:
            return False
    return True


def recipe(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """What must be the same for every trial in one study: the base config minus the
    sampled values, the run-specific paths, and the per-worker settings."""
    base = copy.deepcopy({k: v for k, v in cfg.items()
                          if k not in ("paths", "run_name", "sweep", "mlflow", "storage")})
    for section, key in PARAMS.values():
        base[section].pop(key, None)
    base["train"].pop("device", None)
    base["train"].pop("num_workers", None)
    base["soft_labels"].pop("local_root", None)
    base["soft_labels"].pop("check_file", None)
    base["soft_labels"].pop("arms", None)
    return base


def study_attrs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    text = json.dumps(recipe(cfg), sort_keys=True, default=str)
    return {"space": json.dumps(cfg["sweep"]["space"], sort_keys=True),
            "recipe_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "data_source": cfg["data"]["source"],
            "soft_label_arm": cfg["soft_labels"]["arm"] if uses_soft_labels(cfg) else "none",
            "model": cfg["model"]["name"],
            "objective": "best validation mean AUC (unweighted, six labels)",
            "direction": "maximize"}


def claim_study(study, cfg: Mapping[str, Any]) -> None:
    """Record this setup on a new study, or fail if an existing study used another one."""
    want = study_attrs(cfg)
    have = study.user_attrs
    if not have:
        for key, value in want.items():
            study.set_user_attr(key, value)
        return
    diff = sorted(k for k in want if have.get(k) != want[k])
    if diff:
        raise ValueError(f"study {study.study_name!r} was started with a different setup ({diff}); "
                         "use the same config and space, or another sweep.study_name")


def trial_config(cfg: Mapping[str, Any], params: Mapping[str, Any], number: int) -> dict:
    out = copy.deepcopy(dict(cfg))
    for name, value in params.items():
        section, key = PARAMS[name]
        out[section][key] = value
    sw = cfg["sweep"]
    out["run_name"] = f"{cfg['run_name']}-{sw['study_name']}-w{sw['worker']}-t{number:03d}"
    out["mlflow"]["tags"] = {**(cfg["mlflow"].get("tags") or {}),
                             "study": sw["study_name"], "trial": str(number),
                             "sweep_worker": str(sw["worker"])}
    return out


def drop_weights(run_dir: str | Path | None) -> None:
    if not run_dir:
        return
    for name in WEIGHT_FILES:
        path = Path(run_dir) / "model" / name
        if path.exists():
            path.unlink()


def best_completed(study, exclude: int | None = None):
    import optuna

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            and t.number != exclude and t.value is not None]
    return max(done, key=lambda t: t.value) if done else None


def make_objective(cfg: Mapping[str, Any], train_run=None):
    """The Optuna objective: one training run per trial, returning its best val mean AUC."""
    if train_run is None:
        from src.student.train import run as train_run
    keep = cfg["sweep"].get("keep_models", "best")
    if keep not in ("best", "all", "none"):
        raise ValueError(f"sweep.keep_models must be best, all, or none, got {keep!r}")

    def objective(trial) -> float:
        params = suggest(trial, cfg["sweep"]["space"])
        previous = best_completed(trial.study)
        summary = None
        try:
            summary = train_run(trial_config(cfg, params, trial.number))
        finally:
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        value = float(summary["best_val_mean_auc"])
        for key in ("run_name", "run_dir", "model_version", "mlflow_run_id", "best_epoch",
                    "best_val_mean_pr_auc", "duration_s", "truncated_share_train"):
            trial.set_user_attr(key, summary.get(key))
        trial.set_user_attr("worker", int(cfg["sweep"]["worker"]))
        if keep == "none":
            drop_weights(summary["run_dir"])
        elif keep == "best":
            if previous is not None and previous.value >= value:
                drop_weights(summary["run_dir"])
            elif previous is not None:
                drop_weights(previous.user_attrs.get("run_dir"))   # only if it is on this disk
        logger.info("trial %d: %s -> val mean AUC %.4f", trial.number, json.dumps(params), value)
        return value

    return objective


def run(cfg: Mapping[str, Any], train_run=None) -> dict[str, Any]:
    """Run this worker's trials; return (and write to reports/summaries/) the sweep summary."""
    import optuna

    start = time.perf_counter()
    sw = cfg["sweep"]
    check_space(sw["space"])
    if not uses_soft_labels(cfg):
        raise ValueError("the sweep tunes the mixed loss: the base config needs soft labels "
                         "(any loss.alpha below 1; the sampled alpha replaces it)")
    url = storage_url(cfg)
    worker = int(sw["worker"])
    sampler = optuna.samplers.TPESampler(seed=int(cfg["seed"]) + worker,
                                         n_startup_trials=int(sw.get("n_startup_trials", 8)))
    study = optuna.create_study(study_name=sw["study_name"], storage=url, sampler=sampler,
                                direction="maximize", load_if_exists=True)
    claim_study(study, cfg)
    if sw.get("enqueue_default") and worker == 0 and not study.trials:
        default = {k: v for k, v in DEFAULT_TRIAL.items()}
        if in_space(default, sw["space"]):
            study.enqueue_trial(default)
    timeout = float(sw["timeout_h"]) * 3600 if sw.get("timeout_h") else None
    logger.info("sweep %s on %s: worker %d, %d trials", sw["study_name"], redact(url), worker,
                int(sw["n_trials"]))
    # A CUDA out-of-memory error (a RuntimeError) fails that trial, not the sweep.
    study.optimize(make_objective(cfg, train_run), n_trials=int(sw["n_trials"]),
                   timeout=timeout, catch=(RuntimeError,), gc_after_trial=True)
    summary = study_summary(study, url)
    summary["worker"] = worker
    summary["duration_s"] = round(time.perf_counter() - start, 1)
    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{TASK_ID}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def study_summary(study, url: str) -> dict[str, Any]:
    import optuna

    states = {}
    for t in study.trials:
        states[t.state.name] = states.get(t.state.name, 0) + 1
    best = best_completed(study)
    return {
        "task": TASK_ID, "study_name": study.study_name, "storage": redact(url),
        "trials": len(study.trials), "states": states, "setup": dict(study.user_attrs),
        "best": None if best is None else {
            "trial": best.number, "value": best.value, "params": best.params,
            "overrides": overrides(best.params), **best.user_attrs},
        "trials_table": [
            {"trial": t.number, "state": t.state.name, "value": t.value, **t.params,
             "worker": t.user_attrs.get("worker"), "run_name": t.user_attrs.get("run_name")}
            for t in study.trials if t.state != optuna.trial.TrialState.RUNNING],
    }


def overrides(params: Mapping[str, Any]) -> list[str]:
    """``["loss.alpha=0.43", ...]``: the best values as ``--set`` flags for train.py."""
    return [f"{'.'.join(PARAMS[name])}={value}" for name, value in params.items()]


def merge(into: str, sources: Sequence[str], study_name: str) -> dict[str, Any]:
    """Copy every finished trial of ``study_name`` in ``sources`` into the study at ``into``.

    Trials already there (same run_name) are skipped, so merging twice is harmless. The
    sources must have been started with the same setup.
    """
    import optuna

    target = optuna.create_study(study_name=study_name, storage=into, direction="maximize",
                                 load_if_exists=True)
    seen = {t.user_attrs.get("run_name") for t in target.trials}
    copied = 0
    for url in sources:
        source = optuna.load_study(study_name=study_name, storage=url)
        attrs = dict(source.user_attrs)
        if target.user_attrs:
            diff = sorted(k for k in attrs if target.user_attrs.get(k) != attrs[k])
            if diff:
                raise ValueError(f"{redact(url)}: setup differs from the merged study ({diff})")
        else:
            for key, value in attrs.items():
                target.set_user_attr(key, value)
        for t in source.trials:
            if t.state not in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL):
                continue
            if t.user_attrs.get("run_name") in seen and t.user_attrs.get("run_name"):
                continue
            target.add_trial(optuna.trial.create_trial(
                state=t.state, value=t.value if t.state == optuna.trial.TrialState.COMPLETE
                else None, params=t.params, distributions=t.distributions,
                user_attrs=t.user_attrs, system_attrs={"merged_from": redact(url),
                                                       "source_number": t.number}))
            seen.add(t.user_attrs.get("run_name"))
            copied += 1
    logger.info("merged %d trials into %s", copied, redact(into))
    return {**study_summary(target, into), "copied": copied}


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna sweep of the student recipe.")
    parser.add_argument("command", nargs="?", default="run", choices=("run", "merge", "best"))
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--worker", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--storage", default=None, help="Optuna storage URL (run, best)")
    parser.add_argument("--study", default=None, help="study name (default: sweep.study_name)")
    parser.add_argument("--into", default=None, help="merge: the target storage URL")
    parser.add_argument("--from", dest="sources", nargs="*", default=[],
                        help="merge: source storage URLs")
    args = parser.parse_args()

    sets = list(args.set)
    for flag, key in ((args.worker, "sweep.worker"), (args.n_trials, "sweep.n_trials"),
                      (args.storage, "sweep.storage"), (args.study, "sweep.study_name")):
        if flag is not None:
            sets.append(f"{key}={flag}")
    cfg = load_config(args.config, sets)
    if args.command == "merge":
        if not args.into or not args.sources:
            parser.error("merge needs --into and --from")
        result = merge(args.into, args.sources, cfg["sweep"]["study_name"])
    elif args.command == "best":
        import optuna

        url = storage_url(cfg)
        result = study_summary(optuna.load_study(study_name=cfg["sweep"]["study_name"],
                                                 storage=url), url)["best"]
    else:
        result = run(cfg)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
