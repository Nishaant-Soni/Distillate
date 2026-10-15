"""One entry point every pipeline task goes through.

A task asks for a stage by name. If the lane that owns it has merged a module
with a ``run(cfg)`` function, that runs. If not, the contract-shaped stub runs
instead. Nothing in the DAG changes on the day a real module lands -- the next
run just picks it up, which is what makes Oct 21 a merge rather than a wait.

Both environments can run this: the main Airflow environment imports
``dispatch`` directly, and the two TensorFlow tasks call the CLI with the
/opt/tfenv interpreter, which never needs Airflow installed.

    python -m scripts.stage --stage dedupe --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any, Callable

from scripts import stubs
from scripts.utils import get_logger, load_config, write_summary

logger = get_logger(__name__)

# Stage -> module that owns it. Task ids in the DAG match these names, and the
# modules match the CLI in CONTRACTS.md section 3 (python -m scripts.<name>).
STAGE_MODULES: dict[str, str] = {
    "acquire": "scripts.acquire",            # Praniti
    "preprocess": "scripts.preprocess",      # Praniti
    "dedupe": "scripts.dedupe",              # Praniti
    "split": "scripts.split",                # Sai
    "subsets": "scripts.subsets",            # Sai
    "obfuscate": "scripts.obfuscate",        # Sai
    "schema_stats": "scripts.schema_stats",  # Sai
    "anomalies": "scripts.anomalies",        # Sai
    "bias": "scripts.bias",                  # Ruchita
}

# Stages whose stub writes under a different name than the stage itself.
_STUB_ALIASES = {
    "acquire": ["acquire_wikipedia", "acquire_civil"],
    "preprocess": ["preprocess_wikipedia", "preprocess_civil"],
    "dedupe": ["dedupe_wikipedia"],
    "split": ["split_wikipedia"],
    "subsets": ["civil_subsets"],
}


def _real_run(stage: str) -> Callable[[dict[str, Any]], dict[str, Any]] | None:
    """Return the owning module's ``run`` if it has been merged, else None."""
    module_name = STAGE_MODULES[stage]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Only treat "the module itself is missing" as not-yet-merged. A missing
        # dependency inside a real module must fail loudly, not fall back.
        if exc.name in (module_name, module_name.rsplit(".", 1)[-1]):
            return None
        raise
    run = getattr(module, "run", None)
    if not callable(run):
        logger.warning("%s exists but has no run(cfg); using the stub", module_name)
        return None
    return run


def dispatch(stage: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Run ``stage``: the real module if it exists, the stub otherwise."""
    if stage not in STAGE_MODULES:
        raise KeyError(f"Unknown stage {stage!r}; known: {sorted(STAGE_MODULES)}")

    run = _real_run(stage)
    if run is None:
        logger.info("stage=%s mode=stub (%s not merged yet)", stage, STAGE_MODULES[stage])
        summaries = [stubs.run(cfg, name) for name in _STUB_ALIASES.get(stage, [stage])]
        merged: dict[str, Any] = {
            "task": stage,
            "mode": "stub",
            "rows_out": sum(s.get("rows_out", 0) for s in summaries),
            "duration_s": round(sum(s.get("duration_s", 0.0) for s in summaries), 3),
            "parts": {s["task"]: s for s in summaries},
        }
        write_summary(stage, merged, cfg)
        return merged

    logger.info("stage=%s mode=real (%s)", stage, STAGE_MODULES[stage])
    summary = run(cfg)
    if not isinstance(summary, dict):
        raise TypeError(
            f"{STAGE_MODULES[stage]}.run must return a dict, got {type(summary).__name__}"
        )
    summary.setdefault("task", stage)
    summary["mode"] = "real"
    write_summary(stage, summary, cfg)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=sorted(STAGE_MODULES))
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    summary = dispatch(args.stage, load_config(args.config))
    print(json.dumps(summary, indent=2, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
