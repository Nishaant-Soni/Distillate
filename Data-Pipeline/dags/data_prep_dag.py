"""data_prep_dag: raw TFDS datasets to training-ready, versioned, validated data.

Every stage task routes through ``scripts.stage.dispatch``, which runs the real
module when its owner has merged one and a contract-shaped stub when they have
not. That is why this DAG runs green end to end from day three and why no task
here changes on integration day.

Two tasks run in the TensorFlow environment at /opt/tfenv (Python 3.11, TFDV)
rather than in Airflow's own Python 3.12. They go through BashOperator and the
CLI from CONTRACTS.md section 3, so that environment never needs Airflow
installed in it.

Shape of the run is controlled by PIPELINE_MODE:
  optimized (default) parallel Wikipedia and Civil branches
  baseline            the same tasks as one sequential chain, for the Gantt
                      before/after comparison in the README
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG

from scripts.alerts import task_failure_callback
from scripts.pipeline_graph import BY_ID, TASKS, TFENV, edges

PIPELINE_MODE = os.environ.get("PIPELINE_MODE", "optimized")
# Where Data-Pipeline sits inside the container. The repo root is mounted one
# level above it so .dvc and .git are visible to the dvc_version task.
PIPELINE_DIR = os.environ.get("PIPELINE_DIR", "/opt/airflow/repo/Data-Pipeline")
CONFIG = "config/pipeline.yaml"


def _run_stage(stage: str, **_: Any) -> dict[str, Any]:
    """PythonOperator target for a stage that runs in the main environment."""
    from scripts.stage import dispatch
    from scripts.utils import load_config

    return dispatch(stage, load_config())


def _run_callable(dotted: str, **_: Any) -> dict[str, Any]:
    """PythonOperator target for an infra task, e.g. ``scripts.run_summary:run``."""
    import importlib

    from scripts.utils import load_config

    module_name, func_name = dotted.split(":")
    return getattr(importlib.import_module(module_name), func_name)(load_config())


with DAG(
    dag_id="data_prep_dag",
    description="Acquire, clean, dedupe, split, validate, de-bias and version the toxicity data.",
    doc_md=__doc__,
    # Triggered by hand and by retrain_dag in Phase 3; never on a clock, and
    # never backfilled -- there is one dataset, not a time series of them.
    schedule=None,
    catchup=False,
    start_date=datetime(2026, 10, 12),
    max_active_runs=1,
    default_args={
        "owner": "distillate",
        # Posts dag id, task id, error and log link to Slack. Set here so no new
        # task can be added without it.
        "on_failure_callback": task_failure_callback,
        "retries": 1,
        "retry_delay": timedelta(minutes=1),
        "depends_on_past": False,
    },
    tags=["distillate", "phase1", "data", PIPELINE_MODE],
) as dag:
    operators = {}
    for spec in TASKS:
        common = {
            "task_id": spec.task_id,
            "doc_md": f"**Owner:** {spec.owner}\n\n{spec.doc}",
            # Every task is bounded, so a hung stage shows up as a failure with
            # an alert instead of an Airflow run that never ends.
            "execution_timeout": timedelta(minutes=spec.timeout_minutes),
            "retries": spec.retries,
            "retry_delay": timedelta(minutes=spec.retry_delay_minutes),
        }
        if spec.env == TFENV:
            operators[spec.task_id] = BashOperator(
                bash_command=(
                    f'cd "{PIPELINE_DIR}" && '
                    f'"$TFENV_PYTHON" -m scripts.stage --stage {spec.task_id} --config {CONFIG}'
                ),
                cwd=None,
                **common,
            )
        elif spec.callable:
            operators[spec.task_id] = PythonOperator(
                python_callable=_run_callable,
                op_kwargs={"dotted": spec.callable},
                **common,
            )
        else:
            operators[spec.task_id] = PythonOperator(
                python_callable=_run_stage,
                op_kwargs={"stage": spec.task_id},
                **common,
            )

    for upstream, downstream in edges(PIPELINE_MODE):
        operators[upstream] >> operators[downstream]

    # Referenced so a typo in the graph fails DAG parsing rather than silently
    # leaving a task unwired.
    assert set(operators) == set(BY_ID), "pipeline_graph and the DAG disagree on task ids"
