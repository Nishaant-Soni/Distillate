"""retrain_dag: retrain the student, gate it and register it (Phase 2 skeleton).

pull_data -> submit_training -> gate -> register -> notify, each a step of scripts/retrain.py.
The record of the run (the trigger, then one entry per step) is passed from task to task
through XCom and written to reports/retrain/<run_id>.json by notify.

In Phase 2 the steps are stubs: pull_data records the data revision and the delivered drift
batches, submit_training builds the Vertex AI job spec model-ci uses without submitting it, and
gate and register report that nothing ran. Phase 3 completes them (work plan v4 section 6): the
monitoring alerts trigger this DAG through Airflow's REST API with a conf such as
``{"reason": "flag_rate PSI 0.31", "source": "alert"}``, the job trains on the original data plus
the delivered drift batches, and the candidate reaches traffic only through the same gate and
registry as CI.

Never on a clock and created paused: trigger it by hand (or through the REST API) to run it.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG

from scripts.alerts import task_failure_callback
from scripts.retrain import STEPS


def _step(func: str, previous: str | None, ti=None, dag_run=None, **_: Any) -> dict[str, Any]:
    """PythonOperator target: one step of scripts/retrain.py, on the record so far."""
    from scripts import retrain
    from scripts.utils import load_config

    if previous is None:
        run = retrain.new_run(getattr(dag_run, "conf", None) or {},
                              run_id=f"retrain-{dag_run.run_id}" if dag_run else None)
    else:
        run = ti.xcom_pull(task_ids=previous)
    run["steps"][func.removeprefix("step_")] = getattr(retrain, func)(load_config(), run)
    return run


with DAG(
    dag_id="retrain_dag",
    description="Retrain the student on Vertex AI, gate it, register it, notify (skeleton).",
    doc_md=__doc__,
    schedule=None,
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=True,
    default_args={
        "owner": "distillate",
        "retries": 0,          # a retry would submit a second paid training job
        "execution_timeout": timedelta(minutes=30),
        "on_failure_callback": task_failure_callback,
    },
    tags=["distillate", "retrain", "phase2"],
) as dag:
    previous = None
    upstream = None
    for task_id, func in STEPS:
        task = PythonOperator(task_id=task_id, python_callable=_step,
                              op_kwargs={"func": func, "previous": previous})
        if upstream is not None:
            upstream >> task
        upstream, previous = task, task_id
