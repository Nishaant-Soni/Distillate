"""retrain_dag: retrain the student on the drift stream, check it, register and deploy it.

pull_data -> submit_training -> gate -> promote_check -> register -> dispatch -> notify, each a
step of scripts/retrain.py (work plan v5 6.1, Model-Pipeline/docs/retraining.md). The record of
the run (the trigger, then one entry per step) is passed from task to task through XCom and
written by notify to reports/retrain/<run_id>.json and to
gs://distillate-mlops-data/deployments/retrain/ (for the demo app's Status page).

* pull_data reads drift_ingest's state.json (next_batch = n): train on drift batches 0..n-3,
  hold out n-2. With fewer than three batches out it skips the run (Slack), and the later steps
  do nothing.
* submit_training runs model-ci's Vertex AI job with those overrides and waits for it
  (retrain.submit: false only records the job spec).
* gate and promote_check: the candidate must pass the CI gate and beat @production on the
  held-out batch; production's scores are rescored through the live service if @production has
  moved since drift_ingest scored the batch.
* register (@staging) and dispatch (repository_dispatch to deploy.yml) run only on a pass.

Slack: "triggered" (or "skipped") from pull_data, "deployment requested" from dispatch, the
result from notify, and a failed task's message from the failure callback.

Never on a clock and created paused. check_thresholds in drift_ingest_dag triggers it with a conf
such as ``{"reason": "flag_rate PSI 0.31", "source": "alert"}``; it can also be triggered by hand.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG

from scripts.alerts import task_failure_callback
from scripts.retrain import STEPS

# submit_training waits for the Vertex AI job (retrain.job_timeout_h); the rest are minutes.
TIMEOUTS = {"submit_training": timedelta(hours=5, minutes=15)}


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
    description="Retrain the student on the drift stream, gate, promote_check, register, deploy.",
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
    tags=["distillate", "retrain", "phase3"],
) as dag:
    previous = None
    upstream = None
    for task_id, func in STEPS:
        task = PythonOperator(task_id=task_id, python_callable=_step,
                              op_kwargs={"func": func, "previous": previous},
                              **({"execution_timeout": TIMEOUTS[task_id]}
                                 if task_id in TIMEOUTS else {}))
        if upstream is not None:
            upstream >> task
        upstream, previous = task, task_id
