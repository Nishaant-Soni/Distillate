"""drift_ingest_dag: release the Civil drift stream one batch per run (Phase 2 skeleton).

Each run takes the next of the 20 drift batches through scripts/drift_ingest.py:
release (schema, size and id checks), drift statistics, scoring through the serving API,
evaluation of the previous batch with its delayed gold labels, the moderation overrides, and
finish (the batch record, and the stream moved on by one). Phase 2 scores against a mock of the
v4 section 2.4 API; Phase 3 sets drift_ingest.service_url to the live endpoint. The drift and
overrides steps are stubs until src/monitoring/drift.py and the feedback log exist.

The DAG is created paused and only gets a schedule when drift_ingest.enabled is true
(drift_ingest.schedule picks demo or real), so it never advances the stream by itself:
trigger it by hand to release one batch.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from airflow.providers.standard.operators.python import PythonOperator, ShortCircuitOperator
from airflow.sdk import DAG

from scripts.alerts import task_failure_callback

STEPS = [
    ("compute_drift", "step_drift"),
    ("score_batch", "step_score"),
    ("evaluate_previous", "step_evaluate"),
    ("merge_overrides", "step_overrides"),
    ("finish_batch", "step_finish"),
]


def _step(name: str, **_: Any) -> Any:
    """PythonOperator target: one step of scripts/drift_ingest.py."""
    from scripts import drift_ingest
    from scripts.utils import load_config

    return getattr(drift_ingest, name)(load_config())


def _schedule() -> str | None:
    """None (manual only) unless drift_ingest.enabled is true."""
    try:
        from scripts.utils import load_config

        dcfg = load_config().get("drift_ingest", {})
    except Exception:  # a broken config must not break DAG parsing
        return None
    if not dcfg.get("enabled", False):
        return None
    return dcfg.get("schedules", {}).get(dcfg.get("schedule", "demo"))


with DAG(
    dag_id="drift_ingest_dag",
    description="Release the Civil drift stream one batch per run: check, score, evaluate.",
    doc_md=__doc__,
    schedule=_schedule(),
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=True,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=1),
        "execution_timeout": timedelta(minutes=30),
        "on_failure_callback": task_failure_callback,
    },
    tags=["drift", "phase2"],
) as dag:
    previous = ShortCircuitOperator(
        task_id="release_batch", python_callable=_step, op_kwargs={"name": "step_release"}
    )
    for task_id, func in STEPS:
        task = PythonOperator(task_id=task_id, python_callable=_step, op_kwargs={"name": func})
        previous >> task
        previous = task
