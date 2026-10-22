"""Checks on the data_prep_dag task graph.

Split in two on purpose. The structural half imports only scripts.pipeline_graph
and runs anywhere, including the lint job and a laptop with no Airflow. The
second half needs a real Airflow install and skips without one, so a missing
Airflow never looks like a passing test suite.
"""

import pytest

from scripts import pipeline_graph as graph
from scripts.stage import STAGE_MODULES


# --------------------------------------------------------------------------
# Structure: no Airflow required.
# --------------------------------------------------------------------------
def test_every_stage_task_has_an_owning_module():
    stage_tasks = {t.task_id for t in graph.TASKS if t.callable is None}
    assert stage_tasks == set(STAGE_MODULES), (
        "scripts/stage.py and pipeline_graph disagree about which tasks are stages"
    )


def test_infra_tasks_point_at_importable_callables():
    import importlib

    for task in graph.TASKS:
        if task.callable is None:
            continue
        module_name, func_name = task.callable.split(":")
        assert callable(getattr(importlib.import_module(module_name), func_name))


@pytest.mark.parametrize("mode", sorted(graph.MODES))
def test_graph_is_acyclic_and_covers_every_task(mode):
    order = graph.topological_order(mode)
    assert sorted(order) == sorted(graph.TASK_IDS)
    position = {task_id: i for i, task_id in enumerate(order)}
    for upstream, downstream in graph.edges(mode):
        assert position[upstream] < position[downstream]


@pytest.mark.parametrize("mode", sorted(graph.MODES))
def test_notify_success_runs_last(mode):
    assert graph.topological_order(mode)[-1] == "notify_success"
    assert not [d for u, d in graph.edges(mode) if u == "notify_success"]


def test_every_task_is_bounded_by_a_timeout():
    assert all(t.timeout_minutes > 0 for t in graph.TASKS)


def test_acquisition_retries_twice_with_a_two_minute_delay():
    # The only task that reaches the network; work plan task defaults.
    acquire = graph.BY_ID["acquire"]
    assert (acquire.retries, acquire.retry_delay_minutes) == (2, 2)


def test_only_tensorflow_stages_use_the_tfenv_interpreter():
    assert {t.task_id for t in graph.TASKS if t.env == graph.TFENV} == {"acquire", "schema_stats"}


def test_optimized_graph_actually_parallelises():
    # Wikipedia and Civil branches must not be chained to each other, or the
    # Gantt before/after in the README has nothing to show.
    optimized = graph.edges("optimized")
    assert ("dedupe", "subsets") not in optimized and ("subsets", "dedupe") not in optimized
    assert graph.critical_path_minutes("optimized") < graph.critical_path_minutes("baseline")


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        graph.edges("sideways")


def test_a_cycle_is_detected(monkeypatch):
    monkeypatch.setitem(graph.MODES, "broken", (("bias", "acquire"), *graph.EDGES_OPTIMIZED))
    with pytest.raises(ValueError, match="Cycle"):
        graph.topological_order("broken")


# --------------------------------------------------------------------------
# Wiring: needs Airflow.
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dagbag():
    pytest.importorskip("airflow", reason="Airflow is only installed in the Docker image")
    try:
        from airflow.dag_processing.dagbag import DagBag
    except ImportError:  # older Airflow
        from airflow.models.dagbag import DagBag

    bag = DagBag(dag_folder="dags")
    assert not bag.import_errors, bag.import_errors
    return bag


def test_dag_imports_cleanly(dagbag):
    assert "data_prep_dag" in dagbag.dags


def test_dag_has_the_expected_task_ids(dagbag):
    assert sorted(dagbag.dags["data_prep_dag"].task_ids) == sorted(graph.TASK_IDS)


def test_dag_dependencies_match_the_declared_graph(dagbag):
    dag = dagbag.dags["data_prep_dag"]
    wired = {(u, d) for u in dag.task_ids for d in dag.get_task(u).downstream_task_ids}
    assert wired == set(graph.edges("optimized"))


def test_dag_is_manual_only_and_never_backfills(dagbag):
    dag = dagbag.dags["data_prep_dag"]
    assert dag.schedule is None
    assert dag.catchup is False


def test_every_task_has_a_failure_alert_and_a_timeout(dagbag):
    from scripts.alerts import task_failure_callback

    for task in dagbag.dags["data_prep_dag"].tasks:
        assert task.execution_timeout is not None, task.task_id
        callbacks = task.on_failure_callback
        callbacks = callbacks if isinstance(callbacks, list) else [callbacks]
        assert task_failure_callback in callbacks, task.task_id
