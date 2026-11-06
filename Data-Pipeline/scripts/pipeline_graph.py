"""The data_prep_dag task graph, declared as plain data.

Kept out of the DAG module on purpose:

  * it can be imported and tested without Airflow installed, so the structure is
    checked on every laptop and in the lint job, not only where Airflow exists;
  * the Gantt deliverable needs the same tasks wired two ways -- one sequential
    chain for the baseline run and parallel branches for the optimized run --
    and that is a one-word switch here (PIPELINE_MODE) rather than two DAG files.

Task ids match the module names in scripts/stage.py, which match the CLI in
CONTRACTS.md section 3, so a task id is always traceable to one script and one
owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MAIN = "main"      # Airflow's own Python 3.12 environment
TFENV = "tfenv"    # /opt/tfenv, Python 3.11 with TensorFlow and TFDV


@dataclass(frozen=True)
class Task:
    task_id: str
    owner: str
    env: str
    timeout_minutes: int
    doc: str
    retries: int = 1
    retry_delay_minutes: int = 1
    # Stages route through scripts.stage (real module or stub). The three tasks
    # below with callable=... are infra and always run their own code.
    callable: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


TASKS: tuple[Task, ...] = (
    Task("acquire", "Praniti", MAIN, 120,
         "Download both Kaggle competition datasets to Parquet with a checksum manifest.",
         # Acquisition is the one task that reaches the network, so it is the
         # one task worth retrying patiently (work plan, task defaults).
         retries=2, retry_delay_minutes=2, tags=("network",)),
    Task("preprocess", "Praniti", MAIN, 60,
         "NFKC, HTML entities, whitespace, label casting, derived columns."),
    Task("dedupe", "Praniti", MAIN, 90,
         "Exact and MinHash-LSH near-duplicate removal; never drops gate rows."),
    Task("split", "Sai", MAIN, 30,
         "Seeded multilabel stratified train/val split, canary carve-out, gate."),
    Task("subsets", "Sai", MAIN, 30,
         "Civil subsets: H2 eval, canary, bias set, 20 sorted drift batches."),
    Task("obfuscate", "Sai", MAIN, 15,
         "Seeded character substitution over toxic gate rows; labels unchanged."),
    Task("schema_stats", "Sai", TFENV, 60,
         "TFDV statistics and schema validation over every raw and processed file."),
    Task("anomalies", "Sai", MAIN, 30,
         "Every anomaly check with its fail/warn severity."),
    Task("bias", "Ruchita", MAIN, 60,
         "Slice metrics, reweighting, sample_weight into train.parquet."),
    Task("dvc_version", "Vaishnavi", MAIN, 60,
         "dvc add data/raw data/processed, then push when DVC_PUSH is true.",
         callable="scripts.dvc_version:run"),
    Task("notify_success", "Vaishnavi", MAIN, 10,
         "Merge every task summary into run_summary.json and post one Slack message.",
         retries=0, callable="scripts.run_summary:run"),
)

TASK_IDS: tuple[str, ...] = tuple(t.task_id for t in TASKS)
BY_ID: dict[str, Task] = {t.task_id: t for t in TASKS}

# Optimized shape: once preprocessing is done the Wikipedia chain (dedupe ->
# split -> obfuscate) and the Civil chain (subsets) are independent, so they run
# side by side and rejoin at schema_stats.
EDGES_OPTIMIZED: tuple[tuple[str, str], ...] = (
    ("acquire", "preprocess"),
    ("preprocess", "dedupe"),
    ("preprocess", "subsets"),
    ("dedupe", "split"),
    ("split", "obfuscate"),
    ("obfuscate", "schema_stats"),
    ("subsets", "schema_stats"),
    ("schema_stats", "anomalies"),
    ("anomalies", "bias"),
    ("bias", "dvc_version"),
    ("dvc_version", "notify_success"),
)

# Baseline shape for the Gantt comparison: the same tasks, one after another.
# strict=False is intentional: the last task has no downstream neighbour.
EDGES_BASELINE: tuple[tuple[str, str], ...] = tuple(zip(TASK_IDS, TASK_IDS[1:], strict=False))

MODES = {"optimized": EDGES_OPTIMIZED, "baseline": EDGES_BASELINE}


def edges(mode: str = "optimized") -> tuple[tuple[str, str], ...]:
    """Return the dependency edges for ``mode`` ("optimized" or "baseline")."""
    try:
        return MODES[mode]
    except KeyError:
        raise ValueError(f"mode must be one of {sorted(MODES)}, got {mode!r}") from None


def critical_path_minutes(mode: str = "optimized") -> int:
    """Longest timeout path through the graph -- the shape the Gantt shows."""
    longest: dict[str, int] = {}
    for task_id in topological_order(mode):
        upstream = [longest[u] for u, d in edges(mode) if d == task_id]
        longest[task_id] = max(upstream, default=0) + BY_ID[task_id].timeout_minutes
    return max(longest.values())


def topological_order(mode: str = "optimized") -> list[str]:
    """Task ids in a runnable order. Raises ValueError if the graph has a cycle."""
    graph = edges(mode)
    indegree = {t: 0 for t in TASK_IDS}
    for _, downstream in graph:
        indegree[downstream] += 1
    # Walk in declaration order so the result is stable, not set-ordered.
    ready = [t for t in TASK_IDS if indegree[t] == 0]
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for upstream, downstream in graph:
            if upstream == current:
                indegree[downstream] -= 1
                if indegree[downstream] == 0:
                    ready.append(downstream)
    if len(order) != len(TASK_IDS):
        raise ValueError(f"Cycle in the {mode} graph: {set(TASK_IDS) - set(order)}")
    return order
