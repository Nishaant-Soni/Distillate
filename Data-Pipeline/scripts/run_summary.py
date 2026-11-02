"""Merge every task's summary into one run report and announce the result.

Each task writes ``reports/summaries/<task>.json`` (utils.write_summary). This
is the final task in the DAG: it collects them into ``reports/run_summary.json``
and posts one Slack message, so a run produces a single notification rather than
one per task.

The merged report also answers the question that matters during integration
week: which stages are real and which are still stubs.

    python -m scripts.run_summary --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.alerts import send_alert
from scripts.utils import atomic_write, get_logger, load_config

logger = get_logger(__name__)


def collect(cfg: dict[str, Any]) -> dict[str, Any]:
    """Read every per-task summary and fold it into one run report."""
    summaries_dir = Path(cfg["paths"]["summaries_dir"])
    tasks: dict[str, Any] = {}
    unreadable: list[str] = []

    for path in sorted(summaries_dir.glob("*.json")):
        try:
            tasks[path.stem] = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            # A corrupt summary must not lose the rest of the run report.
            logger.error("Could not read %s: %s", path.name, exc)
            unreadable.append(path.name)

    stub = sorted(t for t, s in tasks.items() if s.get("mode") == "stub")
    real = sorted(t for t, s in tasks.items() if s.get("mode") == "real")
    return {
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tasks_reported": len(tasks),
        "duration_s": round(sum(float(s.get("duration_s") or 0) for s in tasks.values()), 3),
        "rows_out": sum(int(s.get("rows_out") or 0) for s in tasks.values()),
        "stages_real": real,
        "stages_stubbed": stub,
        "unreadable_summaries": unreadable,
        "tasks": tasks,
    }


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Write run_summary.json and post one Slack message. Never fails the DAG."""
    report = collect(cfg)
    details = {
        "tasks": report["tasks_reported"],
        "rows out": f"{report['rows_out']:,}",
        "task time": f"{report['duration_s']:.1f}s",
        "real stages": ", ".join(report["stages_real"]) or "none yet",
        "still stubbed": ", ".join(report["stages_stubbed"]) or "none",
    }
    # Stubs mean the run is green but not yet meaningful -- say so rather than
    # reporting an unqualified success.
    level = "warn" if report["stages_stubbed"] else "ok"
    title = ("data_prep_dag finished with stubbed stages" if report["stages_stubbed"]
             else "data_prep_dag finished")
    # Alert first, then write, so the report on disk records whether Slack
    # actually received it. send_alert never raises.
    report["slack_posted"] = send_alert(level, title, details)
    target = Path(cfg["paths"]["reports_dir"]) / "run_summary.json"
    atomic_write(target, json.dumps(report, indent=2, default=str, sort_keys=True))
    logger.info("Wrote %s (%d tasks)", target, report["tasks_reported"])
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config)), indent=2, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
