"""What the Status page shows, built without Streamlit so it can be tested on its own.

Each source is optional, and a missing one shows as "not yet" rather than an error:

* the registry: ``@production`` and ``@staging`` with their gate results (MLflow);
* the deployment: from ``deploy.sh`` / ``rollback.sh``'s ``deployments.jsonl``, the newest
  ``deployed`` or ``rolled-back`` line (what is serving), and a newer line that didn't succeed
  (a blocked deploy or a test package) as the latest attempt;
* the last retraining: the newest ``retrain_dag`` record (``<run_id>.json``);
* Grafana: a link.

It also raises the flag from work plan v5, section 6.1: a deploy run that waits in the
``deploy`` concurrency group and is replaced by a newer request never starts, so it leaves no
deployment line and no Slack message. So when ``@staging`` is newer than ``@production`` and no
deployment line names it, the page says so. When a line does name it but the deploy failed,
the page says that instead, since a re-run alone wouldn't help. A version registered before
the first deployment record isn't flagged for a missing line: it predates the deploy pipeline,
so no deploy of it was requested (v2, ``@staging`` since the Phase 2 rollback demo).

``deployments.jsonl`` follows #64 (``status``: ``deployed`` and ``rolled-back`` on success;
``smoke-failed``, ``error``, ``partial`` and ``smoke-passed-not-promoted`` otherwise). The
retraining record format isn't final yet (Vaishnavi's ``retrain_dag``); this reads the fields it
is expected to have and shows the rest as it is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings
from .registry import Registry, RegistryError, VersionInfo

SERVING = {"deployed", "rolled-back"}     # deployments.jsonl statuses that leave it serving
FAILED = {"smoke-failed", "error", "partial"}


@dataclass
class Status:
    production: VersionInfo | None = None
    staging: VersionInfo | None = None
    registry_note: str | None = None          # why the registry part is empty, if it is
    serving: dict[str, Any] | None = None          # the newest deployed / rolled-back line
    latest_attempt: dict[str, Any] | None = None   # a newer line that didn't succeed, if any
    deployment_note: str | None = None
    last_retraining: dict[str, Any] | None = None
    retraining_note: str | None = None
    grafana_url: str | None = None
    warnings: list[str] = field(default_factory=list)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    """Every JSON object line of a local file; blank or broken lines are skipped."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _status(line: dict[str, Any]) -> str:
    return str(line.get("status", "")).lower()


def deployed_versions(deployments: list[dict[str, Any]]) -> set[str]:
    """The model versions with at least one successful deployment line."""
    return {str(d["version"]) for d in deployments if d.get("version") and _status(d) in SERVING}


def split_deployments(deployments: list[dict[str, Any]]
                      ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """(what is serving, the latest attempt if it is newer and didn't succeed)."""
    serving_at = next((i for i in range(len(deployments) - 1, -1, -1)
                       if _status(deployments[i]) in SERVING), None)
    serving = deployments[serving_at] if serving_at is not None else None
    last = deployments[-1] if deployments else None
    attempt = last if last is not None and last is not serving else None
    return serving, attempt


def staging_warning(production: VersionInfo | None, staging: VersionInfo | None,
                    deployments: list[dict[str, Any]]) -> str | None:
    """The v5 section 6.1 flag, or None."""
    if production is None or staging is None:
        return None
    try:
        newer = int(staging.version) > int(production.version)
    except ValueError:
        return None
    if not newer or staging.version in deployed_versions(deployments):
        return None
    head = f"@staging (v{staging.version}) is newer than @production (v{production.version})"
    tried = [d for d in deployments if str(d.get("version", "")) == staging.version
             and _status(d) in FAILED]
    if tried:
        last = tried[-1]
        when = f" at {last['at']}" if last.get("at") else ""
        step = f", step {last['step']}" if last.get("step") else ""
        if _status(last) == "partial":
            return (f"{head}: its deploy moved traffic but not @production (partial{when}). "
                    f"Move the alias with rollback_model --to {staging.version} --alias "
                    "production.")
        return (f"{head}: its deploy ran and failed ({_status(last)}{step}{when}). Fix the "
                "cause before deploying it again; see the deploy run.")
    if registered_before_records(staging, deployments):
        return None
    return (f"{head} and no deployment line names it. If a deploy was requested, its run may "
            "have been replaced in the deploy queue; re-run it by hand.")


def _moment(at: Any) -> datetime | None:
    """A deployments.jsonl ``at`` as an aware UTC time (naive is taken as UTC), or None."""
    try:
        moment = datetime.fromisoformat(str(at).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def registered_before_records(staging: VersionInfo, deployments: list[dict[str, Any]]) -> bool:
    """True when ``staging`` was registered before the first deployment record.

    Such a version predates the deploy pipeline (v2, @staging since the Phase 2 rollback demo),
    so no deploy of it was ever requested and the missing line means nothing. Every version
    registered since (by model-ci or retrain_dag, which request the deploy right after
    registering) is still flagged. Without a registration time or any dated record, False:
    the flag stays as it was."""
    times = [m for m in (_moment(d.get("at")) for d in deployments if d.get("at")) if m]
    if staging.created_ms is None or not times:
        return False
    registered = datetime.fromtimestamp(staging.created_ms / 1000, tz=timezone.utc)
    return registered < min(times)


def latest_retraining(folder: str) -> dict[str, Any] | None:
    """The newest retraining record in ``folder`` (by modification time), or None."""
    records = sorted(Path(folder).glob("*.json"), key=lambda p: p.stat().st_mtime)
    for path in reversed(records):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if isinstance(record, dict):
            steps = record.get("steps") or {}
            return {"run_id": record.get("run_id", path.stem),
                    "reason": (record.get("trigger") or {}).get("reason"),
                    "gate": (steps.get("gate") or {}).get("result"),
                    "registered": (steps.get("register") or {}).get("registered")}
    return None


def build(settings: Settings, registry: Registry | None) -> Status:
    """Everything the Status page shows, from whatever sources are configured."""
    status = Status(grafana_url=settings.grafana_url)

    if registry is None:
        status.registry_note = "not yet: no MLflow server configured (MLFLOW_TRACKING_URI)"
    else:
        try:
            status.production = registry.alias(settings.production_alias)
            status.staging = registry.alias(settings.staging_alias)
        except RegistryError as err:
            status.registry_note = str(err)

    deployments: list[dict[str, Any]] = []
    if not settings.deployments_log:
        status.deployment_note = "not yet: no deployment log configured (DEPLOYMENTS_LOG)"
    elif not Path(settings.deployments_log).is_file():
        status.deployment_note = "not yet: the deployment log doesn't exist yet"
    else:
        deployments = read_jsonl(settings.deployments_log)
        status.serving, status.latest_attempt = split_deployments(deployments)
        if status.serving is None:
            status.deployment_note = ("not yet: no successful deployment recorded"
                                      if deployments else "not yet: no deployments recorded")

    if not settings.retrain_records:
        status.retraining_note = "not yet: no retraining records configured (RETRAIN_RECORDS)"
    elif not Path(settings.retrain_records).is_dir():
        status.retraining_note = "not yet: the retraining records folder doesn't exist yet"
    else:
        status.last_retraining = latest_retraining(settings.retrain_records)
        if status.last_retraining is None:
            status.retraining_note = "not yet: no retraining has run"

    warning = staging_warning(status.production, status.staging, deployments)
    if warning:
        status.warnings.append(warning)
    return status
