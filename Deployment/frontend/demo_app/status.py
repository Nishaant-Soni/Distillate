"""What the Status page shows, built without Streamlit so it can be tested on its own.

Each source is optional, and a missing one shows as "not yet" rather than an error:

* the registry: ``@production`` and ``@staging`` with their gate results (MLflow);
* the last deployment: the last line of ``deploy.sh``'s ``deployments.jsonl``;
* the last retraining: the newest ``retrain_dag`` record (``<run_id>.json``);
* Grafana: a link.

It also raises the flag from work plan v5, section 6.1: a deploy run that waits in the
``deploy`` concurrency group and is replaced by a newer request never starts, so it leaves no
deployment line and no Slack message. So when ``@staging`` is newer than ``@production`` and no
deployment line names it, the page says so.

The ``deployments.jsonl`` and retraining record formats aren't final yet (Ruchita's
``deploy.sh``, Vaishnavi's ``retrain_dag``); this reads the fields both are expected to have
and shows the rest as they are.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .registry import Registry, RegistryError, VersionInfo

DEPLOYED = {"deployed", "success", "ok", "passed"}


@dataclass
class Status:
    production: VersionInfo | None = None
    staging: VersionInfo | None = None
    registry_note: str | None = None          # why the registry part is empty, if it is
    last_deployment: dict[str, Any] | None = None
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


def deployed_versions(deployments: list[dict[str, Any]]) -> set[str]:
    """The model versions with at least one successful deployment line."""
    return {str(d["version"]) for d in deployments
            if "version" in d and str(d.get("outcome", d.get("status", ""))).lower() in DEPLOYED}


def staging_warning(production: VersionInfo | None, staging: VersionInfo | None,
                    deployed: set[str]) -> str | None:
    """The v5 section 6.1 flag, or None."""
    if production is None or staging is None:
        return None
    try:
        newer = int(staging.version) > int(production.version)
    except ValueError:
        return None
    if not newer or staging.version in deployed:
        return None
    return (f"@staging (v{staging.version}) is newer than @production (v{production.version}) "
            "and no deployment line names it. If a deploy was requested, its run may have been "
            "replaced in the deploy queue; re-run it by hand.")


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
        if deployments:
            status.last_deployment = deployments[-1]
        else:
            status.deployment_note = "not yet: no deployments recorded"

    if not settings.retrain_records:
        status.retraining_note = "not yet: no retraining records configured (RETRAIN_RECORDS)"
    elif not Path(settings.retrain_records).is_dir():
        status.retraining_note = "not yet: the retraining records folder doesn't exist yet"
    else:
        status.last_retraining = latest_retraining(settings.retrain_records)
        if status.last_retraining is None:
            status.retraining_note = "not yet: no retraining has run"

    warning = staging_warning(status.production, status.staging, deployed_versions(deployments))
    if warning:
        status.warnings.append(warning)
    return status
