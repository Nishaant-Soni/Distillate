"""Move a registry alias back to an earlier model version, in one command.

A candidate worse than production never gets the ``production`` alias (the gate blocks it).
If a bad model gets through anyway, this puts the alias back. Serving and the gate resolve
``models:/distillate-student@production``, so moving the alias is the whole rollback in Phase
2. Phase 3 adds the Cloud Run traffic shift around it. Artifact Registry versions are
immutable, so nothing there changes.

    python -m src.pipeline.rollback_model --list
    python -m src.pipeline.rollback_model --previous --reason "canary AUC dropped"
    python -m src.pipeline.rollback_model --to 3
    python -m src.pipeline.rollback_model --to 3 --alias staging

``--previous`` picks the newest version older than the current one whose gate passed. A
version whose gate didn't pass is refused unless ``--force``. Every move is recorded as tags
(``<alias>_previous`` on the registered model, ``last_alias_move`` on the target) and posted
to Slack.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from typing import Any

from src.pipeline.registry import client, load_config, notify, version_by_alias
from src.teacher.config import get_logger

logger = get_logger(__name__)


class RollbackError(RuntimeError):
    """The alias can't be moved; the message says why."""


def all_versions(mlflow_client, name: str) -> list:
    versions = mlflow_client.search_model_versions(f"name = '{name}'")
    return sorted(versions, key=lambda v: int(v.version))


def previous_version(mlflow_client, name: str, current: str):
    """The newest READY, gate-passed version older than ``current``."""
    older = [v for v in all_versions(mlflow_client, name)
             if int(v.version) < int(current) and v.status == "READY"
             and v.tags.get("gate_passed") == "true"]
    if not older:
        raise RollbackError(f"no gate-passed version of {name} older than v{current}")
    return older[-1]


def move_alias(mlflow_client, name: str, alias: str, to: str | None = None,
               previous: bool = False, force: bool = False, reason: str = "") -> dict[str, Any]:
    """Point ``alias`` at version ``to`` (or the previous good one); return a summary."""
    from mlflow.exceptions import MlflowException

    if (to is None) == (not previous):
        raise RollbackError("give exactly one of --to VERSION or --previous")
    current = version_by_alias(mlflow_client, name, alias)
    if previous:
        if current is None:
            raise RollbackError(f"no version holds @{alias}, so there is no previous one")
        target = previous_version(mlflow_client, name, current.version)
    else:
        try:
            target = mlflow_client.get_model_version(name, str(to))
        except MlflowException:
            raise RollbackError(f"{name} has no version {to}") from None
        if target.status != "READY":
            raise RollbackError(f"v{to} is {target.status}, not READY")
        if target.tags.get("gate_passed") != "true" and not force:
            raise RollbackError(f"v{to} has no passed gate on record; use --force to move "
                                f"@{alias} to it anyway")

    # Some stores return version numbers as int, the REST server as str: compare as str.
    from_version = str(current.version) if current else None
    to_version = str(target.version)
    if from_version == to_version:
        return {"registered_model": name, "alias": alias, "from": from_version,
                "to": to_version, "changed": False}

    mlflow_client.set_registered_model_alias(name, alias, to_version)
    move = {"alias": alias, "from": from_version, "to": to_version, "reason": reason,
            "at": datetime.now(UTC).isoformat(timespec="seconds")}
    mlflow_client.set_model_version_tag(name, to_version, "last_alias_move", json.dumps(move))
    if from_version:
        mlflow_client.set_registered_model_tag(name, f"{alias}_previous", from_version)
    logger.info("@%s moved from v%s to v%s", alias, from_version, to_version)
    return {"registered_model": name, **move,
            "model_version": target.tags.get("model_version"), "changed": True}


def version_table(mlflow_client, name: str) -> str:
    # search_model_versions leaves aliases empty, so read them from the registered model.
    by_version: dict[str, list[str]] = {}
    for alias, version in mlflow_client.get_registered_model(name).aliases.items():
        by_version.setdefault(str(version), []).append(f"@{alias}")
    rows = ["version  aliases               gate    model_version"]
    for v in all_versions(mlflow_client, name):
        aliases = ",".join(sorted(by_version.get(str(v.version), []))) or "-"
        gate = {"true": "passed"}.get(v.tags.get("gate_passed"), "-")
        rows.append(f"v{v.version:<7} {aliases:<21} {gate:<7} {v.tags.get('model_version', '')}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--to", help="version number to point the alias at")
    target.add_argument("--previous", action="store_true",
                        help="the newest gate-passed version older than the current one")
    target.add_argument("--list", action="store_true", help="show versions and aliases")
    parser.add_argument("--alias", default=None, help="default: mlflow.production_alias")
    parser.add_argument("--force", action="store_true",
                        help="allow a version with no passed gate on record")
    parser.add_argument("--reason", default="", help="recorded in the tags and the message")
    parser.add_argument("--config", default=None)
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    name = cfg["mlflow"]["registered_model"]
    alias = args.alias or cfg["mlflow"]["production_alias"]
    mlflow_client = client(args.tracking_uri)
    if args.list:
        print(version_table(mlflow_client, name))
        return 0
    try:
        summary = move_alias(mlflow_client, name, alias, to=args.to, previous=args.previous,
                             force=args.force, reason=args.reason)
    except RollbackError as exc:
        notify(cfg, "fail", f"Rollback of @{alias} refused", {"reason": str(exc)})
        return 1
    if summary["changed"]:
        notify(cfg, "warn", f"Rolled back {name} @{alias}: v{summary['from']} → "
               f"v{summary['to']}", {"model_version": summary["model_version"],
                                     "reason": summary["reason"] or "not given"})
    else:
        logger.info("@%s already points at v%s; nothing to do", alias, summary["to"])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
