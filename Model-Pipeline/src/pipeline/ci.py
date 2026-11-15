"""Helpers model-ci.yml calls between the Vertex job, the gate and registration.

The training job (``python -m src.pipeline.run``, docs/ci.md) leaves one output folder:

    <out>/package/                      the section 2.5 model package (meta.json, thresholds...)
    <out>/predictions/<stem>.parquet    prediction files for every split the gate reads
    <out>/gold/<stem>.parquet           id + labels (+ identity columns for the bias set), no text

with the stems from configs/gate.yaml (``val``, ``gate``, ``bias_eval``). Subcommands:

    python -m src.pipeline.ci check-outputs --dir <out>
    python -m src.pipeline.ci fetch-production --out <dir>     # @production's predictions
    python -m src.pipeline.ci notify --level fail --title "..." --field run=<url> ...

``fetch-production`` writes nothing and prints ``none`` when no version holds the alias yet,
so the gate's regression check is skipped (and says so) on the first run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from src.pipeline.registry import client, load_config, notify, version_by_alias
from src.teacher.config import PIPELINE_ROOT, get_logger

logger = get_logger(__name__)
GATE_CONFIG = PIPELINE_ROOT / "configs" / "gate.yaml"


def gate_stems(gate_config: str | Path = GATE_CONFIG) -> list[str]:
    with Path(gate_config).open(encoding="utf-8") as f:
        return list(yaml.safe_load(f)["splits"].values())


def check_outputs(out: Path, stems: list[str]) -> list[str]:
    """Every file the gate and registration need from the job; returns what is missing."""
    needed = [out / "package" / "meta.json", out / "package" / "thresholds.json"]
    for stem in stems:
        needed += [out / "predictions" / f"{stem}.parquet", out / "gold" / f"{stem}.parquet"]
    return [str(p.relative_to(out)) for p in needed if not p.exists()]


def fetch_production(mlflow_client, cfg: dict[str, Any], out: Path) -> dict[str, Any] | None:
    """Download @production's predictions and thresholds into ``out``; None if no production.

    The gate reads ``thresholds.json`` from next to the predictions, so it is placed there.
    """
    name, alias = cfg["mlflow"]["registered_model"], cfg["mlflow"]["production_alias"]
    mv = version_by_alias(mlflow_client, name, alias)
    if mv is None:
        return None
    paths = {a.path for a in mlflow_client.list_artifacts(mv.run_id, "predictions")}
    if not paths:
        logger.warning("v%s holds @%s but has no predictions logged; skipping the regression "
                       "check", mv.version, alias)
        return None
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        files = Path(mlflow_client.download_artifacts(mv.run_id, "predictions", tmp))
        for item in files.iterdir():
            shutil.copy2(item, out / item.name)
        thresholds = mlflow_client.download_artifacts(mv.run_id, "package/thresholds.json", tmp)
        shutil.copy2(thresholds, out / "thresholds.json")
    return {"version": str(mv.version), "model_version": mv.tags.get("model_version"),
            "run_id": mv.run_id}


def _github_output(**values: str) -> None:
    """Expose step outputs to the workflow when running in GitHub Actions."""
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            for key, value in values.items():
                f.write(f"{key}={value}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check-outputs", help="the job left every file the gate needs")
    check.add_argument("--dir", required=True, type=Path)
    check.add_argument("--gate-config", default=str(GATE_CONFIG))

    fetch = sub.add_parser("fetch-production", help="@production's predictions for the gate")
    fetch.add_argument("--out", required=True, type=Path)
    fetch.add_argument("--config", default=None)
    fetch.add_argument("--tracking-uri", default=None)

    note = sub.add_parser("notify", help="one Slack message (logged without a webhook)")
    note.add_argument("--level", choices=["ok", "warn", "fail"], required=True)
    note.add_argument("--title", required=True)
    note.add_argument("--field", action="append", default=[], metavar="KEY=VALUE")
    note.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    if args.command == "check-outputs":
        missing = check_outputs(args.dir, gate_stems(args.gate_config))
        if missing:
            logger.error("the training job's output in %s is missing: %s (contract: "
                         "docs/ci.md)", args.dir, ", ".join(missing))
            return 1
        logger.info("job outputs complete in %s", args.dir)
        return 0

    cfg = load_config(args.config)
    if args.command == "fetch-production":
        found = fetch_production(client(args.tracking_uri), cfg, args.out)
        _github_output(has_production="true" if found else "false",
                       production_version=found["version"] if found else "")
        print(json.dumps(found) if found else "none")
        return 0

    fields = dict(f.split("=", 1) for f in args.field)
    delivered = notify(cfg, args.level, args.title, fields)
    _github_output(slack_posted=str(delivered).lower())
    return 0


if __name__ == "__main__":
    sys.exit(main())
