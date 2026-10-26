"""Version the pipeline's data with DVC and push it to the GCS remote.

Git holds only the small ``.dvc`` pointer files; the Parquet itself lives in the
bucket. Pushing is controlled by ``DVC_PUSH`` (pipeline.yaml -> dvc.push), which
defaults to false, so a fresh clone and CI both run this task green without any
GCP credentials.

    python -m scripts.dvc_version --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.alerts import send_alert
from scripts.utils import PIPELINE_ROOT, get_logger, load_config, timed, write_summary

logger = get_logger(__name__)
TASK_ID = "dvc_version"


def _repo_root() -> Path:
    """DVC is initialised at the Git root, one level above Data-Pipeline/."""
    return PIPELINE_ROOT.parent


def _dvc(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    logger.info("dvc %s", " ".join(args))
    return subprocess.run(["dvc", *args], cwd=cwd, capture_output=True, text=True, check=False)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    root = _repo_root()
    targets = [Path(cfg["paths"][key]) for key in ("raw_dir", "processed_dir")]
    summary: dict[str, Any] = {"task": TASK_ID, "mode": "real", "tracked": [], "pushed": False}

    with timed(summary):
        if not (root / ".dvc").is_dir():
            # Not an error: the repo may not have been `dvc init`-ed yet.
            summary["skipped"] = "no .dvc directory; run `dvc init` at the repo root"
            logger.warning(summary["skipped"])
            write_summary(TASK_ID, summary, cfg)
            return summary

        for target in targets:
            if not target.exists():
                logger.warning("Skipping %s: not produced by this run", target)
                continue
            result = _dvc(["add", str(target.relative_to(root))], cwd=root)
            if result.returncode != 0:
                summary["error"] = result.stderr.strip()[:500]
                send_alert("fail", "dvc add failed",
                           {"target": target.name, "error": summary["error"]})
                raise RuntimeError(f"dvc add {target} failed: {summary['error']}")
            summary["tracked"].append(str(target.relative_to(root)))

        if cfg.get("dvc", {}).get("push"):
            result = _dvc(["push"], cwd=root)
            summary["pushed"] = result.returncode == 0
            if not summary["pushed"]:
                # A failed push is a warning: the data is versioned locally and
                # the .dvc files are still correct and committable.
                summary["push_error"] = result.stderr.strip()[:500]
                send_alert("warn", "dvc push failed", {"error": summary["push_error"]})
        else:
            summary["skipped_push"] = "DVC_PUSH is false"

    pointers = [f"{t}.dvc" for t in summary["tracked"]]
    if pointers:
        # Info, not a warning: this fires on success and asks for a normal next
        # step. Logging it as a warning made a healthy run look like it had a
        # problem.
        logger.info("Commit the updated pointer files to Git: %s", " ".join(pointers))
    summary["commit_reminder"] = pointers
    write_summary(TASK_ID, summary, cfg)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config)), indent=2, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
