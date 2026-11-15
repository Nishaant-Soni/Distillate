"""Copy the processed tables a CI run needs from DVC into a writable folder.

The training image holds the code and the ``.dvc`` pointers but no data, and ``/app`` is not
writable by its user, so ``dvc pull`` can't fill ``Data-Pipeline/data/processed`` there. This
streams each file through ``dvc.api.open`` (the DVC remote, ``gs://distillate-mlops-data/dvc``,
with the job's own credentials) into ``--out``, in the layout ``load_data`` reads, so the run
can point ``paths.official_dir`` and ``pipeline.data_dir`` at it:

    python -m src.pipeline.fetch_data --out /tmp/processed --tables train val gate bias

Files are read at a Git revision (default ``HEAD``, the image's own commit). Without one, DVC
only looks at the workspace, and the image's workspace has no data, so every read fails.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.data.load_data import REPO_ROOT, TABLES
from src.teacher.config import get_logger

logger = get_logger(__name__)
PROCESSED = "Data-Pipeline/data/processed"


def _dvc_open(path: str, repo: str, rev: str | None):
    import dvc.api

    return dvc.api.open(path, repo=repo, rev=rev, mode="rb")


def fetch(tables: list[str], out: Path, repo: Path = REPO_ROOT, rev: str | None = "HEAD",
          opener=_dvc_open) -> dict[str, int]:
    """Copy each table's file to ``out/<relative path>``; return the bytes written per table."""
    unknown = sorted(set(tables) - set(TABLES))
    if unknown:
        raise ValueError(f"unknown tables {unknown}; choose from {sorted(TABLES)}")
    written = {}
    for name in tables:
        relative = TABLES[name]
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with opener(f"{PROCESSED}/{relative}", str(repo), rev) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
        written[name] = target.stat().st_size
        logger.info("%s: %s (%.1f MB)", name, target, written[name] / 1e6)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tables", nargs="+", required=True, choices=sorted(TABLES))
    parser.add_argument("--rev", default="HEAD", help="Git revision (default: HEAD)")
    args = parser.parse_args(argv)
    fetch(args.tables, args.out, rev=args.rev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
