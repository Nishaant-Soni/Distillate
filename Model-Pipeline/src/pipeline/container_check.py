"""Checks of the training image that only make sense inside it (CI runs them after the build).

1. **Code commit:** git works for the image's non-root user, so every run records
   ``code_commit`` (``meta.json``, MLflow). Root-owned ``/app`` without a system-wide
   ``safe.directory`` makes ``git rev-parse`` fail, and ``code_commit()`` silently returns None.
2. **Only tracked files:** every file in the repository copy is tracked by git, so no comment
   text (local data folders, drift batches), key file or output came in from the build context.
3. **No Git config:** ``.git/config`` is left out of the image (a remote URL there can embed
   a token), so no remote URL is set.
4. **DVC by revision:** ``load_data(rev=...)``'s read gets past DVC's local state
   (``.dvc/tmp``) to the remote. Without the bucket key it then fails on authentication, which
   passes; a local permission error fails.

    docker run --rm --entrypoint python distillate-train -m src.pipeline.container_check \\
        --expect-commit "$GITHUB_SHA"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from src.teacher.config import PIPELINE_ROOT

REPO_ROOT = PIPELINE_ROOT.parent
# A DVC-tracked file (Data-Pipeline/data/processed.dvc), as load_data reads it.
DVC_PATH = "Data-Pipeline/data/processed/wikipedia/gate.parquet"
LOCAL_FAILURES = ("permission denied", "permissionerror", "dubious ownership", "read-only file")


def check_commit(expect: str | None = None) -> str:
    from src.student.train import code_commit

    commit = code_commit()
    if commit is None:
        raise AssertionError("code_commit() is None: git fails for this user (safe.directory?)")
    if expect and commit != expect:
        raise AssertionError(f"code_commit() is {commit}, expected {expect}")
    return commit


def untracked_files(root: Path = REPO_ROOT) -> list[str]:
    """Files in the repository copy that git does not track (ignored ones included)."""
    out = subprocess.check_output(["git", "-C", str(root), "ls-files", "--others", "-z"])
    return [p for p in out.decode().split("\0") if p]


def git_remotes(root: Path = REPO_ROOT) -> list[str]:
    """Remote URLs in the image's Git config (none: .git/config stays out of the image)."""
    proc = subprocess.run(["git", "-C", str(root), "config", "--get-regexp", r"remote\..*\.url"],
                          capture_output=True, text=True)
    return [line.split(maxsplit=1)[0] for line in proc.stdout.splitlines() if line.strip()]


def check_dvc_read(rev: str = "HEAD", path: str = DVC_PATH, root: Path = REPO_ROOT) -> str:
    """'read' with the key, or the remote's error without it; a local failure raises."""
    from src.data.load_data import _dvc_read

    try:
        _dvc_read(path, str(root), rev)
        return "read"
    except Exception as err:   # noqa: BLE001  (any remote error passes; local ones don't)
        chain, e = [], err
        while e is not None:
            chain.append(f"{type(e).__name__}: {e}")
            e = e.__cause__ or e.__context__
        text = " | ".join(chain)
        if isinstance(err, PermissionError) or any(k in text.lower() for k in LOCAL_FAILURES):
            raise AssertionError(f"DVC read failed locally: {text}") from err
        return f"reached the remote ({text[:300]})"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Checks of the training image.")
    parser.add_argument("--expect-commit", default=None, help="the commit the image was built at")
    parser.add_argument("--rev", default="HEAD", help="the revision for the DVC read")
    parser.add_argument("--skip-dvc", action="store_true")
    args = parser.parse_args(argv)
    failed = False

    try:
        print("code_commit:", check_commit(args.expect_commit))
    except AssertionError as err:
        print("FAIL", err)
        failed = True

    extra = untracked_files()
    if extra:
        print(f"FAIL {len(extra)} untracked files in the image:", *extra[:20], sep="\n  ")
        failed = True
    else:
        print("untracked files: none")

    remotes = git_remotes()
    if remotes or (REPO_ROOT / ".git" / "config").exists():
        print("FAIL .git/config is in the image (remotes: {})".format(", ".join(remotes) or "none"))
        failed = True
    else:
        print("git config: not in the image")

    if not args.skip_dvc:
        try:
            print("dvc read:", check_dvc_read(args.rev))
        except AssertionError as err:
            print("FAIL", err)
            failed = True

    print("container check:", "FAILED" if failed else "ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
