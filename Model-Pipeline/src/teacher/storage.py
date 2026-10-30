"""Where shards live: a private Hugging Face dataset now, GCS later (``storage.backend``).

Both backends store files under repository-relative paths such as
``teacher/p2-zs-v2b/shard_0000.parquet``. ``write_shard`` uploads the Parquet first and the
manifest last, so a manifest only exists for a fully uploaded shard.

``LocalStorage`` keeps the same layout in a folder, for tests and local dry runs.
The Hugging Face token comes from the ``HF_TOKEN`` environment variable (Kaggle or Colab
Secrets); it is never passed on the command line or written anywhere.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from src.teacher.config import get_logger

logger = get_logger(__name__)


class Storage(Protocol):
    def exists(self, path: str) -> bool: ...
    def upload(self, local: Path, path: str) -> None: ...
    def download(self, path: str, local_dir: Path) -> Path: ...
    def list(self, prefix: str) -> list[str]: ...
    def delete(self, path: str) -> None: ...


class LocalStorage:
    """Files under ``root``, with the same relative paths as the remote store."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def exists(self, path: str) -> bool:
        return (self.root / path).is_file()

    def upload(self, local: Path, path: str) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        shutil.copyfile(local, partial)
        partial.replace(target)   # atomic, like a remote commit

    def download(self, path: str, local_dir: Path) -> Path:
        target = Path(local_dir) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / path, target)
        return target

    def delete(self, path: str) -> None:
        (self.root / path).unlink()

    def list(self, prefix: str) -> list[str]:
        base = self.root / prefix
        if not base.exists():
            return []
        return sorted(p.relative_to(self.root).as_posix() for p in base.rglob("*")
                      if p.is_file() and not p.name.endswith(".partial"))


class HFStorage:
    """A private Hugging Face dataset repository. Each upload is one commit."""

    def __init__(self, repo_id: str, retries: int = 4, backoff_s: float = 5.0):
        from huggingface_hub import HfApi

        self.api, self.repo_id = HfApi(), repo_id
        self.retries, self.backoff_s = retries, backoff_s

    def _retry(self, what: str, fn):
        for attempt in range(self.retries):
            try:
                return fn()
            except Exception as error:   # network errors, commit conflicts between workers
                if attempt == self.retries - 1:
                    raise
                wait = self.backoff_s * 2 ** attempt
                logger.warning("%s failed (%s); retrying in %.0f s", what, error, wait)
                time.sleep(wait)
        raise AssertionError("unreachable")

    def exists(self, path: str) -> bool:
        return self._retry(f"exists {path}", lambda: self.api.file_exists(
            self.repo_id, path, repo_type="dataset"))

    def upload(self, local: Path, path: str) -> None:
        self._retry(f"upload {path}", lambda: self.api.upload_file(
            path_or_fileobj=str(local), path_in_repo=path, repo_id=self.repo_id,
            repo_type="dataset", commit_message=f"Add {path}"))

    def download(self, path: str, local_dir: Path) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(self._retry(f"download {path}", lambda: hf_hub_download(
            self.repo_id, path, repo_type="dataset", local_dir=str(local_dir))))

    def delete(self, path: str) -> None:
        self._retry(f"delete {path}", lambda: self.api.delete_file(
            path, self.repo_id, repo_type="dataset", commit_message=f"Delete {path}"))

    def list(self, prefix: str) -> list[str]:
        files = self._retry(f"list {prefix}", lambda: self.api.list_repo_files(
            self.repo_id, repo_type="dataset"))
        return sorted(f for f in files if f.startswith(prefix))


def get_storage(cfg: Mapping[str, Any], local_root: str | Path | None = None) -> Storage:
    """``LocalStorage`` if ``local_root`` is given, else the configured backend."""
    if local_root is not None:
        return LocalStorage(local_root)
    backend = cfg["storage"]["backend"]
    if backend == "hf":
        return HFStorage(cfg["storage"]["hf_repo"])
    raise ValueError(f"storage backend {backend!r} is not implemented yet")


def shard_paths(prefix: str, name: str) -> tuple[str, str]:
    """``(<prefix><name>.parquet, <prefix><name>.manifest.json)``."""
    prefix = prefix if prefix.endswith("/") else prefix + "/"
    return f"{prefix}{name}.parquet", f"{prefix}{name}.manifest.json"


def shard_done(storage: Storage, prefix: str, name: str) -> bool:
    """A shard is complete when its manifest exists."""
    return storage.exists(shard_paths(prefix, name)[1])


def write_shard(storage: Storage, prefix: str, name: str, frame,
                manifest: Mapping[str, Any]) -> None:
    """Upload the shard Parquet, then its manifest."""
    parquet_path, manifest_path = shard_paths(prefix, name)
    with tempfile.TemporaryDirectory() as tmp:
        local_parquet = Path(tmp) / f"{name}.parquet"
        local_manifest = Path(tmp) / f"{name}.manifest.json"
        frame.to_parquet(local_parquet, index=False)
        local_manifest.write_text(json.dumps(dict(manifest), indent=2) + "\n")
        storage.upload(local_parquet, parquet_path)
        storage.upload(local_manifest, manifest_path)
