"""Load the data pipeline's versioned tables for training and evaluation (Phase 2).

load_data() returns the processed Wikipedia splits (train, val, gate), the Civil evaluation sets
(h2, bias) and the obfuscation set, with the teacher labels from teacher_labels.parquet joined
onto train and val by id, plus a data hash for MLflow:

    from src.data.load_data import load_data

    data = load_data()                 # the files on disk
    data = load_data(rev="a6d6ceb")    # the files as of a Git commit, through DVC
    data["train"], data.data_hash

The tables come from Data-Pipeline/data/processed/ (CONTRACTS.md section 2). With rev=None they
are read from disk; with a Git revision they are read through DVC exactly as that commit
versioned them, so a training run can be reproduced from the revision it logged.

The data hash is the SHA-256 of every file read (name and content), so it changes whenever any
input changes, and is the same for the same data however it was read.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.teacher.config import LABELS

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "Data-Pipeline" / "data" / "processed"

TABLES: Mapping[str, str] = {
    "train": "wikipedia/train.parquet",
    "val": "wikipedia/val.parquet",
    "gate": "wikipedia/gate.parquet",
    "h2": "civil/h2_eval.parquet",
    "bias": "civil/bias_eval.parquet",
    "obfuscation": "obfuscation.parquet",
}
TEACHER_FILE = "teacher_labels.parquet"
WITH_TEACHER = ("train", "val")
TARGETS = [f"p_{label}" for label in LABELS]
VERSION_COLUMNS = ["prompt_version", "model_version", "teacher_version"]


@dataclass(frozen=True)
class LoadedData:
    """The loaded tables, and what is needed to log them."""

    tables: dict[str, pd.DataFrame]
    data_hash: str
    file_hashes: dict[str, str]
    rev: str | None
    teacher_versions: dict[str, str] = field(default_factory=dict)

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.tables[name]


def find_dvc_root(start: Path) -> Path:
    """The nearest folder at or above ``start`` that holds a .dvc folder."""
    for folder in (start, *start.parents):
        if (folder / ".dvc").is_dir():
            return folder
    raise FileNotFoundError(f"no DVC repository (.dvc folder) at or above {start}")


def _dvc_read(path: str, repo: str, rev: str) -> bytes:
    """The file's bytes as versioned at ``rev`` (DVC is imported only when needed)."""
    try:
        import dvc.api
    except ImportError as err:
        raise ImportError("load_data(rev=...) needs DVC: pip install 'dvc[gs]'") from err
    return dvc.api.read(path, repo=repo, rev=rev, mode="rb")


def read_bytes(data_dir: Path, name: str, rev: str | None) -> bytes:
    """One file's bytes, from disk (rev=None) or through DVC at ``rev``."""
    path = data_dir / name
    if rev is None:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found: run the data pipeline, or dvc pull, first"
            )
        return path.read_bytes()
    root = find_dvc_root(data_dir)
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return _dvc_read(relative, str(root), rev)


def combine_hashes(file_hashes: Mapping[str, str]) -> str:
    """One hash for a set of files: SHA-256 of the sorted name:sha256 lines."""
    text = "\n".join(f"{name}:{digest}" for name, digest in sorted(file_hashes.items()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def attach_teacher(table: pd.DataFrame, teacher: pd.DataFrame, name: str) -> pd.DataFrame:
    """Join the teacher labels onto ``table`` by id; every row must get its targets."""
    joined = table.merge(teacher, on="id", how="left", validate="one_to_one")
    missing = joined[TARGETS].isna().any(axis=1)
    if missing.any():
        examples = joined.loc[missing, "id"].head(3).tolist()
        raise ValueError(
            f"{name}: {int(missing.sum())} rows have no teacher labels, e.g. {examples}. "
            "Rebuild teacher_labels.parquet (merge_teacher_labels)."
        )
    return joined


def load_data(
    rev: str | None = None,
    *,
    data_dir: str | Path | None = None,
    with_teacher: bool = True,
) -> LoadedData:
    """Load every table, with the teacher labels joined onto train and val.

    rev: a Git revision (commit, branch or tag) to read through DVC; None reads the files on
        disk.
    data_dir: the processed-data folder (default: Data-Pipeline/data/processed).
    with_teacher: join teacher_labels.parquet onto train and val (default True).
    """
    folder = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    names = dict(TABLES)
    if with_teacher:
        names["teacher"] = TEACHER_FILE

    frames: dict[str, pd.DataFrame] = {}
    file_hashes: dict[str, str] = {}
    for key, name in names.items():
        content = read_bytes(folder, name, rev)
        file_hashes[name] = hashlib.sha256(content).hexdigest()
        frames[key] = pd.read_parquet(io.BytesIO(content))

    versions: dict[str, str] = {}
    if with_teacher:
        teacher = frames.pop("teacher")
        for column in VERSION_COLUMNS:
            values = teacher[column].astype(str).unique()
            if len(values) != 1:
                raise ValueError(f"teacher labels mix several {column} values: {list(values)}")
            versions[column] = values[0]
        teacher = teacher.drop(columns=VERSION_COLUMNS)
        for key in WITH_TEACHER:
            frames[key] = attach_teacher(frames[key], teacher, key)

    return LoadedData(
        tables=frames,
        data_hash=combine_hashes(file_hashes),
        file_hashes=file_hashes,
        rev=rev,
        teacher_versions=versions,
    )
