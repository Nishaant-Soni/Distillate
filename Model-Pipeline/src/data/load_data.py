"""Load the data pipeline's versioned tables for training and evaluation (Phase 2).

load_data() returns the processed Wikipedia splits (train, val, gate), the Civil evaluation sets
(h2, bias) and the obfuscation set, with the teacher labels from teacher_labels.parquet joined
onto train and val by id, plus data hashes for MLflow:

    from src.data.load_data import load_data

    data = load_data()                          # the files on disk, every table
    data = load_data(rev="a6d6ceb")             # the files as of a Git commit, through DVC
    data = load_data(tables=("train", "val"))   # only what training needs
    data["train"], data.train_data_hash, data.data_hash

The tables come from Data-Pipeline/data/processed/ (CONTRACTS.md section 2). With rev=None they
are read from disk; with a Git revision they are read through DVC exactly as that commit
versioned them, so a training run can be reproduced from the revision it logged.

data_hash is the SHA-256 over every file read (name and content). train_data_hash covers only
the files training reads (train, val and the teacher labels), so a change to an evaluation
table does not change a training run's lineage hash.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterable, Mapping
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
TRAINING_FILES = (TABLES["train"], TABLES["val"], TEACHER_FILE)
TARGETS = [f"p_{label}" for label in LABELS]
VERSION_COLUMNS = ["prompt_version", "model_version", "teacher_version"]


@dataclass(frozen=True)
class LoadedData:
    """The loaded tables, and what is needed to log them."""

    tables: dict[str, pd.DataFrame]
    data_hash: str
    train_data_hash: str
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
    """The file's bytes as versioned at ``rev`` (DVC is imported only when needed).

    A file missing from that version raises FileNotFoundError, as it does with rev=None.
    """
    try:
        import dvc.api
        from dvc.exceptions import PathMissingError
    except ImportError as err:
        raise ImportError("load_data(rev=...) needs DVC: pip install 'dvc[gs]'") from err
    try:
        return dvc.api.read(path, repo=repo, rev=rev, mode="rb")
    except PathMissingError as err:
        raise FileNotFoundError(
            f"{path} is not in the DVC version at {rev!r}: check the revision, or that the "
            "data pipeline versioned this file there"
        ) from err


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
    """Join the teacher labels onto ``table`` by id.

    Every row must get every teacher score (the targets and the ablation arms' columns), so a
    missing value anywhere stops loading instead of passing silently into training.
    """
    joined = table.merge(teacher, on="id", how="left", validate="one_to_one")
    scores = [column for column in teacher.columns if column != "id"]
    per_column = joined[scores].isna().sum()
    per_column = per_column[per_column > 0]
    if len(per_column):
        rows = joined[scores].isna().any(axis=1)
        examples = joined.loc[rows, "id"].head(3).tolist()
        details = ", ".join(f"{column}: {int(count)}" for column, count in per_column.items())
        raise ValueError(
            f"{name}: {int(rows.sum())} rows have no teacher labels or scores ({details}), "
            f"e.g. {examples}. Rebuild teacher_labels.parquet (merge_teacher_labels)."
        )
    return joined


def load_data(
    rev: str | None = None,
    *,
    data_dir: str | Path | None = None,
    with_teacher: bool = True,
    tables: Iterable[str] | None = None,
) -> LoadedData:
    """Load the tables, with the teacher labels joined onto train and val.

    rev: a Git revision (commit, branch or tag) to read through DVC; None reads the files on
        disk.
    data_dir: the processed-data folder (default: Data-Pipeline/data/processed).
    with_teacher: join teacher_labels.parquet onto train and val (default True).
    tables: which tables to load (default: all of TABLES), e.g. ("train", "val") for training.
    """
    folder = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    wanted = list(TABLES) if tables is None else list(tables)
    unknown = sorted(set(wanted) - set(TABLES))
    if unknown:
        raise ValueError(f"unknown tables {unknown}; choose from {list(TABLES)}")
    names = {key: TABLES[key] for key in wanted}
    joined_tables = [key for key in WITH_TEACHER if key in names]
    if with_teacher and joined_tables:
        names["teacher"] = TEACHER_FILE

    frames: dict[str, pd.DataFrame] = {}
    file_hashes: dict[str, str] = {}
    for key, name in names.items():
        content = read_bytes(folder, name, rev)
        file_hashes[name] = hashlib.sha256(content).hexdigest()
        frames[key] = pd.read_parquet(io.BytesIO(content))

    versions: dict[str, str] = {}
    if "teacher" in frames:
        teacher = frames.pop("teacher")
        for column in VERSION_COLUMNS:
            values = teacher[column].astype(str).unique()
            if len(values) != 1:
                raise ValueError(f"teacher labels mix several {column} values: {list(values)}")
            versions[column] = values[0]
        teacher = teacher.drop(columns=VERSION_COLUMNS)
        for key in joined_tables:
            frames[key] = attach_teacher(frames[key], teacher, key)

    training_hashes = {name: h for name, h in file_hashes.items() if name in TRAINING_FILES}
    return LoadedData(
        tables=frames,
        data_hash=combine_hashes(file_hashes),
        train_data_hash=combine_hashes(training_hashes),
        file_hashes=file_hashes,
        rev=rev,
        teacher_versions=versions,
    )
