"""Parquet writing and checksums for Praniti's stages.

Every write goes through the team's ``atomic_path`` (scripts/utils.py): the file is
written to a temporary path and renamed at the end, so a crash never leaves a
half-written file behind.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.utils import atomic_path


def write_frame(df: pd.DataFrame, path: Path) -> int:
    """Write one DataFrame to Parquet atomically. Returns the row count."""
    with atomic_path(path) as tmp:
        df.to_parquet(tmp, index=False)
    return len(df)


def write_chunks(chunks: Iterable[pd.DataFrame], path: Path) -> int:
    """Write DataFrames one after another into a single Parquet file, atomically.

    Used for the 1.8M-row Civil train file so it never sits in memory at once.
    Returns the total row count.
    """
    rows = 0
    with atomic_path(path) as tmp:
        writer = None
        try:
            for chunk in chunks:
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(tmp, table.schema)
                writer.write_table(table.cast(writer.schema))
                rows += table.num_rows
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            raise ValueError(f"No data to write for {path}")
    return rows


def sha256_file(path: Path) -> str:
    """SHA-256 checksum of a file, read in 1 MB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
