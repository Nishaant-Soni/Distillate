"""Shard plan, shard frames, and manifests (docs/teacher_prompts_and_shard_spec.md §5).

The plan sorts the source ids and cuts consecutive shards of up to ``labeling.shard_size``
rows, so the same ids always give the same shards. ``teacher_shards.json`` records each
shard's row count, first and last id, and id hash; it is small enough to commit, and the
worker recomputes the plan from the source ids and checks it against the file.

A shard is complete when its manifest exists. The worker uploads the Parquet first and the
manifest last, so a crash in between leaves no manifest and the shard is redone.

    python -m src.teacher.shards plan   # writes artifacts/teacher_shards.json for raw train
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.teacher.config import LABELS, load_config

SHARD_COLUMNS = ["id", *[f"p_{label}" for label in LABELS],
                 *[f"logit_{label}" for label in LABELS], "status", "prompt_version",
                 "model_version"]
REQUIRED_MANIFEST = ("row_count", "ids_sha256", "prompt_version", "model_version")


def shard_name(index: int) -> str:
    return f"shard_{index:04d}"


def ids_sha256(ids: Sequence[str]) -> str:
    """SHA-256 of the sorted ids joined by newlines."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def plan_shards(ids: Sequence[str], shard_size: int) -> list[list[str]]:
    """Sorted ids cut into consecutive shards of up to ``shard_size``."""
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    ordered = sorted(str(i) for i in ids)
    if len(set(ordered)) != len(ordered):
        raise ValueError("source ids contain duplicates")
    return [ordered[k:k + shard_size] for k in range(0, len(ordered), shard_size)]


def plan_record(plan: Sequence[Sequence[str]], source: str, shard_size: int) -> dict[str, Any]:
    """The content of ``teacher_shards.json``."""
    return {
        "source": source,
        "shard_size": shard_size,
        "num_shards": len(plan),
        "total_rows": sum(len(s) for s in plan),
        "ids_sha256": ids_sha256([i for s in plan for i in s]),
        "shards": [{"name": shard_name(k), "row_count": len(s), "first_id": s[0],
                    "last_id": s[-1], "ids_sha256": ids_sha256(s)}
                   for k, s in enumerate(plan)],
    }


def check_plan(plan: Sequence[Sequence[str]], record: Mapping[str, Any]) -> None:
    """Raise if ``plan`` differs from a saved ``teacher_shards.json``."""
    if [s["ids_sha256"] for s in record["shards"]] != [ids_sha256(s) for s in plan]:
        raise ValueError("the shard plan differs from the saved plan; are the ids the same?")


def worker_shards(num_shards: int, worker: int, num_workers: int) -> list[int]:
    """Shard indices for one worker: index modulo ``num_workers`` equals ``worker``."""
    if not 0 <= worker < num_workers:
        raise ValueError(f"worker must be in [0, {num_workers}), got {worker}")
    return [k for k in range(num_shards) if k % num_workers == worker]


def shard_frame(pred, prompt_version: str, model_version: str):
    """``score_texts`` output reduced to the spec's shard columns and types, sorted by id."""
    import numpy as np
    import pandas as pd

    out = pd.DataFrame({"id": pd.Series(pred["id"].astype(str).to_numpy(), dtype="string")})
    for column in SHARD_COLUMNS[1:13]:
        out[column] = pred[column].to_numpy(dtype=np.float32)
    out["status"] = pd.Series(pred["status"].to_numpy(), dtype="string")
    out["prompt_version"] = pd.Series([prompt_version] * len(out), dtype="string")
    out["model_version"] = pd.Series([model_version] * len(out), dtype="string")
    return out.sort_values("id", kind="stable").reset_index(drop=True)


def build_manifest(frame, prompt_version: str, model_version: str,
                   **optional: Any) -> dict[str, Any]:
    """Required fields from the frame, plus optional fields (validators ignore unknown ones)."""
    return {
        "row_count": len(frame),
        "ids_sha256": ids_sha256(frame["id"].tolist()),
        "prompt_version": prompt_version,
        "model_version": model_version,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **{k: v for k, v in optional.items() if v is not None},
    }


def read_ids(path: str | Path) -> list[str]:
    """The ``id`` column of a Parquet or CSV file."""
    import pandas as pd

    path = Path(path)
    frame = (pd.read_csv(path, dtype=str, keep_default_na=False) if path.suffix == ".csv"
             else pd.read_parquet(path, columns=["id"]))
    return frame["id"].astype(str).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher shard plan.")
    parser.add_argument("what", choices=["plan"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--ids", default=None, help="Parquet/CSV with an id column "
                        "(default: raw train from paths.data_dir)")
    parser.add_argument("--out", default=None, help="default: paths.shard_plan")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ids_path = Path(args.ids or Path(cfg["paths"]["data_dir"]) / cfg["data"]["files"]["train"])
    plan = plan_shards(read_ids(ids_path), cfg["labeling"]["shard_size"])
    record = plan_record(plan, ids_path.name, cfg["labeling"]["shard_size"])
    out = Path(args.out or cfg["paths"]["shard_plan"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "shards"}, indent=2))


if __name__ == "__main__":
    main()
