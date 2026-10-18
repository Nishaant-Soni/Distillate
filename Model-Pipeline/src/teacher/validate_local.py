"""Shard checks (docs/teacher_prompts_and_shard_spec.md §6) for one labeling run.

Reads every shard Parquet and manifest under a prefix, from the configured store or a local
folder, and checks them against the source ids. Exits non-zero and lists every failed check
with the shard name, so it works by hand and as a DAG task.

With ``--partial`` (after a session, before all shards exist), the coverage check is
skipped; every shard that does exist must still match its planned ids exactly.

    python -m src.teacher.validate_local --prefix teacher/p2-zs-v2b/
    python -m src.teacher.validate_local --prefix teacher/p2-zs-v2b/ --partial
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.teacher.config import LABELS, load_config, model_version
from src.teacher.prompts import prompt_version
from src.teacher.shards import (
    REQUIRED_MANIFEST,
    SHARD_COLUMNS,
    check_plan,
    ids_sha256,
    plan_shards,
    read_ids,
)
from src.teacher.storage import Storage, get_storage

SHARD_FILE = re.compile(r"(shard_\d{4})\.(parquet|manifest\.json)$")
FLOAT_COLUMNS = SHARD_COLUMNS[1:13]
STRING_COLUMNS = ["id", "status", "prompt_version", "model_version"]


def load_run(storage: Storage, prefix: str, cache_dir: Path):
    """``(frames, manifests, schemas, errors)`` for the shards directly under ``prefix``."""
    import pandas as pd
    import pyarrow.parquet as pq

    prefix = prefix if prefix.endswith("/") else prefix + "/"
    frames, manifests, schemas, errors = {}, {}, {}, []
    for path in storage.list(prefix):
        rest = path[len(prefix):]
        if "/" in rest:          # gate/ and drift/ runs live under the training prefix
            continue
        match = SHARD_FILE.fullmatch(rest)
        if match is None:
            errors.append(f"{rest}: unexpected file")
            continue
        name, kind = match.groups()
        local = storage.download(path, cache_dir)
        if kind == "parquet":
            schemas[name] = pq.read_schema(local)
            frames[name] = pd.read_parquet(local)
        else:
            manifests[name] = json.loads(Path(local).read_text())
    return frames, manifests, schemas, errors


def check_run(frames: Mapping[str, Any], manifests: Mapping[str, Mapping[str, Any]],
              schemas: Mapping[str, Any], source_ids: Sequence[str], shard_size: int,
              expected_prompt: str, expected_model: str, max_failure: float = 0.01,
              partial: bool = False, plan_record: Mapping[str, Any] | None = None) -> list[str]:
    """Every failed check, as ``"<shard>: <problem>"`` strings. Empty means valid."""
    import numpy as np
    import pyarrow as pa

    errors: list[str] = []
    plan = plan_shards(source_ids, shard_size)
    if plan_record is not None:
        try:
            check_plan(plan, plan_record)
        except ValueError as error:
            errors.append(f"plan: {error}")
    planned = {f"shard_{k:04d}": ids for k, ids in enumerate(plan)}
    source = set(source_ids)

    for name in sorted(set(frames) | set(manifests)):
        if name not in manifests:
            errors.append(f"{name}: Parquet without a manifest")
        if name not in frames:
            errors.append(f"{name}: manifest without a Parquet")
    seen: dict[str, str] = {}
    failed_total = rows_total = 0
    for name in sorted(set(frames) & set(manifests)):
        frame, manifest, schema = frames[name], manifests[name], schemas[name]
        missing = [k for k in REQUIRED_MANIFEST if k not in manifest]
        if missing:
            errors.append(f"{name}: manifest misses {missing}")
            continue
        if list(frame.columns) != SHARD_COLUMNS:
            errors.append(f"{name}: columns {list(frame.columns)}, expected {SHARD_COLUMNS}")
            continue
        ids = frame["id"].astype(str).tolist()
        if len(frame) != manifest["row_count"]:
            errors.append(f"{name}: {len(frame)} rows, manifest says {manifest['row_count']}")
        if ids_sha256(ids) != manifest["ids_sha256"]:
            errors.append(f"{name}: id hash differs from the manifest")
        if name not in planned:
            errors.append(f"{name}: not in the shard plan ({len(plan)} shards)")
        elif sorted(ids) != planned[name]:
            errors.append(f"{name}: ids differ from the planned shard")
        outside = [i for i in ids if i not in source]
        if outside:
            errors.append(f"{name}: {len(outside)} ids not in the source list, e.g. {outside[0]}")
        if len(set(ids)) != len(ids):
            errors.append(f"{name}: duplicate ids within the shard")
        for i in ids:
            if i in seen and seen[i] != name:
                errors.append(f"{name}: id {i} also in {seen[i]}")
                break
            seen.setdefault(i, name)

        for column in STRING_COLUMNS:
            if not pa.types.is_string(schema.field(column).type) and \
                    not pa.types.is_large_string(schema.field(column).type):
                errors.append(f"{name}: {column} is {schema.field(column).type}, not string")
        for column in FLOAT_COLUMNS:
            if schema.field(column).type != pa.float32():
                errors.append(f"{name}: {column} is {schema.field(column).type}, not float32")

        status = frame["status"].astype(str)
        bad_status = sorted(set(status) - {"ok", "failed"})
        if bad_status:
            errors.append(f"{name}: status values {bad_status}")
        ok, failed = (status == "ok").to_numpy(), (status == "failed").to_numpy()
        p = frame[[f"p_{label}" for label in LABELS]].to_numpy(dtype="float64")
        z = frame[[f"logit_{label}" for label in LABELS]].to_numpy(dtype="float64")
        if ok.any() and not (np.isfinite(p[ok]).all() and (p[ok] >= 0).all()
                             and (p[ok] <= 1).all()):
            errors.append(f"{name}: ok rows with p outside [0, 1] or NaN")
        if ok.any() and not np.isfinite(z[ok]).all():
            errors.append(f"{name}: ok rows with non-finite logits")
        if failed.any() and not (np.isnan(p[failed]).all() and np.isnan(z[failed]).all()):
            errors.append(f"{name}: failed rows with non-NaN scores")
        if len(frame) and failed.mean() > max_failure:
            errors.append(f"{name}: failure rate {failed.mean():.2%} above {max_failure:.0%}")
        failed_total += int(failed.sum())
        rows_total += len(frame)

        for key, expected in (("prompt_version", expected_prompt),
                              ("model_version", expected_model)):
            values = set(frame[key].astype(str)) | {str(manifest[key])}
            if values != {expected}:
                errors.append(f"{name}: {key} {sorted(values)}, expected {expected}")

    if rows_total and failed_total / rows_total > max_failure:
        errors.append(f"run: failure rate {failed_total / rows_total:.2%} above "
                      f"{max_failure:.0%}")
    if not partial:
        missing_shards = sorted(set(planned) - (set(frames) & set(manifests)))
        if missing_shards:
            errors.append(f"run: {len(missing_shards)} planned shards missing, e.g. "
                          f"{missing_shards[0]}")
        uncovered = source - set(seen)
        if uncovered:
            errors.append(f"run: {len(uncovered)} source ids have no label")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate teacher shards.")
    parser.add_argument("--prefix", required=True, help="e.g. teacher/p2-zs-v2b/")
    parser.add_argument("--ids", default=None, help="source ids (default: raw train)")
    parser.add_argument("--plan", default=None, help="teacher_shards.json to check against")
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="first N sorted source ids only (dry runs)")
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--local-root", default=None, help="local folder instead of the store")
    parser.add_argument("--partial", action="store_true", help="skip the coverage check")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ids_path = args.ids or Path(cfg["paths"]["data_dir"]) / cfg["data"]["files"]["train"]
    source = sorted(read_ids(ids_path))
    if args.limit:
        source = source[:args.limit]
    p = cfg["prompt"]
    expected_prompt = args.prompt_version or prompt_version(p["protocol"], p["shots"],
                                                            p.get("text_version", "v1"))
    plan_record = json.loads(Path(args.plan).read_text()) if args.plan else None
    storage = get_storage(cfg, args.local_root)
    with tempfile.TemporaryDirectory() as tmp:
        frames, manifests, schemas, errors = load_run(storage, args.prefix, Path(tmp))
        errors += check_run(frames, manifests, schemas, source,
                            args.shard_size or cfg["labeling"]["shard_size"], expected_prompt,
                            model_version(cfg), partial=args.partial, plan_record=plan_record)
    summary = {"prefix": args.prefix, "shards": len(set(frames) & set(manifests)),
               "rows": sum(len(f) for f in frames.values()), "errors": len(errors)}
    print(json.dumps(summary))
    for error in errors:
        print("FAIL", error)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
