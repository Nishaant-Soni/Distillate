"""One labeling worker: score its share of the shards and upload them.

Loop (implementation plan, Step 6):
1. take the shards whose index modulo ``--num-workers`` equals ``--worker``;
2. skip any shard whose manifest exists;
3. score the shard and write it (Parquet first, manifest last);
4. stop before the time budget runs out, so the notebook can finish cleanly.

Run one worker per GPU (``label_shards.ipynb`` sets ``CUDA_VISIBLE_DEVICES``). Rerunning the
same command resumes: finished shards are skipped and an interrupted one is redone.

    python -m src.teacher.label_worker --worker 0 --num-workers 2 \\
        --data /kaggle/input/.../wiki_train_raw.parquet --model-dir /kaggle/input/.../model \\
        --prefix teacher/p2-zs-v2b/

The prompt comes from teacher.yaml (``prompt.*``); the Hugging Face token from ``HF_TOKEN``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

FIRST_SHARD_ROWS_PER_S = 4.0   # time estimate before the first shard is measured; cautious
SAFETY_S = 300                 # leave this much of the budget unused


def select_rows(data_path: str | Path, ids_path: str | Path | None = None,
                limit: int | None = None):
    """``id`` and ``text`` of the rows to label, sorted by id.

    ``data_path``: a Parquet with ``id`` and cleaned ``text`` (e.g. ``wiki_train_raw``).
    ``ids_path``: optional Parquet/CSV of ids to keep (gate subset, drift). ``limit``: the
    first N sorted ids only (dry runs).
    """
    import pandas as pd

    from src.teacher.shards import read_ids

    rows = pd.read_parquet(data_path, columns=["id", "text"])
    rows["id"] = rows["id"].astype(str)
    if ids_path is not None:
        keep = set(read_ids(ids_path))
        rows = rows[rows["id"].isin(keep)]
        if len(rows) != len(keep):
            raise ValueError(f"{len(keep) - len(rows)} ids are not in {data_path}")
    rows = rows.sort_values("id", kind="stable").reset_index(drop=True)
    return rows.iloc[:limit].reset_index(drop=True) if limit else rows


def run_worker(rows, plan: Sequence[Sequence[str]], worker: int, num_workers: int, storage,
               prefix: str, builder: Any, score_fn: Callable, model_version: str, *,
               deadline: float, min_coverage: float = 0.5, max_retries: int = 1,
               clock: Callable[[], float] = time.monotonic,
               extra_manifest: Mapping[str, Any] | None = None,
               log: Callable[[str], None] = print) -> dict[str, Any]:
    """Label this worker's unfinished shards until done or ``deadline`` (a ``clock`` time)."""
    from src.teacher.scoring import score_texts
    from src.teacher.shards import build_manifest, shard_frame, shard_name, worker_shards
    from src.teacher.storage import shard_done, write_shard

    text = dict(zip(rows["id"], rows["text"], strict=True))
    summary: dict[str, Any] = {"worker": worker, "written": [], "skipped": [],
                               "stopped_before": None, "rows": 0, "seconds": 0.0}
    rows_per_s = FIRST_SHARD_ROWS_PER_S
    for k in worker_shards(len(plan), worker, num_workers):
        name, ids = shard_name(k), list(plan[k])
        if shard_done(storage, prefix, name):
            summary["skipped"].append(name)
            log(json.dumps({"worker": worker, "shard": name, "event": "skipped"}))
            continue
        if clock() + len(ids) / rows_per_s + SAFETY_S > deadline:
            summary["stopped_before"] = name
            log(json.dumps({"worker": worker, "shard": name, "event": "stopped",
                            "reason": "time budget"}))
            break
        start = clock()
        pred, info = score_texts(ids, [text[i] for i in ids], builder, score_fn,
                                 min_coverage, max_retries)
        frame = shard_frame(pred, builder.prompt_version, model_version)
        ok = pred["status"] == "ok"
        seconds = clock() - start
        manifest = build_manifest(
            frame, builder.prompt_version, model_version, **(extra_manifest or {}),
            worker=worker, seconds=round(seconds, 1), failed_rows=info["failed_rows"],
            empty_rows=info["empty_rows"], truncated_rows=info["truncated_rows"],
            retried_cells=info["retried_cells"], censored_cells=info["censored_cells"],
            coverage_min=round(float(pred.loc[ok, "coverage"].min()), 6) if ok.any() else None)
        write_shard(storage, prefix, name, frame, manifest)
        summary["written"].append(name)
        summary["rows"] += len(ids)
        summary["seconds"] += seconds
        rows_per_s = summary["rows"] / max(summary["seconds"], 1e-9)
        log(json.dumps({"worker": worker, "shard": name, "event": "written", "rows": len(ids),
                        "seconds": round(seconds, 1), "failed_rows": info["failed_rows"]}))
    summary["seconds"] = round(summary["seconds"], 1)
    return summary


def load_scorer(cfg: Mapping[str, Any], model_dir: str | Path):
    """``(tokenizer, score_fn, vllm_version)`` with the pilot's vLLM settings."""
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM

    from src.teacher.scoring import answer_token_ids, vllm_score_fn

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    yes_ids, no_ids = answer_token_ids(tokenizer)
    llm = LLM(model=str(model_dir), dtype=cfg["model"]["dtype"],
              max_model_len=cfg["model"]["max_model_len"], seed=cfg["seed"],
              gpu_memory_utilization=0.90, enable_prefix_caching=True,
              generation_config="vllm")
    return tokenizer, vllm_score_fn(llm, yes_ids, no_ids, cfg["scoring"]["logprobs"]), \
        vllm.__version__


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher labeling worker.")
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data", required=True, help="Parquet with id and cleaned text")
    parser.add_argument("--ids", default=None, help="ids to keep (gate subset, drift)")
    parser.add_argument("--prefix", required=True, help="e.g. teacher/p2-zs-v2b/")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--plan", default=None, help="teacher_shards.json to check against")
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="first N sorted ids (dry run)")
    parser.add_argument("--started-at", type=float, default=None,
                        help="session start (Unix time); the budget counts from here")
    parser.add_argument("--stop-after-hours", type=float, default=None)
    parser.add_argument("--local-root", default=None, help="local folder instead of the store")
    parser.add_argument("--code-commit", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    # Before anything imports transformers (Kaggle's torchaudio does not match vLLM's torch),
    # and a fresh engine process rather than a fork.
    from src.teacher.compat import hide_torchaudio

    hide_torchaudio()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from src.teacher.config import load_config, model_version
    from src.teacher.prompts import PromptBuilder
    from src.teacher.shards import check_plan, plan_shards
    from src.teacher.storage import get_storage

    cfg = load_config(args.config)
    hours = args.stop_after_hours or cfg["labeling"]["stop_after_hours"]
    started_at = args.started_at or time.time()
    deadline = time.monotonic() + (started_at + hours * 3600 - time.time())

    rows = select_rows(args.data, args.ids, args.limit)
    plan = plan_shards(rows["id"].tolist(), args.shard_size or cfg["labeling"]["shard_size"])
    if args.plan:
        check_plan(plan, json.loads(Path(args.plan).read_text()))
    storage = get_storage(cfg, args.local_root)
    tokenizer, score_fn, vllm_version = load_scorer(cfg, args.model_dir)
    builder = PromptBuilder.from_config(cfg, tokenizer)
    print(json.dumps({"worker": args.worker, "event": "ready", "rows": len(rows),
                      "shards": len(plan), "prompt_version": builder.prompt_version}))
    summary = run_worker(
        rows, plan, args.worker, args.num_workers, storage, args.prefix, builder, score_fn,
        model_version(cfg), deadline=deadline, min_coverage=cfg["scoring"]["min_coverage"],
        max_retries=cfg["scoring"]["max_retries"],
        extra_manifest={"code_commit": args.code_commit, "vllm_version": vllm_version})
    print(json.dumps({"event": "done", **summary}))


if __name__ == "__main__":
    main()
