"""Score a trained run on the evaluation sets, measure CPU latency, and the truncation sweep.

Prediction files (CONTRACTS.md §2.2) go next to the run's ``predictions/val.parquet``, named by
the stems in ``configs/gate.yaml`` (``gate``, ``bias_eval``) plus ``h2_eval`` and
``obfuscation``, with the run's ``thresholds.json`` copied beside them, so the folder is what
``src.eval.gate --candidate`` reads. No text is written.

Latency (work plan v4: batch size 1, measured in the evaluation step) runs the §2.5 package's
``model.onnx`` with ONNX Runtime on CPU, one comment per call, tokenization included, on real
comments (their own lengths, truncated at the model's ``max_length``).

    python -m src.student.evaluate latency --package outputs/student/<run>/package \\
        --texts val_texts.parquet --threads 2 --out outputs/student/<run>/latency.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.teacher.config import get_logger

logger = get_logger(__name__)

# Prediction file stem for each evaluation table (src.data.load_data's table names).
STEMS = {"gate": "gate", "bias": "bias_eval", "h2": "h2_eval", "obfuscation": "obfuscation"}


def score_logits(model, tokenizer, texts: Sequence[str], max_length: int, batch_size: int,
                 device, fp16: bool = True) -> np.ndarray:
    """Logits in input order, batched by length (less padding, same numbers)."""
    from src.student.predict import predict_logits

    texts = list(texts)
    order = np.argsort([len(t) for t in texts], kind="stable")
    logits = predict_logits(model, tokenizer, [texts[i] for i in order], max_length,
                            batch_size, device, fp16=fp16)
    out = np.empty_like(logits)
    out[order] = logits
    return out


def score_run(run_dir: str | Path, tables: Mapping[str, Any], batch_size: int = 256,
              device=None, fp16: bool = True) -> dict[str, Any]:
    """Write ``predictions/<stem>.parquet`` for each table (``stem -> frame with id, text``)."""
    import torch

    from src.student.model import load_trained
    from src.student.predict import check_prediction_frame, prediction_frame

    run_dir = Path(run_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((run_dir / "model" / "meta.json").read_text())
    model, tokenizer = load_trained(run_dir / "model")
    model.to(device)
    (run_dir / "predictions").mkdir(exist_ok=True)
    files = {}
    for stem, frame in tables.items():
        start = time.perf_counter()
        logits = score_logits(model, tokenizer, frame["text"].astype(str).tolist(),
                              int(meta["max_length"]), batch_size, device, fp16)
        pred = prediction_frame(frame["id"].astype(str), logits, meta["model_version"])
        check_prediction_frame(pred)
        pred.to_parquet(run_dir / "predictions" / f"{stem}.parquet", index=False)
        files[stem] = {"rows": len(pred), "seconds": round(time.perf_counter() - start, 1)}
        logger.info("%s: %s scored, %d rows", run_dir.name, stem, len(pred))
    shutil.copy(run_dir / "model" / "thresholds.json", run_dir / "predictions" / "thresholds.json")
    record = {"model_version": meta["model_version"], "code_commit": meta.get("code_commit"),
              "device": str(device), "fp16": fp16, "files": files}
    (run_dir / "scoring.json").write_text(json.dumps(record, indent=2) + "\n")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def truncation_sweep(run_dir: str | Path, frame, lengths: Sequence[int] = (64, 128, 256),
                     batch_size: int = 256, device=None) -> dict[str, str]:
    """Prediction files of ``frame`` at each maximum length (``<stem>_len<L>.parquet``).

    For the sensitivity analysis (performance against input length). The model is the same;
    only the truncation changes, so lengths above the training ``max_length`` are allowed.
    """
    import torch

    from src.student.model import load_trained
    from src.student.predict import check_prediction_frame, prediction_frame

    run_dir = Path(run_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((run_dir / "model" / "meta.json").read_text())
    model, tokenizer = load_trained(run_dir / "model")
    model.to(device)
    out_dir = run_dir / "predictions" / "truncation"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for length in lengths:
        logits = score_logits(model, tokenizer, frame["text"].astype(str).tolist(), int(length),
                              batch_size, device)
        pred = prediction_frame(frame["id"].astype(str), logits, meta["model_version"])
        check_prediction_frame(pred)
        path = out_dir / f"gate_len{int(length)}.parquet"
        pred.to_parquet(path, index=False)
        written[str(length)] = str(path)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return written


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def measure_latency(package_dir: str | Path, texts: Sequence[str], threads: int = 2,
                    n: int = 300, warmup: int = 20, seed: int = 42) -> dict[str, Any]:
    """Batch-size-1 CPU latency of the package: tokenize + ONNX Runtime, per comment, in ms."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    package_dir = Path(package_dir)
    meta = json.loads((package_dir / "meta.json").read_text())
    max_length = int(meta["max_length"])
    tokenizer = AutoTokenizer.from_pretrained(package_dir / "tokenizer")
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(threads)
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(package_dir / "model.onnx"), options,
                                   providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    texts = list(texts)
    picks = [texts[i] for i in rng.choice(len(texts), size=min(n + warmup, len(texts)),
                                          replace=len(texts) < n + warmup)]
    times, tokens = [], []
    for i, text in enumerate(picks):
        start = time.perf_counter()
        batch = tokenizer([text], truncation=True, max_length=max_length, return_tensors="np")
        feed = {k: batch[k].astype("int64") for k in ("input_ids", "attention_mask")}
        session.run(["logits"], feed)
        elapsed = (time.perf_counter() - start) * 1000
        if i >= warmup:
            times.append(elapsed)
            tokens.append(int(batch["input_ids"].shape[1]))
    t = np.array(times)
    return {
        "model_version": meta.get("model_version"), "format": "onnx fp32",
        "batch_size": 1, "threads": int(threads), "requests": len(times), "warmup": warmup,
        "max_length": max_length, "mean_tokens": float(np.mean(tokens)),
        "max_tokens": int(np.max(tokens)),
        **{f"p{q}_ms": float(np.percentile(t, q)) for q in (50, 90, 95, 99)},
        "mean_ms": float(t.mean()), "cpu": cpu_name(), "cpu_count": os.cpu_count(),
        "onnxruntime": ort.__version__,
    }


def main(argv: Sequence[str] | None = None) -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    lat = sub.add_parser("latency", help="batch-1 CPU latency of a package")
    lat.add_argument("--package", required=True)
    lat.add_argument("--texts", required=True, help="parquet file with a text column")
    lat.add_argument("--threads", type=int, default=2)
    lat.add_argument("--n", type=int, default=300)
    lat.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    texts = pd.read_parquet(args.texts, columns=["text"])["text"].astype(str).tolist()
    result = measure_latency(args.package, texts, args.threads, args.n)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
