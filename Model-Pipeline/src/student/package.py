"""The model package (work plan v4 §2.5): what the registry stores and serving loads.

    <package>/
        model.onnx          inputs input_ids, attention_mask (int64, dynamic batch and
                            sequence); output logits (batch, 6), in LABELS order
        tokenizer/          save_pretrained tokenizer
        thresholds.json     per label, chosen on validation (thresholds.py)
        meta.json           §2.5 lineage plus max_length, labels, the ONNX export and its
                            parity check, and the SHA-256 of every other file

Built from a training run folder (``<run>/model``: weights, tokenizer, meta.json,
thresholds.json). The ONNX graph is exported with torch's dynamo exporter and checked against
PyTorch on the same inputs at several sequence lengths and batch sizes (``package.parity``),
including lengths other than the training ``max_length``, so the truncation sweep can score at
64, 128 and 256. A failed check stops the build. int8 quantization is Phase 3.

    python -m src.student.package --run-dir outputs/student/<run> --out outputs/packages/<run>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.teacher.config import LABELS, get_logger

logger = get_logger(__name__)

INPUTS = ("input_ids", "attention_mask")
OUTPUT = "logits"
DEFAULTS = {"opset": 18, "parity_lengths": [16, 64, 128, 256], "parity_batch_sizes": [1, 4],
            "max_abs_logit_diff": 1e-3}
# Meta fields v4 §2.5 requires, copied from the run's meta.json.
SECTION_2_5 = ("model_version", "training_data_dvc_hash", "alpha", "temperature",
               "teacher_version", "teacher_model_version", "prompt_version", "stack_sha256")
# Neutral filler for the parity check when no texts are given: no comment text needed.
PARITY_TEXTS = ("thanks for the edit to the article", "please see the talk page section",
                "the source link in the image note was changed back", "revert",
                "user page fact check " * 80)    # longer than every parity length


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logits_module(model):
    """``(input_ids, attention_mask) -> logits``: the graph ONNX sees, without extra outputs."""
    import torch

    class Logits(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits

    return Logits(model).eval()


def single_file(path: Path) -> None:
    """Fold external weight files (``model.onnx.data``) back into ``model.onnx``."""
    import onnx

    extra = [p for p in path.parent.iterdir() if p.name.startswith(path.name) and p != path]
    if not extra:
        return
    graph = onnx.load(str(path), load_external_data=True)
    onnx.save_model(graph, str(path), save_as_external_data=False)
    for leftover in extra:
        leftover.unlink()


def export_onnx(model, tokenizer, path: str | Path, opset: int = 18,
                max_positions: int | None = None) -> dict[str, Any]:
    """Export ``model`` to ``path`` with dynamic batch and sequence; return the export record."""
    import torch

    model = model.eval().cpu()
    # Example inputs of batch 2 and length 17: neither 0, 1, nor the training length, so the
    # exporter can't specialize a dimension on them.
    example = tokenizer(["export example one", "export example two is longer"],
                        padding="max_length", max_length=17, truncation=True,
                        return_tensors="pt")
    args = tuple(example[name] for name in INPUTS)
    limit = max_positions or getattr(model.config, "max_position_embeddings", 512)
    batch = torch.export.Dim("batch")
    sequence = torch.export.Dim("sequence", max=int(limit))
    start = time.perf_counter()
    torch.onnx.export(logits_module(model), args, str(path), input_names=list(INPUTS),
                      output_names=[OUTPUT], opset_version=opset, dynamo=True,
                      dynamic_shapes={name: {0: batch, 1: sequence} for name in INPUTS},
                      optimize=True, verbose=False)
    single_file(Path(path))
    return {"exporter": "torch.onnx dynamo", "torch": torch.__version__, "opset": opset,
            "inputs": {name: ["batch", "sequence"] for name in INPUTS},
            "outputs": {OUTPUT: ["batch", len(LABELS)]}, "max_sequence": int(limit),
            "seconds": round(time.perf_counter() - start, 1)}


def parity(model, tokenizer, onnx_path: str | Path, texts: Sequence[str],
           lengths: Sequence[int], batch_sizes: Sequence[int]) -> dict[str, Any]:
    """Max |logit difference| between PyTorch and ONNX Runtime per (length, batch size)."""
    import numpy as np
    import onnxruntime
    import torch

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]
    if tuple(names) != INPUTS or [o.name for o in session.get_outputs()] != [OUTPUT]:
        raise ValueError(f"model.onnx has inputs {names}, outputs "
                         f"{[o.name for o in session.get_outputs()]}")
    model = model.eval().cpu()
    # Longest first, so every batch reaches its max_length and the rest are padded.
    ordered = sorted(texts, key=len, reverse=True)
    cases, worst = [], 0.0
    for length in lengths:
        for size in batch_sizes:
            batch_texts = [ordered[i % len(ordered)] for i in range(size)]
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=int(length),
                            return_tensors="pt")
            with torch.inference_mode():
                ref = model(input_ids=enc["input_ids"],
                            attention_mask=enc["attention_mask"]).logits.numpy()
            out = session.run([OUTPUT], {k: enc[k].numpy().astype("int64") for k in INPUTS})[0]
            diff = float(np.abs(out - ref).max())
            worst = max(worst, diff)
            cases.append({"max_length": int(length), "batch": int(size),
                          "sequence": int(enc["input_ids"].shape[1]), "max_abs_diff": diff})
    return {"max_abs_logit_diff": worst, "cases": cases}


def build_package(run_dir: str | Path, out: str | Path,
                  settings: Mapping[str, Any] | None = None,
                  texts: Sequence[str] | None = None) -> dict[str, Any]:
    """Build the §2.5 package from ``<run_dir>/model`` into ``out``; return its meta.json."""
    from src.student.model import load_trained

    s = {**DEFAULTS, **(settings or {})}
    model_dir = Path(run_dir) / "model"
    for name in ("meta.json", "thresholds.json", "config.json"):
        if not (model_dir / name).exists():
            raise FileNotFoundError(f"{model_dir / name} not found: package a finished run")
    run_meta = json.loads((model_dir / "meta.json").read_text())
    if tuple(run_meta.get("labels", LABELS)) != LABELS:
        raise ValueError(f"{model_dir}: labels {run_meta['labels']}, expected {list(LABELS)}")
    thresholds = json.loads((model_dir / "thresholds.json").read_text())
    if sorted(thresholds) != sorted(LABELS):
        raise ValueError(f"{model_dir}/thresholds.json must have exactly the six labels")

    model, tokenizer = load_trained(model_dir)
    names = tuple(model.config.id2label[k] for k in sorted(model.config.id2label, key=int))
    if names != LABELS:
        raise ValueError(f"{model_dir}: model outputs {list(names)}, expected {list(LABELS)}")

    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "tokenizer").mkdir(parents=True)
    export = export_onnx(model, tokenizer, out / "model.onnx", int(s["opset"]))
    # The given texts (validation comments, when training builds the package) plus the
    # neutral filler, whose longest line reaches every parity length.
    check = parity(model, tokenizer, out / "model.onnx", [*(texts or ()), *PARITY_TEXTS],
                   [int(x) for x in s["parity_lengths"]], [int(x) for x in s["parity_batch_sizes"]])
    check["limit"] = float(s["max_abs_logit_diff"])
    check["passed"] = check["max_abs_logit_diff"] <= check["limit"]
    if not check["passed"]:
        raise ValueError(f"ONNX parity failed: max |logit diff| {check['max_abs_logit_diff']:.2e}"
                         f" above {check['limit']:.0e}")
    tokenizer.save_pretrained(out / "tokenizer")
    shutil.copy(model_dir / "thresholds.json", out / "thresholds.json")
    if (model_dir / "thresholds_report.json").exists():
        shutil.copy(model_dir / "thresholds_report.json", out / "thresholds_report.json")

    files = {p.relative_to(out).as_posix(): sha256_file(p)
             for p in sorted(out.rglob("*")) if p.is_file()}
    meta = {
        **{key: run_meta.get(key) for key in SECTION_2_5},
        "max_length": run_meta["max_length"], "labels": list(LABELS),
        "model_id": run_meta.get("model_id"), "soft_label_arm": run_meta.get("soft_label_arm"),
        "training_data_sha256": run_meta.get("training_data_sha256"),
        "data_source": run_meta.get("data_source"), "seed": run_meta.get("seed"),
        "code_commit": run_meta.get("code_commit"),
        "mlflow_run_id": run_meta.get("mlflow_run_id"),
        "thresholds": thresholds, "onnx": {**export, "parity": check},
        "files": files, "source_run": str(Path(run_dir).name),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    logger.info("package %s: %d files, ONNX parity %.2e", out, len(files),
                check["max_abs_logit_diff"])
    return meta


def verify_package(folder: str | Path) -> list[str]:
    """Problems with a package folder (missing files, changed hashes); empty means valid."""
    folder = Path(folder)
    problems = [f"missing {name}" for name in ("model.onnx", "tokenizer", "thresholds.json",
                                                "meta.json") if not (folder / name).exists()]
    if problems:
        return problems
    meta = json.loads((folder / "meta.json").read_text())
    for key in (*SECTION_2_5, "max_length", "labels", "files"):
        if key not in meta:
            problems.append(f"meta.json has no {key}")
    for name, digest in (meta.get("files") or {}).items():
        path = folder / name
        if not path.exists():
            problems.append(f"{name} listed in meta.json but missing")
        elif sha256_file(path) != digest:
            problems.append(f"{name}: SHA-256 differs from meta.json")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the §2.5 model package from a run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opset", type=int, default=DEFAULTS["opset"])
    args = parser.parse_args()
    meta = build_package(args.run_dir, args.out, {"opset": args.opset})
    print(json.dumps({k: meta[k] for k in ("model_version", "max_length")} |
                     {"parity": meta["onnx"]["parity"]["max_abs_logit_diff"]}, indent=2))


if __name__ == "__main__":
    main()
