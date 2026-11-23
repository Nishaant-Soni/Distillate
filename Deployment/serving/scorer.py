"""Scores comments with a model package (work plan v5 §2.5): the inference behind /predict.

A package folder holds ``model.onnx`` (inputs ``input_ids`` and ``attention_mask``, int64,
dynamic batch and sequence; output ``logits`` (batch, 6) in LABELS order), ``tokenizer/``,
``thresholds.json`` (one float per label) and ``meta.json`` (``model_version``,
``max_length``, ``labels``, and the SHA-256 of every other file).

Scoring matches the model pipeline's ``predict_onnx_logits`` exactly (same tokenizer call,
same truncation at the package's ``max_length``, sigmoid of the logits), so the service's
scores equal the package's own outputs. One text is scored at a time, with no padding.

``reference_outputs(package_dir, texts)`` computes expected scores from the package itself;
Deployment/deploy/deploy_tools.py uses it to write ``reference_outputs.json`` into packages that
don't have one, for the smoke test. The CLI is the image build's package check:

    python scorer.py --package <dir> [--out reference_outputs.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

LABELS = ("toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack")
INPUTS = ("input_ids", "attention_mask")
OUTPUT = "logits"
REQUIRED = ("model.onnx", "tokenizer", "thresholds.json", "meta.json")

# Neutral filler, no comment text: kept equal to PARITY_TEXTS in
# Model-Pipeline/src/student/package.py (a test checks), the inputs of its own ONNX parity check.
REFERENCE_TEXTS = ("thanks for the edit to the article", "please see the talk page section",
                   "the source link in the image note was changed back", "revert",
                   "user page fact check " * 80)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_package(folder: str | Path) -> list[str]:
    """Problems with a package folder (missing files, changed hashes, wrong labels or
    thresholds); empty means it can be served."""
    folder = Path(folder)
    problems = [f"missing {name}" for name in REQUIRED if not (folder / name).exists()]
    if problems:
        return problems
    meta = json.loads((folder / "meta.json").read_text())
    for key in ("model_version", "max_length", "labels", "files"):
        if key not in meta:
            problems.append(f"meta.json has no {key}")
    if tuple(meta.get("labels") or ()) != LABELS:
        problems.append(f"meta.json labels {meta.get('labels')}, expected {list(LABELS)}")
    for name, digest in (meta.get("files") or {}).items():
        path = folder / name
        if not path.exists():
            problems.append(f"{name} listed in meta.json but missing")
        elif sha256_file(path) != digest:
            problems.append(f"{name}: SHA-256 differs from meta.json")
    if "model.onnx" not in (meta.get("files") or {}):
        problems.append("meta.json files has no model.onnx hash")
    thresholds = json.loads((folder / "thresholds.json").read_text())
    if sorted(thresholds) != sorted(LABELS):
        problems.append(f"thresholds.json labels {sorted(thresholds)}, expected the six labels")
    elif not all(isinstance(v, (int, float)) and 0 < v < 1 for v in thresholds.values()):
        problems.append("thresholds.json values must be numbers between 0 and 1")
    return problems


def decide(scores: Mapping[str, float], thresholds: Mapping[str, float],
           margin: float) -> dict[str, Any]:
    """§2.4 decisions: ``flagged`` per label (score at or above its threshold), ``flag``
    (any label flagged), ``low_confidence`` (any score within ``margin`` of its threshold)."""
    flagged = {label: bool(scores[label] >= thresholds[label]) for label in LABELS}
    return {
        "flagged": flagged,
        "flag": any(flagged.values()),
        "low_confidence": any(abs(scores[label] - thresholds[label]) < margin
                              for label in LABELS),
    }


class Scorer:
    """A loaded package: ``scores(text)`` returns the six label probabilities."""

    def __init__(self, package_dir: str | Path, intra_op_threads: int = 0,
                 inter_op_threads: int = 0):
        import onnxruntime
        from transformers import AutoTokenizer

        folder = Path(package_dir)
        missing = [name for name in REQUIRED if not (folder / name).exists()]
        if missing:
            raise FileNotFoundError(f"{folder} is not a model package: missing {missing}")
        meta = json.loads((folder / "meta.json").read_text())
        if tuple(meta.get("labels") or ()) != LABELS:
            raise ValueError(f"package labels {meta.get('labels')}, expected {list(LABELS)}")
        thresholds = json.loads((folder / "thresholds.json").read_text())
        if sorted(thresholds) != sorted(LABELS):
            raise ValueError(f"thresholds.json has {sorted(thresholds)}, expected the six labels")

        self.meta = meta
        self.model_version = str(meta["model_version"])
        self.max_length = int(meta["max_length"])
        self.thresholds = {label: float(thresholds[label]) for label in LABELS}

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = int(intra_op_threads)
        options.inter_op_num_threads = int(inter_op_threads)
        self.session = onnxruntime.InferenceSession(str(folder / "model.onnx"), options,
                                                    providers=["CPUExecutionProvider"])
        inputs = tuple(i.name for i in self.session.get_inputs())
        outputs = [o.name for o in self.session.get_outputs()]
        if inputs != INPUTS or outputs != [OUTPUT]:
            raise ValueError(f"model.onnx has inputs {list(inputs)}, outputs {outputs}; "
                             f"expected {list(INPUTS)} and ['{OUTPUT}']")
        self.tokenizer = AutoTokenizer.from_pretrained(folder / "tokenizer")

    def logits(self, text: str) -> list[float]:
        """The six logits for one text, in LABELS order."""
        batch = self.tokenizer([text], truncation=True, max_length=self.max_length,
                               padding=True, return_tensors="np")
        feed = {name: batch[name].astype("int64") for name in INPUTS}
        out = self.session.run([OUTPUT], feed)[0]
        return [float(x) for x in out[0]]

    def scores(self, text: str) -> dict[str, float]:
        """The six probabilities for one text: the sigmoid of the logits."""
        return {label: 1.0 / (1.0 + math.exp(-z))
                for label, z in zip(LABELS, self.logits(text), strict=True)}


def reference_outputs(package_dir: str | Path,
                      texts: Sequence[str] = REFERENCE_TEXTS) -> dict[str, Any]:
    """Expected scores for ``texts`` from the package itself, in the smoke test's format:
    ``{"model_version", "max_length", "items": [{"text", "scores": {label: p}}]}``."""
    scorer = Scorer(package_dir)
    return {"model_version": scorer.model_version, "max_length": scorer.max_length,
            "items": [{"text": text, "scores": scorer.scores(text)} for text in texts]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Expected scores for the reference texts.")
    parser.add_argument("--package", required=True, help="model package folder")
    parser.add_argument("--out", help="write JSON here (default: print it)")
    args = parser.parse_args()
    problems = check_package(args.package)
    if problems:
        raise SystemExit("package problems: " + "; ".join(problems))
    body = json.dumps(reference_outputs(args.package), indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(body)
    else:
        print(body, end="")


if __name__ == "__main__":
    main()
