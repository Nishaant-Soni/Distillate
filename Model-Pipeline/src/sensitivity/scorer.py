"""The student as a black box for the sensitivity analysis: texts in, six scores out.

Every analysis (truncation, obfuscation, SHAP) needs only ``logits(texts, max_length)``, so it
works the same for a Hugging Face ``save_pretrained`` folder (what the student's training writes)
and for the CONTRACTS.md §5 model package (``model.onnx`` plus ``tokenizer/``). Either folder
holds a ``meta.json`` with the training ``max_length`` and the ``model_version``.

Scores come back as logits: AUC and PR-AUC rank on them, because float32 probabilities tie at
exactly 1.0 for confident predictions. ``probabilities`` applies the sigmoid, for thresholds,
recall, and ECE.

    scorer = load_scorer("outputs/student/<run>/model")
    scorer.logits(texts)                  # the training max_length
    scorer.logits(texts, max_length=64)   # truncated harder
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.teacher.config import LABELS, get_logger

logger = get_logger(__name__)

TOKEN_INPUTS = ("input_ids", "attention_mask")
ONNX_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")


class Scorer:
    """Batching shared by both model formats; subclasses implement ``_forward``.

    Texts are scored in batches of similar length (less padding), and the logits come back in
    input order.
    """

    def __init__(self, tokenizer, max_length: int, model_version: str, batch_size: int = 64):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.model_version = model_version
        self.batch_size = int(batch_size)

    def logits(self, texts: Sequence[str | None], max_length: int | None = None) -> np.ndarray:
        """``(rows, 6)`` float32 logits, in input order; ``None`` texts score as empty."""
        texts = ["" if text is None else str(text) for text in texts]
        length = int(max_length or self.max_length)
        out = np.empty((len(texts), len(LABELS)), dtype="float32")
        order = np.argsort([len(text) for text in texts], kind="stable")
        for start in range(0, len(texts), self.batch_size):
            rows = order[start:start + self.batch_size]
            batch = self.tokenizer([texts[i] for i in rows], truncation=True, max_length=length,
                                   padding=True, return_tensors="np")
            out[rows] = self._forward({k: np.asarray(batch[k], dtype="int64")
                                       for k in TOKEN_INPUTS if k in batch})
        return out

    def probabilities(self, texts: Sequence[str | None],
                      max_length: int | None = None) -> np.ndarray:
        """``(rows, 6)`` float64 sigmoid of the logits."""
        return 1 / (1 + np.exp(-self.logits(texts, max_length).astype("float64")))

    def _forward(self, batch: Mapping[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class HFScorer(Scorer):
    """A ``save_pretrained`` sequence-classification model, run with PyTorch."""

    def __init__(self, model, tokenizer, max_length: int, model_version: str,
                 batch_size: int = 64, device: str = "auto", fp16: bool = False):
        import torch

        super().__init__(tokenizer, max_length, model_version, batch_size)
        self.device = torch.device(resolve_device(device))
        self.model = model.to(self.device).eval()
        self.fp16 = fp16 and self.device.type == "cuda"

    def _forward(self, batch: Mapping[str, np.ndarray]) -> np.ndarray:
        import torch

        tensors = {k: torch.from_numpy(v).to(self.device) for k, v in batch.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16,
                                                    enabled=self.fp16):
            logits = self.model(**tensors).logits
        return logits.float().cpu().numpy()


class OnnxScorer(Scorer):
    """A §5 package's ``model.onnx``, run with ONNX Runtime."""

    def __init__(self, session, tokenizer, max_length: int, model_version: str,
                 batch_size: int = 64):
        super().__init__(tokenizer, max_length, model_version, batch_size)
        self.session = session
        self.inputs = [i.name for i in session.get_inputs()]
        missing = sorted(set(self.inputs) - set(TOKEN_INPUTS))
        if missing:
            raise ValueError(f"model.onnx wants inputs {missing} that the tokenizer can't give; "
                             f"supported: {list(TOKEN_INPUTS)}")
        outputs = [o.name for o in session.get_outputs()]
        self.output = "logits" if "logits" in outputs else outputs[0]

    def _forward(self, batch: Mapping[str, np.ndarray]) -> np.ndarray:
        (logits,) = self.session.run([self.output], {k: batch[k] for k in self.inputs})
        return np.asarray(logits, dtype="float32")


def resolve_device(device: str = "auto") -> str:
    """``auto`` picks CUDA, then Apple MPS, then CPU."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_meta(model_dir: Path, max_length: int | None) -> dict[str, Any]:
    """``meta.json``, checked: the label order, and a max_length from it or the caller."""
    path = model_dir / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: the model folder must hold its meta.json")
    meta = json.loads(path.read_text())
    if "labels" in meta and tuple(meta["labels"]) != LABELS:
        raise ValueError(f"{path}: labels {meta['labels']}, expected {list(LABELS)}")
    meta["max_length"] = max_length or meta.get("max_length")
    if not meta["max_length"]:
        raise ValueError(f"{path} has no max_length: pass max_length to load_scorer")
    meta["model_version"] = meta.get("model_version") or meta.get("version")
    if not meta["model_version"]:
        raise ValueError(f"{path} has no model_version (or version)")
    return meta


def check_label_order(id2label: Mapping[Any, str], where: Path) -> None:
    """The model's outputs must be the six labels in ``LABELS`` order."""
    names = tuple(id2label[k] for k in sorted(id2label, key=int))
    if names != LABELS:
        raise ValueError(f"{where}: model outputs {list(names)}, expected {list(LABELS)}")


def load_scorer(model_dir: str | Path, batch_size: int = 64, device: str = "auto",
                fp16: bool = False, max_length: int | None = None) -> Scorer:
    """The scorer for a model folder: ``model.onnx`` if present, else ``save_pretrained``."""
    from transformers import AutoTokenizer

    model_dir = Path(model_dir)
    meta = read_meta(model_dir, max_length)
    if (model_dir / "model.onnx").exists():
        import onnxruntime

        tokenizer_dir = model_dir / "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir if tokenizer_dir.is_dir() else model_dir)
        # CUDA if present, else CPU. Not every available provider: CoreML on a Mac fails on
        # some transformer graphs that CPU runs fine.
        providers = [p for p in ONNX_PROVIDERS if p in onnxruntime.get_available_providers()]
        session = onnxruntime.InferenceSession(str(model_dir / "model.onnx"),
                                               providers=providers)
        scorer: Scorer = OnnxScorer(session, tokenizer, meta["max_length"],
                                    meta["model_version"], batch_size)
    elif (model_dir / "config.json").exists():
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        check_label_order(model.config.id2label, model_dir)
        scorer = HFScorer(model, AutoTokenizer.from_pretrained(model_dir), meta["max_length"],
                          meta["model_version"], batch_size, device, fp16)
    else:
        raise FileNotFoundError(f"{model_dir}: neither model.onnx nor a save_pretrained model")
    logger.info("loaded %s (%s, max_length %d)", meta["model_version"],
                type(scorer).__name__, scorer.max_length)
    return scorer
