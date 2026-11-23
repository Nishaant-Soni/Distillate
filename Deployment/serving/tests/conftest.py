"""A tiny model package built offline: a WordLevel tokenizer and a small ONNX graph with the
real package's interface (input_ids and attention_mask in, logits (batch, 6) out)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scorer import LABELS, sha256_file

WORDS = ["thanks", "for", "the", "edit", "you", "are", "an", "idiot", "please", "see", "talk",
         "page", "revert", "hello", "world", "user", "fact", "check"]
THRESHOLDS = {"toxicity": 0.55, "severe_toxicity": 0.6, "obscene": 0.5, "threat": 0.7,
              "insult": 0.45, "identity_attack": 0.65}
MAX_LENGTH = 16


def build_tokenizer(folder: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    vocab = {token: i for i, token in enumerate(specials + WORDS)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]")
    fast.save_pretrained(folder)


def build_onnx(path: Path) -> None:
    """logits = mean of the attended token ids * w + b: different texts, different scores."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    w = np.array([[0.25, 0.1, 0.2, 0.05, 0.3, 0.15]], dtype=np.float32)
    b = np.array([-2.5, -2.0, -2.2, -1.5, -3.0, -1.0], dtype=np.float32)
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["ids_f", "mask_f"], ["masked"]),
        helper.make_node("ReduceSum", ["masked", "axes"], ["total"], keepdims=1),
        helper.make_node("ReduceSum", ["mask_f", "axes"], ["count"], keepdims=1),
        helper.make_node("Div", ["total", "count"], ["mean"]),
        helper.make_node("MatMul", ["mean", "w"], ["scaled"]),
        helper.make_node("Add", ["scaled", "b"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info(n, TensorProto.INT64, ["batch", "sequence"])
         for n in ("input_ids", "attention_mask")],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", len(LABELS)])],
        [numpy_helper.from_array(np.array([1], dtype=np.int64), "axes"),
         numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def build_package(folder: Path, model_version: str = "student-test-v1") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    build_tokenizer(folder / "tokenizer")
    build_onnx(folder / "model.onnx")
    (folder / "thresholds.json").write_text(json.dumps(THRESHOLDS, indent=2) + "\n")
    files = {p.relative_to(folder).as_posix(): sha256_file(p)
             for p in sorted(folder.rglob("*")) if p.is_file()}
    meta = {"model_version": model_version, "max_length": MAX_LENGTH, "labels": list(LABELS),
            "thresholds": THRESHOLDS, "files": files}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return folder


@pytest.fixture(scope="session")
def package(tmp_path_factory) -> Path:
    return build_package(tmp_path_factory.mktemp("pkg") / "package")


@pytest.fixture
def config(package) -> dict:
    from app import load_config

    cfg = load_config()
    cfg["package_dir"] = str(package)
    cfg["onnx"] = {"intra_op_threads": 1, "inter_op_threads": 1}
    return cfg


@pytest.fixture
def client(config):
    from fastapi.testclient import TestClient

    from app import create_app

    with TestClient(create_app(config)) as c:
        yield c
