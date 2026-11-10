"""The §2.5 model package from a trained tiny model: files, meta.json, ONNX, verification."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.student.package import INPUTS, SECTION_2_5, build_package, verify_package
from src.student.predict import predict_logits, predict_onnx_logits
from src.student.train import run
from src.teacher.config import LABELS
from tests.student.conftest import smoke_cfg

onnxruntime = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

FAST = ("mlflow.enabled=false", "train.max_steps=8", "train.epochs=1")


@pytest.fixture(scope="module")
def packaged(tmp_path_factory):
    from src.student.synthetic import build

    root = tmp_path_factory.mktemp("package")
    build(root, n=300)
    summary = run(smoke_cfg(root, *FAST, "package.enabled=true"))
    return summary


def test_training_writes_thresholds_and_the_package(packaged):
    run_dir = packaged["run_dir"]
    pkg = packaged["package"]["dir"]
    for name in ("model.onnx", "tokenizer/tokenizer_config.json", "thresholds.json", "meta.json"):
        assert (Path(pkg) / name).exists(), name
    run_thresholds = json.loads((Path(run_dir) / "model" / "thresholds.json").read_text())
    assert sorted(run_thresholds) == sorted(LABELS) and packaged["thresholds"] == run_thresholds
    assert packaged["package"]["onnx_parity"] < 1e-3


def test_meta_has_the_contract_fields_and_what_scorers_need(packaged):
    meta = json.loads((Path(packaged["package"]["dir"]) / "meta.json").read_text())
    for key in (*SECTION_2_5, "max_length", "labels", "files", "thresholds"):
        assert key in meta, key
    assert meta["labels"] == list(LABELS) and meta["max_length"] == 32
    assert meta["model_version"] == packaged["model_version"]
    assert meta["onnx"]["inputs"] == {k: ["batch", "sequence"] for k in INPUTS}
    assert {c["max_length"] for c in meta["onnx"]["parity"]["cases"]} == {16, 64, 128, 256}
    for case in meta["onnx"]["parity"]["cases"]:    # each case reached its full length
        assert case["sequence"] == case["max_length"], case


def test_onnx_has_dynamic_axes_and_matches_pytorch(packaged):
    import torch
    from transformers import AutoTokenizer

    from src.student.model import load_trained

    pkg = Path(packaged["package"]["dir"])
    session = onnxruntime.InferenceSession(str(pkg / "model.onnx"),
                                           providers=["CPUExecutionProvider"])
    assert [(i.name, i.shape) for i in session.get_inputs()] == \
        [("input_ids", ["batch", "sequence"]), ("attention_mask", ["batch", "sequence"])]
    assert [o.name for o in session.get_outputs()] == ["logits"]

    texts = ["the page marktoxicity edit", "see", "thanks " * 90]   # the last is truncated
    tokenizer = AutoTokenizer.from_pretrained(pkg / "tokenizer")
    model, torch_tokenizer = load_trained(Path(packaged["run_dir"]) / "model")
    for max_length in (8, 32, 128):
        onnx_out = predict_onnx_logits(session, tokenizer, texts, max_length, batch_size=2)
        torch_out = predict_logits(model, torch_tokenizer, texts, max_length, 2,
                                   torch.device("cpu"))
        assert np.abs(onnx_out - torch_out).max() < 1e-4


def test_verify_catches_a_changed_file(packaged, tmp_path):
    assert verify_package(packaged["package"]["dir"]) == []
    copy = tmp_path / "pkg"
    shutil.copytree(packaged["package"]["dir"], copy)
    (copy / "thresholds.json").write_text(json.dumps(dict.fromkeys(LABELS, 0.5)))
    assert verify_package(copy) == ["thresholds.json: SHA-256 differs from meta.json"]
    (copy / "model.onnx").unlink()
    assert verify_package(copy) == ["missing model.onnx"]
    assert Path(packaged["package"]["dir"]).exists()


def test_a_run_without_thresholds_cannot_be_packaged(packaged, tmp_path):
    run_copy = tmp_path / "run"
    shutil.copytree(Path(packaged["run_dir"]) / "model", run_copy / "model")
    (run_copy / "model" / "thresholds.json").unlink()
    with pytest.raises(FileNotFoundError, match="thresholds.json"):
        build_package(run_copy, tmp_path / "out")
