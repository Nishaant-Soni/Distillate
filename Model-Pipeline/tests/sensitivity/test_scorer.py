import json
import shutil

import numpy as np
import pytest

from src.sensitivity.scorer import OnnxScorer, load_scorer
from src.teacher.config import LABELS
from tests.sensitivity.conftest import MODEL_VERSION, WORDS, save_student, tiny_model

TEXTS = ["you are an idiot", "thanks for the edit", None,
         "please talk on the page " * 6, "kill", "this article is stupid and i hate it"]


def test_save_pretrained_folder_gives_logits_in_input_order(student_dir):
    scorer = load_scorer(student_dir, batch_size=2)
    logits = scorer.logits(TEXTS)
    assert logits.shape == (len(TEXTS), len(LABELS)) and logits.dtype == np.float32
    assert scorer.model_version == MODEL_VERSION and scorer.max_length == 32
    # Batched (sorted by length, padded) equals one at a time: order and padding are handled.
    one_by_one = np.vstack([scorer.logits([text]) for text in TEXTS])
    np.testing.assert_allclose(logits, one_by_one, atol=1e-5)
    # None scores as the empty string.
    np.testing.assert_allclose(logits[2], scorer.logits([""])[0], atol=1e-6)


def test_max_length_truncates(student_dir):
    scorer = load_scorer(student_dir)
    long_text = " ".join(WORDS)
    # max_length 8 keeps [CLS], the first six words and [SEP].
    np.testing.assert_allclose(scorer.logits([long_text], max_length=8),
                               scorer.logits([" ".join(WORDS[:6])]), atol=1e-5)
    assert not np.allclose(scorer.logits([long_text], max_length=8), scorer.logits([long_text]))


def test_probabilities_are_the_sigmoid(student_dir):
    scorer = load_scorer(student_dir)
    p = scorer.probabilities(TEXTS)
    assert p.dtype == np.float64 and ((p > 0) & (p < 1)).all()
    np.testing.assert_allclose(p, 1 / (1 + np.exp(-scorer.logits(TEXTS).astype("float64"))))


def test_no_texts(student_dir):
    assert load_scorer(student_dir).logits([]).shape == (0, len(LABELS))


def test_refuses_a_model_with_other_outputs(tmp_path):
    folder = save_student(tmp_path, labels=tuple(reversed(LABELS)))
    with pytest.raises(ValueError, match="model outputs"):
        load_scorer(folder)


def test_meta_checks(tmp_path):
    folder = save_student(tmp_path, meta={"model_version": "x", "labels": list(LABELS)})
    with pytest.raises(ValueError, match="max_length"):
        load_scorer(folder)
    assert load_scorer(folder, max_length=16).max_length == 16
    (folder / "meta.json").write_text(json.dumps({"version": "pkg-v1", "max_length": 16,
                                                  "labels": ["toxicity"]}))
    with pytest.raises(ValueError, match="labels"):
        load_scorer(folder)
    (folder / "meta.json").unlink()
    with pytest.raises(FileNotFoundError, match="meta.json"):
        load_scorer(folder)


def test_folder_without_a_model(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"model_version": "x", "max_length": 8}))
    with pytest.raises(FileNotFoundError, match="neither model.onnx"):
        load_scorer(tmp_path)


class FakeSession:
    """Stands in for onnxruntime.InferenceSession: logits from the token ids, feeds recorded."""

    def __init__(self, inputs=("input_ids", "attention_mask")):
        self._inputs = [type("Arg", (), {"name": name}) for name in inputs]
        self.feeds = []

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return [type("Arg", (), {"name": "logits"})]

    def run(self, names, feed):
        self.feeds.append(feed)
        ids = feed["input_ids"] * feed.get("attention_mask", 1)
        return [np.repeat(ids.sum(axis=1, keepdims=True), len(LABELS), axis=1).astype("float64")]


def test_onnx_scorer_feeds_the_model_inputs_and_keeps_order(tmp_path):
    _, tokenizer = tiny_model(tmp_path)
    session = FakeSession()
    scorer = OnnxScorer(session, tokenizer, 32, "pkg-v1", batch_size=2)
    logits = scorer.logits(TEXTS)
    assert logits.dtype == np.float32
    expected = [sum(tokenizer(text or "", truncation=True, max_length=32)["input_ids"])
                for text in TEXTS]
    np.testing.assert_array_equal(logits[:, 0], expected)
    assert all(set(feed) == {"input_ids", "attention_mask"} for feed in session.feeds)
    assert all(feed["input_ids"].dtype == np.int64 for feed in session.feeds)


def test_onnx_scorer_refuses_inputs_it_cannot_give(tmp_path):
    _, tokenizer = tiny_model(tmp_path)
    with pytest.raises(ValueError, match="token_type_ids"):
        OnnxScorer(FakeSession(("input_ids", "token_type_ids")), tokenizer, 32, "pkg-v1")


def test_onnx_package_matches_the_pytorch_model(tmp_path):
    """A real §5 package (model.onnx, tokenizer/, meta.json) scores like the model it came from."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import torch

    student = save_student(tmp_path)
    model, tokenizer = tiny_model(tmp_path)
    package = tmp_path / "package"
    tokenizer.save_pretrained(package / "tokenizer")
    shutil.copy(student / "meta.json", package / "meta.json")
    ids = torch.ones(2, 5, dtype=torch.long)
    axes = {0: "batch", 1: "sequence"}
    torch.onnx.export(model, (ids, torch.ones_like(ids)), package / "model.onnx",
                      input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes={"input_ids": axes, "attention_mask": axes,
                                    "logits": {0: "batch"}}, dynamo=False)
    onnx_scorer = load_scorer(package)
    assert isinstance(onnx_scorer, OnnxScorer)
    np.testing.assert_allclose(onnx_scorer.logits(TEXTS), load_scorer(student).logits(TEXTS),
                               atol=1e-4)
