"""Package checks, scoring, decisions, and the smoke test's reference outputs."""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pytest

from scorer import (
    LABELS,
    REFERENCE_TEXTS,
    Scorer,
    check_package,
    decide,
    main,
    reference_outputs,
)

PACKAGE_PY = Path(__file__).resolve().parents[3] / "Model-Pipeline/src/student/package.py"


def test_a_valid_package_has_no_problems(package):
    assert check_package(package) == []


def test_check_catches_a_changed_file_and_bad_thresholds(package, tmp_path):
    copy = tmp_path / "pkg"
    shutil.copytree(package, copy)
    (copy / "model.onnx").write_bytes((copy / "model.onnx").read_bytes() + b"\0")
    assert check_package(copy) == ["model.onnx: SHA-256 differs from meta.json"]

    shutil.rmtree(copy)
    shutil.copytree(package, copy)
    (copy / "thresholds.json").write_text(json.dumps({"toxicity": 0.5}))
    assert any("thresholds.json" in p for p in check_package(copy))

    shutil.rmtree(copy / "tokenizer")
    assert check_package(copy) == ["missing tokenizer"]


def test_scorer_matches_the_pipeline_batch_scoring(package):
    """The pipeline's predict_onnx_logits scores in batches with padding; one text at a time
    must give the same logits (the smoke test compares the two)."""
    import numpy as np

    scorer = Scorer(package)
    texts = list(REFERENCE_TEXTS)
    batch = scorer.tokenizer(texts, truncation=True, max_length=scorer.max_length,
                             padding=True, return_tensors="np")
    feed = {k: batch[k].astype("int64") for k in ("input_ids", "attention_mask")}
    batched = scorer.session.run(["logits"], feed)[0]
    single = np.array([scorer.logits(t) for t in texts])
    assert np.abs(batched - single).max() < 1e-5


def test_decide():
    thresholds = {label: 0.5 for label in LABELS}
    low = {label: 0.1 for label in LABELS}
    assert decide(low, thresholds, 0.1) == {"flagged": {label: False for label in LABELS},
                                            "flag": False, "low_confidence": False}
    one = {**low, "threat": 0.5}                             # at the threshold: flagged
    out = decide(one, thresholds, 0.1)
    assert out["flagged"]["threat"] and out["flag"] and out["low_confidence"]
    near = {**low, "insult": 0.45}
    assert decide(near, thresholds, 0.1)["low_confidence"]
    assert not decide(near, thresholds, 0.1)["flag"]


def test_scorer_refuses_a_package_with_wrong_labels(package, tmp_path):
    copy = tmp_path / "pkg"
    shutil.copytree(package, copy)
    meta = json.loads((copy / "meta.json").read_text())
    meta["labels"] = list(reversed(LABELS))
    (copy / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="labels"):
        Scorer(copy)


def test_reference_outputs_format_and_cli(package, tmp_path):
    ref = reference_outputs(package)
    assert ref["model_version"] == "student-test-v1" and ref["max_length"] == 16
    assert [item["text"] for item in ref["items"]] == list(REFERENCE_TEXTS)
    for item in ref["items"]:
        assert list(item["scores"]) == list(LABELS)

    out = tmp_path / "reference_outputs.json"
    import sys
    argv = sys.argv
    sys.argv = ["scorer.py", "--package", str(package), "--out", str(out)]
    try:
        main()
    finally:
        sys.argv = argv
    assert json.loads(out.read_text()) == json.loads(json.dumps(ref))


@pytest.mark.skipif(not PACKAGE_PY.exists(), reason="Model-Pipeline not in this checkout")
def test_reference_texts_equal_the_package_parity_texts():
    tree = ast.parse(PACKAGE_PY.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "PARITY_TEXTS" for t in node.targets):
            # literal_eval can't do the "..." * 80; the expression is string literals only.
            value = eval(compile(ast.Expression(node.value), str(PACKAGE_PY), "eval"),
                         {"__builtins__": {}})
            assert value == REFERENCE_TEXTS
            return
    raise AssertionError("PARITY_TEXTS not found in package.py")
