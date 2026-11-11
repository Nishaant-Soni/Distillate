"""Student models, chosen by ``model.name``: a Hugging Face encoder with six sigmoid outputs.

``tiny`` is a one-layer DistilBERT with a word-level vocabulary built from the synthetic
data's words, made offline from a config, so tests and CI never download weights.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.teacher.config import LABELS

SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")


def _label_maps() -> dict[str, Any]:
    return {"num_labels": len(LABELS), "problem_type": "multi_label_classification",
            "id2label": dict(enumerate(LABELS)), "label2id": {k: i for i, k in enumerate(LABELS)}}


def tiny_tokenizer():
    from transformers import BertTokenizerFast

    from src.student.synthetic import VOCAB

    # from_pretrained on a folder works on transformers 4 and 5 alike (their constructors differ).
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "vocab.txt").write_text("\n".join((*SPECIAL_TOKENS, *VOCAB)) + "\n")
        return BertTokenizerFast.from_pretrained(tmp)


def build(cfg: Mapping[str, Any]):
    """``(model, tokenizer, model_id)`` for ``cfg["model"]["name"]``."""
    name = cfg["model"]["name"]
    if name == "tiny":
        from transformers import DistilBertConfig, DistilBertForSequenceClassification

        tokenizer = tiny_tokenizer()
        config = DistilBertConfig(vocab_size=len(tokenizer), dim=32, n_layers=1, n_heads=2,
                                  hidden_dim=64, max_position_embeddings=512,
                                  pad_token_id=tokenizer.pad_token_id, **_label_maps())
        return DistilBertForSequenceClassification(config), tokenizer, "tiny-distilbert"

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_id = cfg["models"][name]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, **_label_maps())
    return model, tokenizer, model_id


def load_trained(model_dir: str | Path):
    """``(model, tokenizer)`` saved by ``train.py`` with ``save_pretrained``."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(model_dir)
    return (AutoModelForSequenceClassification.from_pretrained(model_dir),
            AutoTokenizer.from_pretrained(model_dir))
