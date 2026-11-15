"""A tiny offline student, saved the way the student's training saves one: no network, no GPU."""

from __future__ import annotations

import json

import pytest

from src.teacher.config import LABELS

WORDS = ("the", "page", "edit", "article", "talk", "thanks", "please", "idiot", "stupid", "kill",
         "hate", "you", "are", "an", "this", "is")
SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
PIECES = ("##ic",)        # so "idiotic" splits into word pieces, as with a real WordPiece vocab
MODEL_VERSION = "tiny-student-test"


def tiny_model(tmp_path, labels=LABELS):
    """A one-layer DistilBERT and a word-level tokenizer, seeded, as an untrained model."""
    import torch
    from transformers import (
        BertTokenizerFast,
        DistilBertConfig,
        DistilBertForSequenceClassification,
    )

    vocab = tmp_path / "vocab"
    vocab.mkdir(parents=True, exist_ok=True)
    (vocab / "vocab.txt").write_text("\n".join((*SPECIAL_TOKENS, *WORDS, *PIECES)) + "\n")
    tokenizer = BertTokenizerFast.from_pretrained(vocab)
    torch.manual_seed(0)
    config = DistilBertConfig(vocab_size=len(tokenizer), dim=32, n_layers=1, n_heads=2,
                              hidden_dim=64, max_position_embeddings=512,
                              pad_token_id=tokenizer.pad_token_id, num_labels=len(labels),
                              problem_type="multi_label_classification",
                              id2label=dict(enumerate(labels)),
                              label2id={k: i for i, k in enumerate(labels)})
    return DistilBertForSequenceClassification(config).eval(), tokenizer


def save_student(tmp_path, labels=LABELS, meta=None):
    """A ``save_pretrained`` folder with meta.json, like the student's ``model/``."""
    model, tokenizer = tiny_model(tmp_path, labels)
    folder = tmp_path / "student"
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    default = {"model_version": MODEL_VERSION, "max_length": 32, "labels": list(LABELS)}
    (folder / "meta.json").write_text(json.dumps(default if meta is None else meta))
    return folder


@pytest.fixture
def student_dir(tmp_path):
    return save_student(tmp_path)
