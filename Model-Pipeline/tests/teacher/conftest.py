"""Offline stand-ins for the Qwen tokenizer, so the teacher tests need no network or GPU."""

from __future__ import annotations

import pytest


class FakeTokenizer:
    """One token per whitespace-separated word, plus a Qwen-style chat template.

    Words the tokenizer has not seen get new ids, so encode/decode round-trip.
    ``multi_token`` words encode to two ids, to test the single-token filter.
    """

    def __init__(self, multi_token: tuple[str, ...] = ()):
        self.vocab: dict[str, int] = {}
        self.multi_token = set(multi_token)
        self.pad_token_id = 0
        self.eos_token_id = 0

    def _id(self, word: str) -> int:
        return self.vocab.setdefault(word, len(self.vocab) + 1)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text in self.multi_token:
            return [self._id(text + "#1"), self._id(text + "#2")]
        # A leading space is part of the word, as in byte-level BPE.
        if text.startswith(" ") and text.strip() and " " not in text.strip():
            return [self._id(text)]
        return [self._id(word) for word in text.split()]

    def decode(self, ids: list[int]) -> str:
        words = {i: w for w, i in self.vocab.items()}
        return " ".join(words[i] for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()
