"""Masking: listed words keep their first and last letters; everything else is unchanged.

The tests use the presets (our own text) and words built from their letters, so this file
adds no offensive words of its own.
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

from demo_app.masking import MASKED_TERMS, is_masked, mask, mask_word

HERE = Path(__file__).resolve().parents[1]
PRESETS = {p["id"]: p["text"] for p in
           json.loads((HERE / "presets.json").read_text(encoding="utf-8"))["presets"]}


def test_a_listed_word_keeps_its_first_and_last_letters():
    assert mask(PRESETS["insult"]) == "You're an i***t if you think that source is reliable."


def test_punctuation_and_case_are_kept():
    assert mask_word("Hell!") == "H**l!"
    assert mask_word("(damn),") == "(d**n),"


def test_the_obfuscation_tricks_are_undone_before_matching():
    assert mask(PRESETS["obfuscated"]) == "y0u are such an 1***t, st0p editing this page."
    cyrillic_o = "о"
    assert is_masked(f"m{cyrillic_o}r{cyrillic_o}n")
    assert is_masked("$tupid")                         # $ for s


def test_every_preset_shows_masked_where_it_should():
    expected = {
        "insult": "i***t", "obscene": "d**n", "identity-attack": "s****d", "obfuscated": "1***t",
    }
    for preset, hidden in expected.items():
        assert hidden in mask(PRESETS[preset])
    assert "h**l" in mask(PRESETS["obscene"])


@pytest.mark.parametrize("preset", ["clean", "clean-strong", "threat", "identity-clean-1",
                                    "identity-clean-2"])
def test_text_without_listed_words_is_unchanged(preset):
    assert mask(PRESETS[preset]) == PRESETS[preset]


def test_only_whole_words_match():
    assert mask("hello shell hellish") == "hello shell hellish"


def test_spacing_is_unchanged():
    text = "  line one\n\tline  two  "
    assert mask(text) == text


def test_the_source_holds_hashes_not_words():
    """No word in masking.py is one of the words it masks."""
    source = (HERE / "demo_app" / "masking.py").read_text(encoding="utf-8")
    words = set(re.findall(r"[a-z]+", source.lower()))
    assert not {w for w in words if hashlib.sha256(w.encode()).hexdigest() in MASKED_TERMS}
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in MASKED_TERMS)
