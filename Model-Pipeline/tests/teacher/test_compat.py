import importlib
import importlib.util
import sys

import pytest

from src.teacher.compat import hide_torchaudio


@pytest.fixture
def clean_modules(monkeypatch):
    # monkeypatch restores both entries afterwards, whatever hide_torchaudio does to them.
    monkeypatch.delitem(sys.modules, "torchaudio", raising=False)
    monkeypatch.delitem(sys.modules, "transformers", raising=False)


@pytest.mark.usefixtures("clean_modules")
def test_hidden_torchaudio_is_not_found_or_importable():
    hide_torchaudio()
    assert importlib.util.find_spec("torchaudio") is None
    with pytest.raises(ImportError):
        importlib.import_module("torchaudio")


@pytest.mark.usefixtures("clean_modules")
def test_too_late_after_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", object())
    with pytest.raises(RuntimeError, match="transformers"):
        hide_torchaudio()
