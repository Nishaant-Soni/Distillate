"""Workarounds for the Kaggle GPU image.

Installing vLLM replaces Kaggle's torch with a CUDA 13 build but leaves Kaggle's torchaudio,
built for CUDA 12.8. transformers imports torchaudio whenever it can find it, and that import
fails on the CUDA mismatch, which breaks ``import vllm``. The teacher never uses audio, so
the fix is to make torchaudio invisible rather than reinstall it.
"""

from __future__ import annotations

import sys


def hide_torchaudio() -> None:
    """Make ``import torchaudio`` fail and ``find_spec("torchaudio")`` return None.

    transformers caches whether torchaudio is available the first time it checks, so this
    must run before anything imports transformers or vLLM.
    """
    if sys.modules.get("torchaudio") is not None:
        raise RuntimeError("torchaudio is already imported; call hide_torchaudio() earlier")
    if "transformers" in sys.modules:
        raise RuntimeError("transformers is already imported; call hide_torchaudio() earlier")
    # The import system treats a None entry as "not importable".
    sys.modules["torchaudio"] = None  # type: ignore[assignment]
