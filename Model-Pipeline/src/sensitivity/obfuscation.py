"""Substitution obfuscation at a chosen strength, for the obfuscation sweep.

Uses the data pipeline's substitution table (``Data-Pipeline/scripts/obfuscate.py``: a -> 4,
e -> 3, i -> 1, ...), so the disguise is the pipeline's own. Two differences from the pipeline's
obfuscation set, which rewrites toxic gate comments with one of three tricks at a fixed 0.5:

* the strength is a parameter: each eligible character is replaced with probability ``rate``;
* it runs on validation comments, toxic or not, so AUC stays defined (the gate set is for
  final numbers only).

Every comment gets one uniform draw per character, seeded by its id, so the rates are nested: a
character replaced at 0.25 is also replaced at 0.5, and a higher rate is never easier.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Mapping
from pathlib import Path

TASK_ID = "sensitivity-obfuscation"


def substitution_table(data_pipeline_dir: str | Path) -> dict[str, str]:
    """``SUBSTITUTIONS`` from the data pipeline's ``scripts/obfuscate.py``.

    The folder is appended (not prepended) to ``sys.path``, so it never shadows this pipeline's
    packages.
    """
    root = str(Path(data_pipeline_dir).resolve())
    if not (Path(root) / "scripts" / "obfuscate.py").exists():
        raise FileNotFoundError(f"{root}/scripts/obfuscate.py not found: set paths.data_pipeline")
    if root not in sys.path:
        sys.path.append(root)
    from scripts.obfuscate import SUBSTITUTIONS

    return dict(SUBSTITUTIONS)


def substitute(text: str, rate: float, key: str, seed: int, table: Mapping[str, str]) -> str:
    """``text`` with each character in ``table`` (any case) replaced with probability ``rate``."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")
    rng = random.Random(f"{seed}:{TASK_ID}:{key}")
    return "".join(
        table[ch.lower()] if rng.random() < rate and ch.lower() in table else ch
        for ch in text
    )


def changed_share(original: str, obfuscated: str) -> float:
    """The share of characters that differ (both strings have the same length)."""
    if not original:
        return 0.0
    return sum(a != b for a, b in zip(original, obfuscated, strict=True)) / len(original)
