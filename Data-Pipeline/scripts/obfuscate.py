"""Build the obfuscation set: toxic gate comments disguised the way evaders write.

Input (written by scripts/split.py):
    data/processed/wikipedia/gate.parquet

Output:
    data/processed/obfuscation.parquet

Takes up to ``splits.obfuscation_size`` toxic gate comments and rewrites each
with one of three tricks, chosen per comment:

    substitution  idiot -> 1d10t     (look-alike digits and symbols)
    spacing       idiot -> i d i o t (letters split apart)
    homoglyph     idiot looks unchanged, but some letters are Cyrillic look-alikes

Labels are unchanged: the comment is still toxic, just harder to read. Ids become
``<original_id>_obf``. The derived columns (char_len, has_identity_term,
has_obfuscation) are recomputed on the new text. Every random choice is seeded
per comment id, so the output does not depend on row order.

    python -m scripts.obfuscate --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.utils import atomic_path, get_logger, load_config, timed, write_summary

logger = get_logger(__name__)
TASK_ID = "obfuscate"
SOURCE_LABEL = "toxicity"
CHANGE_PROB = 0.5  # share of eligible characters or words each trick rewrites

SUBSTITUTIONS = {"a": "4", "e": "3", "i": "1", "l": "1", "o": "0", "s": "5", "t": "7"}
HOMOGLYPHS = {  # Latin -> Cyrillic characters that render almost identically
    "a": "\u0430", "c": "\u0441", "e": "\u0435", "i": "\u0456", "o": "\u043e",
    "p": "\u0440", "x": "\u0445", "y": "\u0443",
}


def _swap_chars(text: str, table: dict[str, str], rng: random.Random) -> str:
    return "".join(
        table[ch.lower()] if ch.lower() in table and rng.random() < CHANGE_PROB else ch
        for ch in text
    )


def substitute(text: str, rng: random.Random) -> str:
    """Replace some letters with look-alike digits: idiot -> 1d10t."""
    return _swap_chars(text, SUBSTITUTIONS, rng)


def homoglyph(text: str, rng: random.Random) -> str:
    """Replace some Latin letters with identical-looking Cyrillic ones."""
    return _swap_chars(text, HOMOGLYPHS, rng)


def space_out(text: str, rng: random.Random) -> str:
    """Split the letters of some words (3+ letters) with spaces: idiot -> i d i o t."""
    return " ".join(
        " ".join(word) if len(word) >= 3 and word.isalpha() and rng.random() < CHANGE_PROB
        else word
        for word in text.split(" ")
    )


TECHNIQUES: dict[str, Callable[[str, random.Random], str]] = {
    "substitution": substitute,
    "spacing": space_out,
    "homoglyph": homoglyph,
}


def obfuscate_text(text: str, comment_id: str, seed: int) -> tuple[str, str]:
    """Return (new_text, technique). new_text equals text if no trick could apply."""
    rng = random.Random(f"{seed}:{TASK_ID}:{comment_id}")
    names = list(TECHNIQUES)
    first = rng.randrange(len(names))
    # Try the chosen trick first; if it changes nothing (a short comment, say),
    # fall through to the others in a fixed order.
    for name in names[first:] + names[:first]:
        for _ in range(3):  # a few seeded retries, since each trick is probabilistic
            new = TECHNIQUES[name](text, rng)
            if new != text:
                return new, name
    return text, "none"


def load_identity_terms(path: Path) -> re.Pattern[str] | None:
    """Word-boundary pattern for the lexicon; None if the lexicon is empty."""
    terms = [line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not terms:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b")


def nonalpha_share(text: str) -> float:
    """Share of non-whitespace characters that are not letters."""
    chars = [ch for ch in text if not ch.isspace()]
    return sum(not ch.isalpha() for ch in chars) / len(chars) if chars else 0.0


def recompute_derived(df: pd.DataFrame, identity: re.Pattern[str] | None,
                      nonalpha_threshold: float) -> pd.DataFrame:
    """Recompute the contract's derived columns from the (new) text."""
    # TODO: switch to the shared helper once preprocess.py merges, so both stages
    # compute these columns with exactly the same rules.
    df = df.copy()
    df["char_len"] = df["text"].str.len().astype("int32")
    df["has_identity_term"] = (
        df["text"].str.lower().str.contains(identity) if identity is not None
        else pd.Series(False, index=df.index)
    ).astype("bool")
    df["has_obfuscation"] = (df["text"].map(nonalpha_share) > nonalpha_threshold).astype("bool")
    return df


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = cfg["seed"]
    target = cfg["splits"]["obfuscation_size"]
    processed = Path(cfg["paths"]["processed_dir"])
    gate_path = processed / "wikipedia" / "gate.parquet"
    summary: dict[str, Any] = {"task": TASK_ID, "mode": "real", "dropped": {}}

    with timed(summary):
        if not gate_path.exists():
            raise FileNotFoundError(
                f"{gate_path.name} not found in {gate_path.parent.as_posix()}; run split first"
            )
        gate = pd.read_parquet(gate_path)
        missing = [c for c in ("id", "text", SOURCE_LABEL) if c not in gate.columns]
        if missing:
            raise ValueError(f"{gate_path.name} is missing required columns: {missing}")

        toxic = gate[gate[SOURCE_LABEL] == 1].sort_values("id", kind="stable")
        toxic = toxic.reset_index(drop=True)
        summary["rows_in"] = {"gate": len(gate), "gate_toxic": len(toxic)}
        if len(toxic) < target:
            logger.warning("Only %d toxic gate comments; obfuscation set will be smaller than %d",
                           len(toxic), target)

        # Seeded shuffle, then take comments in that order until the target is
        # met. Comments no trick can change are skipped and counted.
        order = np.random.default_rng(seed).permutation(len(toxic))
        rows, techniques, unchanged = [], Counter(), 0
        for i in order:
            if len(rows) == target:
                break
            row = toxic.iloc[i]
            new_text, technique = obfuscate_text(row["text"], row["id"], seed)
            if technique == "none":
                unchanged += 1
                continue
            rows.append({**row.to_dict(), "id": f"{row['id']}_obf", "text": new_text})
            techniques[technique] += 1

        out = pd.DataFrame(rows, columns=toxic.columns).astype(toxic.dtypes.to_dict())
        identity = load_identity_terms(Path(cfg["paths"]["identity_terms"]))
        out = recompute_derived(out, identity, cfg["preprocess"]["obfuscation_nonalpha_share"])
        out = out.sort_values("id", kind="stable").reset_index(drop=True)

        with atomic_path(processed / "obfuscation.parquet") as tmp:
            out.to_parquet(tmp, index=False)

        if unchanged:
            summary["dropped"]["no_trick_could_change_text"] = unchanged
        summary["rows_out"] = len(out)
        summary["techniques"] = dict(sorted(techniques.items()))
        summary["has_obfuscation_share"] = (
            round(float(out["has_obfuscation"].mean()), 4) if len(out) else 0.0
        )

    logger.info("obfuscate: %d rows (%s)", len(out), summary["techniques"])
    write_summary(TASK_ID, summary, cfg)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config)), indent=2, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
