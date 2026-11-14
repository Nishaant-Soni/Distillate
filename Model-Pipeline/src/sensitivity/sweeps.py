"""Input length and obfuscation strength: how the student's validation metrics move.

Both sweeps rescore the same validation rows under a changed input and report, per label and
as a mean over labels with positives: AUC and PR-AUC (ranked on the logits), ECE and Brier (on
the probabilities), and recall and flag rate at the decision threshold. The metric definitions
are the teacher lane's (``src.teacher.metrics.label_metrics``), unweighted: validation is a
plain random split.

* ``truncation_sweep``: the rows truncated at each ``max_length``; plus the same split by each
  row's untruncated token length, so the cost of truncation shows where it happens.
* ``obfuscation_sweep``: the rows obfuscated at each substitution rate (``obfuscation.py``);
  plus how many toxic comments caught when clean are missed once obfuscated.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.sensitivity.obfuscation import changed_share, substitute
from src.sensitivity.scorer import Scorer
from src.teacher.config import LABELS
from src.teacher.metrics import label_metrics

MEAN_COLUMNS = ["auc", "pr_auc", "ece", "brier", "recall", "flag_rate"]


def load_thresholds(model_dir: str | Path | None, default: float) -> dict[str, float]:
    """Per-label thresholds from the model's ``thresholds.json``, else ``default`` for all."""
    path = Path(model_dir) / "thresholds.json" if model_dir is not None else None
    if path is None or not path.exists():
        return dict.fromkeys(LABELS, float(default))
    thresholds = json.loads(path.read_text())
    missing = [label for label in LABELS if label not in thresholds]
    if missing:
        raise ValueError(f"{path}: no threshold for {missing}")
    return {label: float(thresholds[label]) for label in LABELS}


def check_frame(frame: pd.DataFrame) -> None:
    """Validation rows: ``id``, ``text`` and the six labels, ids unique."""
    missing = [c for c in ("id", "text", *LABELS) if c not in frame]
    if missing:
        raise ValueError(f"validation frame has no {missing}")
    if frame["id"].duplicated().any():
        raise ValueError("validation frame has duplicate ids")


def label_table(truth: pd.DataFrame, logits: np.ndarray, thresholds: Mapping[str, float],
                bins: int = 10) -> pd.DataFrame:
    """One row per label plus ``mean`` (over labels with both classes present)."""
    logits = np.asarray(logits, dtype="float64")
    p = 1 / (1 + np.exp(-logits))
    w = np.ones(len(truth))
    rows = {}
    for k, label in enumerate(LABELS):
        y = truth[label].to_numpy(dtype="float64")
        flagged = p[:, k] >= thresholds[label]
        n_pos = int(y.sum())
        rows[label] = {
            "n": len(y), "n_pos": n_pos,
            **label_metrics(y, p[:, k], w, bins, score=logits[:, k]),
            "recall": float(flagged[y == 1].mean()) if n_pos else float("nan"),
            "flag_rate": float(flagged.mean()) if len(y) else float("nan"),
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    both = table["auc"].notna()
    table.loc["mean"] = {"n": len(truth), "n_pos": int(table["n_pos"].sum()),
                         **table.loc[both, MEAN_COLUMNS].mean().to_dict()}
    table.index.name = "label"
    return table


def token_lengths(scorer: Scorer, texts: Sequence[str]) -> np.ndarray:
    """Each text's token count with special tokens and no truncation."""
    encoded = scorer.tokenizer([str(t) for t in texts], truncation=False, verbose=False)
    return np.array([len(ids) for ids in encoded["input_ids"]])


def band_names(edges: Sequence[int]) -> list[str]:
    """``<=64``, ``65-128``, ..., ``>256`` for edges 64, 128, 256."""
    edges = sorted(edges)
    names = [f"<={edges[0]}"]
    names += [f"{lo + 1}-{hi}" for lo, hi in zip(edges, edges[1:], strict=False)]
    return names + [f">{edges[-1]}"]


def assign_bands(lengths: np.ndarray, edges: Sequence[int]) -> np.ndarray:
    """The band name of each length (a band includes its upper edge)."""
    names = band_names(edges)
    return np.array([names[i] for i in np.searchsorted(sorted(edges), lengths, side="left")])


def truncation_sweep(scorer: Scorer, frame: pd.DataFrame, max_lengths: Sequence[int],
                     band_edges: Sequence[int], thresholds: Mapping[str, float],
                     bins: int = 10) -> dict[str, pd.DataFrame]:
    """``overall``: per max_length and label. ``bands``: the same per token-length band."""
    check_frame(frame)
    texts = frame["text"].fillna("").astype(str).tolist()
    lengths = token_lengths(scorer, texts)
    bands = assign_bands(lengths, band_edges)
    overall, by_band = [], []
    for max_length in max_lengths:
        logits = scorer.logits(texts, max_length=max_length)
        table = label_table(frame, logits, thresholds, bins)
        table.insert(0, "truncated_share", float((lengths > max_length).mean()))
        overall.append(table.reset_index().assign(max_length=max_length))
        for band in band_names(band_edges):
            rows = bands == band
            if not rows.any():
                continue
            part = label_table(frame[rows], logits[rows], thresholds, bins)
            part.insert(0, "truncated_share", float((lengths[rows] > max_length).mean()))
            by_band.append(part.reset_index().assign(max_length=max_length, band=band))
    first = ["max_length"]
    return {"overall": _ordered(pd.concat(overall, ignore_index=True), first),
            "bands": _ordered(pd.concat(by_band, ignore_index=True), ["band", *first])}


def obfuscation_sweep(scorer: Scorer, frame: pd.DataFrame, rates: Sequence[float],
                      table: Mapping[str, str], seed: int, thresholds: Mapping[str, float],
                      bins: int = 10, clean: np.ndarray | None = None) -> pd.DataFrame:
    """Per rate and label; ``missed``: share of positives flagged when clean, not at ``rate``.

    ``clean``: the rows' probabilities at the model's own length, if already computed.
    """
    check_frame(frame)
    texts = frame["text"].fillna("").astype(str).tolist()
    ids = frame["id"].astype(str).tolist()
    if clean is None:
        clean = scorer.probabilities(texts)
    caught = {label: (clean[:, k] >= thresholds[label]) & (frame[label].to_numpy() == 1)
              for k, label in enumerate(LABELS)}
    out = []
    for rate in rates:
        changed = [substitute(t, rate, i, seed, table) for t, i in zip(texts, ids, strict=True)]
        logits = scorer.logits(changed)
        result = label_table(frame, logits, thresholds, bins)
        p = 1 / (1 + np.exp(-logits.astype("float64")))
        missed = {}
        for k, label in enumerate(LABELS):
            n = int(caught[label].sum())
            still = p[caught[label], k] >= thresholds[label]
            missed[label] = float(1 - still.mean()) if n else float("nan")
        result["missed"] = pd.Series(missed)
        result.loc["mean", "missed"] = pd.Series(missed).mean()   # skips labels never caught
        result.insert(0, "chars_changed", float(np.mean([changed_share(a, b) for a, b in
                                                         zip(texts, changed, strict=True)])))
        out.append(result.reset_index().assign(rate=float(rate)))
    return _ordered(pd.concat(out, ignore_index=True), ["rate"])


def _ordered(frame: pd.DataFrame, first: list[str]) -> pd.DataFrame:
    return frame[first + [c for c in frame.columns if c not in first]]
