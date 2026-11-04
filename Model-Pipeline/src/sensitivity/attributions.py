"""SHAP token attributions on validation comments, reported without their text.

* **Selection** (``select_examples``): per label, seeded samples of three outcomes at the
  threshold: ``caught`` (toxic and flagged, a correct prediction), ``missed`` (toxic, not
  flagged) and ``false_alarm`` (flagged, not toxic). Correct non-toxic comments are left out:
  their attributions say little. A comment chosen for several labels is explained once, since
  SHAP returns all six outputs together.
* **Attributions** (``explain``): SHAP's partition explainer with the text masker, on the
  student's logits, so each label's values sum to its logit minus the base value. Word pieces
  are merged into words and their values summed (SHAP values add).
* **Output, no comment text:**

  * ``token_table``: per label and outcome, the words that push hardest toward the label;
  * ``masked_example``: a few comments per label and outcome with only their top words
    visible, everything else masked (``▒``).

  A word is shown only if it occurs in at least ``min_doc_freq`` validation comments, so a rare
  string (a username, say) never reaches the report, even as a top attribution.
"""

from __future__ import annotations

import string
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.sensitivity.scorer import Scorer
from src.teacher.config import LABELS

GROUPS = ("caught", "missed", "false_alarm")
CORRECT = {"caught": True, "missed": False, "false_alarm": False}
MASK = "▒▒▒"


@dataclass(frozen=True)
class Attribution:
    """One comment's words and their SHAP values, ``(words, 6)``, in ``LABELS`` order."""

    words: tuple[str, ...]
    values: np.ndarray


def normalize(word: str) -> str:
    """Lowercase, without surrounding whitespace and punctuation (kept if that empties it)."""
    word = word.strip().lower()
    return word.strip(string.punctuation) or word


def document_frequency(texts: Iterable[str]) -> Counter:
    """How many comments each normalized word occurs in."""
    counts: Counter = Counter()
    for text in texts:
        counts.update({normalize(w) for w in str(text).split()} - {""})
    return counts


def select_examples(frame: pd.DataFrame, probabilities: np.ndarray,
                    thresholds: Mapping[str, float], per_group: int, seed: int) -> pd.DataFrame:
    """``id``, ``label``, ``group``, ``p``, ``y``: up to ``per_group`` rows per label and group."""
    rng = np.random.default_rng(seed)
    ids = frame["id"].astype(str).to_numpy()
    rows = []
    for k, label in enumerate(LABELS):
        y = frame[label].to_numpy() == 1
        flagged = probabilities[:, k] >= thresholds[label]
        for group, mask in (("caught", y & flagged), ("missed", y & ~flagged),
                            ("false_alarm", ~y & flagged)):
            where = np.flatnonzero(mask)
            chosen = np.sort(rng.choice(where, min(per_group, len(where)), replace=False))
            rows += [{"id": ids[i], "label": label, "group": group,
                      "p": float(probabilities[i, k]), "y": int(y[i])} for i in chosen]
    return pd.DataFrame(rows, columns=["id", "label", "group", "p", "y"])


def merge_words(pieces: Sequence[str], values: np.ndarray) -> Attribution:
    """SHAP's token strings to words: a piece ending in whitespace ends a word.

    Empty pieces (``[CLS]``, ``[SEP]``) are dropped; a word's values are the sum of its pieces'.
    """
    values = np.asarray(values, dtype="float64")
    words, out = [], []
    current, total = "", np.zeros(values.shape[1])
    for piece, value in zip(pieces, values, strict=True):
        if piece == "":
            continue
        current, total = current + piece, total + value
        if piece[-1].isspace():
            words.append(current.strip())
            out.append(total)
            current, total = "", np.zeros(values.shape[1])
    if current.strip():
        words.append(current.strip())
        out.append(total)
    return Attribution(tuple(words), np.array(out).reshape(len(words), values.shape[1]))


def explain(scorer: Scorer, texts: Sequence[str], max_evals: int,
            batch_size: int) -> list[Attribution]:
    """Word attributions for each text, toward each label's logit."""
    import shap

    try:
        explainer = shap.explainers.Partition(lambda x: scorer.logits(list(x)),
                                              shap.maskers.Text(scorer.tokenizer),
                                              output_names=list(LABELS))
    except ValueError as err:
        if "Keras" in str(err):
            raise RuntimeError("transformers was imported with TensorFlow enabled: import "
                               "src.sensitivity (or set USE_TF=0) before transformers") from err
        raise
    explanation = explainer(list(texts), max_evals=max_evals, batch_size=batch_size,
                            silent=True)
    return [merge_words(list(explanation[i].data), explanation[i].values)
            for i in range(len(texts))]


def token_table(selection: pd.DataFrame, attributions: Mapping[str, Attribution],
                doc_freq: Mapping[str, int], min_doc_freq: int, min_occurrences: int,
                top_k: int) -> pd.DataFrame:
    """Per label and group, the ``top_k`` words by mean value toward the label (positive only).

    A word ranks only if it occurs ``min_occurrences`` times in the group's comments and in at
    least ``min_doc_freq`` validation comments.
    """
    rows = []
    for (label, group), part in selection.groupby(["label", "group"], sort=False):
        k = LABELS.index(label)
        values: dict[str, list[float]] = {}
        for comment in part["id"]:
            attribution = attributions[comment]
            for word, value in zip(attribution.words, attribution.values[:, k], strict=True):
                values.setdefault(normalize(word), []).append(float(value))
        ranked = sorted(
            ((w, float(np.mean(v)), len(v)) for w, v in values.items()
             if len(v) >= min_occurrences and doc_freq.get(w, 0) >= min_doc_freq
             and np.mean(v) > 0),
            key=lambda item: item[1], reverse=True)
        rows += [{"label": label, "group": group, "correct": CORRECT[group], "rank": rank,
                  "word": w, "mean_value": mean, "occurrences": n, "comments": len(part)}
                 for rank, (w, mean, n) in enumerate(ranked[:top_k], start=1)]
    return pd.DataFrame(rows, columns=["label", "group", "correct", "rank", "word",
                                       "mean_value", "occurrences", "comments"])


def masked_example(attribution: Attribution, label: str, visible: int,
                   doc_freq: Mapping[str, int], min_doc_freq: int) -> str:
    """The comment with only its ``visible`` top words toward ``label`` shown, with values.

    Every run of hidden words becomes one ``▒▒▒``, so the line doesn't give away the length.
    """
    values = attribution.values[:, LABELS.index(label)]
    allowed = [i for i, w in enumerate(attribution.words)
               if doc_freq.get(normalize(w), 0) >= min_doc_freq and values[i] > 0]
    shown = set(sorted(allowed, key=lambda i: values[i], reverse=True)[:visible])
    parts: list[str] = []
    for i, word in enumerate(attribution.words):
        if i in shown:
            parts.append(f"**{normalize(word)}** ({values[i]:+.2f})")
        elif not parts or parts[-1] != MASK:
            parts.append(MASK)
    return " ".join(parts)


def examples(selection: pd.DataFrame, attributions: Mapping[str, Attribution],
             per_group: int, visible: int, doc_freq: Mapping[str, int],
             min_doc_freq: int) -> pd.DataFrame:
    """The first ``per_group`` selected comments of each label and group, masked."""
    rows = []
    for (label, group), part in selection.groupby(["label", "group"], sort=False):
        for _, row in part.head(per_group).iterrows():
            rows.append({"label": label, "group": group, "correct": CORRECT[group],
                         "p": row["p"], "y": row["y"],
                         "masked": masked_example(attributions[row["id"]], label, visible,
                                                  doc_freq, min_doc_freq)})
    return pd.DataFrame(rows, columns=["label", "group", "correct", "p", "y", "masked"])


def shap_analysis(scorer: Scorer, frame: pd.DataFrame, probabilities: np.ndarray,
                  thresholds: Mapping[str, float], cfg: Mapping[str, Any],
                  seed: int) -> dict[str, pd.DataFrame]:
    """Select, explain once per comment, and build the token tables and masked examples.

    ``selection`` keeps the ids, for the run's own records; the report uses only ``tokens``
    and ``examples``, which hold no comment text.
    """
    selection = select_examples(frame, probabilities, thresholds, int(cfg["per_group"]), seed)
    unique = list(dict.fromkeys(selection["id"]))
    text_by_id = dict(zip(frame["id"].astype(str), frame["text"].fillna("").astype(str),
                          strict=True))
    attributions = dict(zip(unique, explain(scorer, [text_by_id[i] for i in unique],
                                            int(cfg["max_evals"]), int(cfg["batch_size"])),
                            strict=True))
    doc_freq = document_frequency(frame["text"].fillna(""))
    min_doc_freq = int(cfg["min_doc_freq"])
    return {
        "selection": selection,
        "tokens": token_table(selection, attributions, doc_freq, min_doc_freq,
                              int(cfg["min_occurrences"]), int(cfg["top_k"])),
        "examples": examples(selection, attributions, int(cfg["examples_per_group"]),
                             int(cfg["visible_words"]), doc_freq, min_doc_freq),
    }
