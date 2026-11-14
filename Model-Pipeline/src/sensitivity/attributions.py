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
    visible (never more than a third of a comment's words), everything else masked (``▒``).
    No two visible words are adjacent, and the first and last words are always hidden, so a
    line never reproduces a fragment of a comment or shows where it starts or ends.

  A word is shown only if it occurs in at least ``min_doc_freq`` validation comments, so a rare
  string (a username, say) never reaches the report, even as a top attribution. Slurs
  (``MASKED_TERMS``) are shown with their inner letters masked (``n****r``), in the tables and
  the examples alike.
"""

from __future__ import annotations

import hashlib
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
# SHA-256 of the normalized slurs shown with their inner letters masked: kept as hashes so the
# list itself adds no slurs to the repository. Each line's comment is the word as reported. To
# add one: hashlib.sha256(normalize(word).encode()).hexdigest().
MASKED_TERMS = frozenset({
    "120f6e5b4ea32f65bda68452fcfaaef06b0136e1d0e4a6f60bc3771fa0936dd6",   # n****r
    "5b3ae48be122f7ed19b4cc587649f41f9d2565df51cfa332f8e7806f4ebb9032",   # n*****s
    "08a841e996781e9e77d30a4e4420a8f501a280b00624e6d1224bf54aaff73eba",   # n***a
    "341d56384afc0f47b34ca18273e793be555507a49444c30d3d0588688de46cb3",   # n****s
    "8f5083e3e5c7dc8932f2bf58212f963f3a44752618c96297f82623f736c52738",   # f****t
    "1e02eec4f1095143be282056557034d4e8ab915342c1af508223141d472d2347",   # f*****s
    "9915ba2d822280f22c283df4e76584a40e0119fc58f73c5f84d4fdb04d04fa6f",   # f**
    "17bde8b4064612e266eaa1b5b1dfa054d1e6dd263f66b392159006d2c9a8ae8a",   # f**s
    "158869a97379229b7681efae9d7f9c9214134e836d649ba53477c0c111414d59",   # r****d
    "c1cabb6f6e431f9c4dea4c3b3264d6fe3829241d674f3496a2dfff6658f0363e",   # r*****s
    "c3de533e9b7fe63b79f648687a30d2861edd92fe7c3cd1f2c485e0a605367624",   # k**e
    "268651b3ece980102f18871fde07189372961e056f858ca147a28d004f876b03",   # k***s
    "98b52c4b6b7d1f48e7477a5ccc10955dd195d0ac5a38c8281bfeb08762634909",   # s**c
    "044eb98b18769b887d5a0258f675e413058b8e9fc9b9786b418bfc6f03f26c98",   # s***s
    "f9d0d9b18ae9033a5ea36df19bf279b059e887a9ae785db81117bceaecc95933",   # c***k
    "45cd3e1bb472d8e285a160a95f6409ec5ce86102e9caaaf0b63fc11db04ab559",   # c****s
    "16ea09fc78ca83ca502cbcf2377acdf280bf18f61e259153f0868405eedab5ef",   # t****y
    "402f7ebd98864afed4817ff4718d357dbfa95458aa4236d6c5959536956fa4e5",   # t******s
    "886d51e97ad7931d0d2af8439ca6d9e4887e3c2b469ed247cbd68ceb3649ccde",   # d**e
    "92a9bf818e7e00021b6d6959f877f5df11b0f989e486db38f35853ebbd552086",   # d***s
    "eef3bd091670c3447022d619c06ad15de96da72b5a66f28bb8b75d1b1c12a05f",   # w*****k
    "0ce875d620076b533c6e681abaa0dc5d9a941366bb000a02ec5351f341b96c61",   # w******s
    "e7b98c6aa5b944e0b315d350d423f895ac9e44fb84f1534b18c2572370a67b9e",   # c**n
    "daec0c235d3135fc1c60aee787f784c59c4d53b2041ff2bacd0d3bebda7246a8",   # c***s
    "cc02032349c833ac5e97bac094560ed40e09acf34cb1978ab7a9840b9bf15b4d",   # g**k
    "82159dd02870c13ec30e4f146db7c317aca568469d0e7e22b8610a2eaa3d3924",   # g***s
    "333f7618092958c75b8c5af6f1ec77b42803922a0fc6ff1570a8af3a3aab3b4a",   # t*******d
    "da0a6b1b9213d48ff10cc45733cc8a925dbd2afb26320ed488ac2efa9a8ae33a",   # t********s
    "22fc75e65a0e9d34324092a7c6a8dba961853294abca4e5914e60c550f48e0c2",   # r*****d
    "17a110e5332848721f27510e1e2bca710a321663bb8a5632e0e46a820fc9e28a",   # r******s
})


@dataclass(frozen=True)
class Attribution:
    """One comment's words and their SHAP values, ``(words, 6)``, in ``LABELS`` order."""

    words: tuple[str, ...]
    values: np.ndarray


def normalize(word: str) -> str:
    """Lowercase, without surrounding whitespace and punctuation (kept if that empties it)."""
    word = word.strip().lower()
    return word.strip(string.punctuation) or word


def display(word: str) -> str:
    """The normalized word as reported: a slur keeps only its first and last letters."""
    word = normalize(word)
    if hashlib.sha256(word.encode()).hexdigest() not in MASKED_TERMS:
        return word
    if len(word) <= 3:
        return word[0] + "*" * (len(word) - 1)
    return word[0] + "*" * (len(word) - 2) + word[-1]


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
                  "word": display(w), "mean_value": mean, "occurrences": n, "comments": len(part)}
                 for rank, (w, mean, n) in enumerate(ranked[:top_k], start=1)]
    return pd.DataFrame(rows, columns=["label", "group", "correct", "rank", "word",
                                       "mean_value", "occurrences", "comments"])


def masked_example(attribution: Attribution, label: str, visible: int,
                   doc_freq: Mapping[str, int], min_doc_freq: int) -> str:
    """The comment with only its top words toward ``label`` shown, with values.

    At most ``visible`` words, and never more than a third of the comment's words, so a short
    comment can't be read off its top words. Taken by value, skipping the first and last words
    and any word next to one already shown, so every visible word has hidden words on both
    sides: no fragment of the comment, and not where it starts or ends. Every run of hidden
    words becomes one ``▒▒▒``, so the line doesn't give away the length.
    """
    values = attribution.values[:, LABELS.index(label)]
    last = len(attribution.words) - 1
    allowed = [i for i, w in enumerate(attribution.words)
               if 0 < i < last and doc_freq.get(normalize(w), 0) >= min_doc_freq
               and values[i] > 0]
    cap = min(visible, len(attribution.words) // 3)
    shown: set[int] = set()
    for i in sorted(allowed, key=lambda i: values[i], reverse=True):
        if len(shown) == cap:
            break
        if i - 1 not in shown and i + 1 not in shown:
            shown.add(i)
    parts: list[str] = []
    for i, word in enumerate(attribution.words):
        if i in shown:
            parts.append(f"**{display(word)}** ({values[i]:+.2f})")
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
