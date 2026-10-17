"""Turning the teacher's yes/no answer into a probability.

P(yes) is the summed probability of every single-token "yes" variant, P(no) likewise;
``coverage = P(yes) + P(no)`` and ``p = P(yes) / coverage``. A cell with coverage below
``min_coverage`` is retried; if any label of a row still fails, the row is ``failed``
and all six probabilities are NaN.

Two backends produce the same per-prompt ``(p, coverage)`` pairs:
* vLLM, reading the top-``logprobs`` entries of the first generated token (labeling);
* Hugging Face, an exact softmax at the last prompt position (smoke-test cross-check).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

from src.teacher.config import LABELS

YES_VARIANTS = ("yes", "Yes", "YES", " yes", " Yes", " YES")
NO_VARIANTS = ("no", "No", "NO", " no", " No", " NO")

# Takes prompt strings, returns one (p, coverage) per prompt, in order.
ScoreFn = Callable[[Sequence[str]], list[tuple[float, float]]]


def answer_token_ids(tokenizer: Any) -> tuple[frozenset[int], frozenset[int]]:
    """``YES_IDS`` and ``NO_IDS``: the variants that encode to exactly one token."""

    def single_tokens(variants: Sequence[str]) -> frozenset[int]:
        ids = (tokenizer.encode(v, add_special_tokens=False) for v in variants)
        return frozenset(t[0] for t in ids if len(t) == 1)

    yes_ids, no_ids = single_tokens(YES_VARIANTS), single_tokens(NO_VARIANTS)
    if not yes_ids or not no_ids:
        raise ValueError(f"no single-token answer variants: yes={yes_ids}, no={no_ids}")
    if yes_ids & no_ids:
        raise ValueError(f"yes and no token ids overlap: {yes_ids & no_ids}")
    return yes_ids, no_ids


def yes_probability(top_logprobs: Mapping[int, float], yes_ids: Collection[int],
                    no_ids: Collection[int]) -> tuple[float, float]:
    """``(p, coverage)`` from a token-id -> logprob map. ``p`` is NaN when coverage is 0."""
    p_yes = sum(math.exp(lp) for tid, lp in top_logprobs.items() if tid in yes_ids)
    p_no = sum(math.exp(lp) for tid, lp in top_logprobs.items() if tid in no_ids)
    coverage = p_yes + p_no
    return (p_yes / coverage if coverage > 0 else math.nan), coverage


def score_texts(ids: Sequence[str], texts: Sequence[str], builder: Any, score_fn: ScoreFn,
                min_coverage: float = 0.5, max_retries: int = 1):
    """Score cleaned texts with P2 and return ``(frame, info)``.

    ``frame`` has ``id``, ``p_<label>`` (float32, raw), ``cov_<label>`` (float32),
    ``coverage`` (the row's lowest label coverage), and ``status`` (``ok``/``failed``).
    Empty texts are ``failed`` without being scored.
    """
    import numpy as np
    import pandas as pd

    if builder.protocol != "p2":
        # P1 answer-slot parsing is not built: the pilot runs P2 only.
        raise NotImplementedError("score_texts supports the p2 protocol only")
    if len(ids) != len(texts):
        raise ValueError("ids and texts differ in length")

    n, k = len(texts), len(LABELS)
    p = np.full((n, k), np.nan)
    cov = np.zeros((n, k))
    prompts: dict[int, list[str]] = {}
    truncated = 0
    for i, text in enumerate(texts):
        if text.strip():
            prompts[i], was_truncated = builder.build(text)
            truncated += was_truncated

    def run(cells: list[tuple[int, int]]) -> None:
        results = score_fn([prompts[i][j] for i, j in cells])
        if len(results) != len(cells):
            raise RuntimeError(f"score_fn returned {len(results)} results for {len(cells)}")
        for (i, j), (p_ij, cov_ij) in zip(cells, results, strict=True):
            p[i, j], cov[i, j] = p_ij, cov_ij

    cells = [(i, j) for i in prompts for j in range(k)]
    if cells:
        run(cells)
    retried = 0
    for _ in range(max_retries):
        low = [(i, j) for i, j in cells if not cov[i, j] >= min_coverage]
        if not low:
            break
        retried += len(low)
        run(low)

    failed = ~(cov >= min_coverage).all(axis=1)
    p[failed] = np.nan

    frame = pd.DataFrame({"id": pd.Series(list(ids), dtype="string")})
    for j, label in enumerate(LABELS):
        frame[f"p_{label}"] = p[:, j].astype("float32")
    for j, label in enumerate(LABELS):
        frame[f"cov_{label}"] = cov[:, j].astype("float32")
    frame["coverage"] = cov.min(axis=1).astype("float32")
    frame["status"] = pd.Series(np.where(failed, "failed", "ok"), dtype="string")

    info = {
        "rows": n,
        "failed_rows": int(failed.sum()),
        "empty_rows": n - len(prompts),
        "truncated_rows": int(truncated),
        "prompts_scored": len(cells) + retried,
        "retried_cells": retried,
    }
    return frame, info


def vllm_score_fn(llm: Any, yes_ids: Collection[int], no_ids: Collection[int],
                  logprobs: int = 20) -> ScoreFn:
    """Greedy one-token generation; reads the top-``logprobs`` of that token."""
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=logprobs)

    def score(prompts: Sequence[str]) -> list[tuple[float, float]]:
        outputs = llm.generate(list(prompts), params, use_tqdm=False)
        results = []
        for out in outputs:
            first = out.outputs[0].logprobs[0]  # token id -> Logprob, for the first token
            top = {tid: entry.logprob for tid, entry in first.items()}
            results.append(yes_probability(top, yes_ids, no_ids))
        return results

    return score


def hf_score_fn(model: Any, tokenizer: Any, yes_ids: Collection[int],
                no_ids: Collection[int], batch_size: int = 8) -> ScoreFn:
    """Exact probabilities from a forward pass; the reference the vLLM path is checked against.

    Pads on the left so the last position is always the end of the prompt, and passes
    position ids from the attention mask so padding does not shift positions.
    """
    import torch

    yes = torch.tensor(sorted(yes_ids))
    no = torch.tensor(sorted(no_ids))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
        else tokenizer.eos_token_id

    def encode_batch(batch: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [tokenizer.encode(text, add_special_tokens=False) for text in batch]
        width = max(len(e) for e in encoded)
        input_ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(encoded), width), dtype=torch.long)
        for row, e in enumerate(encoded):
            input_ids[row, width - len(e):] = torch.tensor(e)
            mask[row, width - len(e):] = 1
        return input_ids, mask

    @torch.inference_mode()
    def score(prompts: Sequence[str]) -> list[tuple[float, float]]:
        results: list[tuple[float, float]] = []
        for start in range(0, len(prompts), batch_size):
            input_ids, mask = encode_batch(prompts[start:start + batch_size])
            input_ids, mask = input_ids.to(model.device), mask.to(model.device)
            positions = (mask.cumsum(-1) - 1).clamp(min=0)
            logits = model(input_ids=input_ids, attention_mask=mask, position_ids=positions,
                           logits_to_keep=1).logits[:, -1, :].float()
            probs = torch.softmax(logits, dim=-1).cpu()
            p_yes, p_no = probs[:, yes].sum(-1), probs[:, no].sum(-1)
            for y, n in zip(p_yes.tolist(), p_no.tolist(), strict=True):
                coverage = y + n
                results.append((y / coverage if coverage > 0 else math.nan, coverage))
        return results

    return score
