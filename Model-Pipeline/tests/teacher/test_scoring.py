import math

import numpy as np
import pytest

from src.teacher.config import LABELS
from src.teacher.prompts import PromptBuilder
from src.teacher.scoring import Score, answer_token_ids, score_texts, yes_probability

from .conftest import FakeTokenizer

YES = {1, 2, 3}
NO = {4, 5}


def test_answer_token_ids_keeps_single_tokens_only():
    tok = FakeTokenizer(multi_token=("YES", " NO"))
    yes_ids, no_ids = answer_token_ids(tok)
    assert len(yes_ids) == 5 and len(no_ids) == 5
    assert tok.vocab["YES#1"] not in yes_ids
    assert not yes_ids & no_ids


def test_answer_token_ids_fails_without_variants():
    tok = FakeTokenizer(multi_token=("no", "No", "NO", " no", " No", " NO"))
    with pytest.raises(ValueError):
        answer_token_ids(tok)


def test_variants_are_summed_and_renormalised():
    top = {1: math.log(0.3), 2: math.log(0.1), 4: math.log(0.2), 5: math.log(0.2),
           99: math.log(0.15)}
    p, coverage, logit, censored = yes_probability(top, YES, NO)
    assert coverage == pytest.approx(0.8)
    assert p == pytest.approx(0.4 / 0.8)
    assert logit == pytest.approx(math.log(0.4 / 0.4)) and not censored


def test_logit_does_not_saturate():
    # P(no) = 1e-12: p rounds to exactly 1.0 in float32, the logit keeps the information.
    top = {1: math.log(1 - 1e-12), 4: math.log(1e-12)}
    weaker = {1: math.log(1 - 1e-9), 4: math.log(1e-9)}
    a, b = yes_probability(top, YES, NO), yes_probability(weaker, YES, NO)
    assert np.float32(a.p) == np.float32(b.p) == 1.0
    assert a.logit == pytest.approx(math.log((1 - 1e-12) / 1e-12))
    assert a.logit > b.logit > 20


def test_absent_side_gives_finite_bound():
    # No "no" token in the list: its mass is at most len(NO) * the lowest listed probability.
    top = {1: math.log(0.99), 99: math.log(1e-6), 98: math.log(1e-9)}
    p, coverage, logit, censored = yes_probability(top, YES, NO)
    assert p == 1.0 and censored
    assert logit == pytest.approx(math.log(0.99) - math.log(len(NO) * 1e-9))


def test_no_answer_tokens_gives_nan():
    p, coverage, logit, censored = yes_probability({99: math.log(0.9)}, YES, NO)
    assert coverage == 0 and math.isnan(p) and math.isnan(logit)


def scripted(answers):
    """A fake score_fn: ``answers(label, call_number)`` gives each prompt's (p, coverage)."""
    calls = []

    def score(prompts):
        calls.append(len(prompts))
        out = []
        for prompt in prompts:
            label = next(lab for lab in LABELS if f"definition of {lab}?" in prompt)
            p, coverage = answers(label, len(calls))
            out.append(Score(p, coverage, math.log(p / (1 - p))))
        return out

    return score, calls


@pytest.fixture
def builder(tokenizer):
    return PromptBuilder(tokenizer, "p2", "zs")


def test_ok_rows(builder):
    score, calls = scripted(lambda label, call: (0.25, 0.95))
    frame, info = score_texts(["a", "b"], ["first text", "second text"], builder, score)
    assert list(frame["status"]) == ["ok", "ok"]
    assert np.allclose(frame[[f"p_{lab}" for lab in LABELS]].to_numpy(), 0.25)
    assert np.allclose(frame[[f"logit_{lab}" for lab in LABELS]].to_numpy(), math.log(1 / 3))
    assert calls == [12]
    assert info["failed_rows"] == 0 and info["retried_cells"] == 0
    assert info["censored_cells"] == 0


def test_low_coverage_retried_once_then_ok(builder):
    # threat has low coverage on the first call only.
    score, calls = scripted(
        lambda label, call: (0.9, 0.3) if (label == "threat" and call == 1) else (0.1, 0.9))
    frame, info = score_texts(["a"], ["some text"], builder, score)
    assert frame.loc[0, "status"] == "ok"
    assert frame.loc[0, "p_threat"] == pytest.approx(0.1)
    assert calls == [6, 1] and info["retried_cells"] == 1


def test_low_coverage_after_retry_fails_whole_row(builder):
    score, calls = scripted(
        lambda label, call: (0.9, 0.3) if label == "threat" else (0.1, 0.9))
    frame, info = score_texts(["a"], ["some text"], builder, score, max_retries=1)
    assert frame.loc[0, "status"] == "failed"
    assert frame[[f"p_{lab}" for lab in LABELS]].isna().all(axis=None)
    assert frame[[f"logit_{lab}" for lab in LABELS]].isna().all(axis=None)
    assert calls == [6, 1] and info["failed_rows"] == 1


def test_empty_text_fails_without_scoring(builder):
    score, calls = scripted(lambda label, call: (0.5, 0.99))
    frame, info = score_texts(["a", "b"], ["", "real text"], builder, score)
    assert list(frame["status"]) == ["failed", "ok"]
    assert calls == [6] and info["empty_rows"] == 1


def test_output_types(builder):
    score, _ = scripted(lambda label, call: (0.5, 0.99))
    frame, _ = score_texts(["a"], ["text"], builder, score)
    assert str(frame["id"].dtype) == "string" and str(frame["status"].dtype) == "string"
    for label in LABELS:
        assert frame[f"p_{label}"].dtype == np.float32
        assert frame[f"logit_{label}"].dtype == np.float32


def test_p1_not_supported(tokenizer):
    score, _ = scripted(lambda label, call: (0.5, 0.99))
    with pytest.raises(NotImplementedError):
        score_texts(["a"], ["text"], PromptBuilder(tokenizer, "p1", "zs"), score)


def test_vllm_output_parsing(monkeypatch):
    """vllm_score_fn reads the first generated token's top logprobs, in prompt order."""
    import sys
    import types
    from types import SimpleNamespace as NS

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    from src.teacher.scoring import vllm_score_fn

    tables = {"a": {1: math.log(0.6), 4: math.log(0.2), 99: math.log(0.1)},
              "b": {5: math.log(0.9), 99: math.log(1e-3), 98: math.log(1e-7)}}
    seen = {}

    def request_output(prompt):
        first_token = {tid: NS(logprob=lp) for tid, lp in tables[prompt].items()}
        return NS(outputs=[NS(logprobs=[first_token])])

    class FakeLLM:
        def generate(self, prompts, params, use_tqdm):
            seen["params"] = params
            return [request_output(p) for p in prompts]

    results = vllm_score_fn(FakeLLM(), YES, NO, logprobs=20)(["a", "b"])
    assert seen["params"] == {"temperature": 0.0, "max_tokens": 1, "logprobs": 20}
    assert results[0][:3] == pytest.approx((0.75, 0.8, math.log(0.6 / 0.2)))
    assert not results[0].censored
    assert results[1][:2] == pytest.approx((0.0, 0.9)) and results[1].censored
    assert results[1].logit == pytest.approx(math.log(len(YES) * 1e-7) - math.log(0.9))
