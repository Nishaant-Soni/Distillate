"""The Hugging Face scorer on a tiny random Qwen3 model (CPU, no download).

Skipped when torch is not installed. The key property: with left padding and position ids
from the attention mask, a prompt scored inside a padded batch gives the same probabilities
as when scored alone, so batching cannot distort the vLLM comparison.
"""

import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.teacher.scoring import hf_score_fn  # noqa: E402

from .conftest import FakeTokenizer  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=200, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
    )
    return transformers.Qwen3ForCausalLM(config).eval()


@pytest.fixture(scope="module")
def setup(tiny_model):
    tok = FakeTokenizer()
    yes_ids = {tok.encode(" yes")[0], tok.encode("Yes")[0]}
    no_ids = {tok.encode(" no")[0], tok.encode("No")[0]}
    return tok, yes_ids, no_ids


def test_padding_does_not_change_probabilities(tiny_model, setup):
    tok, yes_ids, no_ids = setup
    short = "is this toxic"
    long = "a much longer prompt with many more words so the short one gets padded a lot"
    alone = hf_score_fn(tiny_model, tok, yes_ids, no_ids, batch_size=1)([short])[0]
    batched = hf_score_fn(tiny_model, tok, yes_ids, no_ids, batch_size=2)([long, short])[1]
    assert batched[0] == pytest.approx(alone[0], abs=1e-5)
    assert batched[1] == pytest.approx(alone[1], abs=1e-5)
    assert batched[2] == pytest.approx(alone[2], abs=1e-4)


def test_matches_manual_softmax(tiny_model, setup):
    tok, yes_ids, no_ids = setup
    prompt = "score this comment please"
    p, coverage, logit, _ = hf_score_fn(tiny_model, tok, yes_ids, no_ids)([prompt])[0]
    ids = torch.tensor([tok.encode(prompt)])
    with torch.no_grad():
        probs = torch.softmax(tiny_model(input_ids=ids).logits[0, -1].float(), dim=-1)
    p_yes = float(probs[sorted(yes_ids)].sum())
    p_no = float(probs[sorted(no_ids)].sum())
    assert coverage == pytest.approx(p_yes + p_no, rel=1e-5)
    assert p == pytest.approx(p_yes / (p_yes + p_no), rel=1e-5)
    assert logit == pytest.approx(math.log(p_yes / p_no), abs=1e-4)
    assert 0 < coverage < 1 and not math.isnan(p)


def test_one_result_per_prompt_in_order(tiny_model, setup):
    tok, yes_ids, no_ids = setup
    prompts = [f"prompt number {i} " + "word " * i for i in range(5)]
    score = hf_score_fn(tiny_model, tok, yes_ids, no_ids, batch_size=2)
    batched = score(prompts)
    singles = [score([p])[0] for p in prompts]
    assert len(batched) == 5
    for got, want in zip(batched, singles, strict=True):
        assert got[0] == pytest.approx(want[0], abs=1e-5)
