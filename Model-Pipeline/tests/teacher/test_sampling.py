import numpy as np
import pandas as pd
import pytest

from src.teacher.config import LABELS
from src.teacher.sampling import assign_strata, enriched_sample, pick_fewshot

ENRICH = {"threat": 30, "identity_attack": 30, "severe_toxicity": 30}


def synthetic(n=20000, seed=0):
    """Labels with realistic rarity, plus short and long texts."""
    rng = np.random.default_rng(seed)
    rates = {"toxicity": 0.10, "severe_toxicity": 0.01, "obscene": 0.05, "threat": 0.004,
             "insult": 0.05, "identity_attack": 0.009}
    df = pd.DataFrame({"id": [f"{i:08x}" for i in rng.permutation(n)],
                       "text": ["short text" if i % 2 else "long " * 100 for i in range(n)]})
    for label in LABELS:
        df[label] = (rng.random(n) < rates[label]).astype("int8")
    return df


@pytest.fixture(scope="module")
def train():
    return synthetic()


def test_strata_are_disjoint_and_ordered():
    df = pd.DataFrame({"threat": [1, 1, 0, 0], "identity_attack": [1, 0, 1, 0],
                       "severe_toxicity": [1, 0, 1, 1]})
    assert assign_strata(df, ENRICH).tolist() == ["threat", "threat", "identity_attack",
                                                  "severe_toxicity"]


def test_fixed_seed_gives_fixed_sample(train):
    a = enriched_sample(train, 500, ENRICH, seed=42)
    b = enriched_sample(train.sample(frac=1, random_state=1), 500, ENRICH, seed=42)  # reordered
    c = enriched_sample(train, 500, ENRICH, seed=43)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["id"]) != set(c["id"])


def test_sample_shape(train):
    s = enriched_sample(train, 500, ENRICH, seed=42)
    assert len(s) == 500 and s["id"].is_unique
    assert (s["stratum"] != "rest").sum() >= sum(ENRICH.values())
    assert ((s["inclusion_prob"] > 0) & (s["inclusion_prob"] <= 1)).all()
    assert np.allclose(s["weight"], 1 / s["inclusion_prob"])


def test_inclusion_probabilities_sum_to_sample_size(train):
    # Sum of inclusion probabilities over the population equals the expected sample size.
    strata = assign_strata(train, ENRICH)
    s = enriched_sample(train, 500, ENRICH, seed=42)
    prob_by_stratum = s.groupby("stratum")["inclusion_prob"].first()
    expected = sum(prob_by_stratum[h] * n for h, n in strata.value_counts().items())
    assert expected == pytest.approx(500)


def test_weights_reproduce_prevalence(train):
    # Averaged over many seeds, weighted prevalence matches the population (unbiasedness);
    # unweighted prevalence of the enriched labels is far too high.
    weighted, raw = {lab: [] for lab in LABELS}, {lab: [] for lab in LABELS}
    for seed in range(200):
        s = enriched_sample(train, 500, ENRICH, seed=seed)
        rows = train.set_index("id").loc[s["id"]]
        for label in LABELS:
            weighted[label].append(np.average(rows[label], weights=s["weight"]))
            raw[label].append(rows[label].mean())
    for label in LABELS:
        truth = train[label].mean()
        assert np.mean(weighted[label]) == pytest.approx(truth, rel=0.05), label
    assert np.mean(raw["threat"]) > 5 * train["threat"].mean()


def test_too_small_stratum_rejected(train):
    with pytest.raises(ValueError, match="threat"):
        enriched_sample(train, 15_000, {"threat": 10_000}, seed=42)


def word_counts(texts):
    return [len(t.split()) for t in texts]


def test_fewshot(train):
    pilot = enriched_sample(train, 500, ENRICH, seed=42)
    picked = pick_fewshot(train, set(pilot["id"]), 2, 60, word_counts, seed=42)
    assert list(picked) == list(LABELS)
    ids = [i for group in picked.values() for i in group]
    assert len(ids) == 12 and len(set(ids)) == 12
    assert not set(ids) & set(pilot["id"])
    rows = train.set_index("id").loc[ids]
    assert (rows["text"].map(lambda t: len(t.split())) <= 60).all()   # the 100-word texts are out
    for label, group in picked.items():
        assert (train.set_index("id").loc[group, label] == 1).all()
    assert picked == pick_fewshot(train, set(pilot["id"]), 2, 60, word_counts, seed=42)
