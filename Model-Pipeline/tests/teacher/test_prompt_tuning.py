import numpy as np
import pandas as pd
import pytest

from src.teacher.config import LABELS, PIPELINE_ROOT, load_config
from src.teacher.prompt_tuning import acceptance, label_auc, pilot_halves, select


def pilot(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"{i:06x}" for i in range(n)]
    truth = pd.DataFrame({"id": ids})
    for label in LABELS:
        truth[label] = (rng.random(n) < 0.2).astype("int8")
    pilot_ids = pd.DataFrame({"id": ids, "stratum": rng.choice(["rest", "threat"], n),
                              "weight": rng.uniform(1, 20, n)})
    return truth, pilot_ids, rng


def run(truth, rng, noise):
    """A fake run whose obscene logit has the given noise around the truth."""
    pred = pd.DataFrame({"id": truth["id"], "status": "ok"})
    for label in LABELS:
        pred[f"logit_{label}"] = 4 * truth[label] + rng.normal(0, noise, len(truth))
    return pred


def test_halves_are_balanced_and_deterministic():
    truth, pilot_ids, _ = pilot()
    halves = pilot_halves(truth, pilot_ids, "obscene", seed=42)
    assert halves.equals(pilot_halves(truth, pilot_ids.sample(frac=1, random_state=1),
                                      "obscene", seed=42))
    joined = halves.merge(truth, on="id").merge(pilot_ids, on="id")
    counts = joined.groupby(["stratum", "obscene", "half"]).size().unstack()
    assert ((counts["A"] - counts["B"]).abs() <= 1).all().all()
    assert set(halves["half"]) == {"A", "B"} and len(halves) == len(truth)


def test_committed_halves_match_the_pilot():
    halves = pd.read_parquet(PIPELINE_ROOT / "artifacts" / "prompt_v2" / "halves.parquet")
    pilot_ids = pd.read_parquet(PIPELINE_ROOT / "artifacts" / "pilot" / "pilot_ids.parquet")
    assert set(halves["id"]) == set(pilot_ids["id"])
    assert abs((halves["half"] == "A").sum() - (halves["half"] == "B").sum()) <= 10


def test_select_uses_half_a_and_requires_beating_v1():
    truth, pilot_ids, rng = pilot()
    halves = pilot_halves(truth, pilot_ids, "obscene", seed=42)
    v1 = run(truth, rng, noise=4.0)
    runs = {"p2-zs-v2a": run(truth, rng, 3.0), "p2-zs-v2b": run(truth, rng, 1.0),
            "p2-zs-v2c": run(truth, rng, 6.0)}
    chosen = select(runs, v1, truth, pilot_ids, halves, "obscene")
    assert chosen["chosen"] == "p2-zs-v2b" and chosen["rows"] == (halves["half"] == "A").sum()
    worse = {"p2-zs-v2a": run(truth, rng, 9.0)}
    assert select(worse, v1, truth, pilot_ids, halves, "obscene")["chosen"] is None


def test_label_auc_skips_failed_rows():
    truth, pilot_ids, rng = pilot(400)
    pred = run(truth, rng, 1.0)
    pred.loc[:9, "status"] = "failed"
    pred.loc[:9, "logit_obscene"] = np.nan
    assert 0.5 < label_auc(pred, truth, pilot_ids, "obscene") <= 1.0


def checks(half_b_lo=0.01, civil_lo=0.02, drops=0.0, stack=0.003):
    diff = dict.fromkeys(LABELS, drops) | {"obscene": 0.05}
    return {"label": "obscene", "half_b_auc_diff": diff,
            "half_b_target": {"mean_auc_ci95": [half_b_lo, 0.1]},
            "civil_target": {"mean_auc_ci95": [civil_lo, 0.2]},
            "civil_stack": {"mean_auc_diff": stack}}


@pytest.mark.parametrize("changes, failed", [
    ({}, None),
    ({"half_b_lo": -0.001}, "a_half_b_target_improves"),
    ({"civil_lo": -0.01}, "b_civil_target_improves"),
    ({"drops": -0.006}, "c_no_other_label_drops"),
    ({"stack": -0.0001}, "d_civil_stack_not_worse"),
])
def test_acceptance_rules(changes, failed):
    verdict = acceptance(checks(**changes), max_other_drop=0.005)
    assert [k for k, ok in verdict.items() if not ok] == ([failed] if failed else [])


def test_config_candidates_exist():
    from src.teacher.prompts import P2_QUESTIONS

    t = load_config()["prompt_tuning"]
    assert set(t["candidates"]) <= set(P2_QUESTIONS)
    assert all(t["label"] in P2_QUESTIONS[c] for c in t["candidates"])
