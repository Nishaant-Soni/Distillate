"""The words the pages show (demo_app.display)."""

import pytest

from demo_app import display
from demo_app.api import LABELS

# v1's thresholds and the scores the live API gave the "Insult" preset (Nov 24).
THRESHOLDS = {"toxicity": 0.3510, "severe_toxicity": 0.2249, "obscene": 0.2071,
              "threat": 0.2320, "insult": 0.3252, "identity_attack": 0.2902}
INSULT = {"scores": {"toxicity": 0.9924, "severe_toxicity": 0.0050, "obscene": 0.1694,
                     "threat": 0.0001, "insult": 0.9668, "identity_attack": 0.0005},
          "flagged": {"toxicity": True, "severe_toxicity": False, "obscene": False,
                      "threat": False, "insult": True, "identity_attack": False},
          "flag": True, "low_confidence": True}


def test_every_label_has_a_plain_name():
    assert [display.name(label) for label in LABELS] == [
        "Toxicity", "Severe toxicity", "Obscene", "Threat", "Insult", "Identity attack"]
    assert display.name("new_label") == "New label"


def test_the_verdict_names_the_flagged_labels_in_order():
    assert display.verdict(INSULT) == "**Flagged for review:** Toxicity, Insult."
    clean = {**INSULT, "flag": False, "flagged": dict.fromkeys(LABELS, False)}
    assert display.verdict(clean).startswith("**Not flagged:**")


def test_borderline_names_the_close_scores():
    assert display.borderline(INSULT, THRESHOLDS) == [("Obscene", 0.1694, 0.2071)]
    line = display.borderline_line(INSULT, THRESHOLDS)
    assert line.startswith("**Borderline:** Obscene (0.17, threshold 0.21).")
    assert "this label" in line


def test_borderline_uses_the_apis_strict_margin():
    # serving: abs(score - threshold) < margin, so exactly 0.1 away is not borderline
    # (0.2 - 0.1 == 0.1 exactly in floating point; most pairs that look 0.1 apart aren't)
    response = {**INSULT, "scores": {**INSULT["scores"], "obscene": 0.1}}
    assert all(label != "Obscene" for label, _, _ in
               display.borderline(response, {**THRESHOLDS, "obscene": 0.2}))
    assert ("Obscene", 0.1, 0.19) in display.borderline(response, {**THRESHOLDS, "obscene": 0.19})


def test_borderline_without_thresholds_falls_back_to_the_general_line():
    assert display.borderline(INSULT, None) == []
    assert "a score is close to its threshold" in display.borderline_line(INSULT, None)


@pytest.mark.parametrize("at, shown", [
    ("2026-11-24T15:56:43+00:00", "2026-11-24 15:56 UTC"),
    ("2026-11-24T10:56:43-05:00", "2026-11-24 15:56 UTC"),     # converted to UTC
    ("2026-11-24T15:56:43Z", "2026-11-24 15:56 UTC"),
    ("2026-11-24T15:56:43", "2026-11-24 15:56 UTC"),           # naive: taken as UTC
    ("yesterday", "yesterday"),
    (None, "an unknown time"),
])
def test_times_are_shown_in_utc(at, shown):
    assert display.when(at) == shown


def test_the_serving_line_never_shows_the_actor():
    line = {"actor": "someone-on-the-team", "at": "2026-11-24T15:56:43+00:00",
            "status": "deployed", "version": "1", "run_url": "https://example.test/run/1",
            "smoke": {"n_inputs": 50, "passed": True, "max_abs_diff": 0.0}}
    shown = display.serving_line(line)
    assert shown == ("**v1** live since 2026-11-24 15:56 UTC · smoke test passed on 50 inputs"
                     " · [deploy run](https://example.test/run/1)")
    assert "someone" not in shown


def test_a_rollback_line_says_so():
    line = {"status": "rolled-back", "version": "1", "at": "2026-11-25T09:00:00+00:00"}
    assert display.serving_line(line) == "**v1** live since 2026-11-25 09:00 UTC (rolled back)"


def test_the_retraining_line():
    record = {"run_id": "r1", "reason": "drift", "gate": "passed", "registered": False}
    assert display.retraining_line(record) == \
        "Run `r1` · triggered by drift · gate passed · not registered"
