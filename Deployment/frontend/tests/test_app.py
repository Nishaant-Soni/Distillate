"""Both pages end to end with Streamlit's AppTest, on the mock and with no registry."""

import json
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
SETTINGS = ("DISTILLATE_API_URL", "MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME",
            "MLFLOW_TRACKING_PASSWORD", "DEPLOYMENTS_LOG", "RETRAIN_RECORDS", "GRAFANA_URL",
            "DISTILLATE_SESSION_PER_MINUTE")


@pytest.fixture
def app(monkeypatch):
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    st.cache_resource.clear()
    st.cache_data.clear()
    yield lambda: AppTest.from_file(APP, default_timeout=60)
    st.cache_resource.clear()
    st.cache_data.clear()


def texts(at):
    return [t.value for t in at.text]


def test_own_text_is_scored_on_the_mock(app):
    at = app().run()
    assert not at.exception
    assert any("mock" in c.value for c in at.caption)
    at.text_area[0].input("Thanks for the edit").run()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.dataframe) == 1
    assert "Thanks for the edit" in texts(at)            # this text has nothing to mask
    assert at.success or at.error                         # the flag verdict


def test_a_preset_is_shown_masked_and_never_in_a_text_box(app):
    at = app().run()
    at.selectbox[0].set_value(3).run()                   # "Insult"
    assert not at.text_area
    shown = texts(at)
    assert any("i***t" in t for t in shown) and not any("idiot" in t for t in shown)
    at.button[0].click().run()
    assert not at.exception
    assert not any("idiot" in t for t in texts(at))      # the result is masked too
    table = at.dataframe[0].value.set_index("Label")["Teacher (stack)"]
    assert table["Insult"] == pytest.approx(0.9679)      # presets.json's cached stack score
    assert not any("haven't been computed" in c.value for c in at.caption)
    assert any(c.value.startswith("**Teacher:**") for c in at.caption)   # what the column is


def test_the_verdict_names_the_labels(app):
    at = app().run()
    assert any("student" in m.value and "teacher" in m.value for m in at.markdown)   # intro
    at.selectbox[0].set_value(3).run()
    at.button[0].click().run()
    flagged = at.dataframe[0].value.set_index("Label")["Flagged"]
    named = [label for label, is_flagged in flagged.items() if is_flagged]
    if named:     # the banner lists exactly the flagged labels, by the table's plain names
        verdict = at.error[0].value
        assert verdict == f"**Flagged for review:** {', '.join(named)}."
    else:
        assert at.success[0].value.startswith("**Not flagged:**")


def test_the_page_switch_is_at_the_top(app):
    at = app().run()
    assert not at.sidebar.radio
    assert at.radio[0].options == ["Try it", "Status"]


def test_the_rate_limit_message(app, monkeypatch):
    monkeypatch.setenv("DISTILLATE_SESSION_PER_MINUTE", "1")
    at = app().run()
    at.text_area[0].input("first").run()
    at.button[0].click().run()
    at.text_area[0].input("second").run()
    at.button[0].click().run()
    assert any("Too many requests" in w.value for w in at.warning)


def test_status_without_sources_says_not_yet(app):
    at = app().run()
    at.radio[0].set_value("Status").run()
    assert not at.exception
    assert any("MLFLOW_TRACKING_URI" in i.value for i in at.info)
    page = " ".join(m.value for m in at.markdown)
    assert "DEPLOYMENTS_LOG" in page and "RETRAIN_RECORDS" in page and "GRAFANA_URL" in page


def test_status_summarises_the_deployment_without_the_actor(app, monkeypatch, tmp_path):
    log = tmp_path / "deployments.jsonl"
    log.write_text(json.dumps({
        "action": "deploy", "actor": "someone-on-the-team", "at": "2026-11-24T15:56:43+00:00",
        "status": "deployed", "version": "1", "run_url": "https://github.com/o/r/actions/runs/1",
        "smoke": {"max_abs_diff": 0.0, "n_inputs": 50, "passed": True}}) + "\n")
    monkeypatch.setenv("DEPLOYMENTS_LOG", str(log))
    at = app().run()
    at.radio[0].set_value("Status").run()
    assert not at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "**v1** live since 2026-11-24 15:56 UTC" in page
    assert "smoke test passed on 50 inputs" in page and "[deploy run](" in page
    assert "someone-on-the-team" not in page
    assert not at.json                                     # no raw record dumps
