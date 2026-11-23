"""Both pages end to end with Streamlit's AppTest, on the mock and with no registry."""

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
    assert table["insult"] == pytest.approx(0.9679)      # presets.json's cached stack score
    assert not any("haven't been computed" in c.value for c in at.caption)


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
    at.sidebar.radio[0].set_value("Status").run()
    assert not at.exception
    assert any("MLFLOW_TRACKING_URI" in i.value for i in at.info)
    page = " ".join(m.value for m in at.markdown)
    assert "DEPLOYMENTS_LOG" in page and "RETRAIN_RECORDS" in page and "GRAFANA_URL" in page
