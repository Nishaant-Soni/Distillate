"""Distillate demo app: Try-it and Status (work plan v5, section 6).

    streamlit run app.py      # from Deployment/frontend/; settings come from the environment

Try-it scores a comment through the serving API (or the mock when ``DISTILLATE_API_URL`` isn't
set) and shows it masked by default. Status shows the registry's ``@production`` and
``@staging``, the last deployment, the last retraining and the Grafana links, with "not yet"
for whatever doesn't exist yet. The app never logs or stores the text it scores.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import streamlit as st

from demo_app import api, config, display, masking, rate_limit
from demo_app import status as status_page
from demo_app.registry import Registry, RegistryError

HERE = Path(__file__).resolve().parent
OWN = "Write your own"

st.set_page_config(page_title="Distillate", page_icon="🛡️", layout="centered")


@st.cache_resource
def settings() -> config.Settings:
    return config.load()


@st.cache_resource
def client() -> api.LiveAPI | api.MockAPI:
    return api.make_api(settings())


@st.cache_resource
def instance_window() -> rate_limit.SlidingWindow:
    """Shared by every session on this instance (one Python process)."""
    return rate_limit.SlidingWindow(settings().instance_per_minute)


@st.cache_resource
def registry() -> Registry | None:
    s = settings()
    if not s.mlflow_uri:
        return None
    return Registry(s.mlflow_uri, s.registered_model, s.mlflow_username, s.mlflow_password)


@st.cache_data(ttl=600, show_spinner=False)
def thresholds() -> dict[str, float] | None:
    """The thresholds the scores are compared with: the mock's, or @production's package's."""
    if client().mock:
        return dict(api.MockAPI.thresholds)
    reg = registry()
    if reg is None:
        return None
    try:
        production = reg.alias(settings().production_alias)
    except RegistryError:  # the page still works without threshold markers
        return None
    return reg.thresholds(production.run_id) if production and production.run_id else None


@st.cache_data(show_spinner=False)
def presets() -> list[dict]:
    return json.loads((HERE / "presets.json").read_text(encoding="utf-8"))["presets"]


@st.cache_data(ttl=30, show_spinner=False)
def health() -> tuple[str | None, str | None]:
    """(model_version, error) from the serving API's /health."""
    try:
        return client().health().get("model_version"), None
    except api.APIError as err:
        return None, str(err)


def session_window() -> rate_limit.SlidingWindow:
    if "window" not in st.session_state:
        st.session_state["window"] = rate_limit.SlidingWindow(settings().session_per_minute)
    return st.session_state["window"]


INTRO = ("Distillate scores an English comment for six kinds of toxicity. The model is a small "
         "DistilBERT **student**, trained on human labels and on the scores of a larger "
         "**teacher** (an LLM, Qwen3-4B, combined with a TF-IDF model). A flag means *send it "
         "to a human moderator*, never *remove it*.")
TEACHER_NOTE = ("**Teacher:** the teacher's scores for this example (the stack of Qwen3-4B and "
                "TF-IDF), computed once in advance. The student learned from scores like "
                "these, so the Score and Teacher columns show how closely it follows the "
                "teacher.")


def source_caption() -> None:
    if client().mock:
        st.caption("Scoring with the **mock** service: made-up scores, for trying the page "
                   "before the live endpoint exists.")
        return
    version, _ = health()
    st.caption(f"Scoring with the live model `{version}`." if version
               else "Scoring with the live model.")


def try_it() -> None:
    st.title("Try it")
    st.markdown(INTRO)
    source_caption()
    items = presets()
    names = [OWN] + [p["label"] for p in items]
    choice = st.selectbox("Example", range(len(names)), format_func=lambda i: names[i])
    preset = items[choice - 1] if choice else None
    if preset:
        # Shown masked, never in an editable box, so a preset is never on screen in full.
        text = preset["text"]
        st.text(masking.mask(text))
    else:
        text = st.text_area("Comment", max_chars=settings().max_chars,
                            placeholder="Type a comment to score")
    if st.button("Score", type="primary", disabled=not text.strip()):
        wait = rate_limit.check(session_window(), instance_window())
        if wait:
            st.warning(f"Too many requests; try again in {math.ceil(wait)} s.")
        else:
            try:
                st.session_state["result"] = {"text": text, "preset": preset,
                                              "response": client().predict(text)}
            except api.APIError as err:
                st.error(f"Couldn't score it: {err}")
    result = st.session_state.get("result")
    if result:
        show_result(result)


def show_result(result: dict) -> None:
    response, preset = result["response"], result["preset"]
    st.divider()
    original = st.toggle("Show the original text", value=False)
    st.text(result["text"] if original else masking.mask(result["text"]))
    marks = thresholds()
    if response["flag"]:
        st.error(display.verdict(response))
    else:
        st.success(display.verdict(response))
    if response["low_confidence"]:
        st.warning(display.borderline_line(response, marks))
    stack = (preset or {}).get("stack_scores")
    rows = []
    for label in api.LABELS:
        row = {"Label": display.name(label), "Score": response["scores"][label]}
        if marks:     # without the registry there are no thresholds: no column of "None"
            row["Threshold"] = marks[label]
        row["Flagged"] = response["flagged"][label]
        if preset:
            row["Teacher (stack)"] = stack.get(label) if stack else None
        rows.append(row)
    # Narrow columns kept small, so the whole table fits a phone's width.
    st.dataframe(rows, hide_index=True, use_container_width=True, column_config={
        "Label": st.column_config.TextColumn(width="small"),
        "Score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f",
                                                 width="small"),
        "Threshold": st.column_config.NumberColumn(format="%.3f", width="small"),
        "Flagged": st.column_config.CheckboxColumn(width="small"),
        "Teacher (stack)": st.column_config.NumberColumn("Teacher", format="%.3f",
                                                         width="small"),
    })
    if preset and stack:
        st.caption(TEACHER_NOTE)
    elif preset:
        st.caption("The teacher's scores for this example haven't been computed yet.")
    st.caption(f"Model `{response['model_version']}` · {response['latency_ms']:.0f} ms")


def version_card(alias: str, info) -> None:
    st.subheader(f"@{alias}")
    if info is None:
        st.write("not set")
        return
    gate = {True: "passed", False: "failed", None: "unknown"}[info.gate_passed]
    st.metric("Version", f"v{info.version}")
    st.write(f"`{info.model_version or 'unknown'}`")
    auc = f", gate mean AUC {info.gate_mean_auc:.4f}" if info.gate_mean_auc else ""
    st.write(f"Gate {gate}{auc}")
    st.write(f"Artifact Registry upload: {info.ar_upload or 'unknown'}")
    if info.last_alias_move:
        move = info.last_alias_move
        st.caption(f"Last alias move: v{move.get('from')} → v{move.get('to')} "
                   f"({move.get('reason') or 'no reason given'})")


def status() -> None:
    st.title("Status")
    s = settings()
    version, error = health()
    if error:
        st.error(f"Serving API: {error}")
    elif client().mock:
        st.caption("Serving API: the mock (no live endpoint configured).")
    else:
        st.caption(f"Serving API at `{s.api_url}` is up, serving `{version}`.")
    info = status_page.build(s, registry())
    for warning in info.warnings:
        st.warning(warning)
    if info.registry_note:
        st.info(f"Registry: {info.registry_note}")
    else:
        left, right = st.columns(2)
        with left:
            version_card(s.production_alias, info.production)
        with right:
            version_card(s.staging_alias, info.staging)
    st.subheader("Deployment")
    if info.serving:
        st.markdown(display.serving_line(info.serving))
    else:
        st.write(info.deployment_note)
    if info.latest_attempt:
        attempt = info.latest_attempt
        text = (f"Latest attempt: {attempt.get('action', 'deploy')} "
                f"{'v' + str(attempt['version']) if attempt.get('version') else 'test package'} "
                f"{attempt.get('status', '')} at {attempt.get('at', 'unknown time')}")
        if str(attempt.get("status", "")).lower() in status_page.FAILED:
            st.warning(f"{text}: {attempt.get('detail') or 'see the deploy run'}")
        else:
            st.caption(text)
    st.subheader("Last retraining")
    if info.last_retraining:
        st.markdown(display.retraining_line(info.last_retraining))
    else:
        st.write(info.retraining_note)
    st.subheader("Dashboards")
    if info.grafana_url:
        st.link_button("Open Grafana", info.grafana_url)
    else:
        st.write("not yet: no Grafana URL configured (GRAFANA_URL)")


# At the top of the page, not in the sidebar, which phones hide behind a menu.
if st.radio("Page", ["Try it", "Status"], horizontal=True,
            label_visibility="collapsed") == "Try it":
    try_it()
else:
    status()
