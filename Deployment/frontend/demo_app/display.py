"""The words the pages show, built without Streamlit so they can be tested on their own.

* label names in plain words;
* the verdict lines: which labels are flagged, and which scores are borderline (the reason
  behind the API's ``low_confidence``);
* one-line summaries of the deployment and retraining records for the Status page.

Who ran a deploy (``actor``) is never shown: the page is public, and the full record stays in
``deployments.jsonl`` and the deploy run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .api import LABELS

NAMES = {"toxicity": "Toxicity", "severe_toxicity": "Severe toxicity", "obscene": "Obscene",
         "threat": "Threat", "insult": "Insult", "identity_attack": "Identity attack"}

# The serving app's low_confidence_margin (Deployment/serving/config.yaml): a score is
# borderline when it is less than this far from its threshold, either side.
MARGIN = 0.1


def name(label: str) -> str:
    return NAMES.get(label, label.replace("_", " ").capitalize())


def flagged_labels(response: dict[str, Any]) -> list[str]:
    """The flagged labels' names, in the API's label order."""
    return [name(label) for label in LABELS if response["flagged"].get(label)]


def borderline(response: dict[str, Any], thresholds: dict[str, float] | None
               ) -> list[tuple[str, float, float]]:
    """(name, score, threshold) for every score within MARGIN of its threshold, as the API
    decides low_confidence. Empty without thresholds."""
    if not thresholds:
        return []
    rows = []
    for label in LABELS:
        score, mark = response["scores"][label], thresholds.get(label)
        if mark is not None and abs(score - mark) < MARGIN:
            rows.append((name(label), score, mark))
    return rows


def verdict(response: dict[str, Any]) -> str:
    labels = flagged_labels(response)
    if response["flag"] and labels:
        return f"**Flagged for review:** {', '.join(labels)}."
    if response["flag"]:
        return "**Flagged for review.**"
    return "**Not flagged:** every score is below its threshold."


def borderline_line(response: dict[str, Any], thresholds: dict[str, float] | None) -> str:
    """The low-confidence line: which scores are close to their thresholds."""
    rows = borderline(response, thresholds)
    if not rows:     # no thresholds to name them by (the API still says low_confidence)
        return "**Borderline:** a score is close to its threshold, so a person should check it."
    parts = [f"{label} ({score:.2f}, threshold {mark:.2f})" for label, score, mark in rows]
    return (f"**Borderline:** {', '.join(parts)}. A small change in wording could flip "
            f"{'this label' if len(rows) == 1 else 'these labels'}, so a person should check it.")


def when(at: str | None) -> str:
    """An ISO time from the records as "2026-11-24 15:56 UTC"; as given if it doesn't parse."""
    if not at:
        return "an unknown time"
    try:
        moment = datetime.fromisoformat(at.replace("Z", "+00:00"))
    except ValueError:
        return at
    if moment.tzinfo is not None:            # naive times are taken as UTC already
        moment = moment.astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%d %H:%M UTC")


def serving_line(line: dict[str, Any]) -> str:
    """What is serving, from its deployments.jsonl line."""
    version = f"v{line['version']}" if line.get("version") else "a test package"
    parts = [f"**{version}** live since {when(line.get('at'))}"]
    if str(line.get("status", "")).lower() == "rolled-back":
        parts[0] += " (rolled back)"
    smoke = line.get("smoke") or {}
    if smoke.get("n_inputs"):
        result = "passed" if smoke.get("passed") else "did not pass"
        parts.append(f"smoke test {result} on {smoke['n_inputs']} inputs")
    if line.get("run_url"):
        parts.append(f"[deploy run]({line['run_url']})")
    return " · ".join(parts)


def retraining_line(record: dict[str, Any]) -> str:
    """The last retraining, from status.latest_retraining's summary."""
    parts = [f"Run `{record.get('run_id', 'unknown')}`"]
    if record.get("reason"):
        parts.append(f"triggered by {record['reason']}")
    if record.get("gate"):
        parts.append(f"gate {record['gate']}")
    if record.get("registered") is not None:
        parts.append("registered" if record["registered"] else "not registered")
    return " · ".join(parts)
