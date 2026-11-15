"""Consistency checks: the alert rules, dashboard and thresholds doc agree with each other, and
active rules and panels only use metrics that exist today.

"Exist today" means emitted by the mock (the section 2.4 names) or by scripts/canary.py, whose
names are read from canary.py itself rather than a hand-copied list. Metrics that only Phase 3
will emit may appear only in alert_rules_planned.yml and in dashboard panels titled "(planned)".
"""

import json
import re
import sys
from pathlib import Path

import pytest
import yaml

import mock_service as mock

HERE = Path(__file__).resolve().parents[1]
DATA_PIPELINE = HERE.parents[1] / "Data-Pipeline"
PROMETHEUS = HERE / "prometheus"
DASHBOARD = HERE / "grafana" / "dashboards" / "distillate_dashboard.json"
THRESHOLDS = HERE / "thresholds.md"

# Exported only in Phase 3 (named in thresholds.md and alert_rules_planned.yml).
PLANNED_METRICS = {"distillate_drift_psi", "distillate_delayed_label_mean_auc",
                   "distillate_canary_reference_mean_auc"}
METRIC = re.compile(r"\b(distillate_[a-z_]+)\b")


def _family(name):
    """distillate_x_bucket / _sum / _count belong to the histogram distillate_x."""
    return re.sub(r"_(bucket|sum|count)$", "", name)


def _families(exposition):
    return {line.split()[2] for line in exposition.splitlines() if line.startswith("# TYPE")}


def _used(expr):
    return {_family(m) for m in METRIC.findall(expr)}


def _rules(name):
    return [rule for group in yaml.safe_load((PROMETHEUS / name).read_text())["groups"]
            for rule in group["rules"]]


@pytest.fixture(scope="module")
def canary_metrics():
    """The metric families scripts/canary.py actually writes, from its own prometheus_text."""
    sys.path.insert(0, str(DATA_PIPELINE))
    canary = pytest.importorskip("scripts.canary", reason="needs Data-Pipeline/scripts/canary.py")
    report = {"finished_at": 1.0, "sets": {"civil": {
        "labels": {"toxicity": {"auc": 0.9, "f1": 0.5}}, "mean_auc": 0.9, "rows_scored": 1,
        "errors": 0,
        "flag": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "false_positive_rate": 0.0}}}}
    return _families(canary.prometheus_text(report))


@pytest.fixture(scope="module")
def known(canary_metrics):
    return _families(mock.Metrics().text()) | canary_metrics


@pytest.fixture(scope="module")
def active():
    return _rules("alert_rules.yml")


@pytest.fixture(scope="module")
def planned():
    return _rules("alert_rules_planned.yml")


@pytest.fixture(scope="module")
def dashboard():
    return json.loads(DASHBOARD.read_text(encoding="utf-8"))


def _targets(dashboard):
    for panel in dashboard["panels"]:
        for target in panel.get("targets", []):
            yield panel, target["expr"]


def test_every_rule_is_well_formed(active, planned):
    names = [rule["alert"] for rule in active + planned]
    assert len(names) == len(set(names)), "alert names must be unique across both files"
    for rule in active + planned:
        assert rule["expr"].strip() and rule["annotations"]["summary"], rule["alert"]
        assert rule["labels"]["severity"] in {"warning", "critical"}, rule["alert"]
        assert rule["labels"].get("action", "retrain") == "retrain", rule["alert"]


def test_active_rules_only_use_metrics_that_exist_today(active, known):
    for rule in active:
        assert _used(rule["expr"]) <= known, (rule["alert"], _used(rule["expr"]) - known)


def test_planned_rules_only_use_existing_or_named_planned_metrics(planned, known):
    for rule in planned:
        extra = _used(rule["expr"]) - known - PLANNED_METRICS
        assert not extra, (rule["alert"], extra)


def test_no_active_rule_can_trigger_a_retrain(active, planned):
    """Every retrain rule waits on a Phase 3 metric, so none is active yet."""
    assert not [r["alert"] for r in active if r["labels"].get("action") == "retrain"]
    retrain = {r["alert"] for r in planned if r["labels"].get("action") == "retrain"}
    assert retrain == {"DistillateCanaryAUCDropVsPromoted", "DistillateInputDriftAlert",
                       "DistillateDelayedLabelAUCDrop"}


def test_dashboard_marks_every_panel_that_needs_a_planned_metric(dashboard, known):
    for panel, expr in _targets(dashboard):
        used = _used(expr)
        if "(planned)" in panel["title"]:
            assert used <= known | PLANNED_METRICS, (panel["title"], used)
        else:
            assert used <= known, (panel["title"], used - known)


def test_dashboard_is_well_formed(dashboard):
    ids = [panel["id"] for panel in dashboard["panels"]]
    assert len(ids) == len(set(ids)), "panel ids must be unique"
    for panel in dashboard["panels"]:
        if panel["type"] == "timeseries":
            assert panel["datasource"]["uid"] == "prometheus", panel["title"]
            assert panel["targets"] and all(t["expr"] for t in panel["targets"]), panel["title"]


def test_dashboard_has_the_v4_panels(dashboard):
    titles = " ".join(panel["title"].lower() for panel in dashboard["panels"])
    for needed in ("latency", "requests per second", "instances", "flagged rate per label",
                   "overall flag rate", "canary mean auc", "drift psi", "gate history"):
        assert needed in titles, needed


def test_thresholds_doc_covers_every_alert(active, planned):
    doc = THRESHOLDS.read_text(encoding="utf-8")
    missing = [rule["alert"] for rule in active + planned if f"`{rule['alert']}`" not in doc]
    assert not missing, f"thresholds.md doesn't document {missing}"


def test_prometheus_loads_only_the_active_rules():
    config = yaml.safe_load((PROMETHEUS / "prometheus.yml").read_text())
    assert config["rule_files"] == ["/etc/prometheus/alert_rules.yml"]
    # DistillateServiceDown watches up{job="distillate"}, so the job name must match.
    assert config["scrape_configs"][0]["job_name"] == "distillate"
    compose = (HERE / "docker-compose.yml").read_text()
    assert "alert_rules_planned.yml" not in compose
