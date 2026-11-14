"""Consistency checks: the alert rules, dashboard and thresholds doc agree with each other and
only use metrics that exist (the mock's section 2.4 names, the canary's, or the two drift
metrics drift_ingest_dag exports in Phase 3)."""

import json
import re
from pathlib import Path

import pytest
import yaml

import mock_service as mock

HERE = Path(__file__).resolve().parents[1]
RULES = HERE / "prometheus" / "alert_rules.yml"
DASHBOARD = HERE / "grafana" / "dashboards" / "distillate_dashboard.json"
THRESHOLDS = HERE / "thresholds.md"

# From scripts/canary.py's prometheus_text (Data-Pipeline, PR for canary.py).
CANARY_METRICS = {
    "distillate_canary_auc", "distillate_canary_f1", "distillate_canary_mean_auc",
    "distillate_canary_flag_precision", "distillate_canary_flag_recall",
    "distillate_canary_flag_f1", "distillate_canary_flag_false_positive_rate",
    "distillate_canary_rows_scored", "distillate_canary_errors",
    "distillate_canary_last_run_timestamp_seconds",
}
# Exported by drift_ingest_dag in Phase 3 (named in thresholds.md).
PHASE3_DRIFT_METRICS = {"distillate_drift_psi", "distillate_delayed_label_mean_auc"}
METRIC = re.compile(r"\b(distillate_[a-z_]+)\b")


def _family(name):
    """distillate_x_bucket / _sum / _count belong to the histogram distillate_x."""
    return re.sub(r"_(bucket|sum|count)$", "", name)


@pytest.fixture(scope="module")
def known():
    families = {line.split()[2] for line in mock.Metrics().text().splitlines()
                if line.startswith("# TYPE")}
    return families | CANARY_METRICS | PHASE3_DRIFT_METRICS


@pytest.fixture(scope="module")
def rules():
    return [rule for group in yaml.safe_load(RULES.read_text())["groups"]
            for rule in group["rules"]]


@pytest.fixture(scope="module")
def dashboard():
    return json.loads(DASHBOARD.read_text(encoding="utf-8"))


def test_every_rule_is_well_formed(rules):
    names = [rule["alert"] for rule in rules]
    assert len(names) == len(set(names)), "alert names must be unique"
    for rule in rules:
        assert rule["expr"].strip() and rule["annotations"]["summary"], rule["alert"]
        assert rule["labels"]["severity"] in {"warning", "critical"}, rule["alert"]
        assert rule["labels"].get("action", "retrain") == "retrain", rule["alert"]


def test_rules_only_use_known_metrics(rules, known):
    for rule in rules:
        used = {_family(m) for m in METRIC.findall(rule["expr"])}
        assert used <= known, (rule["alert"], used - known)


def test_dashboard_only_uses_known_metrics(dashboard, known):
    for panel in dashboard["panels"]:
        for target in panel.get("targets", []):
            used = {_family(m) for m in METRIC.findall(target["expr"])}
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


def test_the_retrain_alerts_are_the_documented_ones(rules):
    retrain = {rule["alert"] for rule in rules if rule["labels"].get("action") == "retrain"}
    assert retrain == {"DistillateCanaryAUCDrop", "DistillateInputDriftAlert",
                       "DistillateDelayedLabelAUCDrop"}


def test_thresholds_doc_covers_every_alert(rules):
    doc = THRESHOLDS.read_text(encoding="utf-8")
    missing = [rule["alert"] for rule in rules if f"`{rule['alert']}`" not in doc]
    assert not missing, f"thresholds.md doesn't document {missing}"


def test_prometheus_config_loads_the_rules():
    config = yaml.safe_load((HERE / "prometheus" / "prometheus.yml").read_text())
    assert config["rule_files"] == ["/etc/prometheus/alert_rules.yml"]
    # DistillateServiceDown watches up{job="distillate"}, so the job name must match.
    assert config["scrape_configs"][0]["job_name"] == "distillate"
