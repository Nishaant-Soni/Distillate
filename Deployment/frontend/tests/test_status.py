"""The Status page's data: each source optional, and the @staging-newer flag."""

import json
import os

from demo_app.config import load
from demo_app.registry import RegistryError, VersionInfo
from demo_app.status import build, deployed_versions, staging_warning


def version(alias, number):
    return VersionInfo(alias=alias, version=str(number), run_id=f"run{number}",
                       model_version=f"student-v{number}", gate_passed=True,
                       gate_mean_auc=0.98, ar_upload="ok", code_commit="abc1234")


class FakeRegistry:
    def __init__(self, aliases=None, error=None):
        self.aliases, self.error = aliases or {}, error

    def alias(self, alias):
        if self.error:
            raise self.error
        return self.aliases.get(alias)


def settings(**env):
    return load({key.upper(): str(value) for key, value in env.items()})


def test_nothing_configured_shows_not_yet_everywhere():
    status = build(settings(), None)
    assert "MLFLOW_TRACKING_URI" in status.registry_note
    assert "DEPLOYMENTS_LOG" in status.deployment_note
    assert "RETRAIN_RECORDS" in status.retraining_note
    assert status.grafana_url is None and status.warnings == []


def test_staging_newer_without_a_deployment_line_is_flagged():
    reg = FakeRegistry({"production": version("production", 1), "staging": version("staging", 2)})
    status = build(settings(), reg)
    assert status.production.version == "1" and status.staging.version == "2"
    assert len(status.warnings) == 1 and "@staging (v2)" in status.warnings[0]


def test_a_deployment_line_for_staging_clears_the_flag(tmp_path):
    log = tmp_path / "deployments.jsonl"
    log.write_text("\n".join([json.dumps({"version": "1", "outcome": "deployed"}),
                              "not json",
                              json.dumps({"version": "2", "outcome": "deployed"})]) + "\n")
    reg = FakeRegistry({"production": version("production", 1), "staging": version("staging", 2)})
    status = build(settings(deployments_log=log), reg)
    assert status.warnings == []
    assert status.last_deployment == {"version": "2", "outcome": "deployed"}


def test_a_failed_deployment_doesnt_count():
    assert deployed_versions([{"version": 2, "outcome": "smoke test failed"}]) == set()
    assert deployed_versions([{"version": 2, "status": "OK"}]) == {"2"}


def test_no_flag_when_staging_isnt_newer():
    assert staging_warning(version("production", 2), version("staging", 2), set()) is None
    assert staging_warning(version("production", 3), version("staging", 2), set()) is None
    assert staging_warning(None, version("staging", 2), set()) is None


def test_a_registry_error_is_a_note_not_a_crash():
    status = build(settings(), FakeRegistry(error=RegistryError("registry unreachable (Timeout)")))
    assert status.registry_note == "registry unreachable (Timeout)"
    assert status.production is None and status.warnings == []


def test_missing_files_are_not_yet(tmp_path):
    status = build(settings(deployments_log=tmp_path / "none.jsonl",
                            retrain_records=tmp_path / "missing"), None)
    assert "doesn't exist yet" in status.deployment_note
    assert "doesn't exist yet" in status.retraining_note


def test_the_newest_retraining_record(tmp_path):
    older, newer = tmp_path / "retrain-a.json", tmp_path / "retrain-b.json"
    older.write_text(json.dumps({"run_id": "retrain-a", "trigger": {"reason": "old"},
                                 "steps": {}}))
    newer.write_text(json.dumps({"run_id": "retrain-b", "trigger": {"reason": "flag_rate PSI"},
                                 "steps": {"gate": {"result": "passed"},
                                           "register": {"registered": True}}}))
    os.utime(older, (1, 1))
    status = build(settings(retrain_records=tmp_path), None)
    assert status.last_retraining == {"run_id": "retrain-b", "reason": "flag_rate PSI",
                                      "gate": "passed", "registered": True}


def test_grafana_link():
    assert build(settings(grafana_url="https://grafana.example"), None).grafana_url == \
        "https://grafana.example"
