"""The Status page's data: each source optional, and the @staging-newer flag."""

import json
import os
from dataclasses import replace
from datetime import datetime

import pytest

from demo_app.config import load
from demo_app.registry import RegistryError, VersionInfo
from demo_app.status import build, deployed_versions, split_deployments, staging_warning


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


def line(version, status, **extra):
    """A deployments.jsonl line as deploy.sh / rollback.sh (#64) write it."""
    return {"at": "2026-11-21T10:00:00+00:00", "action": "deploy", "status": status,
            "version": str(version), **extra}


def write_log(tmp_path, *lines):
    log = tmp_path / "deployments.jsonl"
    log.write_text("\n".join(json.dumps(x) if isinstance(x, dict) else x for x in lines) + "\n")
    return log


def test_a_deployment_line_for_staging_clears_the_flag(tmp_path):
    log = write_log(tmp_path, line(1, "deployed"), "not json", line(2, "deployed"))
    reg = FakeRegistry({"production": version("production", 1), "staging": version("staging", 2)})
    status = build(settings(deployments_log=log), reg)
    assert status.warnings == []
    assert status.serving == line(2, "deployed") and status.latest_attempt is None


def test_only_deployed_and_rolled_back_count():
    assert deployed_versions([line(2, "smoke-failed"), line(3, "error"), line(4, "partial"),
                              line(5, "smoke-passed-not-promoted")]) == set()
    assert deployed_versions([line(2, "deployed"), line(1, "rolled-back")]) == {"1", "2"}
    assert deployed_versions([line(2, "ok"), line(3, "success")]) == set()


def test_a_blocked_deploy_is_the_latest_attempt_not_what_is_serving(tmp_path):
    log = write_log(tmp_path, line(1, "deployed"), line(2, "smoke-failed", step="smoke-test"))
    status = build(settings(deployments_log=log), None)
    assert status.serving["version"] == "1"
    assert status.latest_attempt["status"] == "smoke-failed"


def test_a_test_package_is_an_attempt_too():
    serving, attempt = split_deployments([line(1, "deployed"),
                                          {"status": "smoke-passed-not-promoted", "tag": "t"}])
    assert serving["version"] == "1" and attempt["tag"] == "t"


def test_a_rollback_is_what_is_serving():
    serving, attempt = split_deployments([line(2, "deployed"),
                                          line(1, "rolled-back", action="rollback")])
    assert serving["version"] == "1" and attempt is None


def test_no_successful_line_yet(tmp_path):
    status = build(settings(deployments_log=write_log(tmp_path, line(1, "error"))), None)
    assert status.serving is None and "no successful deployment" in status.deployment_note
    assert status.latest_attempt["status"] == "error"


def test_a_failed_deploy_of_staging_says_failed_not_rerun():
    prod, stage = version("production", 1), version("staging", 2)
    warning = staging_warning(prod, stage, [line(1, "deployed"),
                                            line(2, "smoke-failed", step="smoke-test")])
    assert "ran and failed (smoke-failed, step smoke-test" in warning
    assert "re-run it by hand" not in warning


def test_a_partial_deploy_of_staging_says_to_move_the_alias():
    warning = staging_warning(version("production", 1), version("staging", 2),
                              [line(2, "partial", step="move-alias")])
    assert "moved traffic but not @production" in warning
    assert "rollback_model --to 2 --alias production" in warning


def test_no_line_for_staging_still_suggests_a_rerun():
    warning = staging_warning(version("production", 1), version("staging", 2),
                              [line(1, "deployed")])
    assert "no deployment line names it" in warning and "re-run it by hand" in warning


NOV_15 = 1_763_200_000_000            # 2025-11-15T09:46:40Z in ms; any time before NOV_24
NOV_24 = "2026-11-24T15:56:43+00:00"   # the first real deployment line


def registered(alias, number, at_ms):
    return replace(version(alias, number), created_ms=at_ms)


def ms(iso):
    return int(datetime.fromisoformat(iso).timestamp() * 1000)


def test_a_version_older_than_every_record_is_not_flagged(tmp_path):
    # The live case: v2 has been @staging since the Phase 2 rollback demo, before deploy.sh
    # wrote its first line, so nobody ever asked to deploy it.
    log = write_log(tmp_path, line(1, "deployed", at=NOV_24))
    reg = FakeRegistry({"production": version("production", 1),
                        "staging": registered("staging", 2, ms("2026-11-15T12:00:00+00:00"))})
    assert build(settings(deployments_log=log), reg).warnings == []


def test_a_version_registered_after_the_first_record_is_still_flagged():
    # model-ci / retrain_dag register and then request the deploy: no line means it never ran.
    warning = staging_warning(version("production", 1),
                              registered("staging", 3, ms("2026-12-02T10:00:00+00:00")),
                              [line(1, "deployed", at=NOV_24)])
    assert "no deployment line names it" in warning and "re-run it by hand" in warning


def test_the_first_record_is_the_earliest_not_the_first_line():
    lines = [line(1, "deployed", at="2026-12-01T00:00:00+00:00"), line(1, "error", at=NOV_24)]
    stage = registered("staging", 2, ms("2026-11-30T00:00:00+00:00"))   # after Nov 24
    assert staging_warning(version("production", 1), stage, lines) is not None


def test_an_old_version_whose_deploy_failed_still_says_so():
    # The exemption is only for a missing line: a line that names it and failed is real.
    stage = registered("staging", 2, ms("2026-11-15T12:00:00+00:00"))
    warning = staging_warning(version("production", 1), stage,
                              [line(1, "deployed", at=NOV_24),
                               line(2, "smoke-failed", step="smoke-test", at=NOV_24)])
    assert "ran and failed" in warning


@pytest.mark.parametrize("lines", [
    [],                                                   # no records: nothing to compare with
    [line(1, "deployed", at="not a time")],               # no dated record
    [{k: v for k, v in line(1, "deployed").items() if k != "at"}],
])
def test_without_a_dated_record_the_flag_stays(lines):
    stage = registered("staging", 2, NOV_15)
    assert staging_warning(version("production", 1), stage, lines) is not None


def test_naive_and_z_times_are_utc():
    stage = registered("staging", 2, ms("2026-11-24T15:00:00+00:00"))
    for at in ("2026-11-24T16:00:00Z", "2026-11-24T16:00:00"):        # an hour after it
        assert staging_warning(version("production", 1), stage,
                               [line(1, "deployed", at=at)]) is None


def test_no_flag_when_staging_isnt_newer():
    assert staging_warning(version("production", 2), version("staging", 2), []) is None
    assert staging_warning(version("production", 3), version("staging", 2), []) is None
    assert staging_warning(None, version("staging", 2), []) is None


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
