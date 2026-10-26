"""Run every data anomaly check over the pipeline's outputs.

Each check has a severity (work plan, Phase 1):

    fail  raw row counts differ from the expected counts in config
    fail  an id appears in two sets that must not share comments
    fail  more than validation.max_missing_fraction of rows miss text or a label
          (any missing rows under that limit give a warning instead)
    fail  a label outside {0, 1}, or a *_frac column outside [0, 1]
    fail  duplicate ids within one file
    warn  text longer than preprocess.max_char_len_flag characters
    warn  train and validation label prevalence differ by more than
          validation.prevalence_tolerance_pts points
    fail  a created_date that is missing or cannot be parsed (Civil files)
    fail  a fixed-size subset (canaries, H2, obfuscation, drift batches) whose row
          count differs from config; only a warning while an upstream data stage
          is still a stub, since stub outputs are small on purpose

Every check runs, even after one fails, so a single run reports everything.
Warnings go out as one alert through alerts.send_alert("warn", ...). If any
check fails, the summary is still written and the task then raises, which fails
the DAG and triggers the failure alert.

    python -m scripts.anomalies --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.acquire import raw_location
from scripts.alerts import send_alert
from scripts.utils import get_logger, load_config, timed, write_summary

logger = get_logger(__name__)
TASK_ID = "anomalies"

WIKIPEDIA_SETS = ("train", "val", "gate", "canary_wikipedia")
CIVIL_SETS = ("h2_eval", "canary_civil", "bias_eval")
# Pairs of sets that may share ids. Identities test is an annotated subset of
# Civil test, so bias_eval can contain h2_eval comments. The work plan's second
# allowed exception is obfuscation against gate; it needs no entry here because
# obfuscate rewrites every id as "<original_id>_obf", so the sets cannot collide.
ALLOWED_OVERLAPS = frozenset({frozenset({"h2_eval", "bias_eval"})})
# Raw files acquire writes, keyed to config datasets.<name>.expected_rows.<split>.
RAW_FILES = {
    ("wikipedia", "train"): "wikipedia/train.parquet",
    ("wikipedia", "test"): "wikipedia/test.parquet",
    ("civil_comments", "train"): "civil/train.parquet",
    ("civil_comments", "validation"): "civil/validation.parquet",
    ("civil_comments", "test"): "civil/test.parquet",
    ("civil_identities", "test"): "civil/identities_test.parquet",
}
# Stages that produce the data checked here. If any ran as a stub, its outputs are
# small on purpose, so size checks drop to a warning instead of failing the run.
UPSTREAM_DATA_TASKS = ("acquire", "preprocess", "dedupe", "split", "subsets", "obfuscate")


class AnomalyError(RuntimeError):
    """Raised after the summary is written when at least one fail-level check fails."""


class Checks:
    """Collects check results so every check runs before anything is raised."""

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def record(self, check: str, target: str, severity: str, passed: bool,
               detail: str = "") -> None:
        self.results.append({"check": check, "target": target, "severity": severity,
                             "passed": bool(passed), "detail": detail})

    def failed(self, severity: str) -> list[dict[str, Any]]:
        return [r for r in self.results if not r["passed"] and r["severity"] == severity]


def load_sets(cfg: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Every processed set the checks cover, keyed by name; plus any missing files."""
    processed = Path(cfg["paths"]["processed_dir"])
    paths = {name: processed / "wikipedia" / f"{name}.parquet" for name in WIKIPEDIA_SETS}
    paths["obfuscation"] = processed / "obfuscation.parquet"
    paths.update({name: processed / "civil" / f"{name}.parquet" for name in CIVIL_SETS})
    sets, missing = {}, []
    for name, path in paths.items():
        if path.exists():
            sets[name] = pd.read_parquet(path)
        else:
            missing.append(path.relative_to(processed).as_posix())
    drift_files = sorted((processed / "civil" / "drift").glob("batch_*.parquet"))
    if drift_files:
        sets["drift"] = pd.concat([pd.read_parquet(p) for p in drift_files], ignore_index=True)
    else:
        missing.append("civil/drift/batch_*.parquet")
    return sets, missing


def check_raw_counts(cfg: dict[str, Any], checks: Checks) -> None:
    """Raw row counts must match config. Skipped while acquire is still a stub."""
    acquire = Path(cfg["paths"]["summaries_dir"]) / "acquire.json"
    if acquire.exists() and json.loads(acquire.read_text()).get("mode") == "stub":
        checks.record("raw_row_counts", "data/raw", "warn", True,
                      "skipped: acquire is still a stub, so counts are placeholders")
        return
    for (dataset, split), rel in RAW_FILES.items():
        expected = cfg["datasets"][dataset]["expected_rows"][split]
        # Not always under raw_dir: acquire keeps the 1.8M-row Civil train in the
        # untracked cache instead. raw_location owns that rule.
        path = raw_location(cfg, rel)
        if not path.exists():
            checks.record("raw_row_counts", rel, "fail", False, "file missing")
            continue
        rows = len(pd.read_parquet(path, columns=["id"]))
        checks.record("raw_row_counts", rel, "fail", rows == expected,
                      f"{rows:,} rows, expected {expected:,}")


def check_id_overlap(sets: dict[str, pd.DataFrame], checks: Checks) -> None:
    ids = {name: set(df["id"]) for name, df in sets.items()}
    for a, b in itertools.combinations(sorted(ids), 2):
        shared = len(ids[a] & ids[b])
        if frozenset({a, b}) in ALLOWED_OVERLAPS:
            checks.record("id_overlap", f"{a} & {b}", "fail", True,
                          f"{shared:,} shared ids (allowed)")
        else:
            checks.record("id_overlap", f"{a} & {b}", "fail", shared == 0,
                          f"{shared:,} shared ids")


def check_missing(name: str, df: pd.DataFrame, labels: list[str], limit: float,
                  checks: Checks) -> None:
    empty_text = df["text"].isna() | (df["text"].astype("string").str.strip() == "")
    missing_label = df[labels].isna().any(axis=1)
    bad = int((empty_text | missing_label).sum())
    share = bad / len(df) if len(df) else 0.0
    detail = f"{bad:,} rows ({share:.3%}), limit {limit:.1%}"
    if share > limit:
        checks.record("missing_text_or_labels", name, "fail", False, detail)
    else:
        # Under the limit is not fatal, but config asks for a warning (preprocess drops them).
        checks.record("missing_text_or_labels", name, "warn", bad == 0, detail)


def check_label_domain(name: str, df: pd.DataFrame, labels: list[str], checks: Checks) -> None:
    bad = {label: int((~df[label].dropna().isin([0, 1])).sum()) for label in labels}
    for col in (c for c in df.columns if c.endswith("_frac")):
        values = df[col].dropna()
        bad[col] = int(((values < 0) | (values > 1)).sum())
    bad = {k: v for k, v in bad.items() if v}
    checks.record("label_domain", name, "fail", not bad,
                  f"out of range: {bad}" if bad else "all in range")


def check_duplicate_ids(name: str, df: pd.DataFrame, checks: Checks) -> None:
    dupes = int(df["id"].duplicated().sum())
    checks.record("duplicate_ids", name, "fail", dupes == 0, f"{dupes:,} duplicate ids")


def check_long_text(name: str, df: pd.DataFrame, max_len: int, checks: Checks) -> None:
    long = int((df["text"].astype("string").str.len() > max_len).sum())
    checks.record("long_text", name, "warn", long == 0,
                  f"{long:,} comments over {max_len:,} characters")


def check_prevalence_gap(sets: dict[str, pd.DataFrame], labels: list[str], tolerance: float,
                         checks: Checks) -> None:
    if "train" not in sets or "val" not in sets:
        return
    gaps = {label: abs(sets["train"][label].mean() - sets["val"][label].mean()) * 100
            for label in labels}
    worst = max(gaps, key=gaps.get)
    checks.record("prevalence_gap", "train vs val", "warn", gaps[worst] <= tolerance,
                  f"largest gap {gaps[worst]:.2f} pts ({worst}), tolerance {tolerance} pts")


def check_created_date(name: str, df: pd.DataFrame, checks: Checks) -> None:
    if "created_date" not in df.columns:
        checks.record("created_date", name, "fail", False, "column missing")
        return
    dates = df["created_date"]
    if not pd.api.types.is_datetime64_any_dtype(dates):
        # format="mixed": some Civil timestamps have no fractional seconds.
        dates = pd.to_datetime(dates, format="mixed", utc=True, errors="coerce")
    bad = int(dates.isna().sum())
    checks.record("created_date", name, "fail", bad == 0, f"{bad:,} missing or unparseable")


def stubbed_upstream(cfg: dict[str, Any]) -> list[str]:
    """Upstream data stages whose latest summary says they ran as a stub."""
    summaries = Path(cfg["paths"]["summaries_dir"])
    stubbed = []
    for task in UPSTREAM_DATA_TASKS:
        path = summaries / f"{task}.json"
        if path.exists() and json.loads(path.read_text()).get("mode") == "stub":
            stubbed.append(task)
    return stubbed


def check_subset_sizes(cfg: dict[str, Any], sets: dict[str, pd.DataFrame],
                       checks: Checks) -> None:
    """Every fixed-size subset must have exactly the row count config asks for.

    Regression guard: a run once built the 40,000-row drift stream from 200 rows of
    stub data and still went green. gate and bias_eval are whole splits by design,
    so they are left to the raw row-count check.
    """
    splits, civil = cfg["splits"], cfg["splits"]["civil"]
    stubbed = stubbed_upstream(cfg)
    severity = "warn" if stubbed else "fail"
    note = f" (warn only: stub upstream in {', '.join(stubbed)})" if stubbed else ""

    expected = {
        "canary_wikipedia": splits["canary_wikipedia_size"],
        "obfuscation": splits["obfuscation_size"],
        "h2_eval": civil["h2_size"],
        "canary_civil": civil["canary_size"],
    }
    for name, n in expected.items():
        if name in sets:
            rows = len(sets[name])
            checks.record("subset_size", name, severity, rows == n,
                          f"{rows:,} rows, expected {n:,}{note}")

    drift_dir = Path(cfg["paths"]["processed_dir"]) / "civil" / "drift"
    files = sorted(drift_dir.glob("batch_*.parquet"))
    n_batches, per_batch = civil["drift_batches"], civil["drift_total"] // civil["drift_batches"]
    checks.record("subset_size", "drift batch count", severity, len(files) == n_batches,
                  f"{len(files)} batches, expected {n_batches}{note}")
    sizes = {p.stem: len(pd.read_parquet(p, columns=["id"])) for p in files}
    wrong = {stem: n for stem, n in sizes.items() if n != per_batch}
    detail = (f"wrong sizes: {wrong}, expected {per_batch:,} each" if wrong
              else f"all {len(sizes)} batches have {per_batch:,} rows")
    checks.record("subset_size", "drift batch rows", severity, not wrong, detail + note)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    labels = cfg["labels"]["names"]
    summary: dict[str, Any] = {"task": TASK_ID, "mode": "real", "dropped": {}}
    checks = Checks()

    with timed(summary):
        check_raw_counts(cfg, checks)
        sets, missing = load_sets(cfg)
        for rel in missing:
            checks.record("file_present", rel, "fail", False, "expected output missing")

        check_id_overlap(sets, checks)
        for name, df in sets.items():
            check_missing(name, df, labels, cfg["validation"]["max_missing_fraction"], checks)
            check_label_domain(name, df, labels, checks)
            check_duplicate_ids(name, df, checks)
            check_long_text(name, df, cfg["preprocess"]["max_char_len_flag"], checks)
        check_prevalence_gap(sets, labels, cfg["validation"]["prevalence_tolerance_pts"], checks)
        for name in ("h2_eval", "canary_civil", "bias_eval", "drift"):
            if name in sets:
                check_created_date(name, sets[name], checks)
        check_subset_sizes(cfg, sets, checks)

        fails, warns = checks.failed("fail"), checks.failed("warn")
        summary["rows_out"] = 0  # this stage reads data; it writes none
        summary["rows_checked"] = {name: len(df) for name, df in sets.items()}
        summary["checks"] = {"run": len(checks.results), "warn": len(warns), "fail": len(fails)}
        summary["results"] = checks.results

    write_summary(TASK_ID, summary, cfg)
    if warns:
        send_alert("warn", f"Data anomalies: {len(warns)} warning(s)",
                   {f"{r['check']} / {r['target']}": r["detail"] for r in warns})
    if fails:
        lines = "; ".join(f"{r['check']} / {r['target']}: {r['detail']}" for r in fails)
        raise AnomalyError(f"{len(fails)} anomaly check(s) failed: {lines}")
    logger.info("anomalies: %d checks, %d warnings, 0 failures", len(checks.results), len(warns))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config)), indent=2, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
