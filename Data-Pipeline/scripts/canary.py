"""Canary replay: score the labeled canary sets through a live service and measure it.

Real traffic has no labels, so the canary sets (canary_wikipedia, 2,000 comments, and
canary_civil, 4,000) are replayed against any service that follows the serving API
(CONTRACTS.md / work plan v4 section 2.4, POST <url>/predict), and the answers are
compared with the known labels:

    per label   AUC from the scores, F1 from the flagged decisions
    flag        precision, recall, F1 and false positive rate of the service's flag,
                against "any label is positive" in the gold labels

Civil metrics cover five labels: severe_toxicity is excluded (work plan v4, section 2.2),
since almost no Civil comment reaches 0.5 on it.

Outputs (no comment text in either):
    reports/canary/canary_report.json   full metrics, plus run details
    reports/canary/canary.prom          the same metrics in Prometheus text format, for the
                                        exporter / textfile collector behind the dashboards

A call that fails, or returns a response that doesn't match the API (including a score that
isn't a finite number in [0, 1]), is counted as an error and left out of the metrics. If
more than canary.max_error_fraction of the calls fail, the run fails: metrics from a broken
service would be misleading. Before replaying, GET <url>/health must answer "ok", so a
down service fails fast. From the CLI, a failed run also sends a "fail" alert
(alerts.send_alert), since no new metrics are written and the dashboard would otherwise keep
showing the last good run.

    python -m scripts.canary --url http://localhost:8000 --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from scripts.alerts import send_alert
from scripts.utils import PIPELINE_ROOT, atomic_write, get_logger, load_config, timed, write_summary

logger = get_logger(__name__)
TASK_ID = "canary"
Predict = Callable[[str], dict[str, Any]]


class CanaryError(RuntimeError):
    """The canary could not produce trustworthy metrics (no URL, too many failed calls)."""


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PIPELINE_ROOT / p).resolve()


# ---------------------------------------------------------------- calling the service


def http_predict(url: str, timeout_s: float) -> Predict:
    """A predict function for POST <url>/predict, per the serving API."""
    endpoint = url.rstrip("/") + "/predict"

    def call(text: str) -> dict[str, Any]:
        body = json.dumps({"text": text}).encode("utf-8")
        request = urllib.request.Request(endpoint, data=body, method="POST",
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read())

    return call


def check_health(url: str, timeout_s: float) -> None:
    """GET <url>/health must answer {"status": "ok"}; otherwise fail before replaying anything."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout_s) as response:
            body = json.loads(response.read())
    except Exception as exc:  # noqa: BLE001 - any failure means the service isn't usable
        raise CanaryError(f"Health check failed for {url}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(body, dict) or body.get("status") != "ok":
        raise CanaryError(f"Health check for {url} did not return status ok: {body!r}")


def _is_score(value: Any) -> bool:
    """A finite number in [0, 1] (bools excluded, though Python counts them as ints)."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0.0 <= value <= 1.0)


def _check_response(response: Any, labels: list[str]) -> str | None:
    """Why a response doesn't match the serving API, or None if it does."""
    if not isinstance(response, dict):
        return "not a JSON object"
    for key in ("scores", "flagged"):
        if not isinstance(response.get(key), dict):
            return f"missing {key}"
        missing = [label for label in labels if label not in response[key]]
        if missing:
            return f"{key} missing {missing}"
    bad_scores = [label for label in labels if not _is_score(response["scores"][label])]
    if bad_scores:
        return f"scores not finite numbers in [0, 1]: {bad_scores}"
    bad_flags = [label for label in labels if not isinstance(response["flagged"][label], bool)]
    if bad_flags:
        return f"flagged not true/false: {bad_flags}"
    if not isinstance(response.get("flag"), bool):
        return "missing flag"
    return None


def replay(texts: list[str], predict: Predict, labels: list[str],
           workers: int) -> tuple[pd.DataFrame, list[str]]:
    """Send every text; return one row per text (None where the call failed) and the errors."""
    errors: list[str] = []

    def safe(text: str) -> dict[str, Any] | None:
        try:
            response = predict(text)
        except Exception as exc:  # noqa: BLE001 - any failure is one failed call
            errors.append(type(exc).__name__)
            return None
        problem = _check_response(response, labels)
        if problem:
            errors.append(f"bad_response: {problem}")
            return None
        return response

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        responses = list(pool.map(safe, texts))

    rows = []
    for response in responses:
        if response is None:
            rows.append({"ok": False})
            continue
        row = {"ok": True, "flag": response["flag"],
               "model_version": response.get("model_version")}
        for label in labels:
            row[f"p_{label}"] = float(response["scores"][label])
            row[f"flagged_{label}"] = bool(response["flagged"][label])
        rows.append(row)
    return pd.DataFrame(rows), errors


# ---------------------------------------------------------------- metrics


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def metrics(gold: pd.DataFrame, predictions: pd.DataFrame, labels: list[str]) -> dict[str, Any]:
    """Per-label AUC and F1, and the flag decision's precision, recall, F1 and FPR."""
    ok = predictions["ok"].to_numpy()
    gold, predictions = gold.loc[ok].reset_index(drop=True), predictions.loc[ok].reset_index(
        drop=True)
    per_label = {}
    for label in labels:
        y = gold[label].astype(int).to_numpy()
        both_classes = 0 < y.sum() < len(y)
        per_label[label] = {
            "positives": int(y.sum()),
            "auc": _round(roc_auc_score(y, predictions[f"p_{label}"])) if both_classes else None,
            "f1": _round(f1_score(y, predictions[f"flagged_{label}"].astype(int),
                                  zero_division=0)),
        }
    y_any = gold[labels].max(axis=1).astype(int).to_numpy()
    flag = predictions["flag"].astype(int).to_numpy()
    negatives = int((y_any == 0).sum())
    aucs = [m["auc"] for m in per_label.values() if m["auc"] is not None]
    return {
        "rows_scored": len(gold),
        "labels": per_label,
        "mean_auc": _round(np.mean(aucs)) if aucs else None,
        "flag": {
            "positives": int(y_any.sum()),
            "precision": _round(precision_score(y_any, flag, zero_division=0)),
            "recall": _round(recall_score(y_any, flag, zero_division=0)),
            "f1": _round(f1_score(y_any, flag, zero_division=0)),
            "false_positive_rate": _round(((flag == 1) & (y_any == 0)).sum() / negatives)
            if negatives else None,
        },
    }


# ---------------------------------------------------------------- Prometheus output


def prometheus_text(report: dict[str, Any]) -> str:
    """The report's metrics in Prometheus text exposition format (gauges)."""
    lines: list[str] = []

    def gauge(name: str, help_text: str, samples: list[tuple[dict[str, str], float | None]]):
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        for label_values, value in samples:
            if value is None:
                continue  # Prometheus has no null; a missing sample is the honest encoding
            tags = ",".join(f'{k}="{v}"' for k, v in label_values.items())
            lines.append(f"{name}{{{tags}}} {value}" if tags else f"{name} {value}")

    sets = report["sets"]
    gauge("distillate_canary_auc", "Canary ROC-AUC per set and label.",
          [({"set": s, "label": lbl}, m["auc"])
           for s, r in sets.items() for lbl, m in r["labels"].items()])
    gauge("distillate_canary_f1", "Canary F1 of the flagged decision per set and label.",
          [({"set": s, "label": lbl}, m["f1"])
           for s, r in sets.items() for lbl, m in r["labels"].items()])
    gauge("distillate_canary_mean_auc", "Canary mean ROC-AUC per set.",
          [({"set": s}, r["mean_auc"]) for s, r in sets.items()])
    for key, help_text in (("precision", "precision"), ("recall", "recall"), ("f1", "F1"),
                           ("false_positive_rate", "false positive rate")):
        gauge(f"distillate_canary_flag_{key}", f"Canary flag-decision {help_text} per set.",
              [({"set": s}, r["flag"][key]) for s, r in sets.items()])
    gauge("distillate_canary_rows_scored", "Canary comments scored successfully per set.",
          [({"set": s}, r["rows_scored"]) for s, r in sets.items()])
    gauge("distillate_canary_errors", "Canary calls that failed or returned a bad response.",
          [({"set": s}, r["errors"]) for s, r in sets.items()])
    gauge("distillate_canary_last_run_timestamp_seconds", "When the canary last finished.",
          [({}, report["finished_at"])])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- stage


def run(cfg: dict[str, Any], url: str | None = None, predict: Predict | None = None
        ) -> dict[str, Any]:
    """Replay every canary set and write the report and Prometheus file.

    ``predict`` overrides the HTTP client (tests, or a local service object). Otherwise the
    URL comes from the argument, then CANARY_URL, then canary.service_url.
    """
    ccfg = cfg["canary"]
    if predict is None:
        url = url or os.environ.get("CANARY_URL") or ccfg.get("service_url")
        if not url:
            raise CanaryError("No service URL: pass --url, set CANARY_URL, or canary.service_url")
        check_health(url, ccfg["timeout_s"])
        predict = http_predict(url, ccfg["timeout_s"])

    all_labels = cfg["labels"]["names"]
    summary: dict[str, Any] = {"task": TASK_ID, "mode": "real", "dropped": {}}
    report: dict[str, Any] = {"service_url": url, "sets": {}}
    with timed(summary):
        for name, spec in ccfg["sets"].items():
            labels = [lbl for lbl in all_labels if lbl not in spec.get("exclude_labels", [])]
            gold = pd.read_parquet(_resolve(spec["path"]), columns=["text", *labels])
            if ccfg.get("max_rows"):
                gold = gold.head(ccfg["max_rows"])
            start = time.perf_counter()
            predictions, errors = replay(gold["text"].tolist(), predict, labels,
                                         ccfg["workers"])
            elapsed = time.perf_counter() - start
            if len(gold) and len(errors) / len(gold) > ccfg["max_error_fraction"]:
                raise CanaryError(f"{name}: {len(errors)} of {len(gold)} calls failed "
                                  f"(limit {ccfg['max_error_fraction']:.1%}): {errors[:3]}")
            result = metrics(gold.drop(columns="text"), predictions, labels)
            versions = predictions.get("model_version", pd.Series(dtype=object)).dropna()
            if versions.nunique() > 1:
                # A rollout mid-run: these metrics mix two models, so read them with care.
                logger.warning("canary %s saw %d model versions in one run: %s", name,
                               versions.nunique(), sorted(set(versions)))
            result.update({
                "rows": len(gold), "errors": len(errors), "labels_scored": labels,
                "model_versions": sorted(set(versions)),
                "seconds": round(elapsed, 2),
            })
            report["sets"][name] = result
            logger.info("canary %s: %d scored, %d errors, mean AUC %s, flag recall %s",
                        name, result["rows_scored"], len(errors), result["mean_auc"],
                        result["flag"]["recall"])
        report["finished_at"] = round(time.time(), 3)

        out = _resolve(ccfg["output_dir"])
        atomic_write(out / "canary_report.json", json.dumps(report, indent=2, sort_keys=True))
        atomic_write(out / "canary.prom", prometheus_text(report))
        summary["rows_in"] = sum(r["rows"] for r in report["sets"].values())
        summary["rows_out"] = sum(r["rows_scored"] for r in report["sets"].values())
        summary["dropped"] = {"failed_calls": sum(r["errors"] for r in report["sets"].values())}
        summary["mean_auc"] = {s: r["mean_auc"] for s, r in report["sets"].items()}
    write_summary(TASK_ID, summary, cfg)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--url", default=None, help="service base URL, e.g. http://host:8000")
    args = parser.parse_args(argv)
    try:
        summary = run(load_config(args.config), url=args.url)
    except CanaryError as exc:
        # Nothing new is written on failure, so without this the dashboard would keep
        # showing the last good run.
        send_alert("fail", "Canary run failed", {"error": str(exc), "url": args.url})
        raise
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
