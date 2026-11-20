"""Smoke test a deployed revision before it gets traffic (work plan v5, section 6.1).

Owner: Ruchita. Called by deploy.sh on the revision's tagged URL (no traffic yet):

1. GET /health returns status "ok" and the package's model_version (retried while the
   revision starts).
2. POST /predict for every reference input in the package's reference_outputs.json. Each
   response must report the same model_version, scores within ``--tolerance`` of the
   package's own, flagged[label] exactly when the score reaches that label's threshold, and
   flag exactly when any label is flagged (the API contract, section 2.4).

Exit 0 when everything passes, 1 otherwise; ``--out`` gets a JSON summary either way.
Standard library only, so it runs anywhere deploy.sh does.

    python smoke_test.py --url https://v3-abc1234---distillate-api-xyz.a.run.app \\
        --reference package/reference_outputs.json --thresholds package/thresholds.json \\
        --model-version student-distilbert-mixed-5ep-20261112T210931Z --out smoke.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

LABELS = ("toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack")
MAX_PROBLEMS = 20


def _request(url: str, body: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def wait_for_health(base: str, model_version: str, attempts: int = 6, delay: float = 5.0,
                    timeout: float = 30.0) -> list[str]:
    """Problems with /health after ``attempts`` tries (a new revision may still be starting)."""
    last = "no response"
    for attempt in range(attempts):
        try:
            health = _request(f"{base}/health", None, timeout)
            problems = []
            if health.get("status") != "ok":
                problems.append(f"/health status {health.get('status')!r}, expected 'ok'")
            if model_version and health.get("model_version") != model_version:
                problems.append(f"/health model_version {health.get('model_version')!r}, "
                                f"expected {model_version!r}")
            return problems
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last = str(exc)
            if attempt < attempts - 1:
                time.sleep(delay)
    return [f"/health unreachable after {attempts} tries: {last}"]


def check_prediction(response: dict[str, Any], expected: dict[str, float],
                     thresholds: dict[str, float], model_version: str,
                     tolerance: float) -> tuple[list[str], float]:
    """Problems with one /predict response, and its largest score difference."""
    problems: list[str] = []
    scores, flagged = response.get("scores") or {}, response.get("flagged") or {}
    worst = 0.0
    for label in LABELS:
        if label not in scores or label not in flagged:
            problems.append(f"no score or flag for {label}")
            continue
        diff = abs(float(scores[label]) - float(expected[label]))
        worst = max(worst, diff)
        if diff > tolerance:
            problems.append(f"{label} score {scores[label]:.6f} vs package "
                            f"{expected[label]:.6f} (diff {diff:.2e} > {tolerance:.0e})")
        if bool(flagged[label]) != (float(scores[label]) >= float(thresholds[label])):
            problems.append(f"{label} flagged={flagged[label]} disagrees with score "
                            f"{scores[label]:.4f} vs threshold {thresholds[label]:.4f}")
    if flagged and bool(response.get("flag")) != any(bool(flagged.get(lab)) for lab in LABELS):
        problems.append("flag disagrees with the per-label flags")
    if model_version and response.get("model_version") != model_version:
        problems.append(f"model_version {response.get('model_version')!r}, expected "
                        f"{model_version!r}")
    return problems, worst


def run(base: str, reference: dict[str, Any], thresholds: dict[str, float],
        model_version: str, tolerance: float, timeout: float = 30.0,
        health_attempts: int = 6, health_delay: float = 5.0) -> dict[str, Any]:
    base = base.rstrip("/")
    problems = wait_for_health(base, model_version, health_attempts, health_delay, timeout)
    worst, checked = 0.0, 0
    if not problems:
        for i, (text, expected) in enumerate(zip(reference["texts"], reference["scores"],
                                                 strict=True)):
            try:
                response = _request(f"{base}/predict", {"text": text}, timeout)
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                problems.append(f"input {i}: /predict failed: {exc}")
                continue
            found, diff = check_prediction(response, expected, thresholds, model_version,
                                           tolerance)
            problems += [f"input {i}: {p}" for p in found]
            worst, checked = max(worst, diff), checked + 1
    return {"passed": not problems, "url": base, "n_inputs": len(reference["texts"]),
            "n_checked": checked, "max_abs_diff": worst, "tolerance": tolerance,
            "problems": problems[:MAX_PROBLEMS], "n_problems": len(problems)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="the revision's tagged URL")
    parser.add_argument("--reference", required=True, help="reference_outputs.json")
    parser.add_argument("--thresholds", required=True, help="the package's thresholds.json")
    parser.add_argument("--model-version", default="", help="expected model_version")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out", default=None, help="write the JSON summary here")
    args = parser.parse_args(argv)

    reference = json.loads(Path(args.reference).read_text())
    thresholds = json.loads(Path(args.thresholds).read_text())
    result = run(args.url, reference, thresholds, args.model_version, args.tolerance,
                 args.timeout)
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    print(f"Smoke test {'PASSED' if result['passed'] else 'FAILED'}: "
          f"{result['n_checked']}/{result['n_inputs']} inputs, max score diff "
          f"{result['max_abs_diff']:.2e}", file=sys.stderr)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
