"""The teacher's serving cost for H3 (``reports/h3_teacher_cost.json``), from the T4 sweep.

The sweep (``kaggle/20_teacher_cost_and_presets.ipynb``, written to ``artifacts/h3/``) served
raw Qwen with vLLM on one T4, one request per comment (its six P2 prompts), at a series of
Poisson request rates; a rate "keeps up" when p95 <= 2 s, there are no errors and at least
95% of the offered rate is achieved. One instance (a T4 on an n1-standard-4) is sized at the
highest rate that kept up, and priced at GCP's on-demand list price (``h3.price``).

At a target rate R (comments/s), the instances needed are ceil(R / sustainable rate), and

    cost per million comments = instances * hourly price / (R * 3600) * 1e6

Without a target rate, the report still has the per-instance figures; the student load
test's rate fills in the rest.

    python -m src.teacher.h3_cost [--rate R]   # writes reports/h3_teacher_cost.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.teacher.config import PIPELINE_ROOT, get_logger, load_config

logger = get_logger(__name__)

CAVEATS = [
    "Measured on a T4 (Kaggle), not an L4: Kaggle's free tier has no L4.",
    "The load generator ran in the same session against localhost (Kaggle exposes no ports), "
    "so network time is left out, and the generator shared the host's CPUs with the server.",
    "The Kaggle host isn't an n1-standard-4; the price is for the machine a T4 would attach to "
    "on GCP.",
    "The load is Civil drift comments (the traffic the service sees), 2,000 seeded.",
    "List prices, on demand: sustained-use and committed-use discounts are left out.",
]
DURATION_SHORT = ("Each rate ran for 5 minutes after a 2-minute warm-up; there was no longer "
                  "stability run.")
DURATION_STABLE = ("Each sweep rate ran for 5 minutes after a 2-minute warm-up; the sustainable "
                   "rate then ran for {minutes:g} minutes in a second session (stability).")
# What must match between the sweep session and the stability session.
SAME_SESSION_KEYS = ("model_version", "prompt_version", "code_commit", "vllm_version", "gpu")


def stability_section(sweep: Mapping[str, Any], session: Mapping[str, Any]) -> dict[str, Any]:
    """The long run at the sustainable rate, from a second session's sweep file; checked to be
    the same teacher, code and load as ``sweep``, and run at its sustainable rate."""
    run = session.get("stability")
    if not run:
        raise ValueError("the stability file has no stability run")
    for key in SAME_SESSION_KEYS:
        if session["environment"][key] != sweep["environment"][key]:
            raise ValueError(f"the stability session's {key} ({session['environment'][key]}) "
                             f"differs from the sweep's ({sweep['environment'][key]})")
    if session["texts"] != sweep["texts"]:
        raise ValueError("the stability session's load differs from the sweep's")
    if run["offered_rate"] != sweep["sustainable_rate"]:
        raise ValueError(f"the stability run is at {run['offered_rate']}/s, not the sustainable "
                         f"rate ({sweep['sustainable_rate']}/s)")
    keys = ("offered_rate", "seconds", "sent", "ok", "errors", "achieved_rate", "latency_p50_s",
            "latency_p95_s", "latency_p99_s", "latency_mean_s", "gpu_util_mean", "keeps_up")
    return {**{k: run.get(k) for k in keys},
            "confirms_sustainable_rate": bool(run["keeps_up"]),
            "session_max_throughput": session["saturation"]["achieved_rate"],
            "server_ready_s": session["environment"].get("server_ready_s"),
            "session_hours": session.get("session_hours")}


def hourly_price(price: Mapping[str, Any]) -> dict[str, float]:
    """The GPU, the machine and their total per hour, from ``h3.price``."""
    m = price["machine"]
    gpu = price["gpu"]["per_hour"] * price["gpu"]["count"]
    machine = m["vcpus"] * m["per_vcpu_hour"] + m["memory_gib"] * m["per_gib_hour"]
    return {"gpu": round(gpu, 4), "machine": round(machine, 4), "total": round(gpu + machine, 4)}


def per_million(per_hour: float, rate: float) -> float:
    """Cost per million comments for ``per_hour`` of instances serving ``rate`` comments/s."""
    return per_hour / (rate * 3600) * 1e6


def at_rate(rate: float, sustainable: float, per_hour: float) -> dict[str, Any]:
    """Instances, instance-hours per hour and cost at a target ``rate`` (comments/s)."""
    if rate <= 0:
        raise ValueError(f"the target rate must be positive, got {rate}")
    instances = math.ceil(rate / sustainable)
    return {"target_rate": rate, "instances": instances,
            "instance_hours_per_hour": instances,
            "utilization": round(rate / (instances * sustainable), 4),
            "cost_per_hour": round(instances * per_hour, 4),
            "cost_per_million_comments": round(per_million(instances * per_hour, rate), 2)}


def build(sweep: Mapping[str, Any], price: Mapping[str, Any], target_rate: float | None,
          p95_target: float = 2.0,
          stability_session: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The report's content."""
    sustainable = sweep["sustainable_rate"]
    if not sustainable:
        raise ValueError("the sweep has no rate that kept up")
    hourly = hourly_price(price)
    per_hour = hourly["total"]
    runs = []
    for r in sweep["runs"]:
        sent_rate = r["sent"] / r["seconds"]
        row = {"offered_rate": r["offered_rate"], "sent_rate": round(sent_rate, 3),
               "achieved_rate": r["achieved_rate"],
               "latency_p50_s": r["latency_p50_s"], "latency_p95_s": r["latency_p95_s"],
               "latency_p99_s": r["latency_p99_s"], "errors": r["errors"],
               "gpu_util_mean": r.get("gpu_util_mean"), "keeps_up": r["keeps_up"]}
        # Poisson arrivals: a seed can send noticeably less than the nominal rate. If the
        # server still kept pace with what was sent, the miss is the generator's.
        generator_miss = (not r["keeps_up"] and r["errors"] == 0
                          and r["latency_p95_s"] <= p95_target
                          and r["achieved_rate"] >= 0.95 * sent_rate)
        if generator_miss:
            row["note"] = (f"the seeded arrivals sent {sent_rate:.2f}/s against "
                           f"{r['offered_rate']}/s nominal; the server kept pace with them")
        # At the nominal rate, as for the sustainable rate; none where one instance can't
        # serve that rate.
        row["cost_per_million_one_instance_at_this_rate"] = (
            round(per_million(per_hour, r["offered_rate"]), 2)
            if r["keeps_up"] or generator_miss else None)
        runs.append(row)
    at_sustainable = next(r for r in sweep["runs"] if r["offered_rate"] == sustainable)
    saturation = sweep["saturation"]
    stability = stability_section(sweep, stability_session) if stability_session else None
    sessions = [saturation["achieved_rate"]] + ([stability["session_max_throughput"]]
                                                if stability else [])
    duration = (DURATION_STABLE.format(minutes=stability["seconds"] / 60) if stability
                else DURATION_SHORT)
    return {
        "what": "the teacher (raw Qwen, P2 zero-shot, vLLM) as a service on one T4: throughput, "
                "latency, instances and cost per million comments",
        "teacher": {k: sweep["environment"][k]
                    for k in ("model_version", "prompt_version", "vllm_version", "gpu")},
        "code_commit": sweep["environment"]["code_commit"],
        "keeps_up_rule": sweep["keeps_up_rule"],
        "per_instance": {
            "sustainable_rate": sustainable,
            "latency_at_sustainable_s": {k: at_sustainable[f"latency_{k}_s"]
                                         for k in ("p50", "p95", "p99")},
            "max_throughput": saturation["achieved_rate"],
            "max_throughput_note": f"closed loop, {saturation['concurrency']} concurrent "
                                   f"requests; p95 {saturation['latency_p95_s']} s",
            "max_throughput_by_session": sessions,
            "comments_per_hour": round(sustainable * 3600),
            "cost_per_million_comments": round(per_million(per_hour, sustainable), 2),
            "cost_per_million_at_max_throughput": round(
                per_million(per_hour, saturation["achieved_rate"]), 2),
            "cost_per_month_always_on": round(per_hour * 730, 2),
        },
        "price": {"per_hour": hourly, "region": price["region"], "currency": price["currency"],
                  "gpu": price["gpu"]["name"], "machine": price["machine"]["name"],
                  "checked": price["checked"], "source": price["source"],
                  "skus": {"gpu": price["gpu"]["sku"], "vcpu": price["machine"]["vcpu_sku"],
                           "ram": price["machine"]["ram_sku"]}},
        "at_target_rate": (at_rate(target_rate, sustainable, per_hour)
                           if target_rate else None),
        "stability": stability,
        "runs": runs,
        "parity_with_offline": sweep["parity_with_offline"],
        "load": sweep["texts"],
        "caveats": [*CAVEATS[:3], duration, *CAVEATS[3:]],
    }


def load_ids(path: Path) -> dict[str, Any]:
    """Which comments the load sent, for the report: the ids file, its hash and how it was
    drawn (the file holds ``order``, ``id`` and ``batch``; no text)."""
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if [int(r["order"]) for r in rows] != list(range(len(rows))):
        raise ValueError(f"{path}: 'order' must run 0, 1, 2, ...")
    shown = path.resolve()
    shown = shown.relative_to(PIPELINE_ROOT) if shown.is_relative_to(PIPELINE_ROOT) else path
    return {"ids_file": shown.as_posix(), "ids": len(rows),
            "ids_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "selection": "select_drift_rows over the 20 drift batches (sorted by batch, then id), "
                         "then sample(n=2000, random_state=42); requests cycle through them in "
                         "this order"}


def run(cfg: Mapping[str, Any], target_rate: float | None = None) -> dict[str, Any]:
    h3 = cfg["h3"]
    sweep = json.loads((Path(cfg["paths"]["h3_dir"]) / h3["sweep_file"]).read_text())
    rate = target_rate if target_rate is not None else h3.get("target_rate")
    stability = (json.loads((Path(cfg["paths"]["h3_dir"]) / h3["stability_file"]).read_text())
                 if h3.get("stability_file") else None)
    report = build(sweep, h3["price"], rate, h3["p95_target_s"], stability)
    if h3.get("load_ids_file"):
        report["load"] = {**report["load"],
                          **load_ids(Path(cfg["paths"]["h3_dir"]) / h3["load_ids_file"])}
    out = Path(h3["report"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    p = report["per_instance"]
    logger.info("one T4 instance: %s comments/s, $%s/h, $%s per million comments; target: %s",
                p["sustainable_rate"], report["price"]["per_hour"]["total"],
                p["cost_per_million_comments"], report["at_target_rate"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="The teacher's serving cost for H3.")
    parser.add_argument("--rate", type=float, default=None,
                        help="target request rate, comments/s (default: h3.target_rate)")
    args = parser.parse_args()
    run(load_config(), args.rate)


if __name__ == "__main__":
    main()
