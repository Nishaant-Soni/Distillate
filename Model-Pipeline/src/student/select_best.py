"""The selection rule (work plan v4 §5): which trained student becomes the candidate.

Among the candidates that pass

1. the **bias check**: the bias part of ``src.eval.gate`` on the Civil bias set
   (``configs/gate.yaml``: no identity slice's FPR more than ``max_fpr_gap_pts`` above the
   overall rate; slices under ``min_slice_size`` aren't judged), and
2. the **CPU latency budget**: the package's batch-1 latency (``latency.json``, measured in
   the evaluation step), ``select.latency_stat`` at most ``select.latency_budget_ms``,

choose the highest **validation** mean AUC; candidates within ``select.auc_tie`` of it count
as tied and the higher validation mean PR-AUC wins. The gate set, H2 and the bias set's
accuracy are never used for choosing (only the bias set's slice FPRs, as the rule says).

``select.candidates`` (a list of arm names, or null for every run) limits who competes; the
other runs stay in the table as controls. With ``select.latency_enforced: false`` the budget
is reported for every candidate but does not exclude anyone (``select.note`` says why, and
goes into ``selection.md``).

Each candidate is a run folder of ``src.pipeline.run``: ``summary.json``, ``latency.json``,
``eval/gate_report.json`` (or ``predictions/`` and ``gold/`` to compute it).

    python -m src.student.select_best --runs outputs/student/* --out outputs/selection
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.student.config import load_config
from src.teacher.config import PIPELINE_ROOT, get_logger

logger = get_logger(__name__)


def candidate(run_dir: str | Path, gate_config: str | Path) -> dict[str, Any]:
    """What the rule needs from one run folder."""
    from src.eval.plots import arm_name
    from src.pipeline.run import bias_passed, gate_report

    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    report_path = run_dir / "eval" / "gate_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
    else:
        (run_dir / "eval").mkdir(exist_ok=True)
        report = gate_report(run_dir, Path(gate_config))
    latency_path = run_dir / "latency.json"
    latency = json.loads(latency_path.read_text()) if latency_path.exists() else None
    bias = report.get("bias", {})
    worst = max((s for s in bias.get("slices", []) if s.get("judged")),
                key=lambda s: s["gap_pts"] if s["gap_pts"] is not None else -math.inf,
                default=None)
    return {
        "arm": arm_name(summary["run_name"]), "run_name": summary["run_name"],
        "run_dir": str(run_dir), "model_version": summary["model_version"],
        "model_id": summary.get("model_id"),
        "val_mean_auc": float(summary["best_val_mean_auc"]),
        "val_mean_pr_auc": float(summary["best_val_mean_pr_auc"]),
        "bias_passed": bias_passed(report),
        "bias_worst_slice": {k: worst[k] for k in ("slice", "n", "fpr", "gap_pts")}
        if worst else None,
        "bias_overall_fpr": bias.get("overall_fpr"),
        "latency": {k: latency[k] for k in ("p50_ms", "p95_ms", "p99_ms", "threads", "cpu",
                                             "max_length")} if latency else None,
        "package_ok": (summary.get("package") or {}).get("status") == "ok",
        "gate_passed": report.get("passed"), "gate_failed_checks": report.get("failed_checks"),
        "mlflow_run_id": summary.get("mlflow_run_id"),
    }


def select(candidates: Sequence[Mapping[str, Any]], rule: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the rule; every candidate comes back with ``eligible`` and its reasons."""
    stat, budget = rule["latency_stat"], float(rule["latency_budget_ms"])
    tie = float(rule.get("auc_tie") or 0.0)
    enforced = bool(rule.get("latency_enforced", True))
    allowed = rule.get("candidates")
    allowed = None if allowed is None else [str(a) for a in allowed]
    judged = []
    for c in candidates:
        reasons, notes = [], []
        latency = notes if not enforced else reasons
        if allowed is not None and c["arm"] not in allowed:
            reasons.append("control arm, not a candidate (select.candidates)")
        if not c["bias_passed"]:
            reasons.append("bias check failed")
        if not c["package_ok"]:
            latency.append("no package (latency can't be measured)")
        if c["latency"] is None:
            latency.append("no latency measurement")
        elif c["latency"][stat] > budget:
            latency.append(f"latency {stat} {c['latency'][stat]:.1f} ms over {budget:g} ms"
                           + ("" if enforced else " (reported, not enforced)"))
        judged.append({**c, "eligible": not reasons, "reasons": reasons, "notes": notes,
                       "within_budget": c["latency"] is not None
                       and c["latency"][stat] <= budget})
    eligible = [c for c in judged if c["eligible"]]
    selected, tied = None, []
    if eligible:
        best_auc = max(c["val_mean_auc"] for c in eligible)
        tied = [c for c in eligible if c["val_mean_auc"] >= best_auc - tie]
        selected = max(tied, key=lambda c: (c["val_mean_pr_auc"], c["val_mean_auc"]))
    ranking = sorted(judged, key=lambda c: (not c["eligible"], -c["val_mean_auc"]))
    return {
        "rule": {"bias": "configs/gate.yaml bias section (slice FPR gap), no production model",
                 "latency": f"{stat} <= {budget:g} ms, batch size 1", "auc_tie": tie,
                 "latency_enforced": enforced, "candidates": allowed,
                 "order": "highest validation mean AUC; ties by validation mean PR-AUC",
                 "note": rule.get("note")},
        "selected": selected, "tied": [c["arm"] for c in tied], "candidates": ranking,
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    rule, sel = result["rule"], result["selected"]
    enforced = rule.get("latency_enforced", True)
    who = ("candidates ({})".format(", ".join(rule["candidates"])) if rule.get("candidates")
           else "candidates")
    budget = (f" and the CPU latency budget ({rule['latency']})" if enforced else "")
    lines = ["# Student selection", "",
             f"Rule: among {who} that pass the bias check ({rule['bias']}){budget}, "
             f"{rule['order']} (tie: within {rule['auc_tie']} AUC)."
             + ("" if enforced else f" The CPU latency budget ({rule['latency']}) is "
                "reported, not enforced."), ""]
    if rule.get("note"):
        lines += [f"Note: {' '.join(str(rule['note']).split())}", ""]
    if sel:
        lat = sel["latency"]
        speed = (f"latency p95 {lat['p95_ms']:.1f} ms on {lat['threads']} threads" if lat
                 else "no latency measurement")
        lines += [f"**Selected: `{sel['arm']}`** (`{sel['model_version']}`): validation mean "
                  f"AUC {sel['val_mean_auc']:.5f}, PR-AUC {sel['val_mean_pr_auc']:.4f}, "
                  f"{speed}.", ""]
        if len(result.get("tied") or []) > 1:
            lines += ["Tied within {} AUC: {}; the higher validation PR-AUC wins.".format(
                rule["auc_tie"], ", ".join(result["tied"])), ""]
        if not enforced and not sel.get("within_budget"):
            lines += ["**The selected model misses the latency budget** "
                      f"({rule['latency']}).", ""]
    else:
        lines += ["**No candidate passes the rule.**", ""]
    lines += ["| Arm | Val mean AUC | Val mean PR-AUC | Bias check | Worst slice gap (pts) | "
              "Latency p50 / p95 (ms) | Eligible | Reasons and notes |",
              "|---|---|---|---|---|---|---|---|"]
    for c in result["candidates"]:
        lat, worst = c["latency"], c["bias_worst_slice"]
        gap = "{:+.1f} ({})".format(worst["gap_pts"], worst["slice"]) if worst else "-"
        speed = "{:.1f} / {:.1f}".format(lat["p50_ms"], lat["p95_ms"]) if lat else "-"
        bias = "pass" if c["bias_passed"] else "fail"
        said = [*c["reasons"], *c.get("notes", [])]
        lines.append(
            f"| {c['arm']} | {c['val_mean_auc']:.5f} | {c['val_mean_pr_auc']:.4f} | {bias} | "
            f"{gap} | {speed} | {'yes' if c['eligible'] else 'no'} | "
            f"{'; '.join(said) or '-'} |")
    return "\n".join(lines) + "\n"


def run(runs: Sequence[str | Path], out: str | Path, cfg: Mapping[str, Any],
        log_mlflow: bool = False) -> dict[str, Any]:
    gate_config = Path(cfg["pipeline"]["gate_config"])
    gate_config = gate_config if gate_config.is_absolute() else PIPELINE_ROOT / gate_config
    candidates = []
    for run_dir in runs:
        if not (Path(run_dir) / "summary.json").exists():
            logger.warning("skipping %s: no summary.json", run_dir)
            continue
        candidates.append(candidate(run_dir, gate_config))
    result = select(candidates, cfg["select"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    (out / "selection.md").write_text(render_markdown(result))
    if log_mlflow:
        import mlflow

        from src.student.train import _mlflow_start

        with _mlflow_start(cfg, "selection"):
            mlflow.set_tags({"lane": "student", "step": "selection",
                             "selected": (result["selected"] or {}).get("arm", "none")})
            mlflow.log_params({f"select.{k}": (str(v)[:500] if v is not None else "null")
                               for k, v in cfg["select"].items()})
            if result["selected"]:
                mlflow.set_tag("selected_model_version", result["selected"]["model_version"])
                mlflow.log_metric("selected_val_mean_auc", result["selected"]["val_mean_auc"])
            mlflow.log_artifacts(str(out))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Choose the candidate student.")
    parser.add_argument("--runs", nargs="+", required=True, help="run folders")
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--mlflow", action="store_true", help="log the selection to MLflow")
    args = parser.parse_args(argv)
    result = run(args.runs, args.out, load_config(args.config, args.set), args.mlflow)
    print(render_markdown(result))
    return 0 if result["selected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
