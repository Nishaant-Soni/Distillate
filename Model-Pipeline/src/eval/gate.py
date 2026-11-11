"""The model gate: pass or block a candidate (Ruchita).

Three checks, as in the work plan (section 5.1):

1. Validation thresholds: minimum mean AUC and minimum per-label PR-AUC on validation.
2. No regression: on the gate set, mean AUC and per-label PR-AUC may not fall more than a
   tolerance below the current production model. Skipped (and noted) when there is no
   production model yet.
3. Bias limit: on the Civil bias set, no identity slice's false positive rate may sit more
   than ``max_fpr_gap_pts`` above the overall rate, or rise more than ``max_fpr_rise_pts``
   above production's rate for the same slice. Slices under ``min_slice_size`` rows aren't
   judged.

Inputs are folders of prediction files named by split (``val.parquet``, ``gate.parquet``,
``bias.parquet``, as set in configs/gate.yaml) plus a folder of gold-label files with the
same names (``id``, the label columns, and the identity columns for the bias set; no text).
A ``thresholds.json`` next to the predictions sets per-label thresholds.

Writes gate_report.json and gate_report.md and exits 0 (pass) or 1 (blocked).

    python -m src.eval.gate --config configs/gate.yaml --candidate runs/cand/predictions \\
        --gold runs/cand/gold --production runs/prod/predictions --out runs/cand/gate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.eval.metrics import (
    CIVIL_LABELS,
    LABELS,
    align,
    confusion_rates,
    evaluate,
    load_thresholds,
    read_frame,
    to_jsonable,
)

# Civil Comments identity columns, grouped as in Data-Pipeline/scripts/bias.py.
IDENTITY_GROUPS: dict[str, list[str]] = {
    "gender": ["female", "male", "transgender", "other_gender"],
    "sexual_orientation": [
        "homosexual_gay_or_lesbian", "bisexual", "heterosexual", "other_sexual_orientation",
    ],
    "religion": [
        "christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion",
    ],
    "race_or_ethnicity": ["black", "white", "asian", "latino", "other_race_or_ethnicity"],
    "disability": [
        "physical_disability", "intellectual_or_learning_disability",
        "psychiatric_or_mental_illness", "other_disability",
    ],
}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _split_file(folder: str | Path, stem: str) -> Path:
    path = Path(folder) / f"{stem}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return path


def _thresholds(folder: str | Path | None, override: str | None,
                labels: tuple[str, ...]) -> dict[str, float]:
    """thresholds.json from ``override``, else next to the predictions, else 0.5."""
    if override:
        return load_thresholds(override, labels)
    if folder and (Path(folder) / "thresholds.json").exists():
        return load_thresholds(Path(folder) / "thresholds.json", labels)
    return load_thresholds(None, labels)


def _min_for(limit: float | dict[str, float], label: str) -> float:
    return float(limit[label]) if isinstance(limit, dict) else float(limit)


# ---------------------------------------------------------------------------
# Bias slices
# ---------------------------------------------------------------------------

def identity_slices(gold: pd.DataFrame, threshold: float) -> dict[str, np.ndarray]:
    """One boolean mask per identity attribute (>= threshold) and one per group (any)."""
    slices: dict[str, np.ndarray] = {}
    for group, columns in IDENTITY_GROUPS.items():
        present = [c for c in columns if c in gold.columns]
        if not present:
            continue
        any_mask = np.zeros(len(gold), dtype=bool)
        for column in present:
            hit = (pd.to_numeric(gold[column], errors="coerce").fillna(0.0)
                   >= threshold).to_numpy()
            slices[f"{group}:{column}"] = hit
            any_mask |= hit
        slices[f"{group}:any"] = any_mask
    return slices


def slice_fprs(pred: pd.DataFrame, gold: pd.DataFrame, label: str, threshold: float,
               identity_threshold: float) -> tuple[float, dict[str, dict[str, Any]]]:
    """Overall FPR and each identity slice's size and FPR for one label at its threshold."""
    gold = gold.assign(id=gold["id"].astype(str))
    data = align(pred, gold, (label,))
    # align keeps gold order but drops non-ok rows; rebuild the slice masks to match.
    ok_ids = pred.loc[pred["status"] == "ok", "id"].astype(str) if "status" in pred.columns \
        else pred["id"].astype(str)
    kept = gold[gold["id"].isin(set(ok_ids))].reset_index(drop=True)
    y, decision, w = data.y[label], data.p[label] >= threshold, data.w
    overall = confusion_rates(y, decision, w)["fpr"]
    out = {}
    for name, mask in identity_slices(kept, identity_threshold).items():
        n = int(mask.sum())
        fpr = confusion_rates(y[mask], decision[mask], w[mask])["fpr"] if n else math.nan
        out[name] = {"n": n, "fpr": fpr}
    return overall, out


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check(name: str, passed: bool | None, value: Any, limit: Any, detail: str) -> dict:
    return {"name": name, "passed": passed, "value": value, "limit": limit, "detail": detail}


def validation_checks(metrics: dict[str, Any], cfg: dict[str, Any]) -> list[dict]:
    checks = [_check("validation: mean AUC", metrics["mean_auc"] >= cfg["min_mean_auc"],
                     metrics["mean_auc"], cfg["min_mean_auc"], "at least the minimum")]
    for label, m in metrics["per_label"].items():
        limit = _min_for(cfg["min_pr_auc"], label)
        value = m["pr_auc"]
        passed = not math.isnan(value) and value >= limit
        checks.append(_check(f"validation: PR-AUC {label}", passed, value, limit,
                             "at least the minimum"))
    return checks


def regression_checks(cand: dict[str, Any], prod: dict[str, Any] | None,
                      cfg: dict[str, Any]) -> list[dict]:
    if prod is None:
        return [_check("gate set: no regression", None, None, None,
                       "skipped: no production model to compare against")]
    drop = cfg["max_mean_auc_drop"]
    floor = prod["mean_auc"] - drop
    checks = [_check("gate set: mean AUC vs production", cand["mean_auc"] >= floor,
                     cand["mean_auc"], floor,
                     f"production {prod['mean_auc']:.4f} minus {drop}")]
    for label, m in cand["per_label"].items():
        base = prod["per_label"][label]["pr_auc"]
        floor = base - _min_for(cfg["max_pr_auc_drop"], label)
        passed = not math.isnan(m["pr_auc"]) and m["pr_auc"] >= floor
        checks.append(_check(f"gate set: PR-AUC {label} vs production", passed, m["pr_auc"],
                             floor, f"production {base:.4f} minus tolerance"))
    return checks


def bias_checks(cand: tuple[float, dict], prod: tuple[float, dict] | None,
                cfg: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Returns (checks, slice table rows)."""
    overall, slices = cand
    prod_slices = prod[1] if prod else {}
    gap_limit, rise_limit = cfg.get("max_fpr_gap_pts"), cfg.get("max_fpr_rise_pts")
    rows, failures, judged = [], [], 0
    for name, s in slices.items():
        judge = s["n"] >= cfg["min_slice_size"] and not math.isnan(s["fpr"])
        gap = (s["fpr"] - overall) * 100 if not math.isnan(s["fpr"]) else math.nan
        prod_fpr = prod_slices.get(name, {}).get("fpr", math.nan)
        rise = (s["fpr"] - prod_fpr) * 100 if not math.isnan(prod_fpr) else math.nan
        reasons = []
        if judge:
            judged += 1
            if gap_limit is not None and gap > gap_limit:
                reasons.append(f"FPR {gap:+.1f} pts above overall (limit {gap_limit})")
            if rise_limit is not None and not math.isnan(rise) and rise > rise_limit:
                reasons.append(f"FPR up {rise:+.1f} pts vs production (limit {rise_limit})")
        rows.append({"slice": name, "n": s["n"], "fpr": s["fpr"], "gap_pts": gap,
                     "production_fpr": prod_fpr, "rise_pts": rise, "judged": judge,
                     "passed": (not reasons) if judge else None})
        if reasons:
            failures.append(_check(f"bias: {name}", False, s["fpr"], None, "; ".join(reasons)))
    summary = _check("bias: slice FPR limits", not failures, judged, None,
                     f"{judged} slices judged (n >= {cfg['min_slice_size']}), "
                     f"{len(failures)} over a limit")
    return [summary, *failures], rows


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_gate(cfg: dict[str, Any], candidate: str | Path, gold: str | Path,
             production: str | Path | None = None, thresholds: str | None = None,
             production_thresholds: str | None = None) -> dict[str, Any]:
    """Evaluate the candidate (and production, if given) and apply every check."""
    splits = cfg["splits"]
    designs = cfg.get("design") or {}
    report: dict[str, Any] = {"metrics": {}, "production_metrics": {}}

    def metrics_for(folder, role, labels, override):
        design_path = designs.get(role)
        design = read_frame(design_path) if design_path else None
        return evaluate(read_frame(_split_file(folder, splits[role])),
                        read_frame(_split_file(gold, splits[role])), labels,
                        _thresholds(folder, override, labels), design)

    val = metrics_for(candidate, "validation", LABELS, thresholds)
    gate = metrics_for(candidate, "gate", LABELS, thresholds)
    prod_gate = metrics_for(production, "gate", LABELS, production_thresholds) \
        if production else None
    report["metrics"].update(validation=val, gate=gate)
    report["production_metrics"]["gate"] = prod_gate
    report["model_version"] = val["model_version"]
    report["production_model_version"] = prod_gate["model_version"] if prod_gate else None

    bias_cfg = cfg["bias"]
    label = bias_cfg["label"]
    if label not in CIVIL_LABELS:
        raise ValueError(f"bias.label must be one of {CIVIL_LABELS}, got {label!r}")
    bias_gold = read_frame(_split_file(gold, splits["bias"]))

    def fprs(folder, override):
        th = _thresholds(folder, override, (label,))[label]
        return slice_fprs(read_frame(_split_file(folder, splits["bias"])), bias_gold, label,
                          th, bias_cfg["identity_threshold"])

    cand_bias = fprs(candidate, thresholds)
    prod_bias = fprs(production, production_thresholds) if production else None
    bias, slice_rows = bias_checks(cand_bias, prod_bias, bias_cfg)
    report["bias"] = {"label": label, "overall_fpr": cand_bias[0],
                      "production_overall_fpr": prod_bias[0] if prod_bias else None,
                      "slices": slice_rows}

    checks = [*validation_checks(val, cfg["validation"]),
              *regression_checks(gate, prod_gate, cfg["regression"]), *bias]
    report["checks"] = checks
    report["passed"] = all(c["passed"] is not False for c in checks)
    report["failed_checks"] = [c["name"] for c in checks if c["passed"] is False]
    return report


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def render_markdown(report: dict[str, Any]) -> str:
    verdict = "PASSED" if report["passed"] else "BLOCKED"
    lines = [f"# Gate report: {verdict}", "",
             f"Candidate `{report['model_version']}`; production "
             f"`{report['production_model_version'] or 'none'}`.", ""]
    if report["failed_checks"]:
        lines += ["Failed: " + ", ".join(f"`{n}`" for n in report["failed_checks"]), ""]
    lines += ["| Check | Result | Value | Limit | Detail |", "|---|---|---|---|---|"]
    for c in report["checks"]:
        result = {True: "pass", False: "**FAIL**", None: "skipped"}[c["passed"]]
        lines.append(f"| {c['name']} | {result} | {_fmt(c['value'])} | {_fmt(c['limit'])} "
                     f"| {c['detail']} |")
    for role in ("validation", "gate"):
        m = report["metrics"][role]
        lines += ["", f"## {role.capitalize()} metrics", "",
                  f"n = {m['n']}, mean AUC {_fmt(m['mean_auc'])}, mean PR-AUC "
                  f"{_fmt(m['mean_pr_auc'])}, macro-F1 {_fmt(m['macro_f1'])}, flag precision "
                  f"{_fmt(m['flag']['precision'])}, recall {_fmt(m['flag']['recall'])}, "
                  f"F1 {_fmt(m['flag']['f1'])}, FPR {_fmt(m['flag']['fpr'])}."]
    bias = report["bias"]
    lines += ["", f"## Bias slices (`{bias['label']}`, Civil bias set)", "",
              f"Overall FPR {_fmt(bias['overall_fpr'])}.", "",
              "| Slice | n | FPR | Gap (pts) | Production FPR | Rise (pts) | Result |",
              "|---|---|---|---|---|---|---|"]
    for r in bias["slices"]:
        if r["n"] == 0:
            continue
        result = {True: "pass", False: "**FAIL**", None: "too small"}[r["passed"]]
        lines.append(f"| {r['slice']} | {r['n']} | {_fmt(r['fpr'])} | {_fmt(r['gap_pts'], 1)} "
                     f"| {_fmt(r['production_fpr'])} | {_fmt(r['rise_pts'], 1)} | {result} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/gate.yaml")
    parser.add_argument("--candidate", required=True, help="folder of candidate predictions")
    parser.add_argument("--gold", required=True, help="folder of gold-label files")
    parser.add_argument("--production", default=None, help="folder of production predictions")
    parser.add_argument("--thresholds", default=None, help="candidate thresholds.json")
    parser.add_argument("--production-thresholds", default=None)
    parser.add_argument("--out", required=True, help="folder for gate_report.json/.md")
    args = parser.parse_args(argv)

    report = run_gate(load_config(args.config), args.candidate, args.gold, args.production,
                      args.thresholds, args.production_thresholds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_report.json").write_text(
        json.dumps(to_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "gate_report.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"Gate {'PASSED' if report['passed'] else 'BLOCKED'}: "
          f"{', '.join(report['failed_checks']) or 'all checks passed'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
