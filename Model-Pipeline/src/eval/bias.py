"""Phase 2 bias analysis for model candidates (Ruchita).

For each arm (a folder of prediction files plus ``thresholds.json``), on the Civil bias set:

* per identity attribute and per group (attribute >= 0.5; gender, sexual orientation,
  religion, race or ethnicity, disability): subgroup AUC, BPSN AUC and BNSP AUC
  (Borkan et al., 2019), plus FPR, FNR and F1 at the arm's tuned threshold, computed with a
  Fairlearn MetricFrame (in-slice vs rest);
* on Wikipedia validation, when the gold file carries the derived columns: the lexicon
  slices (identity-term mention, length band in characters, obfuscation flag).

A slice is judged when it has at least ``min_slice_size`` rows and flagged when its FPR is
``report_flag_pts`` or more points above overall (the Phase 1 rule). Comparing arms answers
whether distillation passes on or amplifies the teacher's bias: gold-only vs teacher-only
vs mixed students on the same slices.

Each arm runs at its own tuned threshold, and on Civil those thresholds give a low overall
FPR (about 2%), so a 5-point gap can hide a slice at several times the overall rate. The
tables therefore also show the FPR ratio (slice / overall), lead the arm comparison with
BPSN AUC (which doesn't depend on the threshold), and, with ``--matched-fpr``, compare every
arm at the threshold that gives the same overall Civil FPR. ``--compare A B`` adds a paired
bootstrap of the BPSN difference per slice, so a direction is only claimed with an interval.

Writes bias_phase2.json, bias_tables.md (tables for the report) and, with --plots, one
per-slice FPR bar chart per label with the overall rate marked, before vs after when
``--before`` and ``--after`` name two arms. Missing files or columns are reported as
warnings (printed, and listed at the top of bias_tables.md), never skipped silently.

    python -m src.eval.bias --gold <gold dir> --arm gold=runs/gold/predictions \\
        --arm mixed=runs/mixed/predictions --before mixed --after mixed_weighted \\
        --out reports/bias/phase2 --plots
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
from fairlearn.metrics import MetricFrame
from sklearn.metrics import roc_auc_score

from src.eval.gate import identity_slices, load_config
from src.eval.metrics import (
    CIVIL_LABELS,
    LABELS,
    align,
    confusion_rates,
    load_thresholds,
    read_frame,
    to_jsonable,
)

DEFAULTS = {"identity_threshold": 0.5, "min_slice_size": 100, "report_flag_pts": 5.0,
            "length_bands": [100, 500]}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else math.nan


def bias_aucs(y: np.ndarray, score: np.ndarray, in_slice: np.ndarray) -> dict[str, float]:
    """Subgroup, BPSN and BNSP AUC for one slice.

    * subgroup: rows in the slice only;
    * BPSN (background positive, subgroup negative): the slice's negatives with the rest's
      positives; low when the model scores non-toxic slice comments too high;
    * BNSP (background negative, subgroup positive): the slice's positives with the rest's
      negatives; low when the model scores toxic slice comments too low.
    """
    pos, neg = y == 1, y == 0
    bpsn = (in_slice & neg) | (~in_slice & pos)
    bnsp = (in_slice & pos) | (~in_slice & neg)
    return {"subgroup_auc": _auc(y[in_slice], score[in_slice]),
            "bpsn_auc": _auc(y[bpsn], score[bpsn]),
            "bnsp_auc": _auc(y[bnsp], score[bnsp])}


def _fpr(y_true, y_pred) -> float:
    return confusion_rates(np.asarray(y_true), np.asarray(y_pred), np.ones(len(y_true)))["fpr"]


def _fnr(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    if not (y_true == 1).any():
        return math.nan
    return 1.0 - confusion_rates(y_true, np.asarray(y_pred), np.ones(len(y_true)))["recall"]


def _f1(y_true, y_pred) -> float:
    return confusion_rates(np.asarray(y_true), np.asarray(y_pred), np.ones(len(y_true)))["f1"]


def slice_table(y: np.ndarray, score: np.ndarray, decision: np.ndarray,
                slices: dict[str, np.ndarray], min_size: int, flag_pts: float
                ) -> tuple[pd.DataFrame, dict[str, float]]:
    """One row per slice: size, toxic rate, AUCs, and FPR/FNR/F1 from a MetricFrame."""
    overall = {"n": int(len(y)), "toxic_rate": float(y.mean()),
               "auc": _auc(y, score), "fpr": _fpr(y, decision), "fnr": _fnr(y, decision),
               "f1": _f1(y, decision)}
    rows = []
    for name, mask in slices.items():
        n = int(mask.sum())
        row: dict[str, Any] = {"slice": name, "n": n}
        if n == 0:
            rows.append(row)
            continue
        groups = pd.Series(np.where(mask, "in", "out"), name="slice")
        frame = MetricFrame(metrics={"fpr": _fpr, "fnr": _fnr, "f1": _f1},
                            y_true=y, y_pred=decision, sensitive_features=groups)
        rates = frame.by_group.loc["in"]
        row.update({"toxic_rate": float(y[mask].mean()), **bias_aucs(y, score, mask),
                    "fpr": float(rates["fpr"]), "fnr": float(rates["fnr"]),
                    "f1": float(rates["f1"])})
        rows.append(row)
    table = pd.DataFrame(rows)
    for col in ("toxic_rate", "subgroup_auc", "bpsn_auc", "bnsp_auc", "fpr", "fnr", "f1"):
        if col not in table:
            table[col] = math.nan
    table["toxic_rate_gap_pts"] = (table["toxic_rate"] - overall["toxic_rate"]) * 100
    table["fpr_gap_pts"] = (table["fpr"] - overall["fpr"]) * 100
    table["fpr_ratio"] = (table["fpr"] / overall["fpr"]) if overall["fpr"] > 0 else math.nan
    table["judged"] = table["n"] >= min_size
    table["flagged"] = table["judged"] & (table["fpr_gap_pts"] >= flag_pts)
    return table, overall


# ---------------------------------------------------------------------------
# Slices
# ---------------------------------------------------------------------------

WIKI_COLUMNS = ("has_identity_term", "char_len", "has_obfuscation")


def wikipedia_slices(gold: pd.DataFrame, length_bands: list[int]) -> dict[str, np.ndarray]:
    """Lexicon slices from whichever derived columns the gold file has (else empty)."""
    slices: dict[str, np.ndarray] = {}
    if "has_identity_term" in gold:
        identity = gold["has_identity_term"].astype(bool).to_numpy()
        slices["identity_mention"] = identity
        slices["no_identity_mention"] = ~identity
    if "char_len" in gold:
        edges = [0, *sorted(length_bands)]
        length = gold["char_len"].to_numpy()
        for low, high in zip(edges, [*edges[1:], None], strict=True):
            name = f"chars_{low}+" if high is None else f"chars_{low}_{high - 1}"
            slices[name] = length >= low if high is None else (length >= low) & (length < high)
    if "has_obfuscation" in gold:
        obfuscated = gold["has_obfuscation"].astype(bool).to_numpy()
        slices["obfuscation_flag"] = obfuscated
        slices["no_obfuscation_flag"] = ~obfuscated
    return slices


def _kept_gold(pred: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Gold rows in align()'s order: gold order, minus rows not ``ok`` in the predictions."""
    gold = gold.assign(id=gold["id"].astype(str))
    if "status" in pred.columns:
        ok = set(pred.loc[pred["status"] == "ok", "id"].astype(str))
        gold = gold[gold["id"].isin(ok)]
    return gold.reset_index(drop=True)


# ---------------------------------------------------------------------------
# One arm
# ---------------------------------------------------------------------------

def matched_threshold(y: np.ndarray, p: np.ndarray, target_fpr: float) -> float:
    """The threshold at which this arm's overall FPR on these rows is about ``target_fpr``."""
    negatives = p[y == 0]
    return float(np.quantile(negatives, 1 - target_fpr)) if len(negatives) else math.nan


def analyze_arm(folder: str | Path, gold_dir: str | Path, settings: dict[str, Any],
                labels: tuple[str, ...] = ("toxicity",), bias_stem: str = "bias_eval",
                val_stem: str = "val", matched_fpr: float | None = None) -> dict[str, Any]:
    """Civil identity tables for each label, plus Wikipedia lexicon tables when available.

    With ``matched_fpr``, also Civil tables at each label's matched threshold (for comparing
    arms only; deployment thresholds stay the ones tuned on Wikipedia validation).
    """
    folder, gold_dir = Path(folder), Path(gold_dir)
    thresholds_path = folder / "thresholds.json"
    out: dict[str, Any] = {"civil": {}, "wikipedia": {}, "civil_matched": {},
                           "model_version": None, "warnings": []}
    if not thresholds_path.exists():
        out["warnings"].append(f"{folder.name}: no thresholds.json, so 0.5 is used")
    sets = [("civil", bias_stem, labels),
            ("wikipedia", val_stem, tuple(lab for lab in labels if lab in LABELS))]
    for key, stem, set_labels in sets:
        gold_path, pred_path = gold_dir / f"{stem}.parquet", folder / f"{stem}.parquet"
        missing_files = [str(f) for f in (gold_path, pred_path) if not f.exists()]
        if missing_files:
            out["warnings"].append(f"{key} section skipped: missing {', '.join(missing_files)}")
            continue
        pred, gold = read_frame(pred_path), read_frame(gold_path)
        kept = _kept_gold(pred, gold)
        if key == "wikipedia":
            absent = [c for c in WIKI_COLUMNS if c not in gold.columns]
            if absent:
                out["warnings"].append(
                    f"wikipedia: {gold_path.name} has no {', '.join(absent)}; those slices are "
                    "skipped (load_data's validation table has them)")
        slices = (identity_slices(kept, settings["identity_threshold"]) if key == "civil"
                  else wikipedia_slices(kept, settings["length_bands"]))
        if not slices:
            out["warnings"].append(f"{key} section skipped: no slices could be built")
            continue
        thresholds = load_thresholds(thresholds_path if thresholds_path.exists() else None,
                                     set_labels)
        data = align(pred, gold, set_labels)
        out["model_version"] = out["model_version"] or data.model_version
        for label in set_labels:
            table, overall = slice_table(
                data.y[label], data.score[label], data.p[label] >= thresholds[label],
                slices, settings["min_slice_size"], settings["report_flag_pts"])
            out[key][label] = {"threshold": thresholds[label], "overall": overall,
                               "table": table}
            if key == "civil" and matched_fpr is not None:
                th = matched_threshold(data.y[label], data.p[label], matched_fpr)
                m_table, m_overall = slice_table(
                    data.y[label], data.score[label], data.p[label] > th, slices,
                    settings["min_slice_size"], settings["report_flag_pts"])
                out["civil_matched"][label] = {"threshold": th, "target_fpr": matched_fpr,
                                               "overall": m_overall, "table": m_table}
    return out


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------

def _rank_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties (equal to roc_auc_score, faster)."""
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def paired_bpsn(folder_a: str | Path, folder_b: str | Path, gold_dir: str | Path,
                settings: dict[str, Any], label: str = "toxicity",
                bias_stem: str = "bias_eval", n_boot: int = 1000,
                seed: int = 42) -> dict[str, Any]:
    """Paired bootstrap of BPSN AUC (a minus b) per judged Civil slice.

    Both arms are scored on the same rows; each replicate resamples the bias set once and
    recomputes both arms' BPSN on it. BPSN doesn't depend on either arm's threshold.
    """
    pred_a = read_frame(Path(folder_a) / f"{bias_stem}.parquet")
    pred_b = read_frame(Path(folder_b) / f"{bias_stem}.parquet")
    gold = read_frame(Path(gold_dir) / f"{bias_stem}.parquet")
    ids = set(gold["id"].astype(str))
    for pred in (pred_a, pred_b):
        if "status" in pred.columns:
            ids &= set(pred.loc[pred["status"] == "ok", "id"].astype(str))
    gold = gold.assign(id=gold["id"].astype(str))
    gold = gold[gold["id"].isin(ids)].reset_index(drop=True)
    a, b = align(pred_a, gold, (label,)), align(pred_b, gold, (label,))
    y = a.y[label]
    slices = identity_slices(gold, settings["identity_threshold"])
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    out = {}
    for name, mask in slices.items():
        if mask.sum() < settings["min_slice_size"]:
            continue
        def bpsn(idx, score, mask=mask):
            yy, mm = y[idx], mask[idx]
            keep = (mm & (yy == 0)) | (~mm & (yy == 1))
            return _rank_auc(yy[keep], score[idx][keep])
        full = np.arange(len(y))
        point = bpsn(full, a.score[label]) - bpsn(full, b.score[label])
        diffs = np.array([bpsn(i, a.score[label]) - bpsn(i, b.score[label]) for i in draws])
        diffs = diffs[~np.isnan(diffs)]
        lo, hi = np.percentile(diffs, [2.5, 97.5]) if len(diffs) else (math.nan, math.nan)
        out[name] = {"n": int(mask.sum()), "bpsn_a": bpsn(full, a.score[label]),
                     "bpsn_b": bpsn(full, b.score[label]), "diff": float(point),
                     "ci95": [float(lo), float(hi)],
                     "excludes_zero": bool(lo > 0 or hi < 0)}
    return {"label": label, "replicates": n_boot, "rows": len(y), "slices": out}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt(x: Any, digits: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def _pts(x: Any) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.1f}"


def _ratio(x: Any) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1f}x"


def markdown_table(table: pd.DataFrame, overall: dict[str, float]) -> str:
    head = ("| Slice | n | Toxic rate | Subgroup AUC | BPSN AUC | BNSP AUC | FPR | FPR ratio "
            "| FPR gap (pts) | FNR | F1 | Status |\n" + "|---" * 12 + "|")
    lines = [head, f"| **overall** | {overall['n']} | {_fmt(overall['toxic_rate'])} | "
                   f"{_fmt(overall['auc'])} |  |  | {_fmt(overall['fpr'])} | 1.0x |  | "
                   f"{_fmt(overall['fnr'])} | {_fmt(overall['f1'])} |  |"]
    for _, r in table[table["n"] > 0].iterrows():
        status = "too small" if not r["judged"] else ("**flagged**" if r["flagged"] else "ok")
        lines.append(f"| {r['slice']} | {r['n']} | {_fmt(r['toxic_rate'])} | "
                     f"{_fmt(r['subgroup_auc'])} | {_fmt(r['bpsn_auc'])} | "
                     f"{_fmt(r['bnsp_auc'])} | {_fmt(r['fpr'])} | {_ratio(r['fpr_ratio'])} | "
                     f"{_pts(r['fpr_gap_pts'])} | {_fmt(r['fnr'])} | {_fmt(r['f1'])} | "
                     f"{status} |")
    return "\n".join(lines)


def comparison_table(results: dict[str, dict], label: str = "toxicity",
                     key: str = "civil") -> str:
    """Per judged Civil slice and arm: BPSN AUC first (threshold-free), then FPR ratio and gap."""
    arms = [a for a in results if label in results[a][key]]
    if not arms:
        return ""
    base = results[arms[0]][key][label]["table"]
    names = base.loc[base["judged"], "slice"].tolist()
    cols = " | ".join(f"{a} BPSN | {a} FPR ratio | {a} FPR gap" for a in arms)
    head = f"| Slice | {cols} |"
    lines = [head, "|---|" + "---|---|---|" * len(arms)]
    for name in names:
        cells = []
        for arm in arms:
            t = results[arm][key][label]["table"].set_index("slice")
            if name not in t.index:
                cells.append("n/a | n/a | n/a")
                continue
            flag = " ⚑" if bool(t.loc[name, "flagged"]) else ""
            cells.append(f"{_fmt(t.loc[name, 'bpsn_auc'])} | {_ratio(t.loc[name, 'fpr_ratio'])}"
                         f" | {_pts(t.loc[name, 'fpr_gap_pts'])}{flag}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paired_table(paired: dict[str, Any], arm_a: str, arm_b: str) -> str:
    lines = [f"| Slice | n | {arm_a} BPSN | {arm_b} BPSN | Difference [95% CI] | "
             "Interval excludes 0 |", "|---|---|---|---|---|---|"]
    for name, r in paired["slices"].items():
        lo, hi = r["ci95"]
        lines.append(f"| {name} | {r['n']} | {_fmt(r['bpsn_a'])} | {_fmt(r['bpsn_b'])} | "
                     f"{r['diff']:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                     f"{'yes' if r['excludes_zero'] else 'no'} |")
    return "\n".join(lines)


def render_tables(results: dict[str, dict], settings: dict[str, Any],
                  paired: list[tuple[str, str, dict]] | None = None) -> str:
    parts = ["# Phase 2 bias tables (generated)", "",
             f"Slices with n >= {settings['min_slice_size']} are judged; flagged (⚑) when FPR "
             f"is {settings['report_flag_pts']:g}+ points above overall. FPR, FNR and F1 use "
             "each arm's tuned threshold from Wikipedia validation; on Civil that threshold "
             "gives a low overall FPR, so read the FPR ratio (slice / overall) alongside the "
             "gap. BPSN AUC doesn't depend on the threshold, so it leads the arm comparison. "
             "Wikipedia length bands are in characters."]
    warnings = [w for res in results.values() for w in res.get("warnings", [])]
    if warnings:
        parts += ["", "## Warnings", "", *[f"- {w}" for w in dict.fromkeys(warnings)]]
    if len(results) > 1:
        parts += ["", "## Arms compared (Civil, toxicity, tuned thresholds)", "",
                  comparison_table(results)]
    if any(res.get("civil_matched") for res in results.values()):
        target = next(res["civil_matched"]["toxicity"]["target_fpr"] for res in
                      results.values() if "toxicity" in res.get("civil_matched", {}))
        parts += ["", f"## Arms compared at matched overall FPR ({target:.0%})", "",
                  "Each arm at the threshold giving the same overall Civil FPR, for comparing "
                  "arms only (not a deployment threshold).", "",
                  comparison_table(results, key="civil_matched")]
    for arm_a, arm_b, result in paired or []:
        parts += ["", f"## Paired bootstrap: BPSN AUC, {arm_a} minus {arm_b} "
                      f"({result['replicates']} replicates)", "",
                  paired_table(result, arm_a, arm_b)]
    for arm, res in results.items():
        parts += ["", f"## {arm} (`{res.get('model_version')}`)"]
        for key, title in (("civil", "Civil bias set"), ("wikipedia", "Wikipedia validation")):
            for label, entry in res[key].items():
                parts += ["", f"### {title}: {label} (threshold {entry['threshold']:.2f})", "",
                          markdown_table(entry["table"], entry["overall"])]
    return "\n".join(parts) + "\n"


def plot_fpr(results: dict[str, dict], out_dir: Path, arms: list[str],
             label: str = "toxicity") -> Path | None:
    """Per-slice FPR bars for the judged Civil slices, overall FPR marked, one bar per arm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [a for a in arms if label in results[a]["civil"]]
    if not arms:
        return None
    base = results[arms[0]]["civil"][label]["table"]
    names = base.loc[base["judged"], "slice"].tolist()
    if not names:
        return None
    x = np.arange(len(names))
    width = 0.8 / len(arms)
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.6), 5))
    for i, arm in enumerate(arms):
        t = results[arm]["civil"][label]["table"].set_index("slice")
        values = [t.loc[n, "fpr"] if n in t.index else np.nan for n in names]
        bars = ax.bar(x + i * width - 0.4 + width / 2, values, width, label=arm)
        ax.axhline(results[arm]["civil"][label]["overall"]["fpr"], linestyle="--",
                   linewidth=1, color=bars.patches[0].get_facecolor(),
                   label=f"{arm} overall FPR")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(f"False positive rate ({label})")
    ax.set_title(f"Civil identity slices: FPR at the tuned threshold ({', '.join(arms)})")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"fpr_by_slice_{label}_{'_vs_'.join(arms)}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _as_json(results: dict[str, dict]) -> dict[str, Any]:
    out = {}
    for arm, res in results.items():
        out[arm] = {"model_version": res.get("model_version")}
        out[arm]["warnings"] = res.get("warnings", [])
        for key in ("civil", "wikipedia", "civil_matched"):
            out[arm][key] = {label: {"threshold": e["threshold"], "overall": e["overall"],
                                     "slices": json.loads(e["table"].to_json(orient="records"))}
                             for label, e in res.get(key, {}).items()}
    return to_jsonable(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", required=True, help="folder with bias_eval/val gold files")
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR",
                        help="an arm's prediction folder (repeatable)")
    parser.add_argument("--labels", nargs="+", default=["toxicity"],
                        help=f"labels to analyse (Civil: any of {', '.join(CIVIL_LABELS)})")
    parser.add_argument("--config", default="configs/gate.yaml",
                        help="reads splits and the bias section's slice settings")
    parser.add_argument("--before", default=None, help="arm shown first in the plot")
    parser.add_argument("--after", default=None, help="arm shown second in the plot")
    parser.add_argument("--plots", action="store_true", help="write FPR bar charts")
    parser.add_argument("--matched-fpr", type=float, default=None, metavar="RATE",
                        help="also compare arms at this overall Civil FPR, e.g. 0.05")
    parser.add_argument("--compare", nargs=2, action="append", metavar=("A", "B"),
                        help="paired bootstrap of BPSN AUC, A minus B (repeatable)")
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap replicates")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    bad = [lab for lab in args.labels if lab not in CIVIL_LABELS]
    if bad:
        parser.error(f"not Civil labels: {bad} (severe_toxicity is excluded on Civil)")
    cfg = load_config(args.config) if Path(args.config).exists() else {}
    settings = {**DEFAULTS, **{k: v for k, v in cfg.get("bias", {}).items() if k in DEFAULTS}}
    splits = cfg.get("splits", {})
    arms = {}
    for spec in args.arm:
        name, _, folder = spec.partition("=")
        if not folder:
            parser.error(f"--arm needs NAME=DIR, got {spec!r}")
        arms[name] = folder
    compared = [name for pair in (args.compare or []) for name in pair]
    for chosen in (args.before, args.after, *compared):
        if chosen and chosen not in arms:
            parser.error(f"{chosen!r} is not one of the arms {list(arms)}")
    if args.matched_fpr is not None and not 0 < args.matched_fpr < 1:
        parser.error("--matched-fpr must be between 0 and 1")

    bias_stem = splits.get("bias", "bias_eval")
    results = {name: analyze_arm(folder, args.gold, settings, tuple(args.labels), bias_stem,
                                 splits.get("validation", "val"), args.matched_fpr)
               for name, folder in arms.items()}
    for warning in dict.fromkeys(w for res in results.values() for w in res["warnings"]):
        print(f"WARNING: {warning}", file=sys.stderr)
    paired = [(a, b, paired_bpsn(arms[a], arms[b], args.gold, settings, "toxicity",
                                 bias_stem, n_boot=args.n_boot))
              for a, b in (args.compare or [])]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = _as_json(results)
    if paired:
        payload["paired_bpsn"] = to_jsonable({f"{a}-{b}": r for a, b, r in paired})
    (out / "bias_phase2.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "bias_tables.md").write_text(render_tables(results, settings, paired),
                                        encoding="utf-8")
    if args.plots:
        shown = [a for a in (args.before, args.after) if a] or list(arms)
        for label in args.labels:
            plot_fpr(results, out / "plots", shown, label)
    print(f"Wrote {out / 'bias_tables.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
