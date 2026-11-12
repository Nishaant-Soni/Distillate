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

Writes bias_phase2.json, bias_tables.md (tables for the report) and, with --plots, one
per-slice FPR bar chart per label with the overall rate marked, before vs after when
``--before`` and ``--after`` name two arms.

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
    table["judged"] = table["n"] >= min_size
    table["flagged"] = table["judged"] & (table["fpr_gap_pts"] >= flag_pts)
    return table, overall


# ---------------------------------------------------------------------------
# Slices
# ---------------------------------------------------------------------------

def wikipedia_slices(gold: pd.DataFrame, length_bands: list[int]) -> dict[str, np.ndarray]:
    """Lexicon slices, when the gold file has the derived columns (else empty)."""
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

def analyze_arm(folder: str | Path, gold_dir: str | Path, settings: dict[str, Any],
                labels: tuple[str, ...] = ("toxicity",), bias_stem: str = "bias_eval",
                val_stem: str = "val") -> dict[str, Any]:
    """Civil identity tables for each label, plus Wikipedia lexicon tables when available."""
    folder, gold_dir = Path(folder), Path(gold_dir)
    thresholds_path = folder / "thresholds.json"
    out: dict[str, Any] = {"civil": {}, "wikipedia": {}, "model_version": None}
    sets = [("civil", bias_stem, labels),
            ("wikipedia", val_stem, tuple(lab for lab in labels if lab in LABELS))]
    for key, stem, set_labels in sets:
        gold_path, pred_path = gold_dir / f"{stem}.parquet", folder / f"{stem}.parquet"
        if not (gold_path.exists() and pred_path.exists()):
            continue
        pred, gold = read_frame(pred_path), read_frame(gold_path)
        kept = _kept_gold(pred, gold)
        slices = (identity_slices(kept, settings["identity_threshold"]) if key == "civil"
                  else wikipedia_slices(kept, settings["length_bands"]))
        if not slices:
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
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt(x: Any, digits: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def _pts(x: Any) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.1f}"


def markdown_table(table: pd.DataFrame, overall: dict[str, float]) -> str:
    head = ("| Slice | n | Toxic rate | Subgroup AUC | BPSN AUC | BNSP AUC | FPR | FPR gap (pts) "
            "| FNR | F1 | Status |\n|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [head, f"| **overall** | {overall['n']} | {_fmt(overall['toxic_rate'])} | "
                   f"{_fmt(overall['auc'])} |  |  | {_fmt(overall['fpr'])} |  | "
                   f"{_fmt(overall['fnr'])} | {_fmt(overall['f1'])} |  |"]
    for _, r in table[table["n"] > 0].iterrows():
        status = "too small" if not r["judged"] else ("**flagged**" if r["flagged"] else "ok")
        lines.append(f"| {r['slice']} | {r['n']} | {_fmt(r['toxic_rate'])} | "
                     f"{_fmt(r['subgroup_auc'])} | {_fmt(r['bpsn_auc'])} | "
                     f"{_fmt(r['bnsp_auc'])} | {_fmt(r['fpr'])} | {_pts(r['fpr_gap_pts'])} | "
                     f"{_fmt(r['fnr'])} | {_fmt(r['f1'])} | {status} |")
    return "\n".join(lines)


def comparison_table(results: dict[str, dict], label: str = "toxicity") -> str:
    """FPR gap and BPSN AUC per judged Civil slice, one column pair per arm."""
    arms = [a for a in results if label in results[a]["civil"]]
    if not arms:
        return ""
    base = results[arms[0]]["civil"][label]["table"]
    names = base.loc[base["judged"], "slice"].tolist()
    head = "| Slice | " + " | ".join(f"{a} FPR gap | {a} BPSN" for a in arms) + " |"
    lines = [head, "|---|" + "---|---|" * len(arms)]
    for name in names:
        cells = []
        for arm in arms:
            t = results[arm]["civil"][label]["table"].set_index("slice")
            flag = " ⚑" if name in t.index and bool(t.loc[name, "flagged"]) else ""
            gap = t.loc[name, "fpr_gap_pts"] if name in t.index else math.nan
            bpsn = t.loc[name, "bpsn_auc"] if name in t.index else math.nan
            cells.append(f"{_pts(gap)}{flag} | {_fmt(bpsn)}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_tables(results: dict[str, dict], settings: dict[str, Any]) -> str:
    parts = ["# Phase 2 bias tables (generated)", "",
             f"Slices with n >= {settings['min_slice_size']} are judged; flagged (⚑) when FPR "
             f"is {settings['report_flag_pts']:g}+ points above overall. FPR, FNR and F1 use "
             "each arm's tuned threshold from Wikipedia validation. Wikipedia length bands are "
             "in characters."]
    if len(results) > 1:
        parts += ["", "## Arms compared (Civil, toxicity)", "", comparison_table(results)]
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
    ax.legend(fontsize=8)
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
        for key in ("civil", "wikipedia"):
            out[arm][key] = {label: {"threshold": e["threshold"], "overall": e["overall"],
                                     "slices": json.loads(e["table"].to_json(orient="records"))}
                             for label, e in res[key].items()}
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
    for chosen in (args.before, args.after):
        if chosen and chosen not in arms:
            parser.error(f"{chosen!r} is not one of the arms {list(arms)}")

    results = {name: analyze_arm(folder, args.gold, settings, tuple(args.labels),
                                 splits.get("bias", "bias_eval"),
                                 splits.get("validation", "val"))
               for name, folder in arms.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bias_phase2.json").write_text(
        json.dumps(_as_json(results), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "bias_tables.md").write_text(render_tables(results, settings), encoding="utf-8")
    if args.plots:
        shown = [a for a in (args.before, args.after) if a] or list(arms)
        for label in args.labels:
            plot_fpr(results, out / "plots", shown, label)
    print(f"Wrote {out / 'bias_tables.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
