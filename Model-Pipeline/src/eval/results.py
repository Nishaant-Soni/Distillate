"""Results across the ablation arms: the H1 / H2 tables, comparisons, robustness, and plots.

Reads run folders of ``src.pipeline.run`` (``predictions/``, ``gold/``, ``eval/metrics.json``,
``summary.json``, ``latency.json``) and writes to ``--out``:

* ``results.json`` and ``results.md``: per arm, the model card's H1 (gate set) and H2 (Civil,
  five labels) columns (mean AUC with a 95% bootstrap interval, mean PR-AUC, macro-F1, flag F1,
  flag FPR), validation numbers, the validation-to-gate drop (the soft-label leak check in
  docs/tuning.md), obfuscation recall against the same comments' originals, latency, and the
  truncation sweep where a run has one;
* paired bootstrap differences for the comparisons the work plan names (mixed vs gold; mixed
  vs TF-IDF, the part creditable to the LLM; calibrated Qwen vs TF-IDF; mixed vs calibrated
  Qwen; mixed vs stack, the loss) on the gate set and H2. The teacher comparisons pair arms
  trained with the same loss (gold plus soft labels at the tuned alpha), so they differ only in
  the soft labels; the stack-only arm (alpha 0) is compared with mixed, not with them;
* plots: per-label AUC (dots) and PR-AUC (bars) across arms per split, the H1 / H2 figure,
  and the selected arm's confusion matrices.

Metric definitions are ``src.eval.metrics`` (Ruchita's). Bootstraps run in parallel processes.

    python -m src.eval.results --runs outputs/student/* --out outputs/results --bootstrap 1000
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from src.eval.metrics import (
    CIVIL_LABELS,
    LABELS,
    bootstrap_ci,
    load_thresholds,
    paired_bootstrap,
    read_frame,
    to_jsonable,
)
from src.eval.plots import (
    SPLIT_TITLES,
    arm_name,
    confusion_grid,
    h1_h2,
    ordered,
    per_label_bars,
    per_label_dots,
)

# Each teacher comparison is like-for-like: both arms use the mixed loss at the tuned alpha.
PAIRS = (("distilbert-mixed", "distilbert-gold"), ("distilbert-mixed", "distilbert-tfidf"),
         ("distilbert-calqwen", "distilbert-tfidf"), ("distilbert-mixed", "distilbert-calqwen"),
         ("distilbert-mixed", "distilbert-stack"))
CI_SPLITS = ("gate", "h2_eval")


def labels_for(stem: str) -> tuple[str, ...]:
    return CIVIL_LABELS if stem in ("h2_eval", "bias_eval") else LABELS


def collect(runs: Sequence[str | Path]) -> dict[str, Path]:
    """Arm name -> run folder, for folders with metrics; the newest run wins per arm."""
    arms: dict[str, Path] = {}
    for run_dir in sorted(Path(r) for r in runs):
        if (run_dir / "eval" / "metrics.json").exists() and (run_dir / "summary.json").exists():
            arms[arm_name(run_dir.name)] = run_dir
    return {a: arms[a] for a in ordered(arms)}


def _files(run_dir: Path, stem: str):
    return (read_frame(run_dir / "predictions" / f"{stem}.parquet"),
            read_frame(run_dir / "gold" / f"{stem}.parquet"),
            load_thresholds(run_dir / "predictions" / "thresholds.json", labels_for(stem)))


def _ci_job(args):
    arm, run_dir, stem, n_boot = args
    pred, gold, th = _files(Path(run_dir), stem)
    return arm, stem, bootstrap_ci(pred, gold, labels_for(stem), th, n_boot=n_boot)


def _pair_job(args):
    a, b, dir_a, dir_b, stem, n_boot = args
    pred_a, gold, th_a = _files(Path(dir_a), stem)
    pred_b, _, th_b = _files(Path(dir_b), stem)
    return a, b, stem, paired_bootstrap(pred_a, pred_b, gold, labels_for(stem), th_a, th_b,
                                        n_boot=n_boot)


def obfuscation_recall(run_dir: Path) -> dict[str, Any] | None:
    """Recall on the obfuscated toxic comments vs the same comments' originals in the gate set."""
    import numpy as np

    if not (run_dir / "predictions" / "obfuscation.parquet").exists():
        return None
    obf, obf_gold, th = _files(run_dir, "obfuscation")
    gate = read_frame(run_dir / "predictions" / "gate.parquet")
    obf = obf.assign(original=obf["id"].astype(str).str.replace(r"_obf$", "", regex=True))
    merged = obf.merge(gate.assign(id=gate["id"].astype(str)), left_on="original",
                       right_on="id", suffixes=("_obf", "_orig"), how="inner")

    def rates(suffix):
        flag = np.zeros(len(merged), dtype=bool)
        for label in LABELS:
            flag |= merged[f"p_{label}_{suffix}"].to_numpy() >= th[label]
        tox = merged[f"p_toxicity_{suffix}"].to_numpy() >= th["toxicity"]
        return {"toxicity_recall": float(tox.mean()), "flag_recall": float(flag.mean())}

    orig, ob = rates("orig"), rates("obf")
    return {"rows": len(merged), "original": orig, "obfuscated": ob,
            "toxicity_recall_drop": orig["toxicity_recall"] - ob["toxicity_recall"],
            "flag_recall_drop": orig["flag_recall"] - ob["flag_recall"]}


def truncation(run_dir: Path) -> dict[str, Any] | None:
    from src.eval.metrics import evaluate

    folder = run_dir / "predictions" / "truncation"
    if not folder.exists():
        return None
    gold = read_frame(run_dir / "gold" / "gate.parquet")
    th = load_thresholds(run_dir / "predictions" / "thresholds.json", LABELS)
    out = {}
    for path in sorted(folder.glob("gate_len*.parquet"), key=lambda p: int(p.stem[8:])):
        m = evaluate(read_frame(path), gold, LABELS, th)
        out[path.stem[8:]] = {k: m[k] for k in ("mean_auc", "mean_pr_auc", "macro_f1")}
    return out


def build(runs: Sequence[str | Path], out: str | Path, n_boot: int = 1000,
          workers: int = 4, selected: str | None = None) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    arms = collect(runs)
    table: dict[str, Any] = {}
    for arm, run_dir in arms.items():
        metrics = json.loads((run_dir / "eval" / "metrics.json").read_text())
        summary = json.loads((run_dir / "summary.json").read_text())
        latency_path = run_dir / "latency.json"
        row = {"run_name": summary["run_name"], "model_version": summary["model_version"],
               "best_epoch": summary["best_epoch"],
               "train_minutes": round(summary["duration_s"] / 60, 1),
               "metrics": metrics,
               "latency": json.loads(latency_path.read_text()) if latency_path.exists() else None,
               "obfuscation": obfuscation_recall(run_dir), "truncation": truncation(run_dir)}
        if "val" in metrics and "gate" in metrics:
            row["val_to_gate_auc_drop"] = metrics["val"]["mean_auc"] - metrics["gate"]["mean_auc"]
            row["val_to_gate_pr_auc_drop"] = (metrics["val"]["mean_pr_auc"]
                                              - metrics["gate"]["mean_pr_auc"])
        table[arm] = row

    ci_jobs = [(a, str(d), s, n_boot) for a, d in arms.items() for s in CI_SPLITS
               if (d / "predictions" / f"{s}.parquet").exists()] if n_boot else []
    pair_jobs = [(a, b, str(arms[a]), str(arms[b]), s, n_boot) for a, b in PAIRS
                 if a in arms and b in arms for s in CI_SPLITS] if n_boot else []
    pairs: dict[str, Any] = {}
    if ci_jobs or pair_jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for arm, stem, ci in pool.map(_ci_job, ci_jobs):
                table[arm].setdefault("ci", {})[stem] = ci
            for a, b, stem, diff in pool.map(_pair_job, pair_jobs):
                pairs.setdefault(f"{a} - {b}", {})[stem] = diff

    leak = None
    if "distilbert-mixed" in table and "distilbert-gold" in table:
        m, g = table["distilbert-mixed"], table["distilbert-gold"]
        if "val_to_gate_auc_drop" in m and "val_to_gate_auc_drop" in g:
            leak = {"mixed_drop": m["val_to_gate_auc_drop"],
                    "gold_drop": g["val_to_gate_auc_drop"],
                    "difference": m["val_to_gate_auc_drop"] - g["val_to_gate_auc_drop"],
                    "pr_auc_difference": (m["val_to_gate_pr_auc_drop"]
                                          - g["val_to_gate_pr_auc_drop"])}
    result = to_jsonable({"arms": table, "pairs": pairs, "leak_check": leak,
                          "bootstrap": n_boot, "selected": selected})
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "results.md").write_text(render_markdown(result))
    result["plots"] = plots(result, arms, out / "plots", selected)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _point(row: Mapping[str, Any], stem: str, metric: str):
    ci = (row.get("ci") or {}).get(stem, {}).get(metric)
    if ci:
        return ci
    m = row["metrics"].get(stem)
    if m is None:
        return None
    value = m["flag"][metric[5:]] if metric.startswith("flag_") else m[metric]
    return {"point": value, "ci95": [math.nan, math.nan]}


def plots(result: Mapping[str, Any], arms: Mapping[str, Path], out: Path,
          selected: str | None) -> list[str]:
    files = []
    rows = result["arms"]
    for stem in ("val", "gate", "h2_eval"):
        present = [a for a in rows if stem in rows[a]["metrics"]]
        if not present:
            continue
        labels = labels_for(stem)
        for metric in ("auc", "pr_auc"):
            values = {a: {k: v[metric] for k, v in rows[a]["metrics"][stem]["per_label"].items()}
                      for a in present}
            name = "ROC-AUC" if metric == "auc" else "PR-AUC"
            title = f"Per-label {name}, {SPLIT_TITLES[stem]}"
            fn = per_label_dots if metric == "auc" else per_label_bars
            files.append(fn(values, labels, title, name, out / f"per_label_{metric}_{stem}.png"))
    h_rows = []
    for arm, row in rows.items():
        h_rows.append({"arm": arm, **{stem: {m: _point(row, stem, m) for m in (
            "mean_auc", "mean_pr_auc")} for stem in CI_SPLITS}})
    if h_rows:
        files.append(h1_h2(h_rows, out / "h1_h2.png"))
    if selected and selected in arms:
        for stem in ("gate", "h2_eval"):
            run_dir = arms[selected]
            if (run_dir / "predictions" / f"{stem}.parquet").exists():
                pred, gold, th = _files(run_dir, stem)
                path, _ = confusion_grid(pred, gold, th, labels_for(stem),
                                         f"{selected}: {SPLIT_TITLES[stem]}",
                                         out / f"confusion_{selected}_{stem}.png")
                files.append(path)
    return files


def _fmt(value, digits=4):
    return "-" if value is None or (isinstance(value, float) and math.isnan(value)) \
        else f"{value:.{digits}f}"


def render_markdown(result: Mapping[str, Any]) -> str:
    rows = result["arms"]
    lines = ["# Student results", "",
             f"Bootstrap: {result['bootstrap']} resamples (95% percentile intervals). "
             "Thresholds: each arm's own, chosen on validation.", ""]
    for stem, title in (("gate", "H1: in-domain (Wikipedia gate set)"),
                        ("h2_eval", "H2: cross-domain (Civil Comments, five labels)")):
        lines += [f"## {title}", "",
                  "| Arm | Mean AUC [95% CI] | Mean PR-AUC | Macro-F1 | Flag F1 | Flag FPR |",
                  "|---|---|---|---|---|---|"]
        for arm, row in rows.items():
            auc = _point(row, stem, "mean_auc")
            if auc is None:
                continue
            ci = auc["ci95"]
            ci_text = "" if ci[0] is None or (isinstance(ci[0], float) and math.isnan(ci[0])) \
                else f" [{ci[0]:.4f}, {ci[1]:.4f}]"
            vals = [_point(row, stem, m)["point"] for m in ("mean_pr_auc", "macro_f1",
                                                             "flag_f1", "flag_fpr")]
            lines.append(f"| {arm} | {auc['point']:.4f}{ci_text} | "
                         + " | ".join(_fmt(v) for v in vals) + " |")
        lines.append("")
    lines += ["## Validation, the leak check, robustness, latency", "",
              "| Arm | Val mean AUC | Val mean PR-AUC | Val-to-gate AUC drop | Obfuscation "
              "toxicity recall (orig -> obf) | Latency p50 / p95 (ms) | Best epoch (from 1) |",
              "|---|---|---|---|---|---|---|"]
    for arm, row in rows.items():
        val = row["metrics"].get("val", {})
        obf, lat = row.get("obfuscation"), row.get("latency")
        obf_text = (f"{obf['original']['toxicity_recall']:.3f} -> "
                    f"{obf['obfuscated']['toxicity_recall']:.3f}") if obf else "-"
        lat_text = f"{lat['p50_ms']:.1f} / {lat['p95_ms']:.1f}" if lat else "-"
        lines.append(f"| {arm} | {_fmt(val.get('mean_auc'), 5)} | {_fmt(val.get('mean_pr_auc'))} "
                     f"| {_fmt(row.get('val_to_gate_auc_drop'))} | {obf_text} | {lat_text} | "
                     f"{row['best_epoch'] + 1} |")
    leak = result.get("leak_check")
    if leak:
        lines += ["", f"**Leak check** (docs/tuning.md): validation-to-gate mean AUC drop "
                  f"{leak['mixed_drop']:.4f} for mixed vs {leak['gold_drop']:.4f} for gold "
                  f"(difference {leak['difference']:+.4f}; PR-AUC difference "
                  f"{leak['pr_auc_difference']:+.4f}). A clearly larger drop for mixed would "
                  f"mean the validation labels in the teacher's inputs matter."]
    if result.get("pairs"):
        lines += ["", "## Paired bootstrap differences (a - b)", "",
                  "| Comparison | Set | Mean AUC diff [95% CI] | P(diff > 0) | Mean PR-AUC diff "
                  "[95% CI] |", "|---|---|---|---|---|"]
        for name, by_stem in result["pairs"].items():
            for stem, d in by_stem.items():
                a, p = d["metrics"]["mean_auc"], d["metrics"]["mean_pr_auc"]
                lines.append(f"| {name} | {stem} | {a['diff']:+.4f} [{a['ci95'][0]:+.4f}, "
                             f"{a['ci95'][1]:+.4f}] | {a['p_gt_0']:.3f} | {p['diff']:+.4f} "
                             f"[{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] |")
    trunc = {a: r["truncation"] for a, r in rows.items() if r.get("truncation")}
    if trunc:
        lines += ["", "## Truncation (gate set, same model, shorter inputs)", "",
                  "| Arm | Max tokens | Mean AUC | Mean PR-AUC | Macro-F1 |",
                  "|---|---|---|---|---|"]
        for arm, by_len in trunc.items():
            for length, m in by_len.items():
                lines.append(f"| {arm} | {length} | {m['mean_auc']:.4f} | "
                             f"{m['mean_pr_auc']:.4f} | {m['macro_f1']:.4f} |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Results across the ablation arms.")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--selected", default=None, help="the selected arm's name")
    args = parser.parse_args(argv)
    result = build(args.runs, args.out, args.bootstrap, args.workers, args.selected)
    print(render_markdown(result))


if __name__ == "__main__":
    main()
