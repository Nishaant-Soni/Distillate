"""Results plots (work plan v4 §5): per-label AUC and PR-AUC across arms, per-label confusion
matrices at the tuned thresholds, and the H1 / H2 comparison.

PNG files for MLflow and the README. Each arm keeps one color in every chart (a fixed order,
never by rank). AUC is drawn as dots, since its differences (a few thousandths) are invisible
on bars from zero; PR-AUC as bars from zero.

    python -m src.eval.plots --run-dir outputs/student/<run> --out outputs/student/<run>/plots
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Arm order and colors (the reference categorical palette, light mode, in its fixed order).
ARM_ORDER = ("distilbert-gold", "distilbert-stack", "distilbert-mixed", "distilbert-tfidf",
             "distilbert-calqwen", "distilbert-mixed-5ep", "minilm-mixed", "modernbert-mixed")
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
           "#e34948")
OTHER = "#8a8984"
SPLIT_COLORS = {"val": PALETTE[0], "gate": PALETTE[1], "h2_eval": PALETTE[2]}
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SPLIT_TITLES = {"val": "Wikipedia validation", "gate": "Wikipedia gate set (H1)",
                "h2_eval": "Civil Comments H2 (five labels)", "bias_eval": "Civil bias set",
                "obfuscation": "Obfuscation set"}
STAMP = re.compile(r"-\d{8}T\d{6}Z$")


def arm_name(run_name: str) -> str:
    """``distilbert-mixed-20261112T101500Z`` -> ``distilbert-mixed``."""
    return STAMP.sub("", run_name)


def arm_color(arm: str) -> str:
    return PALETTE[ARM_ORDER.index(arm)] if arm in ARM_ORDER else OTHER


def ordered(arms) -> list[str]:
    known = [a for a in ARM_ORDER if a in arms]
    return known + sorted(a for a in arms if a not in ARM_ORDER)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    })
    return plt


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return str(path)


def per_label_dots(values: Mapping[str, Mapping[str, float]], labels: Sequence[str],
                   title: str, xlabel: str, path: Path,
                   colors: Mapping[str, str] | None = None) -> str:
    """One row per label, one dot per series (arm or split). For AUC."""
    plt = _plt()
    series = list(values)
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(labels) + 1.6))
    offsets = np.linspace(-0.25, 0.25, len(series)) if len(series) > 1 else [0.0]
    for off, name in zip(offsets, series, strict=True):
        ys = [i + off for i in range(len(labels))]
        xs = [values[name].get(label, np.nan) for label in labels]
        color = (colors or {}).get(name, arm_color(name))
        ax.scatter(xs, ys, s=42, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label=name)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    return _save(fig, path)


def per_label_bars(values: Mapping[str, Mapping[str, float]], labels: Sequence[str],
                   title: str, ylabel: str, path: Path,
                   colors: Mapping[str, str] | None = None) -> str:
    """Grouped bars from zero: labels on x, one bar per series. For PR-AUC."""
    plt = _plt()
    series = list(values)
    fig, ax = plt.subplots(figsize=(max(7.5, 1.2 * len(labels) + 0.25 * len(series)), 4.2))
    width = 0.8 / max(len(series), 1)
    x = np.arange(len(labels))
    for k, name in enumerate(series):
        heights = [values[name].get(label, np.nan) for label in labels]
        color = (colors or {}).get(name, arm_color(name))
        ax.bar(x - 0.4 + width * (k + 0.5), heights, width * 0.9, color=color, label=name,
               edgecolor=SURFACE, linewidth=1, zorder=3)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    return _save(fig, path)


def confusion_counts(y: np.ndarray, decision: np.ndarray) -> np.ndarray:
    """[[TN, FP], [FN, TP]]."""
    y, d = np.asarray(y).astype(bool), np.asarray(decision).astype(bool)
    return np.array([[np.sum(~y & ~d), np.sum(~y & d)], [np.sum(y & ~d), np.sum(y & d)]])


def confusion_grid(pred, gold, thresholds: Mapping[str, float], labels: Sequence[str],
                   title: str, path: Path) -> tuple[str, dict[str, list[list[int]]]]:
    """Per-label 2x2 confusion matrices at the tuned thresholds; shade = share of the row."""
    from matplotlib.colors import ListedColormap

    from src.eval.metrics import align

    plt = _plt()
    data = align(pred, gold, tuple(labels))
    cols = 3
    rows = int(np.ceil(len(labels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.1 * rows))
    axes = np.atleast_1d(axes).ravel()
    cmap = ListedColormap(SEQUENTIAL)
    out = {}
    for ax, label in zip(axes, labels, strict=False):
        m = confusion_counts(data.y[label], data.p[label] >= thresholds[label])
        out[label] = m.tolist()
        share = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ink = SURFACE if share[i, j] > 0.55 else INK
                ax.text(j, i, f"{m[i, j]:,}\n{share[i, j]:.1%}", ha="center", va="center",
                        color=ink, fontsize=9)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["gold 0", "gold 1"])
        ax.grid(False)
        ax.set_title(f"{label} (t = {thresholds[label]:.2f})", loc="left", fontsize=10)
    for ax in axes[len(labels):]:
        ax.axis("off")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path), out


def h1_h2(rows: Sequence[Mapping[str, Any]], path: Path) -> str:
    """Mean AUC and mean PR-AUC per arm on the gate set (H1) and Civil H2, with 95% CIs.

    ``rows``: ``{"arm", "gate": {"mean_auc": {"point", "ci95"}, ...}, "h2_eval": {...}}``.
    Each panel has its own x scale (two measures, two axes, never one dual axis).
    """
    plt = _plt()
    arms = [r["arm"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 0.42 * len(arms) * 2 + 2.4), sharey=True)
    for c, split in enumerate(("gate", "h2_eval")):
        for r, metric in enumerate(("mean_auc", "mean_pr_auc")):
            ax = axes[r, c]
            for i, row in enumerate(rows):
                cell = (row.get(split) or {}).get(metric)
                if not cell:
                    continue
                point = cell["point"]
                ci = [np.nan if v is None else v for v in (cell.get("ci95") or [None, None])]
                color = arm_color(row["arm"])
                if not np.isnan(ci[0]):
                    ax.plot(ci, [i, i], color=color, linewidth=2, solid_capstyle="round")
                ax.scatter([point], [i], s=48, color=color, edgecolor=SURFACE, linewidth=1.5,
                           zorder=3)
            ax.set_yticks(range(len(arms)), arms)
            ax.set_ylim(len(arms) - 0.5, -0.5)        # first arm on top
            ax.grid(axis="y", visible=False)
            name = "mean ROC-AUC" if metric == "mean_auc" else "mean PR-AUC"
            ax.set_title(f"{SPLIT_TITLES[split]}: {name}", loc="left", fontsize=10)
    fig.suptitle("H1 (in-domain) and H2 (cross-domain), 95% bootstrap intervals", x=0.01,
                 ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def run_plots(run_dir: str | Path, out: str | Path) -> dict[str, Any]:
    """One run's plots: confusion matrices on each split, per-label AUC and PR-AUC by split."""
    import json

    from src.eval.metrics import CIVIL_LABELS, LABELS, load_thresholds, read_frame

    run_dir, out = Path(run_dir), Path(out)
    metrics = json.loads((run_dir / "eval" / "metrics.json").read_text())
    name = arm_name(run_dir.name)
    files, matrices = [], {}
    for stem in ("val", "gate", "h2_eval"):
        pred_path, gold_path = run_dir / "predictions" / f"{stem}.parquet", \
            run_dir / "gold" / f"{stem}.parquet"
        if not (pred_path.exists() and gold_path.exists()):
            continue
        labels = CIVIL_LABELS if stem == "h2_eval" else LABELS
        th = load_thresholds(run_dir / "predictions" / "thresholds.json", labels)
        path, matrices[stem] = confusion_grid(read_frame(pred_path), read_frame(gold_path), th,
                                              labels, f"{name}: {SPLIT_TITLES[stem]}",
                                              out / f"confusion_{stem}.png")
        files.append(path)
    splits = [s for s in ("val", "gate", "h2_eval") if s in metrics]
    for metric, kind in (("auc", "dots"), ("pr_auc", "bars")):
        values = {s: {k: v[metric] for k, v in metrics[s]["per_label"].items()} for s in splits}
        title = f"{name}: per-label {'ROC-AUC' if metric == 'auc' else 'PR-AUC'} by split"
        fn = per_label_dots if kind == "dots" else per_label_bars
        files.append(fn(values, LABELS, title, "ROC-AUC" if metric == "auc" else "PR-AUC",
                        out / f"per_label_{metric}.png", SPLIT_COLORS))
    (out / "confusion_matrices.json").write_text(json.dumps(matrices, indent=2) + "\n")
    return {"files": files}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="One run's results plots.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    print(run_plots(args.run_dir, args.out))


if __name__ == "__main__":
    main()
