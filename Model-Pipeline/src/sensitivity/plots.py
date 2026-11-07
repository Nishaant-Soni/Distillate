"""The report's plots, as PNG files: one figure per analysis, the same numbers as its tables.

Colors are the six labels in a fixed order (validated for colour-vision deficiency; three sit
under 3:1 contrast on the surface, so every line also has its own marker, every figure a
legend, and the report the same numbers as a table). The mean over labels is dashed black.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.teacher.config import LABELS

SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
MARKERS = ("o", "s", "^", "D", "v", "P")
LABEL_STYLE = {label: (SERIES[k], MARKERS[k]) for k, label in enumerate(LABELS)}


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
        "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 2,
        "lines.markersize": 6,
    })
    return plt


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.clf()
    return path


def _label_lines(ax, table: pd.DataFrame, x: str, metric: str) -> None:
    """One line per label (colour and marker), and the mean dashed in ink."""
    for label in LABELS:
        rows = table[table["label"] == label].sort_values(x)
        if rows[metric].notna().any():
            color, marker = LABEL_STYLE[label]
            ax.plot(rows[x], rows[metric], color=color, marker=marker, label=label)
    mean = table[table["label"] == "mean"].sort_values(x)
    ax.plot(mean[x], mean[metric], color=INK, linestyle="--", linewidth=2.5, label="mean")


def _metric_panels(table: pd.DataFrame, x: str, metrics: Sequence[tuple[str, str]],
                   xlabel: str, title: str, path: Path, xticks=None) -> Path:
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
    for ax, (metric, name) in zip(axes, metrics, strict=True):
        _label_lines(ax, table, x, metric)
        ax.set_title(name)
        ax.set_xlabel(xlabel)
        if xticks is not None:
            ax.set_xticks(xticks)
    # One legend for every panel: a label with no line in the last panel (no positives caught,
    # say) is still in the others.
    handles = {}
    for ax in axes:
        for handle, name in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(name, handle)
    order = [name for name in (*LABELS, "mean") if name in handles]
    axes[-1].legend([handles[n] for n in order], order, loc="center left",
                    bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(title, color=INK, fontsize=11)
    return _save(fig, path)


def truncation_plot(overall: pd.DataFrame, path: Path) -> Path:
    lengths = sorted(overall["max_length"].unique())
    return _metric_panels(overall, "max_length", [("auc", "AUC"), ("pr_auc", "PR-AUC"),
                                                  ("recall", "Recall at threshold")],
                          "max_length (tokens)", "Validation metrics by truncation length",
                          path, xticks=lengths)


def bands_plot(bands: pd.DataFrame, path: Path) -> Path:
    """Mean AUC per token-length band, one line per max_length."""
    plt = _pyplot()
    mean = bands[bands["label"] == "mean"]
    order = list(dict.fromkeys(mean["band"]))
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for k, max_length in enumerate(sorted(mean["max_length"].unique())):
        rows = mean[mean["max_length"] == max_length].set_index("band").reindex(order)
        ax.plot(order, rows["auc"], color=SERIES[k], marker=MARKERS[k],
                label=f"max_length {max_length}")
    ax.set_xlabel("untruncated length (tokens)")
    ax.set_title("Mean AUC by comment length")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    return _save(fig, path)


def obfuscation_plot(table: pd.DataFrame, path: Path) -> Path:
    rates = sorted(table["rate"].unique())
    return _metric_panels(table, "rate", [("auc", "AUC"), ("pr_auc", "PR-AUC"),
                                          ("missed", "Caught when clean, missed now")],
                          "substitution rate", "Validation metrics by obfuscation strength",
                          path, xticks=rates)


def importance_plot(importances: pd.DataFrame, path: Path) -> Path:
    """Horizontal bars per parameter, one bar per evaluator."""
    plt = _pyplot()
    params = list(importances.index)[::-1]
    fig, ax = plt.subplots(figsize=(5.5, 0.5 * len(params) + 1.2))
    height = 0.8 / len(importances.columns)
    for k, evaluator in enumerate(importances.columns):
        # The first evaluator on top within each parameter, as in the legend.
        y = [i + ((len(importances.columns) - 1) / 2 - k) * height for i in range(len(params))]
        ax.barh(y, importances.loc[params, evaluator], height=height * 0.9, color=SERIES[k],
                label=evaluator, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(range(len(params)), params)
    ax.set_xlabel("importance (share of the objective's variance)")
    ax.set_title("Hyperparameter importance")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    return _save(fig, path)


def trials_plot(trials: pd.DataFrame, params: Sequence[str], path: Path,
                log_params: Sequence[str] = ()) -> Path:
    """The objective against each parameter, one small panel each (log axis where sampled so)."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(params), figsize=(3.0 * len(params), 3.0), sharey=True,
                             squeeze=False)
    for ax, param in zip(axes[0], params, strict=True):
        ax.scatter(trials[param], trials["value"], s=16, color=SERIES[0], alpha=0.8,
                   edgecolors=SURFACE, linewidths=0.5)
        if param in log_params:
            ax.set_xscale("log")
        ax.set_xlabel(param + (" (log)" if param in log_params else ""))
    axes[0][0].set_ylabel("objective")
    fig.suptitle("Objective by hyperparameter, completed trials", color=INK, fontsize=11)
    return _save(fig, path)


def words_plot(tokens: pd.DataFrame, group: str, path: Path, top: int = 10) -> Path:
    """Per label, the top words of one outcome group by mean SHAP value toward the label."""
    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, label in zip(axes.flat, LABELS, strict=True):
        rows = tokens[(tokens["label"] == label) & (tokens["group"] == group)].head(top)
        ax.set_title(label)
        if rows.empty:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "no words passed the filters", transform=ax.transAxes,
                    ha="center", color=MUTED)
            continue
        color, _ = LABEL_STYLE[label]
        ax.barh(rows["word"][::-1], rows["mean_value"][::-1], color=color, edgecolor=SURFACE)
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"Words pushing toward each label ({group.replace('_', ' ')}), mean SHAP "
                 "value on the logit", color=INK, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)
