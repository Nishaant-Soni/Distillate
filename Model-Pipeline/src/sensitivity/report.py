"""``reports/sensitivity.md`` from a run's results: numbers, plots and masked examples only.

Every number in it comes from ``results.json`` next to the plots, so the report can be rebuilt
without rerunning the model. What the numbers mean is written by hand in the findings file,
which is included as it is.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.teacher.config import LABELS

GROUP_NAMES = {"caught": "caught (correct)", "missed": "missed (incorrect)",
               "false_alarm": "false alarm (incorrect)"}


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def md_table(rows: Sequence[Sequence[Any]], header: Sequence[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _wide(table: pd.DataFrame, column: str, metric: str) -> str:
    """Labels as rows, one column per value of ``column``."""
    pivot = table.pivot(index="label", columns=column, values=metric).reindex([*LABELS, "mean"])
    return md_table([[label, *row] for label, row in zip(pivot.index, pivot.to_numpy().tolist(),
                                                        strict=True)],
                    ["label", *[f"{c:g}" for c in pivot.columns]])


def _data_source(setup: Mapping[str, Any]) -> str:
    if setup["data_source"] == "load_data":
        where = f"DVC revision `{setup['data_rev']}`" if setup["data_rev"] else "files on disk"
        return f"`load_data`, {where}"
    if setup["data_source"] == "file":
        return "a Parquet file (`data.val_file`)"
    return "given in memory"


def setup_section(results: Mapping[str, Any]) -> str:
    setup = results["setup"]
    positives = ", ".join(f"{k} {v}" for k, v in setup["positives"].items())
    rows = [["model_version", f"`{setup['model_version']}`"],
            ["max_length (the model's own)", setup["max_length"]],
            ["thresholds", setup["thresholds_source"]],
            ["validation rows", f"{setup['rows']:,} ({positives})"],
            ["validation data", _data_source(setup)],
            ["validation file sha256", f"`{setup['data_sha256'][:16]}…`"],
            ["code commit", f"`{setup['code_commit']}`"], ["seed", setup["seed"]]]
    return "## Setup\n\n" + md_table(rows, ["", "value"]) + (
        "\n\nAll results are on **validation** (split roles: the gate set is for final numbers "
        "only). Metrics are unweighted. AUC and PR-AUC rank on the logits, because float32 "
        "probabilities tie at 1.0; ECE, Brier, recall and flag rate use the probabilities.")


def hyperparameter_section(results: Mapping[str, Any], plots: Mapping[str, str]) -> str:
    out = "## 1. Hyperparameter importance\n\n"
    hyper = results.get("hyperparameters")
    if hyper is None:
        return out + f"Not run: {results['skipped']['hyperparameters']}."
    s = hyper["summary"]
    importances = pd.DataFrame(hyper["importances"]).set_index("param")
    out += (f"Study `{s['study_name']}`: {s['completed']} completed trials of {s['trials']}, "
            f"objective {s['direction']}d. Best trial {s['best_trial']}: "
            f"{fmt(float(s['best_value']), 4)} with "
            # Significant digits, not decimals: a learning rate of 7.9e-5 isn't 0.0001.
            + ", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}"
                        for k, v in s["best_params"].items()) + ".\n\n")
    spread = hyper.get("stability")
    if spread is None:
        out += md_table([[p, *row] for p, row in zip(
            importances.index, importances.to_numpy().tolist(), strict=True)],
            ["parameter", *importances.columns])
    else:
        ranges = pd.DataFrame(spread["ranges"]).set_index("param")
        evaluators = list(importances.columns)

        def with_range(param: str, evaluator: str) -> str:
            """The value and its range over the seeds: "0.548 (0.367 to 0.548)"."""
            low, high = ranges.loc[param, f"{evaluator}_min"], ranges.loc[param, f"{evaluator}_max"]
            return f"{fmt(importances.loc[param, evaluator])} ({fmt(low)} to {fmt(high)})"

        rows = [[p, *(with_range(p, e) for e in evaluators)] for p in importances.index]
        out += md_table(rows, ["parameter", *(f"{e} (range over seeds)" for e in evaluators)])
        seeds, first = spread["seeds"], importances.index[0]
        counts = " and ".join(f"{spread['top'][e].get(first, 0)} of {len(seeds)} ({e})"
                              for e in evaluators)
        out += (f"\n\nThe ranges come from recomputing both evaluators under {len(seeds)} seeds "
                f"({seeds[0]} to {seeds[-1]}, the first being the one above). `{first}` is "
                f"ranked first in {counts}.")
    out += ("\n\nfANOVA is Optuna's default; MDI (a random forest's mean decrease in impurity) "
            "is an independent check on the ranking. Only parameters sampled in every "
            "completed trial are compared.\n\n")
    return out + f"![Hyperparameter importance]({plots['importance']})\n\n" \
                 f"![Objective by hyperparameter]({plots['trials']})"


def shap_section(results: Mapping[str, Any], plots: Mapping[str, str]) -> str:
    shap = results["shap"]
    cfg = shap["config"]
    counts = pd.DataFrame(shap["counts"], columns=["label", "group", "n"])
    tokens = pd.DataFrame(shap["tokens"], columns=["label", "group", "word", "mean_value"])
    examples = pd.DataFrame(shap["examples"], columns=["label", "group", "p", "y", "masked"])
    out = ("## 2. Token attributions (SHAP)\n\n"
           f"SHAP's partition explainer with the text masker ({cfg['max_evals']} evaluations "
           "per comment), on the logits. Per label, up to "
           f"{cfg['per_group']} validation comments of each outcome at the threshold: caught "
           "(toxic, flagged), missed (toxic, not flagged) and false alarm (flagged, not "
           f"toxic); {shap['explained']} comments explained in all. Word pieces are merged "
           "into words. No comment text is shown: a word appears only if it occurs in at "
           f"least {cfg['min_doc_freq']} validation comments, and in the examples only the "
           f"top {cfg['visible_words']} words are visible, never more than a third of a "
           "comment's words (`▒▒▒` is any run of hidden words).\n\n")
    pivot = (counts.pivot_table(index="label", columns="group", values="n", aggfunc="sum")
             .reindex(index=LABELS, columns=list(GROUP_NAMES)).fillna(0).astype(int))
    out += "Comments per label and outcome:\n\n" + md_table(
        [[label, *row] for label, row in zip(pivot.index, pivot.to_numpy().tolist(),
                                             strict=True)],
        ["label", *GROUP_NAMES.values()]) + "\n\n"
    out += f"![Top words, caught]({plots['words_caught']})\n\n"
    for label in LABELS:
        out += f"### {label}\n\n"
        part = tokens[tokens["label"] == label]
        columns = []
        for group in GROUP_NAMES:
            rows = part[part["group"] == group]
            columns.append([f"{r.word} ({r.mean_value:+.2f})" for r in rows.itertuples()])
        depth = max([len(c) for c in columns] + [0])
        if depth:
            out += md_table([[i + 1, *[c[i] if i < len(c) else "" for c in columns]]
                             for i in range(depth)], ["rank", *GROUP_NAMES.values()]) + "\n\n"
        else:
            out += "No words passed the filters.\n\n"
        for row in examples[examples["label"] == label].itertuples():
            out += (f"* {GROUP_NAMES[row.group]}, p = {row.p:.2f}, toxic = {row.y}: "
                    f"{row.masked}\n")
        out += "\n"
    return out.rstrip()


def truncation_section(results: Mapping[str, Any], plots: Mapping[str, str]) -> str:
    overall = pd.DataFrame(results["truncation"]["overall"])
    bands = pd.DataFrame(results["truncation"]["bands"])
    mean = overall[overall["label"] == "mean"].sort_values("max_length")
    out = ("## 3. Input length\n\nValidation rescored with the input truncated at each "
           "max_length. Mean over labels:\n\n")
    out += md_table([[int(r.max_length), r.truncated_share, r.auc, r.pr_auc, r.ece, r.recall]
                     for r in mean.itertuples()],
                    ["max_length", "truncated", "AUC", "PR-AUC", "ECE", "recall"])
    out += "\n\nAUC per label:\n\n" + _wide(overall, "max_length", "auc")
    out += "\n\nPR-AUC per label:\n\n" + _wide(overall, "max_length", "pr_auc")
    band_mean = bands[bands["label"] == "mean"]
    pivot = band_mean.pivot(index="band", columns="max_length", values="auc")
    pivot = pivot.reindex(list(dict.fromkeys(band_mean["band"])))
    sizes = band_mean.drop_duplicates("band").set_index("band")["n"]
    out += ("\n\nMean AUC by the comment's **untruncated token length** (these bands are in "
            "tokens of the student's tokenizer, not the characters of the bias slices):\n\n")
    out += md_table([[band, sizes[band], *row] for band, row in
                     zip(pivot.index, pivot.to_numpy().tolist(), strict=True)],
                    ["tokens", "rows", *[f"max_length {c}" for c in pivot.columns]])
    return out + f"\n\n![Truncation]({plots['truncation']})\n\n![Length bands]({plots['bands']})"


def obfuscation_section(results: Mapping[str, Any], plots: Mapping[str, str]) -> str:
    table = pd.DataFrame(results["obfuscation"])
    mean = table[table["label"] == "mean"].sort_values("rate")
    out = ("## 4. Obfuscation strength\n\nThe data pipeline's substitution table "
           "(`Data-Pipeline/scripts/obfuscate.py`: a→4, e→3, i→1, l→1, o→0, s→5, t→7) "
           "applied to every validation comment, toxic or not, each eligible character "
           "replaced with probability `rate`, seeded per comment id. The rates are nested: a "
           "character changed at one rate is changed at every higher one. *Missed* is the "
           "share of toxic comments flagged when clean and not flagged at that rate. Mean over "
           "labels:\n\n")
    out += md_table([[f"{r.rate:g}", r.chars_changed, r.auc, r.pr_auc, r.recall, r.missed]
                     for r in mean.itertuples()],
                    ["rate", "characters changed", "AUC", "PR-AUC", "recall", "missed"])
    out += "\n\nAUC per label:\n\n" + _wide(table, "rate", "auc")
    out += "\n\nMissed per label:\n\n" + _wide(table, "rate", "missed")
    out += ("\n\nThis sweep uses validation rather than `obfuscation.parquet`: that set is "
            "built from toxic gate comments at one fixed rate, so it is scored with the gate "
            "and has no non-toxic rows for an AUC.")
    return out + f"\n\n![Obfuscation]({plots['obfuscation']})"


def render(results: Mapping[str, Any], plots: Mapping[str, str],
           findings: str | None = None) -> str:
    """The whole report. ``plots``: plot name -> path relative to the report."""
    parts = [f"# Sensitivity analysis: `{results['setup']['model_version']}`",
             "Generated by `python -m src.sensitivity.run` from `results.json`; edit the "
             "findings file, not this one."]
    if findings:
        parts.append("## Findings\n\n" + findings.strip())
    parts += [setup_section(results), hyperparameter_section(results, plots),
              shap_section(results, plots), truncation_section(results, plots),
              obfuscation_section(results, plots)]
    return "\n\n".join(parts) + "\n"


def relative(path: Path, start: Path) -> str:
    """``path`` relative to the folder ``start``, with forward slashes (for markdown links)."""
    return Path(os.path.relpath(path, start)).as_posix()
