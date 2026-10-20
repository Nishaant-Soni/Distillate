"""Pilot sample, few-shot examples, and the gate subset, with inclusion weights.

Enriched sampling: rare labels get a fixed number of positives, then a uniform random fill
tops the sample up. To keep inclusion probabilities exact, every row belongs to one stratum:
the first enriched label (in config order) it is positive for, or ``rest``.

* Stratum ``h`` with ``N_h`` rows gets ``k_h`` draws without replacement.
* The fill draws ``m`` rows uniformly from all ``N - sum(k)`` rows not yet drawn.

So a row in enriched stratum ``h`` is included with probability
``k_h/N_h + (1 - k_h/N_h) * m/(N - sum(k))``, a ``rest`` row with ``m/(N - sum(k))``, and
its weight is ``1 / probability``. Weighted means over the sample are unbiased estimates of
population means, which is what the weighted pilot metrics rely on.

Rows are sorted by id before drawing, so the same seed gives the same sample whatever
order the input file is in.

    python -m src.teacher.sampling pilot   # writes pilot_ids.parquet and fewshot_ids.json
    python -m src.teacher.sampling gate    # writes gate_subset_ids.parquet (from wiki_test_raw)
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Any

from src.teacher.config import LABELS, get_logger, load_config

logger = get_logger(__name__)
REST = "rest"


def assign_strata(df, enrich: Mapping[str, int]):
    """Each row's stratum: the first label in ``enrich`` it is positive for, else ``rest``."""
    import numpy as np
    import pandas as pd

    strata = np.full(len(df), REST, dtype=object)
    unassigned = np.ones(len(df), dtype=bool)
    for label in enrich:
        hit = unassigned & (df[label].to_numpy() == 1)
        strata[hit] = label
        unassigned &= ~hit
    return pd.Series(strata, index=df.index, name="stratum")


def enriched_sample(df, size: int, enrich: Mapping[str, int], seed: int):
    """Draw ``size`` rows: ``enrich[label]`` from each label's stratum, the rest uniformly.

    Returns a frame with ``id``, ``stratum``, ``inclusion_prob``, ``weight``, sorted by id.
    """
    import numpy as np
    import pandas as pd

    df = df.sort_values("id", kind="stable").reset_index(drop=True)
    strata = assign_strata(df, enrich)
    rng = np.random.default_rng(seed)
    n_total, n_enriched = len(df), sum(enrich.values())
    n_fill = size - n_enriched
    if n_fill < 0 or size > n_total:
        raise ValueError(f"cannot draw {size} rows ({n_enriched} enriched) from {n_total}")

    chosen = np.zeros(n_total, dtype=bool)
    prob = np.empty(n_total)
    fill_prob = n_fill / (n_total - n_enriched)
    prob[:] = fill_prob
    for label, k in enrich.items():
        members = np.flatnonzero(strata.to_numpy() == label)
        if len(members) < k:
            raise ValueError(f"stratum {label!r} has {len(members)} rows, fewer than {k}")
        chosen[rng.choice(members, size=k, replace=False)] = True
        share = k / len(members)
        prob[members] = share + (1 - share) * fill_prob
    chosen[rng.choice(np.flatnonzero(~chosen), size=n_fill, replace=False)] = True

    sample = pd.DataFrame({
        "id": df.loc[chosen, "id"].astype("string"),
        "stratum": strata[chosen].astype("string"),
        "inclusion_prob": prob[chosen],
        "weight": 1.0 / prob[chosen],
    })
    return sample.reset_index(drop=True)


def pick_fewshot(df, exclude_ids: Collection[str], per_label: int, max_tokens: int,
                 count_tokens: Callable[[list[str]], list[int]],
                 seed: int) -> dict[str, list[str]]:
    """``per_label`` distinct positive examples per label, from rows outside ``exclude_ids``.

    Only rows of at most ``max_tokens`` tokens qualify, so the prompt's few-shot truncation
    never cuts off the part of an example that made it positive. Labels are filled in
    ``LABELS`` order; a row picked for one label is not reused.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    excluded = set(exclude_ids)
    positive = df[list(LABELS)].to_numpy().any(axis=1)
    pool = df[~df["id"].isin(excluded) & positive & (df["text"].str.len() > 0)]
    pool = pool[np.asarray(count_tokens(pool["text"].tolist())) <= max_tokens]
    pool = pool.sort_values("id", kind="stable")
    used: set[str] = set()
    picked: dict[str, list[str]] = {}
    for label in LABELS:
        candidates = pool.loc[(pool[label] == 1) & ~pool["id"].isin(used), "id"].to_numpy()
        if len(candidates) < per_label:
            raise ValueError(f"only {len(candidates)} short enough positives for {label!r}")
        ids = [str(i) for i in rng.choice(candidates, size=per_label, replace=False)]
        picked[label] = ids
        used.update(ids)
    return picked


def pilot_summary(train, pilot) -> dict[str, Any]:
    """Positives per label in a sample (pilot or gate subset), and weighted vs. true prevalence."""
    import numpy as np

    rows = train.set_index("id").loc[pilot["id"]]
    weights = pilot["weight"].to_numpy()
    summary: dict[str, Any] = {"rows": len(pilot), "weight_sum": round(float(weights.sum()), 1),
                               "population": len(train), "labels": {}}
    for label in LABELS:
        y = rows[label].to_numpy()
        summary["labels"][label] = {
            "positives": int(y.sum()),
            "weighted_prevalence": round(float(np.average(y, weights=weights)), 5),
            "true_prevalence": round(float(train[label].mean()), 5),
        }
    summary["strata"] = {str(k): int(v) for k, v in pilot["stratum"].value_counts().items()}
    return summary


def token_counter(cfg: Mapping[str, Any]) -> Callable[[list[str]], list[int]]:
    """Token counts with the teacher's tokenizer at the pinned revision (downloads ~15 MB once)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_id"],
                                              revision=cfg["model"]["revision"])
    return lambda texts: [len(ids) for ids in tokenizer(texts, add_special_tokens=False)
                          ["input_ids"]]


def run_pilot(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Build and save the pilot sample and few-shot ids. Raises if a label has < 100 positives."""
    import pandas as pd

    data_dir, out_dir = Path(cfg["paths"]["data_dir"]), Path(cfg["paths"]["pilot_dir"])
    train = pd.read_parquet(data_dir / cfg["data"]["files"]["train"])
    s = cfg["sampling"]
    pilot = enriched_sample(train, s["pilot_size"], s["pilot_enrich"], cfg["seed"])
    fewshot = pick_fewshot(train, set(pilot["id"]), s["fewshot_per_label"],
                           cfg["prompt"]["fewshot_max_tokens"], token_counter(cfg), cfg["seed"])

    summary = pilot_summary(train, pilot)
    too_few = {k: v["positives"] for k, v in summary["labels"].items() if v["positives"] < 100}
    if too_few:
        raise ValueError(f"pilot has fewer than 100 positives for {too_few}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pilot.to_parquet(out_dir / "pilot_ids.parquet", index=False)
    (out_dir / "fewshot_ids.json").write_text(json.dumps(
        {"seed": cfg["seed"], "per_label": s["fewshot_per_label"],
         "max_tokens": cfg["prompt"]["fewshot_max_tokens"], "ids": fewshot}, indent=2) + "\n")
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("pilot: %s rows, positives %s", len(pilot),
                {k: v["positives"] for k, v in summary["labels"].items()})
    return summary


def run_gate(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Draw and save the gate subset from the test split. Raises if a label has < 100 positives."""
    import pandas as pd

    test = pd.read_parquet(Path(cfg["paths"]["data_dir"]) / cfg["data"]["files"]["test"])
    test["id"] = test["id"].astype(str)
    s = cfg["sampling"]
    gate = enriched_sample(test, s["gate_subset_size"], s["gate_enrich"], cfg["seed"])
    summary = pilot_summary(test, gate)
    too_few = {k: v["positives"] for k, v in summary["labels"].items() if v["positives"] < 100}
    if too_few:
        raise ValueError(f"gate subset has fewer than 100 positives for {too_few}")

    out = Path(cfg["paths"]["gate_ids"])
    out.parent.mkdir(parents=True, exist_ok=True)
    gate.to_parquet(out, index=False)
    out.with_name("gate_subset_summary.json").write_text(json.dumps(
        {"seed": cfg["seed"], "source": cfg["data"]["files"]["test"],
         "enrich": dict(s["gate_enrich"]), **summary}, indent=2) + "\n")
    logger.info("gate subset: %s rows, positives %s", len(gate),
                {k: v["positives"] for k, v in summary["labels"].items()})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher sampling.")
    parser.add_argument("what", choices=["pilot", "gate"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run = {"pilot": run_pilot, "gate": run_gate}[args.what]
    print(json.dumps(run(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
