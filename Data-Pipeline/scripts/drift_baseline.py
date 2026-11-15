"""data_prep_dag stage: build the drift baseline from Wikipedia train.

The baseline is built here, inside data_prep_dag and before dvc_version, so it is
versioned in the same DVC commit as the data it describes. The drift DAG only loads it
(drift.load_baseline, which no longer builds on first use). All the logic is in
scripts/drift.py; this module exists because task ids match module names.

    python -m scripts.drift_baseline --config config/pipeline.yaml
"""

from __future__ import annotations

import sys

from scripts.drift import main, run

__all__ = ["main", "run"]

if __name__ == "__main__":
    sys.exit(main())
