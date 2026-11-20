"""Lets the tests import deploy_tools and smoke_test from this folder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
