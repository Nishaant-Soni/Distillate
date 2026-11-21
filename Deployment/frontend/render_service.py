"""Fill service.yaml's ${...} fields from the environment (deploy_frontend.sh calls this).

Every field must be set, though some may be empty (an empty DISTILLATE_API_URL means the
built-in mock), so a typo can't deploy a service with a literal "${...}" in it. Values are
quoted for YAML. Standard library only.

    SERVICE=... IMAGE=... python render_service.py service.yaml > rendered.yaml
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path

FIELD = re.compile(r"\$\{([A-Z_]+)\}")
# Fields that may be empty; every other one must have a value.
MAY_BE_EMPTY = {"API_URL", "MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "GRAFANA_URL"}


def fields(template: str) -> set[str]:
    return set(FIELD.findall(template))


def render(template: str, env: Mapping[str, str]) -> str:
    missing = sorted(f for f in fields(template) if f not in env)
    empty = sorted(f for f in fields(template) if f in env and not env[f].strip()
                   and f not in MAY_BE_EMPTY)
    if missing or empty:
        raise ValueError(f"unset: {missing}; empty but required: {empty}")

    def value(match: re.Match) -> str:
        text = env[match.group(1)].strip()
        # Inside "..." in the template: a JSON string is a valid YAML double-quoted string.
        return json.dumps(text)[1:-1]

    return FIELD.sub(value, template)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: render_service.py <service.yaml>", file=sys.stderr)
        return 2
    try:
        print(render(Path(args[0]).read_text(encoding="utf-8"), os.environ), end="")
    except ValueError as err:
        print(f"render_service.py: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
