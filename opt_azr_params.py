"""Load AZR parameter YAML for runtime scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _load_params(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_path() -> Path:
    candidates: List[Path] = [] 
    env_value = os.getenv("AZR_PARAMS")
    if env_value:
        candidates.append(Path(env_value))

    repo_dir = Path(__file__).resolve().parent
    candidates.extend(
        [
            Path("/opt/azr/azr.params.yaml"),
            repo_dir / "azr.params.yaml",
            repo_dir / "examples" / "azr.params.yaml",
            repo_dir.parent / "examples" / "azr.params.yaml",
        ]
    )

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise FileNotFoundError("Unable to locate azr.params.yaml")


PARAMS: Dict[str, Any] = _load_params(_resolve_path())

__all__ = ["PARAMS"]
