from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


def require_package(name: str, install_hint: str | None = None) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        hint = install_hint or f"Install dependencies with: pip install -r requirements-litexids.txt"
        raise RuntimeError(f"Missing required package '{name}'. {hint}") from exc


def load_config(path: str | Path) -> dict[str, Any]:
    yaml = require_package("yaml")
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]
