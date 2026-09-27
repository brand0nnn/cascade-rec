"""YAML config loading with a stable content hash."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config. A top-level ``base: other.yaml`` key is merged underneath."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        merged = load_config(path.parent / base)
        merged.update(cfg)
        cfg = merged
    return cfg


def config_hash(cfg: dict[str, Any], n: int = 8) -> str:
    """Short deterministic hash of a config dict (used in cache filenames)."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:n]
