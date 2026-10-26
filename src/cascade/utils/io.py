"""Result saving helpers: every run writes metrics, config, seed and git hash."""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from cascade.utils.config import REPO_ROOT


def git_hash() -> str:
    """Current git commit (with '-dirty' if the tree has uncommitted changes)."""
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "configs", "scripts"], cwd=REPO_ROOT).decode().strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def cpu_model() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        return platform.processor() or platform.machine()


def make_run_dir(experiment: str, run_id: str | None = None, *, exist_ok: bool = True) -> Path:
    """Create a run directory; ``exist_ok=False`` atomically rejects collisions."""
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    d = REPO_ROOT / "results" / experiment / run_id
    d.mkdir(parents=True, exist_ok=exist_ok)
    return d


def save_run(run_dir: Path, metrics: dict[str, Any], config: dict[str, Any]) -> None:
    """Write metrics.json, config.yaml and git_hash.txt into ``run_dir``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    (run_dir / "git_hash.txt").write_text(git_hash() + "\n")
