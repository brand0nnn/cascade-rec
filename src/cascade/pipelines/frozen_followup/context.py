"""Explicit paths and devices shared by the historical R1 phases."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FollowupContext:
    bulk: Path
    device_retrieval: str
    din_runs: dict[str, list[Path]]
    fresh_run: Path
    fresh_run_s1: Path
    fresh_val: Path
    out: Path
    pools: Path
    protocol: Path
    ranker_device: str
    repo_root: Path
    seeds: tuple[int, ...]
    swap_doc: Path
    t1: int
    t2: int
    test_final: Path
    tt_dir: Path
    tt_run: Path
