"""Download raw Amazon Reviews 2023 files and record a MANIFEST (CLAUDE.md §4.1)."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

from cascade.utils.config import REPO_ROOT, load_config
from cascade.utils.logging import get_logger

log = get_logger(__name__)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` (via a .part file so interrupted downloads are not mistaken for complete)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, mininterval=5) as bar:
            for b in r.iter_content(1 << 20):
                f.write(b)
                bar.update(len(b))
    if total and tmp.stat().st_size != total:
        raise IOError(f"size mismatch for {dest.name}: {tmp.stat().st_size} != {total}")
    tmp.rename(dest)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    raw_dir = REPO_ROOT / cfg["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, src in cfg["sources"].items():
        dest = raw_dir / src["file"]
        if dest.exists() and manifest.get(src["file"], {}).get("sha256"):
            log.info("%s already downloaded, skipping", dest.name)
            continue
        log.info("downloading %s", src["url"])
        download(src["url"], dest)
        manifest[src["file"]] = {
            "role": name,
            "url": src["url"],
            "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bytes": dest.stat().st_size,
            "sha256": sha256_file(dest),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        log.info("%s sha256=%s", dest.name, manifest[src["file"]]["sha256"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    main(ap.parse_args().config)
