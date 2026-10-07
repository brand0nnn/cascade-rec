"""Download official source files and record their checksums."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from cascade.utils.config import REPO_ROOT, load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(config_path: str, only: str | None = None) -> Path:
    cfg = load_config(config_path)
    root = REPO_ROOT / cfg["raw_dir"]
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, source in cfg["sources"].items():
        if only is not None and name != only:
            continue
        path = root / source["file"]
        if not path.exists():
            partial = path.with_name(path.name + ".part")
            prior = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={prior}-"} if prior else {}
            with requests.get(source["url"], headers=headers, stream=True, timeout=60) as response:
                response.raise_for_status()
                if prior and response.status_code != 206:
                    prior = 0  # server did not honor Range; safely restart
                with partial.open("ab" if prior else "wb") as stream:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            stream.write(block)
            partial.replace(path)
        manifest[name] = {"url": source["url"], "file": source["file"],
                          "bytes": path.stat().st_size, "sha256": _sha256(path),
                          "checked_at_utc": datetime.now(timezone.utc).isoformat()}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--only", choices=("interactions", "metadata"))
    args = parser.parse_args()
    main(args.config, args.only)
