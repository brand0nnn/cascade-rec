"""R1 runtime operations, extracted without changing the historical calculations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd
from cascade.utils.io import git_hash


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=float)+'\n')


def log(**kw):
    print(json.dumps(kw, default=float), flush=True)


def protocol_guard(ctx):
    lock = ctx.bulk/'protocol_lock.json'
    if not lock.exists(): raise ValueError('lock the written protocol before opening Test')
    if digest(ctx.protocol) != json.loads(lock.read_text())['protocol_sha256']:
        raise ValueError('the locked Test protocol changed')


def record_test_open(ctx, family):
    path = ctx.bulk/f'test_opened_{family}.json'
    if path.exists(): raise ValueError(f'{family}: Test already opened; each system is scored once')
    write_json(path, {'opened_utc': pd.Timestamp.utcnow().isoformat(), 'family': family,
                      'protocol_sha256': digest(ctx.protocol), 'git_hash': git_hash()})


def phase_lock(ctx):
    if not ctx.protocol.exists(): raise ValueError('write the protocol first')
    lock = ctx.bulk/'protocol_lock.json'
    if lock.exists(): raise ValueError('protocol already locked')
    write_json(lock, {'protocol_sha256': digest(ctx.protocol), 'locked_utc': pd.Timestamp.utcnow().isoformat(), 'git_hash': git_hash()})
