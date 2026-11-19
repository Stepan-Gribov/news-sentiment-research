"""Small, versioned H3 execution helpers; never writes to source directories."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('./analysis_outputs/h3_alpha_20261104')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_default(value):
    if hasattr(value, 'item'):
        return value.item()
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   default=json_default) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def checkpoint(stage, **kwargs):
    value = dict(stage=stage, active_pid=os.getpid(), updated_utc=utc(), **kwargs)
    save_json(ROOT / 'runtime/execution_checkpoint.json', value)
    print(json.dumps(value, ensure_ascii=False, default=json_default), flush=True)


def file_record(path, expected=None):
    path = Path(path)
    digest = sha(path)
    if expected and digest != expected:
        raise AssertionError(f'Input hash mismatch: {path}: {digest} != {expected}')
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def seal_directory(path, manifest):
    path = Path(path)
    manifest = dict(manifest)
    manifest['sealed_utc'] = utc()
    manifest['files'] = {str(p.relative_to(path)).replace('\\', '/'): file_record(p)
                         for p in sorted(path.rglob('*'))
                         if p.is_file() and p.name != 'manifest.json'
                         and not p.name.endswith('.tmp')}
    save_json(path / 'manifest.json', manifest)
    return manifest
