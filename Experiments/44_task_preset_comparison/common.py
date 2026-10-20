"""Small, framework-free provenance helpers for the task-preset comparison."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = Path.home() / '.local/share/synrfp-runs/task_preset_comparison_v1'


def need(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def immutable(path, value):
    if Path(path).exists():
        need(read(path) == value, 'Immutable study44 receipt changed: ' + str(path))
    else:
        write(path, value)
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class HashCache:
    """A cached full hash is valid only while every filesystem identity agrees."""
    def __init__(self):
        self.cache = {}

    @staticmethod
    def signature(path):
        stat = Path(path).stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def verify(self, path, expected):
        path = Path(path)
        before = self.signature(path)
        key = str(path.resolve())
        prior = self.cache.get(key)
        if prior is None or prior[0] != before:
            digest = sha(path)
            need(self.signature(path) == before, 'File changed during study44 hashing: ' + str(path))
            self.cache[key] = before, digest
        need(self.cache[key][1] == expected, 'Study44 sealed hash mismatch: ' + str(path))
        return expected


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def verify_catalog(catalog, cache=None, required=True):
    need(isinstance(catalog, dict) and (bool(catalog) or not required), 'Missing study44 hash catalogue')
    cache = cache or HashCache()
    for path, digest in catalog.items():
        need(Path(path).is_absolute(), 'Study44 provenance requires absolute paths')
        cache.verify(path, digest)
    return catalog
