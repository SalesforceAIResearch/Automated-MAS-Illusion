"""Canonical release data and atomic, portable output checkpoints."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def digest(problem):
    return hashlib.sha256(problem.strip().encode()).hexdigest()


def read_jsonl(path):
    path = Path(path)
    with (gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz' else path.open(encoding='utf-8')) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def dataset(investors=None, path=None):
    if path:
        rows = [(r.get('generation_params', {}).get('num_investors'), r) for r in read_jsonl(path)]
    else:
        rows = []
        for n in range(2, 7):
            if investors and n not in investors:
                continue
            rows.extend((n, r) for r in read_jsonl(ROOT / f'dataset/data/balanced_dataset_single_{n}_fixed_v2.jsonl.gz'))
    keys = [digest(r['problem']) for _, r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate problems in input dataset')
    return rows


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False)
        os.replace(filename, path)
    finally:
        Path(filename).unlink(missing_ok=True)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(path)
