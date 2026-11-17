"""Verify the released data, split separation, prompts, and distribution boundary."""
import gzip
import hashlib
import json
from pathlib import Path

from evaluation.common.data import ROOT, digest, read_jsonl


def verify():
    manifest = json.loads((ROOT / 'dataset/MANIFEST.json').read_text())
    splits = {'test': set(), 'validation': set()}
    counts = {}
    for row in manifest:
        path = ROOT / row['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], path
        with gzip.open(path, 'rb') as stream:
            raw = stream.read()
        assert hashlib.sha256(raw).hexdigest() == row['uncompressed_sha256'], path
        records = list(read_jsonl(path))
        assert len(records) == row['rows'], path
        keys = [digest(r['problem']) for r in records]
        assert len(set(keys)) == len(keys), path
        assert not splits[row['split']].intersection(keys), path
        splits[row['split']].update(keys)
        if row['split'] == 'test':
            counts[row['investors']] = len(records)
            assert all(r['generation_params']['num_investors'] == row['investors'] for r in records)
    assert counts == {2: 96, 3: 96, 4: 104, 5: 152, 6: 140}
    assert len(splits['test']) == 588 and len(splits['validation']) == 16
    assert not splits['test'].intersection(splits['validation'])
    forbidden = {'ADAS', 'AFlow', 'DyLAN', 'MaAS', 'MAS-Orchestra', 'MAS-Zero', 'Automatic-MAS'}
    for path in ROOT.rglob('*'):
        if any(part in {'.git', '.venv', '__pycache__', 'outputs', '.pytest_cache'} for part in path.parts):
            continue
        assert not forbidden.intersection(path.relative_to(ROOT).parts), path
    for path in (ROOT / 'evaluation/SAS/SMFR/prompts').glob('*.json'):
        protocol = json.loads(path.read_text())
        assert protocol['messages'] and any('{{problem}}' in m['content'] for m in protocol['messages']), path
        assert protocol['default_k'] >= 1
    result_manifest = ROOT / 'results/MANIFEST.json'
    if result_manifest.exists():
        from collections import defaultdict
        groups = defaultdict(set)
        for row in json.loads(result_manifest.read_text())['artifacts']:
            path = ROOT / row['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
            results = list(read_jsonl(path))
            assert len(results) == row['records']
            for result in results:
                key = (result['model'], result['method'], result['run'], result['scope'])
                assert result['problem_sha256'] in splits['test']
                assert result['problem_sha256'] not in groups[key]
                assert isinstance(result['correct'], bool)
                groups[key].add(result['problem_sha256'])
        assert all(len(keys) == 588 for key, keys in groups.items() if key[3] == 'test')
    print('Verified: 588 canonical v2 test cases; 16 disjoint validation cases; exact source checksums; standalone prompt presets; no automated MAS directories.')


if __name__ == '__main__':
    verify()
