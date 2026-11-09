"""Immutable publication identity, including every published weight file's content."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def build_manifest(path):
    path=Path(path)
    index=json.loads((path/'kt_full_experts.index.json').read_text())
    files={entry['file'] for entry in index['layers'].values()}
    files.update(p.name for p in path.glob('model*.safetensors'))
    files.update(['kt_full_experts.index.json','trainer_state.json'])
    if (path/'model.safetensors.index.json').exists():files.add('model.safetensors.index.json')
    records={}
    for name in sorted(files):
        file=path/name
        if file.resolve().parent!=path.resolve():raise ValueError('Unsafe publication filename')
        records[name]=dict(bytes=file.stat().st_size,sha256=sha256(file))
    manifest=dict(version_id=json.loads((path/'trainer_state.json').read_text())['global_step'],files=records)
    raw=json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()
    with (path/'publication-manifest.json').open('xb') as handle:handle.write(raw)
    return hashlib.sha256(raw).hexdigest()


def validate_manifest(path,verify_content=True):
    path=Path(path)
    raw=(path/'publication-manifest.json').read_bytes()
    manifest=json.loads(raw)
    step=json.loads((path/'trainer_state.json').read_text())['global_step']
    if type(step) is not int or step<1 or manifest['version_id']!=step:raise ValueError('Invalid policy version')
    index=json.loads((path/'kt_full_experts.index.json').read_text())
    expected={entry['file'] for entry in index['layers'].values()}|{p.name for p in path.glob('model*.safetensors')}
    expected.update(['kt_full_experts.index.json','trainer_state.json'])
    if (path/'model.safetensors.index.json').exists():expected.add('model.safetensors.index.json')
    if not expected or not expected.issubset(manifest['files']):raise ValueError('Incomplete publication manifest')
    for name,record in manifest['files'].items():
        file=path/name
        if file.resolve().parent!=path.resolve() or file.stat().st_size!=record['bytes']:
            raise ValueError('Publication size/path mismatch')
        if verify_content and sha256(file)!=record['sha256']:raise ValueError(f'Publication content changed: {name}')
    return step,hashlib.sha256(raw).hexdigest()
