"""Bind online authorization to the tested source, recipe and evidence files."""
import hashlib
import json
from pathlib import Path

REQUIRED={'gradient','fixed_update','restore','publication','scale','clip','master','seed','accumulation'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest():
    root=Path(__file__).resolve().parents[1]
    result=hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        result.update(path.relative_to(root).as_posix().encode()+b'\0')
        result.update(path.read_bytes())
    return result.hexdigest()


def verify(config):
    config=Path(config)
    gate=json.loads((config.parent/'precision-gate.json').read_text())
    if gate.get('accepted') is not True or gate.get('source_sha256')!=source_digest():
        raise RuntimeError('Precision gate missing or source changed after validation')
    if gate.get('config_sha256')!=digest(config):
        raise RuntimeError('Recipe changed after validation')
    if set(gate.get('checks',{}))!=REQUIRED:
        raise RuntimeError('Incomplete precision gate checks')
    for name,record in gate['checks'].items():
        path=Path(record['path'])
        if not path.is_absolute() or digest(path)!=record['sha256']:
            raise RuntimeError(f'Validation evidence missing or changed: {name}')
        evidence=json.loads(path.read_text())
        if evidence.get('accepted') is not True:
            raise RuntimeError(f'Validation not accepted: {name}')
        for artifact in evidence.get('artifacts',[]):
            if digest(artifact['path'])!=artifact['sha256']:
                raise RuntimeError(f'Underlying validation artifact changed: {name}')
    return gate
