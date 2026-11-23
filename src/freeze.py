"""Lock validation decisions and artifact hashes before opening test outcomes."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from src.common import ROOT, write_json

def freeze():
    classical = json.loads((ROOT/'artifacts/classical/selected.json').read_text())
    neural = json.loads((ROOT/'artifacts/neural/selected.json').read_text())
    selected = dict(classical)
    selected['mlp'] = neural
    learned = {k:v for k,v in selected.items() if k != 'dummy'}
    winner = max(learned, key=lambda k: learned[k]['validation_ap'])
    files = [ROOT/'config.json', ROOT/'artifacts/splits/indices.npz',
             ROOT/'data/credit_default.csv']
    for folder in ['classical', 'neural']:
        files += [p for p in (ROOT/'artifacts'/folder).rglob('*')
                  if p.is_file() and p.suffix in ['.json','.joblib','.pt','.pth']]
    payload = {'created_utc': datetime.now(timezone.utc).isoformat(),
               'selection_source': 'validation only', 'selected_model': winner,
               'models': selected,
               'sha256': {str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    write_json(ROOT/'artifacts/frozen_selection.json', payload)
    print(f'Frozen validation winner: {winner}; test has not been scored.')
    return payload

def verify_frozen():
    path = ROOT/'artifacts/frozen_selection.json'
    if not path.exists():
        raise RuntimeError('Run python -m src.freeze before evaluation.')
    payload = json.loads(path.read_text())
    for name, digest in payload['sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen artifact changed: {name}. Review the protocol before re-evaluation.')
    return payload

if __name__ == '__main__':
    freeze()
