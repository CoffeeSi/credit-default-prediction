"""Reload the validation-selected model and predict one held-out client."""
import argparse
import json
import joblib
from threadpoolctl import threadpool_limits
from src.common import ROOT
from src.freeze import verify_frozen
from src.data import load_data
from src.preprocessing import FEATURES, load_splits

def load_selected():
    frozen = verify_frozen()
    name = frozen['selected_model']
    meta = frozen['models'][name]
    if name == 'mlp':
        from src.neural import load_predictor
        predictor = load_predictor()
    else:
        predictor = joblib.load(ROOT/meta['model_path'])
    return name, meta, predictor

def demo(index=0, reveal_label=False):
    name, meta, predictor = load_selected()
    test_indices = load_splits()['test']
    if not 0 <= index < len(test_indices):
        raise ValueError(f'index must be between 0 and {len(test_indices)-1}')
    row_index = int(test_indices[index])
    frame = load_data().iloc[[row_index]]
    with threadpool_limits(limits=2):
        score = float(predictor.predict_proba(frame[FEATURES])[0, 1])
    prediction = {'model': name, 'test_position': index, 'source_row_index': row_index,
                  'estimated_default_probability': score, 'threshold': meta['threshold'],
                  'decision': 'PRIORITIZE FOR REVIEW' if score >= meta['threshold'] else 'NOT PRIORITIZED'}
    print(json.dumps(prediction, indent=2), flush=True)
    if reveal_label:
        print('\nActual label (revealed after prediction):', int(frame['default'].iloc[0]), flush=True)
    return prediction

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int, default=0, help='Position in frozen test partition, not source ID')
    parser.add_argument('--reveal-label', action='store_true')
    args = parser.parse_args()
    demo(args.index, args.reveal_label)
