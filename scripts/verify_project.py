"""Check saved outputs against predictions and presentation content."""
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from pptx import Presentation
from src.common import evaluate_scores, write_json
from src.freeze import verify_frozen
from src.demo import load_selected
from src.preprocessing import load_partitions
from threadpoolctl import threadpool_limits

def main():
    frozen = verify_frozen()
    metrics = pd.read_csv(ROOT/'artifacts/evaluation/test_metrics.csv')
    assert set(metrics.model) == {'dummy','logistic','boosting','mlp'}
    assert len(metrics) >= 7
    assert np.isfinite(metrics[['ap','roc_auc','threshold','precision','recall','f1']]).all().all()
    name, meta, model = load_selected()
    X,y = load_partitions(include_test=True)['test']
    with threadpool_limits(limits=2):
        score = model.predict_proba(X)[:,1]
    actual = evaluate_scores(y, score, meta['threshold'])
    row = metrics[(metrics.model == name) & (metrics.threshold_type == 'working')].iloc[0]
    for key, value in actual.items():
        np.testing.assert_allclose(value, row[key], atol=1e-12, rtol=0)
    prs = Presentation(ROOT/'presentation/credit_default_final.pptx')
    text = '\n'.join(shape.text for slide in prs.slides for shape in slide.shapes if shape.has_text_frame)
    tables = '\n'.join(cell.text for slide in prs.slides for shape in slide.shapes if shape.has_table for r in shape.table.rows for cell in r.cells)
    assert len(prs.slides) == 15
    assert all(s.notes_slide.notes_text_frame.text for s in prs.slides)
    for _,r in metrics[metrics.threshold_type == 'working'].iterrows():
        assert (f'{r.ap:.3f}' in text+tables or f'{r.ap:.4f}' in text+tables), f'AP missing from deck: {r.model}'
    checks = {
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'device': frozen.get('models', {}).get('mlp', {}).get('device', 'cuda'), 'seed': 42,
        'frozen_hashes_verified': True,
        'all_four_model_families_reported': True,
        'selected_model_reloaded_and_test_metrics_recomputed': True,
        'selected_model': name, 'test_rows': len(y),
        'pptx_reopened': True, 'slides': len(prs.slides),
        'speaker_notes_present_all_slides': True,
        'all_working_AP_values_found_in_editable_presentation': True,
        'powerpoint_rendered_slide_count': len(list((ROOT/'presentation/preview').glob('*.PNG')))+len(list((ROOT/'presentation/preview').glob('*.png'))),
        'limitations': ['Single historical client split; no external temporal/bank validation.',
                        'MLP trained on CUDA without forced kernel determinism; retraining may shift MLP metrics slightly.',
                        'Probability scores are not calibrated; single seed and small hyperparameter search.']
    }
    # Windows glob is case insensitive, so count unique paths rather than extensions.
    checks['powerpoint_rendered_slide_count'] = len({p.resolve() for p in (ROOT/'presentation/preview').glob('*') if p.suffix.lower()=='.png'})
    write_json(ROOT/'artifacts/verification.json', checks)
    print(json.dumps(checks, indent=2))

if __name__ == '__main__':
    main()
