"""Shared metrics and the predeclared validation-only threshold rule."""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score,
    precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix)

ROOT = Path(__file__).resolve().parents[1]

def config():
    return json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))

def choose_threshold(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = np.divide(2*precision[:-1]*recall[:-1], precision[:-1]+recall[:-1],
                   out=np.zeros(len(thresholds)), where=(precision[:-1]+recall[:-1]) > 0)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])

def evaluate_scores(y, scores, threshold=0.5):
    pred = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'ap': float(average_precision_score(y, scores)),
            'roc_auc': float(roc_auc_score(y, scores)),
            'threshold': float(threshold),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'positive_rate': float(np.mean(y))}

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')
