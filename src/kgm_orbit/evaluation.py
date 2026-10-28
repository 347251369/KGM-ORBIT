import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, f1_score, confusion_matrix

def prediction_metrics(labels, scores, predictions):
    y = np.asarray(labels)
    s = np.asarray(scores, float)
    p = np.asarray(predictions)
    if y.shape != s.shape or y.shape != p.shape or set(y.tolist()) != {0, 1, 2}:
        raise ValueError('Aligned three-state evaluation required')
    if not np.isfinite(s).all() or not np.isin(p, [0, 1, 2]).all():
        raise ValueError('Invalid predictions')
    auc = []
    for (lo, hi) in [(0, 1), (0, 2), (1, 2)]:
        mask = np.isin(y, [lo, hi])
        auc.append(roc_auc_score(y[mask] == hi, s[mask]))
    return dict(ordering_auc=float(np.mean(auc)),
        macro_recall=recall_score(y, p, average='macro', zero_division=0),
        macro_f1=f1_score(y, p, average='macro', zero_division=0),
        n=len(y),
        confusion_matrix=confusion_matrix(y, p, labels=[0, 1, 2]).tolist())

def repeat_metrics(oof):
    if oof.duplicated(['repeat', 'sample_id']).any():
        raise ValueError('Duplicate OOF patient')
    patient_sets = [set(g.sample_id) for (_, g) in oof.groupby('repeat')]
    if any((s != patient_sets[0] for s in patient_sets)):
        raise ValueError('Repeat populations differ')
    if (oof.groupby('sample_id').label.nunique() != 1).any():
        raise ValueError('Labels differ across repeats')
    return pd.DataFrame([dict(repeat=r,
        **prediction_metrics(g.label, g.score, g.prediction)) for (r,
        g) in oof.groupby('repeat')])
