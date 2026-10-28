from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .evidence.registry import validate_bundle
from .pherm.prediction import ProbPHERM, select_d_threshold
from .pherm.framework import PHERM


def fit_from_scores(expression, labels, scores, reliability, inner, bundle, *, options):
    validate_bundle(bundle)
    if (not inner.index.is_unique or set(inner.index) != set(expression.index)
            or not {'D', 'S', 'label', 'fold'} <= set(inner.columns)):
        raise ValueError('Inner OOF requires D, S, label and fold for every training sample')
    inner = inner.loc[expression.index, ['D', 'S', 'label', 'fold']].astype(float)
    if not np.array_equal(inner.label, np.asarray(labels)):
        raise ValueError('Inner OOF labels do not match training labels')
    if (not np.isfinite(inner.to_numpy()).all() or inner.fold.nunique() < 2
            or not (inner.fold == np.floor(inner.fold)).all()):
        raise ValueError('Finite inner predictions and at least two integer folds required')
    modules = [m['module_id'] for m in bundle['modules'] if m['head'] == 'S']
    if list(scores.columns) != modules or not scores.index.equals(expression.index):
        raise ValueError('Scores must align with training samples and S module order')
    if set(reliability) != set(modules):
        raise ValueError('Reliability must map every score column to its fitted value')
    threshold = select_d_threshold(labels, inner.D, inner.S, options=options)
    return ProbPHERM(options).fit(expression, labels, scores,
        [reliability[name] for name in modules], d_threshold=threshold,
        module_definitions=bundle['modules'])


def nested_oof(expression, labels, module_factory=None, *, options=None, seeds, outer_folds,
               inner_folds=None, model_dir=None, evidence_bundle=None):
    from .pipeline import fit_prediction_model
    y = np.asarray(labels)
    if (y.shape != (len(expression),) or set(y.tolist()) != {0, 1, 2}
            or expression.index.has_duplicates
            or (isinstance(labels, pd.Series) and not labels.index.equals(expression.index))):
        raise ValueError('Aligned three-state labels and unique samples required')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Unique repeat seeds required')
    if type(outer_folds) is not int or outer_folds < 2 or np.bincount(y.astype(int)).min() < outer_folds:
        raise ValueError('Insufficient samples for outer stratification')
    rows, splits = [], []
    destination = Path(model_dir) if model_dir is not None else None
    if destination:
        destination.mkdir(parents=True, exist_ok=True)
        for repeat in range(1, len(seeds)+1):
            for fold in range(1, outer_folds+1):
                for suffix in ['.json', '_scores.csv']:
                    path = destination / f'repeat_{repeat}_fold_{fold}{suffix}'
                    if path.exists():
                        raise FileExistsError(path)
    for repeat, seed in enumerate(seeds, 1):
        splitter = StratifiedKFold(outer_folds, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(splitter.split(expression, y), 1):
            fitted, transform, inner = fit_prediction_model(expression.iloc[tr], y[tr], module_factory,
                options=options, inner_folds=inner_folds, inner_seed=seed, evidence_bundle=evidence_bundle)
            scores = None if isinstance(fitted, PHERM) else transform.transform(expression.iloc[te])
            prediction = fitted.predict(expression.iloc[te]) if scores is None else fitted.predict(expression.iloc[te], scores)
            model_file = scores_file = None
            if destination:
                model_file = destination / f'repeat_{repeat}_fold_{fold}.json'
                fitted.save(model_file)
                if scores is not None:
                    scores_file = destination / f'repeat_{repeat}_fold_{fold}_scores.csv'
                    scores.to_csv(scores_file, index_label='sample_id')
            for j, i in enumerate(te):
                rows.append(dict(sample_id=str(expression.index[i]), repeat=repeat, fold=fold,
                    label=int(y[i]), score=float(prediction['risk'][j]),
                    prediction=int(prediction['prediction'][j])))
            splits.append(dict(repeat=repeat, fold=fold, train=expression.index[tr].tolist(),
                test=expression.index[te].tolist(), d_threshold=getattr(fitted, 'd_threshold', None),
                model_file=str(model_file) if model_file else None,
                module_scores_file=str(scores_file) if scores_file else None))
    return pd.DataFrame(rows), splits
