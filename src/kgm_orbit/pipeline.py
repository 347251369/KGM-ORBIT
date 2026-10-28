from .evidence.workflow import build_evidence
from .coeri.report import explain_prediction, explain_oof_sample, generate_report
from .evidence.scoring import fit_scorer, IterativeScorer
from .evidence.registry import Module, validate_bundle
from .pherm.framework import PHERM, FrameworkOptions
from .pherm.prediction import ProbPHERM, PredictionOptions, select_d_threshold


def fit_prediction_model(expression, labels, module_factory=None, *, options=None, inner_folds=None, inner_seed=None,
                         evidence_bundle=None):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    if options is None:
        options = FrameworkOptions() if module_factory is None else PredictionOptions()
    y = np.asarray(labels)
    if (y.shape != (len(expression),) or set(y.tolist()) != {0, 1, 2}
            or expression.index.has_duplicates
            or (isinstance(labels, pd.Series) and not labels.index.equals(expression.index))):
        raise ValueError('Aligned three-state labels and unique samples required')
    if isinstance(options, FrameworkOptions):
        if evidence_bundle is None:
            raise ValueError('Evidence bundle required')
        validate_bundle(evidence_bundle)
        modules = [Module(**m) for m in evidence_bundle['modules']]
        model = PHERM(modules, options).fit(expression, labels)
        return model, model.standardizer, None
    if not isinstance(options, PredictionOptions) or not callable(module_factory):
        raise ValueError('Training options and module factory do not match')
    if type(inner_folds) is not int or inner_folds < 2 or np.bincount(y.astype(int)).min() < inner_folds:
        raise ValueError('Insufficient samples for inner stratification')
    d, s = np.full(len(y), np.nan), np.full(len(y), np.nan)
    folds = np.zeros(len(y), int)
    for fold, (train, validation) in enumerate(StratifiedKFold(inner_folds, shuffle=True,
                                                              random_state=inner_seed).split(expression, y), 1):
        transform = module_factory().fit(expression.iloc[train], y[train])
        train_scores = transform.transform(expression.iloc[train])
        model = ProbPHERM(options).fit(expression.iloc[train], y[train], train_scores,
                                            transform.reliability, d_threshold=options.threshold_grid[0])
        prediction = model.predict(expression.iloc[validation], transform.transform(expression.iloc[validation]))
        d[validation], s[validation], folds[validation] = prediction['D'], prediction['S'], fold
    threshold = select_d_threshold(y, d, s, options=options)
    transform = module_factory().fit(expression, y)
    model = ProbPHERM(options).fit(expression, labels, transform.transform(expression),
                                        transform.reliability, d_threshold=threshold,
                                        module_definitions=getattr(transform, 'module_definitions', None))
    inner = pd.DataFrame(dict(D=d, S=s, label=y, fold=folds), index=expression.index)
    return model, transform, inner

def interpret(fitted, expression, module_scores, evidence_bundle, *, backend, nodes, module_count, tolerance):
    packet = explain_prediction(fitted, expression, module_scores, evidence_bundle,
                                nodes=nodes, module_count=module_count, tolerance=tolerance)
    return packet, generate_report(packet, backend)

__all__ = ['fit_scorer', 'build_evidence', 'interpret', 'explain_oof_sample',
           'fit_prediction_model', 'PHERM', 'FrameworkOptions', 'IterativeScorer', 'ProbPHERM', 'PredictionOptions']
