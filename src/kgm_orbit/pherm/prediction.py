from dataclasses import dataclass, asdict
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


@dataclass(frozen=True)
class PredictionOptions:
    feature_count: int = 100
    expression_C: float = 1.0
    expression_l1_ratio: float = 0.5
    expression_max_iterations: int = 5000
    expression_tolerance: float = 1e-6
    knowledge_alpha: float = 1.0
    knowledge_max_iterations: int = 1000
    minimum_reliability: float = 0.1
    maximum_weight: float = 3.0
    variance_floor: float = 0.05
    evidence_clip: float = 4.0
    knowledge_fraction: float = 0.5
    low_state_position: float = 0.45
    severity_threshold: float = 0.5
    threshold_grid: tuple = (0.3, 0.5, 0.7)
    epsilon: float = 1e-12
    seed: int = 0

    def __post_init__(self):
        for key in ['feature_count', 'expression_max_iterations', 'knowledge_max_iterations']:
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError('Positive integer required: ' + key)
        values = [v for k, v in asdict(self).items() if k != 'threshold_grid']
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite prediction options')
        if (self.expression_C <= 0 or self.expression_tolerance <= 0 or self.knowledge_alpha < 0
                or self.maximum_weight <= 0 or self.variance_floor <= 0 or self.evidence_clip <= 0
                or not 0 < self.minimum_reliability <= 1 or not 0 < self.epsilon < .5):
            raise ValueError('Invalid prediction options')
        for key in ['expression_l1_ratio', 'knowledge_fraction', 'low_state_position', 'severity_threshold']:
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError('Probability setting outside [0,1]: ' + key)
        if (not self.threshold_grid or len(set(self.threshold_grid)) != len(self.threshold_grid)
                or not np.isfinite(self.threshold_grid).all()
                or any(not 0 < t < 1 for t in self.threshold_grid)):
            raise ValueError('Distinct D thresholds in (0,1) required')


def combine_heads(disease, expression_severity, knowledge_severity, *, options, threshold):
    d, sr, sk = np.broadcast_arrays(np.asarray(disease, float), np.asarray(expression_severity, float),
                                   np.asarray(knowledge_severity, float))
    if any(not np.isfinite(v).all() or ((v < 0) | (v > 1)).any() for v in [d, sr, sk]):
        raise ValueError('Finite head probabilities in [0,1] required')
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('D threshold must be in (0,1)')
    s = (1 - options.knowledge_fraction) * sr + options.knowledge_fraction * sk
    p = np.stack([1-d, d*(1-s), d*s], axis=-1)
    return dict(D=d, raw_S=sr, knowledge_S=sk, S=s, probabilities=p,
                risk=options.low_state_position*p[..., 1]+p[..., 2],
                prediction=np.where(d < threshold, 0, np.where(s < options.severity_threshold, 1, 2)))


def select_d_threshold(labels, disease, severity, *, options):
    y, d, s = np.asarray(labels), np.asarray(disease, float), np.asarray(severity, float)
    if y.ndim != 1 or y.shape != d.shape or y.shape != s.shape or set(y.tolist()) != {0, 1, 2}:
        raise ValueError('Aligned three-state inner OOF predictions required')
    if not np.isfinite([d, s]).all() or ((d < 0) | (d > 1) | (s < 0) | (s > 1)).any():
        raise ValueError('Invalid inner OOF probabilities')
    scores = [(float(f1_score(y, np.where(d < t, 0, np.where(s < options.severity_threshold, 1, 2)),
                              average='macro', labels=[0, 1, 2], zero_division=0)), t)
              for t in options.threshold_grid]
    return sorted(scores, key=lambda pair: (-pair[0], abs(pair[1]-.5), pair[1]))[0][1]


def _frame(frame, name, *, finite):
    if (not isinstance(frame, pd.DataFrame) or frame.empty or frame.index.has_duplicates
            or frame.columns.has_duplicates or frame.index.isna().any()
            or any(not isinstance(g, str) or not g.strip() for g in frame.columns)):
        raise ValueError('Invalid ' + name + ' table')
    values = frame.to_numpy(float)
    if np.isinf(values).any() or (finite and not np.isfinite(values).all()):
        raise ValueError('Invalid numbers in ' + name)
    return values


class ProbPHERM:
    def __init__(self, options=None):
        self.options = PredictionOptions() if options is None else options

    def fit(self, expression, labels, module_scores, reliability, *, d_threshold, module_definitions=None):
        x = _frame(expression, 'expression', finite=False)
        m = _frame(module_scores, 'module score', finite=True)
        y = np.asarray(labels)
        if (not expression.index.equals(module_scores.index) or y.shape != (len(x),)
                or set(y.tolist()) != {0, 1, 2}
                or (isinstance(labels, pd.Series) and not labels.index.equals(expression.index))):
            raise ValueError('Training tables and three-state labels must align')
        rel = np.asarray(reliability, float)
        if rel.shape != (m.shape[1],) or not np.isfinite(rel).all() or ((rel < 0) | (rel > 1)).any():
            raise ValueError('One reliability in [0,1] per module required')
        if d_threshold not in self.options.threshold_grid:
            raise ValueError('Use an inner OOF selected D threshold from the configured grid')
        o = self.options
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            med = np.nan_to_num(np.nanmedian(x, axis=0))
        filled = np.where(np.isfinite(x), x, med)
        center, scale = filled.mean(0), filled.std(0)
        scale[scale < 1e-12] = 1
        z = (filled-center)/scale
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            warnings.simplefilter('ignore', UserWarning)
            scores, _ = f_classif(z, y)
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf)
        selected = np.argsort(-scores, kind='stable')[:o.feature_count]
        learner = LogisticRegression(C=o.expression_C, penalty='elasticnet', l1_ratio=o.expression_l1_ratio,
                                     class_weight='balanced', solver='saga', max_iter=o.expression_max_iterations,
                                     tol=o.expression_tolerance, random_state=o.seed)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            learner.fit(z[:, selected], y)
        positive, negative = m[y == 2], m[y == 1]
        if min(len(positive), len(negative)) < 3:
            raise ValueError('At least three training samples per case state required')
        mu_p, mu_n = positive.mean(0), negative.mean(0)
        variance = ((len(positive)-1)*positive.var(0, ddof=1)+(len(negative)-1)*negative.var(0, ddof=1))
        variance = np.maximum(variance/(len(positive)+len(negative)-2), o.variance_floor)
        coef, intercept = (mu_p-mu_n)/variance, -.5*(mu_p**2-mu_n**2)/variance
        evidence = np.clip(m*coef+intercept, -o.evidence_clip, o.evidence_clip)*rel
        case = y > 0
        ec, es = evidence[case].mean(0), evidence[case].std(0)
        es[es <= o.epsilon] = 1
        design, target = (evidence[case]-ec)/es, (y[case] == 2).astype(float)
        r = np.clip(rel, o.minimum_reliability, 1)
        sample_weight = np.where(target == 1, .5/target.sum(), .5/(len(target)-target.sum()))
        sample_weight /= sample_weight.sum()
        def objective(params):
            b, w = params[0], params[1:]
            logits = b+design@w
            error = sample_weight*(expit(logits)-target)
            loss = np.sum(sample_weight*(np.logaddexp(0, logits)-target*logits))
            loss += .5*o.knowledge_alpha*np.sum(w*w/(r*r))
            gradient = np.r_[error.sum(), design.T@error+o.knowledge_alpha*w/(r*r)]
            return loss, gradient
        fit = minimize(objective, np.zeros(m.shape[1]+1), jac=True, method='L-BFGS-B',
                       bounds=[(None, None)]+[(0, o.maximum_weight*v) for v in r],
                       options=dict(maxiter=o.knowledge_max_iterations, ftol=1e-12, gtol=1e-8))
        if not fit.success:
            raise ValueError('Knowledge head did not converge: '+str(fit.message))
        self.raw = dict(genes=expression.columns[selected].tolist(), median=med[selected].tolist(),
                        center=center[selected].tolist(), scale=scale[selected].tolist(),
                        coef=learner.coef_.tolist(), intercept=learner.intercept_.tolist())
        self.knowledge = dict(modules=module_scores.columns.tolist(), coef=coef.tolist(),
                              intercept=intercept.tolist(), reliability=rel.tolist(), center=ec.tolist(),
                              scale=es.tolist(), weight=fit.x[1:].tolist(), bias=float(fit.x[0]))
        self.d_threshold = float(d_threshold)
        self.module_definitions = module_definitions
        self.fit_iterations = dict(expression=int(learner.n_iter_.max()), knowledge=int(fit.nit))
        self._validate_state()
        return self

    def _validate_state(self):
        r, k = self.raw, self.knowledge
        for names in [r['genes'], k['modules']]:
            if not names or len(names) != len(set(names)) or any(not isinstance(n, str) or not n for n in names):
                raise ValueError('Unique nonempty fitted feature names required')
        for state, fields, count in [(r, ['median', 'center', 'scale'], len(r['genes'])),
                                     (k, ['coef', 'intercept', 'reliability', 'center', 'scale', 'weight'], len(k['modules']))]:
            for name in fields:
                a = np.asarray(state[name], float)
                if a.shape != (count,) or not np.isfinite(a).all():
                    raise ValueError('Invalid fitted field: '+name)
            if (np.asarray(state['scale']) <= 0).any():
                raise ValueError('Positive fitted scales required')
        for name, shape in [('coef', (3, len(r['genes']))), ('intercept', (3,))]:
            if np.asarray(r[name]).shape != shape or not np.isfinite(r[name]).all():
                raise ValueError('Invalid expression coefficients')
        rel, w = np.asarray(k['reliability']), np.asarray(k['weight'])
        if (not np.isfinite(k['bias']) or ((rel < 0) | (rel > 1)).any() or (w < 0).any()
                or (w > self.options.maximum_weight*np.clip(rel, self.options.minimum_reliability, 1)+1e-10).any()
                or self.d_threshold not in self.options.threshold_grid):
            raise ValueError('Invalid knowledge coefficients or threshold')

    def inputs(self, expression, module_scores):
        _frame(expression, 'expression', finite=False)
        _frame(module_scores, 'module score', finite=True)
        if not expression.index.equals(module_scores.index):
            raise ValueError('Prediction tables must have identical sample order')
        if not set(self.raw['genes']) <= set(expression.columns) or not set(self.knowledge['modules']) <= set(module_scores.columns):
            raise ValueError('Missing fitted expression genes or modules')
        r, k, o = self.raw, self.knowledge, self.options
        x = expression[r['genes']].to_numpy(float)
        z = (np.where(np.isfinite(x), x, r['median'])-r['center'])/r['scale']
        m = module_scores[k['modules']].to_numpy(float)
        e = np.clip(m*np.asarray(k['coef'])+k['intercept'], -o.evidence_clip, o.evidence_clip)*k['reliability']
        return z, (e-k['center'])/k['scale']

    def from_inputs(self, expression_z, evidence_z):
        r, k, o = self.raw, self.knowledge, self.options
        p = softmax(np.asarray(expression_z)@np.asarray(r['coef']).T+r['intercept'], axis=-1)
        d = p[..., 1:].sum(-1)
        sr = p[..., 2]/np.maximum(d, o.epsilon)
        sk = np.clip(expit(np.asarray(evidence_z)@k['weight']+k['bias']), o.epsilon, 1-o.epsilon)
        return combine_heads(d, sr, sk, options=o, threshold=self.d_threshold)

    def predict(self, expression, module_scores):
        return self.from_inputs(*self.inputs(expression, module_scores))

    def save(self, path):
        self._validate_state()
        obj = dict(options=asdict(self.options), raw=self.raw, knowledge=self.knowledge,
                   d_threshold=self.d_threshold, fit_iterations=self.fit_iterations,
                   module_definitions=self.module_definitions)
        Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')

    @classmethod
    def load(cls, path):
        obj = json.loads(Path(path).read_text(encoding='utf-8'))
        model = cls(PredictionOptions(**obj['options']))
        for name in ['raw', 'knowledge', 'd_threshold', 'fit_iterations', 'module_definitions']:
            setattr(model, name, obj[name])
        model._validate_state()
        return model

    def attribute(self, expression, module_scores, *, nodes=64, tolerance=1e-8):
        if len(expression) != 1:
            raise ValueError('One sample required')
        z, e = self.inputs(expression, module_scores)
        return self.attribute_inputs(z[0], e[0], sample_id=str(expression.index[0]), nodes=nodes, tolerance=tolerance)

    def attribute_inputs(self, expression_z, evidence_z, *, sample_id, nodes=64, tolerance=1e-8):
        z, e = np.asarray(expression_z, float), np.asarray(evidence_z, float)
        if (z.shape != (len(self.raw['genes']),) or e.shape != (len(self.knowledge['modules']),)
                or not np.isfinite(z).all() or not np.isfinite(e).all()
                or type(nodes) is not int or nodes < 1 or not np.isfinite(tolerance) or tolerance <= 0):
            raise ValueError('Aligned finite coordinates and positive quadrature settings required')
        o, r, k = self.options, self.raw, self.knowledge
        a, eta = o.low_state_position, o.knowledge_fraction
        beta, w = np.asarray(r['coef']), np.asarray(k['weight'])
        def integrate(count):
            points, weights = np.polynomial.legendre.leggauss(count)
            t, weights = (points+1)/2, weights/2
            p = softmax(t[:, None]*(z@beta.T)+r['intercept'], axis=1)
            d = p[:, 1:].sum(1)
            if (d <= o.epsilon).any():
                raise ValueError('Expression probability denominator reaches clipping boundary')
            sr = p[:, 2]/d
            kp = expit(t*(e@w)+k['bias'])
            sk = np.clip(kp, o.epsilon, 1-o.epsilon)
            s = (1-eta)*sr+eta*sk
            dp = p[:, :, None]*(beta[None, :, :]-(p@beta)[:, None, :])
            dd = dp[:, 1:, :].sum(1)
            ds = (dp[:, 2, :]*d[:, None]-p[:, 2, None]*dd)/(d[:, None]**2)
            raw_gradient = (a+(1-a)*s)[:, None]*dd+(d*(1-a)*(1-eta))[:, None]*ds
            active = (kp > o.epsilon) & (kp < 1-o.epsilon)
            knowledge_gradient = (d*(1-a)*eta*kp*(1-kp)*active)[:, None]*w
            return z*(weights@raw_gradient), e*(weights@knowledge_gradient)
        first, second = integrate(nodes), integrate(2*nodes)
        error = max(np.max(np.abs(x-y)) for x, y in zip(first, second))
        baseline = float(self.from_inputs(np.zeros_like(z), np.zeros_like(e))['risk'])
        risk = float(self.from_inputs(z, e)['risk'])
        residual = risk-baseline-sum(v.sum() for v in second)
        if abs(residual) > tolerance or error > tolerance:
            raise ValueError('Increase attribution quadrature nodes')
        return dict(sample_id=str(sample_id), baseline=baseline, risk=risk,
                    expression_contributions=dict(zip(r['genes'], second[0].tolist())),
                    module_contributions=dict(zip(k['modules'], second[1].tolist())),
                    completeness_residual=float(residual), quadrature_difference=float(error), nodes=2*nodes)

    def conditional_module_attribution(self, expression_z, evidence_z, *, sample_id):
        z, e = np.asarray(expression_z, float), np.asarray(evidence_z, float)
        if (z.shape != (len(self.raw['genes']),) or e.shape != (len(self.knowledge['modules']),)
                or not np.isfinite(z).all() or not np.isfinite(e).all()):
            raise ValueError('Aligned finite fitted coordinates required')
        o, k = self.options, self.knowledge
        prediction = self.from_inputs(z, e)
        evidence_baseline = -np.asarray(k['center'])/np.asarray(k['scale'])
        baseline = self.from_inputs(z, evidence_baseline)
        v = (e-evidence_baseline)*np.asarray(k['weight'])
        delta = float(v.sum())
        if abs(delta) > 1e-10:
            gain = (float(prediction['knowledge_S'])-float(baseline['knowledge_S']))/delta
        else:
            p = float(expit(k['bias']+evidence_baseline@k['weight']))
            gain = p*(1-p) if o.epsilon < p < 1-o.epsilon else 0.0
        contribution = float(prediction['D'])*(1-o.low_state_position)*o.knowledge_fraction*gain*v
        residual = float(prediction['risk']-baseline['risk']-contribution.sum())
        if abs(residual) > 1e-9:
            raise ValueError('Conditional attribution completeness failed')
        return dict(sample_id=str(sample_id), attribution_scope='knowledge_conditional',
                    baseline=float(baseline['risk']), risk=float(prediction['risk']),
                    module_contributions=dict(zip(k['modules'], contribution.tolist())),
                    completeness_residual=residual)
