from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from .framework import PHERM
from .prediction import ProbPHERM, PredictionOptions, combine_heads


class MixPHERM:
    def __init__(self, modules, *, framework_options=None, prediction_options=None, framework_fraction=0.5):
        if not np.isfinite(framework_fraction) or not 0 <= framework_fraction <= 1:
            raise ValueError('Framework fraction must be in [0,1]')
        self.framework = PHERM(modules, framework_options)
        self.probability = ProbPHERM(prediction_options)
        self.framework_fraction = float(framework_fraction)

    def fit(self, expression, labels, module_scores, reliability, *, d_threshold):
        expected = [m.module_id for m in self.framework.modules if m.head == 'S']
        if list(module_scores.columns) != expected:
            raise ValueError('Score columns must follow S module order')
        self.probability.fit(expression, labels, module_scores, reliability, d_threshold=d_threshold,
            module_definitions=[m.to_dict() for m in self.framework.modules])
        self.framework.fit(expression, labels)
        return self

    def predict(self, expression, module_scores):
        a = self.framework.predict(expression)
        b = self.probability.predict(expression, module_scores)
        fraction = self.framework_fraction
        p = fraction*a['probabilities']+(1-fraction)*b['probabilities']
        d = p[:, 1:].sum(1)
        severity = p[:, 2]/np.maximum(d, self.probability.options.epsilon)
        result = combine_heads(d, severity, severity, options=self.probability.options,
                               threshold=self.probability.d_threshold)
        result['probabilities'] = p
        result['risk'] = self.probability.options.low_state_position*p[:, 1]+p[:, 2]
        result.pop('raw_S')
        result.pop('knowledge_S')
        result.update(framework_risk=a['risk'], probability_risk=b['risk'],
                      lambda_value=a['lambda_value'])
        return result

    def save(self, path):
        self.probability._validate_state()
        fields = ['raw', 'knowledge', 'd_threshold', 'fit_iterations', 'module_definitions']
        probability = {k: getattr(self.probability, k) for k in fields}
        probability['options'] = asdict(self.probability.options)
        payload = dict(architecture='combined', framework_fraction=self.framework_fraction,
                       framework=self.framework.to_dict(), probability=probability)
        Path(path).write_text(json.dumps(payload, allow_nan=False), encoding='utf-8')

    @classmethod
    def load(cls, path):
        p = json.loads(Path(path).read_text(encoding='utf-8'))
        framework = PHERM.from_dict(p['framework'])
        obj = cls(framework.modules, framework_options=framework.options,
                  prediction_options=PredictionOptions(**p['probability']['options']),
                  framework_fraction=p['framework_fraction'])
        obj.framework = framework
        for k in ['raw', 'knowledge', 'd_threshold', 'fit_iterations', 'module_definitions']:
            setattr(obj.probability, k, p['probability'][k])
        obj.probability._validate_state()
        if obj.probability.module_definitions != [m.to_dict() for m in framework.modules]:
            raise ValueError('Combined module definitions disagree')
        return obj

    def attribute(self, expression, module_scores, *, nodes=64, tolerance=1e-8):
        b = self.probability.attribute(expression, module_scores, nodes=nodes, tolerance=tolerance)
        values = self.framework.predict(expression)
        ed, es = values['evidence_d'][0], values['evidence_s'][0]
        network = self.framework.model
        low = self.probability.options.low_state_position
        def score(d, s):
            risk = network.from_evidence(d, s)['risk']
            lo, hi = network.thresholds()
            return low*torch.sigmoid(risk-lo)+(1-low)*torch.sigmoid(risk-hi)
        def integrate(count):
            points, weights = np.polynomial.legendre.leggauss(count)
            t = torch.tensor((points+1)/2, dtype=torch.float64)
            d = (t[:, None]*torch.tensor(ed)).requires_grad_(True)
            s = (t[:, None]*torch.tensor(es)).requires_grad_(True)
            gd, gs = torch.autograd.grad(score(d, s).sum(), (d, s))
            return ed*((weights/2)@gd.detach().numpy()), es*((weights/2)@gs.detach().numpy())
        first, second = integrate(nodes), integrate(2*nodes)
        error = max(np.max(np.abs(a-c)) for a, c in zip(first, second))
        with torch.no_grad():
            base = float(score(torch.zeros_like(torch.tensor(ed)), torch.zeros_like(torch.tensor(es))))
            risk = float(score(torch.tensor(ed), torch.tensor(es)))
        if error > tolerance or abs(risk-base-sum(x.sum() for x in second)) > tolerance:
            raise ValueError('Increase attribution quadrature nodes')
        fraction = self.framework_fraction
        framework_terms = {}
        for head, contribution in zip(['D', 'S'], second):
            names = [m.module_id for m in self.framework.modules if m.head == head]
            framework_terms[head] = dict(zip(names, (fraction*contribution).tolist()))
        expression_terms = {k:(1-fraction)*v for k,v in b['expression_contributions'].items()}
        module_terms = {k:(1-fraction)*v for k,v in b['module_contributions'].items()}
        baseline = fraction*base+(1-fraction)*b['baseline']
        predicted = float(self.predict(expression, module_scores)['risk'][0])
        total = sum(sum(v.values()) for v in framework_terms.values())+sum(expression_terms.values())+sum(module_terms.values())
        residual = predicted-baseline-total
        if abs(residual) > tolerance:
            raise ValueError('Combined attribution completeness failed')
        return dict(sample_id=str(expression.index[0]), baseline=baseline, risk=predicted,
                    framework_contributions=framework_terms, expression_contributions=expression_terms,
                    module_contributions=module_terms, completeness_residual=residual,
                    quadrature_difference=max(float(error), b['quadrature_difference']), nodes=2*nodes)
