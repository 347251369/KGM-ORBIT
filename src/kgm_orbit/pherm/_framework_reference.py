from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit
from .formulas import finite

@dataclass
class Branch:
    weights: object
    adjacency: object
    hidden_bias: object
    output_weights: object
    output_bias: float

    def value_gradient(self, evidence):
        e = finite(evidence)
        (w, b, v) = map(finite, (self.weights, self.hidden_bias, self.output_weights))
        mask = np.asarray(self.adjacency)
        if e.ndim != 1 or w.shape != (e.size,
            e.size) or mask.shape != w.shape or (mask.dtype != np.bool_) or (b.shape != e.shape) or (v.shape != e.shape):
            raise ValueError('Invalid sparse branch shapes')
        w = w * mask
        t = w @ e + b
        hidden = np.logaddexp(0, t) - np.logaddexp(0, b)
        score = float(finite(self.output_bias)) + float(v @ hidden)
        gradient = v * expit(t) @ w
        return (score, gradient)

@dataclass
class Hierarchy:
    disease: Branch
    severity: Branch
    theta: object
    lambda_bias: float

    def evaluate(self, evidence_d, evidence_s):
        (ed, es, theta) = map(finite, (evidence_d, evidence_s, self.theta))
        if es.shape != theta.shape:
            raise ValueError('lambda weights must match S evidence')
        (d, gd) = self.disease.value_gradient(ed)
        (s, gs) = self.severity.value_gradient(es)
        lam = float(expit(float(finite(self.lambda_bias)) + theta @ es))
        gate = (1 + np.tanh(d)) / 2
        (td, ts) = (np.tanh(d), np.tanh(s))
        return dict(D=d,
            S=s,
            lambda_value=lam,
            R=d + lam * gate * ts,
            grad_D=(1 + lam * (1 - td * td) * ts / 2) * gd,
            grad_S_via_S=lam * gate * (1 - ts * ts) * gs,
            grad_S_via_lambda=gate * ts * lam * (1 - lam) * theta)

    def attribute(self, evidence_d, evidence_s, nodes):
        (ed, es) = map(finite, (evidence_d, evidence_s))
        if type(nodes) is not int or nodes < 2:
            raise ValueError('At least two quadrature nodes required')
        (u, w) = leggauss(nodes)
        sums = dict(D=np.zeros_like(ed), S_via_S=np.zeros_like(es), S_via_lambda=np.zeros_like(es))
        for (t, weight) in zip((u + 1) / 2, w / 2):
            values = self.evaluate(t * ed, t * es)
            for key in sums:
                sums[key] += weight * values['grad_' + key]
        sums['D'] *= ed
        sums['S_via_S'] *= es
        sums['S_via_lambda'] *= es
        base = self.evaluate(np.zeros_like(ed), np.zeros_like(es))['R']
        risk = self.evaluate(ed, es)['R']
        residual = risk - base - sum((float(x.sum()) for x in sums.values()))
        return dict(**sums, baseline=base, risk=risk, residual=residual, nodes=nodes)
