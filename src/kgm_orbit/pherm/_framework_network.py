import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ._framework_reference import Branch, Hierarchy

class Calibration(nn.Module):

    def __init__(self, modules, nodes, confidence):
        super().__init__()
        if nodes < 1:
            raise ValueError('Positive calibration node count required')
        initial = torch.linspace(-0.5, 0.5, nodes).repeat(modules, 1)
        self.a = nn.Parameter(initial.clone())
        self.s = nn.Parameter(initial.clone())
        self.b = nn.Parameter(initial.clone())
        self.register_buffer('confidence', torch.as_tensor(confidence, dtype=torch.float64))

    def forward(self, scores):
        return self.confidence * ((F.softplus(scores[:,
            :,
            None] * F.softplus(self.s) + self.b) - F.softplus(self.b)) * F.softplus(self.a)).sum(-1)

class SparseBranch(nn.Module):

    def __init__(self, mask):
        super().__init__()
        mask = torch.as_tensor(mask, dtype=torch.float64)
        n = mask.shape[0]
        self.register_buffer('mask', mask)
        self.weight = nn.Parameter(torch.full((n, n), -2.0, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros(n, dtype=torch.float64))
        self.output = nn.Parameter(torch.full((n,), -1.0, dtype=torch.float64))
        self.intercept = nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self, evidence):
        hidden = F.softplus(evidence @ (F.softplus(self.weight) * self.mask).T + self.bias) - F.softplus(self.bias)
        return hidden @ F.softplus(self.output) + self.intercept

    def reference(self):
        array = lambda x: x.detach().cpu().numpy()
        return Branch(array(F.softplus(self.weight)),
            array(self.mask).astype(bool),
            array(self.bias),
            array(F.softplus(self.output)),
            float(self.intercept.detach()))

class Network(nn.Module):

    def __init__(self, *, confidence_d, confidence_s, adjacency_d, adjacency_s, nodes):
        super().__init__()
        self.calibration_d = Calibration(len(confidence_d), nodes, confidence_d)
        self.calibration_s = Calibration(len(confidence_s), nodes, confidence_s)
        self.branch_d = SparseBranch(adjacency_d)
        self.branch_s = SparseBranch(adjacency_s)
        self.theta = nn.Parameter(torch.zeros(len(confidence_s), dtype=torch.float64))
        self.lambda_bias = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.cut_low = nn.Parameter(torch.tensor(-0.5, dtype=torch.float64))
        self.cut_gap = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.double()

    def from_evidence(self, ed, es):
        d = self.branch_d(ed)
        s = self.branch_s(es)
        lam = torch.sigmoid(es @ self.theta + self.lambda_bias)
        risk = d + lam * (1 + torch.tanh(d)) / 2 * torch.tanh(s)
        return dict(D=d, S=s, lambda_value=lam, risk=risk)

    def forward(self, md, ms):
        ed = self.calibration_d(md)
        es = self.calibration_s(ms)
        return dict(**self.from_evidence(ed, es), evidence_d=ed, evidence_s=es)

    def thresholds(self):
        return (self.cut_low, self.cut_low + F.softplus(self.cut_gap))

    def classify(self, risk):
        (lo, hi) = self.thresholds()
        if not torch.isfinite(risk).all() or not torch.isfinite(torch.stack([lo, hi])).all():
            raise ValueError('Nonfinite prediction or thresholds')
        return (risk > lo).long() + (risk > hi).long()

    def reference(self):
        return Hierarchy(self.branch_d.reference(),
            self.branch_s.reference(),
            self.theta.detach().cpu().numpy(),
            float(self.lambda_bias.detach()))

    def parameter_dict(self):
        return {k: v.detach().cpu().numpy().tolist() for (k, v) in self.state_dict().items()}

    def load_parameter_dict(self, values):
        state = self.state_dict()
        if set(state) != set(values):
            raise ValueError('Model parameter fields do not match')
        loaded = {k: torch.tensor(values[k], dtype=v.dtype) for k, v in state.items()}
        buffers = dict(self.named_buffers())
        for key, value in loaded.items():
            if value.shape != state[key].shape or not torch.isfinite(value).all():
                raise ValueError('Invalid model parameter: ' + key)
            if key in buffers and not torch.equal(value, buffers[key]):
                raise ValueError('Model structure does not match modules: ' + key)
        self.load_state_dict(loaded)
        return self
