from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from ..data import GeneStandardizer
from ..evidence.registry import Module, module_inputs, adjacency
from ._framework_network import Network

@dataclass(frozen=True)
class FrameworkOptions:
    nodes: int = 4
    jaccard_threshold: float = 0.2
    calibration_epochs: int = 100
    prediction_epochs: int = 200
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    loss_d: float = 1.0
    loss_s: float = 1.0
    loss_ordinal: float = 1.0
    seed: int = 0

    def __post_init__(self):
        for key in ['nodes', 'calibration_epochs', 'prediction_epochs', 'seed']:
            if type(getattr(self, key)) is not int:
                raise ValueError('Integer required: '+key)
        if (
            self.nodes < 1
            or min(self.calibration_epochs, self.prediction_epochs) < 1
            or self.learning_rate <= 0
            or self.weight_decay < 0
            or min(self.loss_d, self.loss_s, self.loss_ordinal) < 0
            or self.loss_d + self.loss_s + self.loss_ordinal <= 0
        ):
            raise ValueError('Invalid training options')
        if not 0 <= self.jaccard_threshold <= 1:
            raise ValueError('Jaccard threshold must be in [0,1]')
        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError('Nonfinite training options')

def balanced_bce(logits, targets):
    counts = torch.stack([(targets == k).sum() for k in [0, 1]])
    if (counts == 0).any():
        raise ValueError('Binary objective needs both classes')
    weights = len(targets) / (2 * counts[targets.long()])
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return (loss * weights).mean()

class PHERM:

    def __init__(self, modules, options=None):
        self.modules = list(modules)
        self.options = FrameworkOptions() if options is None else options
        keys = [(m.module_id, m.head) for m in self.modules]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate module/head')

    def fit(self, expression, labels):
        if isinstance(labels, pd.Series) and not labels.index.equals(expression.index):
            raise ValueError('Expression and label IDs must align')
        y = np.asarray(labels)
        if y.shape != (len(expression),) or set(y.tolist()) != {0, 1, 2}:
            raise ValueError('Aligned three-state labels required')
        if expression.index.has_duplicates:
            raise ValueError('Use one row per patient')
        torch.manual_seed(self.options.seed)
        self.standardizer = GeneStandardizer().fit(expression)
        inputs = module_inputs(self.standardizer, expression, self.modules)
        model = Network(confidence_d=inputs['D']['confidence'],
            confidence_s=inputs['S']['confidence'],
            adjacency_d=adjacency(self.modules, 'D', self.options.jaccard_threshold),
            adjacency_s=adjacency(self.modules, 'S', self.options.jaccard_threshold),
            nodes=self.options.nodes)
        md = torch.tensor(inputs['D']['scores'])
        ms = torch.tensor(inputs['S']['scores'])
        yt = torch.tensor(y)
        case = yt > 0
        trace = []
        calibration_tasks = [
            (model.calibration_d, md, torch.ones(len(y), dtype=torch.bool), (yt > 0).double()),
            (model.calibration_s, ms, case, (yt == 2).double()),
        ]
        for calibration, scores, mask, target in calibration_tasks:
            optimizer = torch.optim.Adam(calibration.parameters(),
                lr=self.options.learning_rate,
                weight_decay=self.options.weight_decay)
            for _ in range(self.options.calibration_epochs):
                optimizer.zero_grad()
                ev = calibration(scores)
                loss = torch.stack([
                    balanced_bce(ev[mask, j], target[mask])
                    for j in range(ev.shape[1])
                ]).mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite calibration loss')
                loss.backward()
                optimizer.step()
            for p in calibration.parameters():
                p.requires_grad_(False)
        with torch.no_grad():
            evidence_d = model.calibration_d(md)
            evidence_s = model.calibration_s(ms)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
            lr=self.options.learning_rate,
            weight_decay=self.options.weight_decay)
        for epoch in range(self.options.prediction_epochs):
            optimizer.zero_grad()
            v = model.from_evidence(evidence_d, evidence_s)
            (lo, hi) = model.thresholds()
            ld = balanced_bce(v['D'], (yt > 0).double())
            ls = balanced_bce(v['S'][case], (yt[case] == 2).double())
            lr = (
                balanced_bce(v['risk'] - lo, (yt > 0).double())
                + balanced_bce(v['risk'] - hi, (yt > 1).double())
            ) / 2
            loss = self.options.loss_d * ld + self.options.loss_s * ls + self.options.loss_ordinal * lr
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite prediction loss')
            loss.backward()
            optimizer.step()
            trace.append(dict(epoch=epoch + 1,
                total=float(loss.detach()),
                D=float(ld.detach()),
                S=float(ls.detach()),
                ordinal=float(lr.detach())))
        self.model = model.eval()
        self.training_trace = trace
        return self

    def predict(self, expression):
        if not hasattr(self, 'model'):
            raise ValueError('Fit or load PHERM before prediction')
        inputs = module_inputs(self.standardizer, expression, self.modules)
        with torch.no_grad():
            values = self.model(torch.tensor(inputs['D']['scores']), torch.tensor(inputs['S']['scores']))
            if any(not torch.isfinite(v).all() for v in values.values()):
                raise ValueError('Nonfinite model output')
            values['prediction'] = self.model.classify(values['risk'])
            lo, hi = self.model.thresholds()
            above_low, above_high = torch.sigmoid(values['risk']-lo), torch.sigmoid(values['risk']-hi)
            values['probabilities'] = torch.stack([1-above_low, above_low-above_high, above_high], dim=1)
        return {k: v.numpy() for (k, v) in values.items()}

    def to_dict(self):
        return dict(architecture='framework', modules=[m.to_dict() for m in self.modules],
            options=asdict(self.options), standardizer=self.standardizer.to_dict(),
            parameters=self.model.parameter_dict(), training_trace=self.training_trace)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), allow_nan=False), encoding='utf-8')

    @classmethod
    def from_dict(cls, p):
        obj = cls([Module(**m) for m in p['modules']], FrameworkOptions(**p['options']))
        obj.standardizer = GeneStandardizer.from_dict(p['standardizer'])
        obj.model = Network(confidence_d=[np.mean(m.weights) for m in obj.modules if m.head == 'D'],
            confidence_s=[np.mean(m.weights) for m in obj.modules if m.head == 'S'],
            adjacency_d=adjacency(obj.modules, 'D', obj.options.jaccard_threshold),
            adjacency_s=adjacency(obj.modules, 'S', obj.options.jaccard_threshold), nodes=obj.options.nodes)
        obj.model.load_parameter_dict(p['parameters']).eval()
        obj.training_trace = p['training_trace']
        return obj

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    def attribute(self, expression, *, nodes=64, tolerance=1e-8):
        if len(expression) != 1 or tolerance <= 0 or not np.isfinite(tolerance):
            raise ValueError('One sample and positive finite tolerance required')
        values = self.predict(expression)
        reference = self.model.reference()
        a = reference.attribute(values['evidence_d'][0], values['evidence_s'][0], nodes)
        b = reference.attribute(values['evidence_d'][0], values['evidence_s'][0], 2*nodes)
        error = max(np.max(np.abs(a[k]-b[k])) for k in ['D','S_via_S','S_via_lambda'])
        if error > tolerance or abs(b['residual']) > tolerance:
            raise ValueError('Increase attribution quadrature nodes')
        return dict(sample_id=str(expression.index[0]), **{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in b.items()},
                    quadrature_difference=float(error))
