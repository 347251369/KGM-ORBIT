from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n',
        encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def claim_text(c):
    return '; '.join((f"{k}: {c.get(k, 'unknown')}" for k in ('subject',
        'relation',
        'object',
        'role',
        'direction',
        'tissue',
        'comparison')))

class PathwayPlanner:

    def __init__(self, edges, modules):
        self.modules = {k: set(v) for (k, v) in modules.items()}
        self.adjacency = {}
        for e in edges:
            if not e.get('source') or e.get('relation') not in {'same_reaction',
                'small_complex',
                'curated_direct_relation',
                'reactome_inferred_interaction'}:
                raise ValueError('A pathway-membership clique is not an eligible edge')
            (a, b) = (e['gene_a'], e['gene_b'])
            if a == b:
                continue
            self.adjacency.setdefault(a, set()).add(b)
            self.adjacency.setdefault(b, set()).add(a)

    def choose(self, evidence, gaps, attempted=(), *, use_graph=True):
        candidates = []
        for (module, gene_gaps) in sorted(gaps.items()):
            members = self.modules[module]
            for (gene, missing) in sorted(gene_gaps.items()):
                if gene not in members or not missing or (module, gene) in attempted:
                    continue
                links = self.adjacency.get(gene, set()) & set(evidence) & members
                if use_graph and (not links):
                    continue
                candidates.append((-len(links) if use_graph else 0,
                    module,
                    gene,
                    sorted(missing),
                    sorted(links)))
        if not candidates:
            return None
        (_, module, gene, missing, links) = min(candidates)
        return dict(module=module, gene=gene, missing=missing, evidence_neighbors=links)

def update_evidence(state, claims, task_key):
    before = set(state)
    for c in claims:
        if c.get('task_key') == task_key and c.get('source_id') and c.get('unit_ids') and (c.get('source_supported') is True) and (c.get('relation') == 'expression_change') and (c.get('measurement') == 'RNA') and (c.get('direction') in ('increasing',
            'decreasing')):
            state.add(c['subject'])
    return len(state - before)

class ContrastiveHead:

    def __init__(self, l2=0.1):
        self.l2 = l2

    def fit(self, features, groups):
        if not groups or any((len(g) < 2 for g in groups)):
            raise ValueError('At least one positive/negative group is required')
        x = np.asarray(features, dtype=np.float64)
        ids = sorted({i for g in groups for i in g})
        self.mean = x[ids].mean(0)
        self.scale = np.maximum(x[ids].std(0), 0.05)
        z = (x - self.mean) / self.scale

        def objective(w):
            loss = 0.0
            grad = np.zeros_like(w)
            for g in groups:
                xx = z[g]
                s = xx @ w
                logz = logsumexp(s)
                loss += logz - s[0]
                grad += np.exp(s - logz) @ xx - xx[0]
            return (loss / len(groups) + self.l2 * np.dot(w, w) / 2, grad / len(groups) + self.l2 * w)
        result = minimize(objective,
            np.zeros(x.shape[1]),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-06})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError('Contrastive optimizer failed: ' + str(result.message))
        self.weight = result.x
        self.training = dict(initial_loss=float(objective(np.zeros(x.shape[1]))[0]),
            final_loss=float(result.fun),
            iterations=int(result.nit),
            groups=len(groups))
        return self

    def score(self, x):
        return (np.asarray(x) - self.mean) / self.scale @ self.weight

    def dump(self, path):
        save(path,
            dict(l2=self.l2, mean=self.mean.tolist(), scale=self.scale.tolist(), weight=self.weight.tolist(), training=self.training))

    @classmethod
    def load(cls, path):
        d = read(path)
        obj = cls(d['l2'])
        for k in ('mean', 'scale', 'weight'):
            setattr(obj, k, np.array(d[k]))
        obj.training = d['training']
        return obj

def contrast_margin(score, positive_paraphrases, negative_paraphrases):
    if len(positive_paraphrases) != 2 or len(negative_paraphrases) != 2:
        raise ValueError('Equation 4 requires two paired paraphrases')
    vals = [score(p) - score(n) for (p, n) in zip(positive_paraphrases, negative_paraphrases)]
    if not np.isfinite(vals).all():
        raise ValueError('Nonfinite support score')
    return float(np.mean(vals))

def verify_margin(support_margin, counter_margins, tau, *, fields_pass=True):
    if not np.isfinite(support_margin) or not np.isfinite(tau) or tau < 0:
        raise ValueError('Invalid margin/threshold')
    seen = {}
    for row in counter_margins:
        if not row.get('source_id') or not np.isfinite(row['margin']):
            raise ValueError('Invalid counter provenance')
        if row.get('comparable') is True:
            sid = row['source_id']
            value = row['margin']
            if sid in seen and seen[sid] != value:
                raise ValueError('Same source/claim scored inconsistently')
            seen[sid] = value
    conflict = support_margin > tau and any((v < -tau for v in seen.values()))
    if conflict:
        return 'conflict'
    if any((v == -tau for v in seen.values())):
        return 'unresolved'
    if support_margin > tau and fields_pass:
        return 'accepted'
    if support_margin < -tau:
        return 'refuted'
    return 'unresolved'
