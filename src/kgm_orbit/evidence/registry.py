from collections import defaultdict
from dataclasses import dataclass, asdict
import numpy as np
from ..pherm.formulas import claim_weight, aggregate

@dataclass(frozen=True)
class Module:
    module_id: str
    head: str
    genes: tuple
    directions: tuple
    weights: tuple
    claim_ids: tuple

    def __post_init__(self):
        n = len(self.genes)
        if not n or self.head not in {'D',
            'S'} or len(set(self.genes)) != n or any((len(x) != n for x in [self.directions, self.weights, self.claim_ids])) or (not np.isin(self.directions, [-1, 1]).all()) or (not np.isfinite(self.weights).all()) or (min(self.weights) <= 0):
            raise ValueError('Invalid module membership')

    def to_dict(self):
        return asdict(self)

def compile_modules(claims, *, tau, task_keys):
    grouped = defaultdict(list)
    ids = set()
    for c in claims:
        if c['claim_id'] in ids:
            raise ValueError('Duplicate claim ID')
        ids.add(c['claim_id'])
        h = c.get('head')
        if h not in task_keys or c.get('task_key') != task_keys[h] or c.get('decision') != 'accepted' or (c.get('relation') != 'expression_change') or (c.get('measurement') != 'RNA') or (not c.get('source_id')) or (not c.get('unit_ids')) or (c.get('direction') not in {'increasing',
            'decreasing'}):
            continue
        grouped[c['module_id'], h, c['subject']].append(c)
    modules = defaultdict(list)
    for ((m, h, g), rows) in sorted(grouped.items()):
        signs = {c['direction'] for c in rows}
        if len(signs) != 1:
            continue
        w = claim_weight([c['support_margin'] for c in rows], tau)
        modules[m,
            h].append((g, 1 if signs.pop() == 'increasing' else -1, w, tuple((c['claim_id'] for c in rows))))
    return [Module(m, h, *map(tuple, zip(*rows))) for ((m, h), rows) in sorted(modules.items())]

def module_inputs(standardizer, expression, modules):
    (z, measured) = standardizer.transform(expression)
    index = {g: i for (i, g) in enumerate(standardizer.genes)}
    result = {}
    for h in ['D', 'S']:
        selected = [m for m in modules if m.head == h]
        if not selected:
            raise ValueError(f'No modules for {h}')
        scores = []
        coverage = []
        confidence = []
        for m in selected:
            v = np.zeros((len(expression), len(m.genes)))
            mask = np.zeros_like(v, dtype=bool)
            for (j, g) in enumerate(m.genes):
                if g in index:
                    v[:, j] = z[:, index[g]]
                    mask[:, j] = measured[:, index[g]]
            out = aggregate(v, m.weights, m.directions, mask)
            scores.append(out['score'])
            coverage.append(out['coverage'])
            confidence.append(out['confidence'])
        result[h] = dict(scores=np.stack(scores, 1),
            coverage=np.stack(coverage, 1),
            confidence=np.asarray(confidence),
            module_ids=[m.module_id for m in selected])
    return result

def adjacency(modules, head, threshold):
    if not 0 <= threshold <= 1:
        raise ValueError('Jaccard threshold must be in [0,1]')
    genes = [set(m.genes) for m in modules if m.head == head]
    return np.array([[i == j or (bool(a & b) and len(a & b) / len(a | b) >= threshold) for (j, b) in enumerate(genes)] for (i, a) in enumerate(genes)],
        bool)


def source_dependencies(claim):
    return {claim['source_id']} | {u['source_id'] for u in claim.get('context_units', [])}


def proposition(claim):
    fields = ['module_id', 'head', 'subject', 'relation', 'measurement', 'direction',
              'task_key', 'comparison', 'tissue', 'species']
    return {key: claim.get(key) for key in fields}


def validate_bundle(bundle, modules=None):
    claims = {c['claim_id']: c for c in bundle['claims']}
    if len(claims) != len(bundle['claims']):
        raise ValueError('Duplicate claim ID')
    stored = [Module(**m) for m in bundle['modules']]
    keys = [(m.module_id, m.head) for m in stored]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate module/head')
    if modules is not None:
        canonical = lambda items: sorted((m.module_id, m.head, tuple(m.genes), tuple(m.directions),
                                         tuple(m.weights), tuple(tuple(ids) for ids in m.claim_ids)) for m in items)
        if canonical(stored) != canonical(modules):
            raise ValueError('Evidence modules do not match the fitted model')
    for claim in claims.values():
        if not source_dependencies(claim) <= set(bundle['sources']):
            raise ValueError('Missing claim source or context source')
        for unit in claim.get('context_units', []):
            if unit.get('text', '') not in bundle['sources'][unit['source_id']]['text']:
                raise ValueError('Context text does not match its source')
    for module in stored:
        for gene, sign, ids in zip(module.genes, module.directions, module.claim_ids):
            if not ids or len(set(ids)) != len(ids):
                raise ValueError('Unique evidence claims required for each gene')
            for cid in ids:
                c = claims.get(cid, {})
                expected = dict(module_id=module.module_id, head=module.head, subject=gene,
                                direction='increasing' if sign == 1 else 'decreasing',
                                decision='accepted', relation='expression_change', measurement='RNA')
                if any(c.get(k) != v for k, v in expected.items()) or not c.get('unit_ids'):
                    raise ValueError('Claim does not match model membership: ' + str(cid))
                task = bundle.get('task', {})
                if task:
                    from .workflow import _same_context
                    context = task['heads'][module.head]
                    if (c.get('task_key') != context['task_key']
                            or not _same_context(c.get('comparison'), context['comparison'], context.get('comparison_aliases', []))
                            or not _same_context(c.get('tissue'), task['tissue'], task.get('tissue_aliases', []))
                            or ('species' in task and (not c.get('species') or not _same_context(c['species'], task['species'], task.get('species_aliases', []))))):
                        raise ValueError('Claim does not match evidence task: ' + str(cid))
    return claims
