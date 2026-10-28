from dataclasses import dataclass, asdict, replace
from importlib.resources import files
import hashlib
import re
import numpy as np
from .core import PathwayPlanner, contrast_margin, verify_margin
from .registry import compile_modules
from .retrieval import chunk_sources, rank_chunks

def prompt(name):
    return files('kgm_orbit.resources').joinpath('prompts', name + '.txt').read_text(encoding='utf-8')

@dataclass(frozen=True)
class EvidenceOptions:
    max_rounds: int
    patience: int
    chunk_characters: int
    chunk_overlap: int
    top_k: int
    tau: float

    def __post_init__(self):
        sizes = [self.max_rounds, self.patience, self.chunk_characters, self.top_k]
        if any(type(v) is not int or v < 1 for v in sizes):
            raise ValueError('Positive integer retrieval settings required')
        if not 0 <= self.chunk_overlap < self.chunk_characters:
            raise ValueError('Chunk overlap must be smaller than chunk size')
        if not np.isfinite(self.tau) or self.tau < 0:
            raise ValueError('Nonnegative finite verification threshold required')


def _store_sources(store, records):
    for source in records:
        record = asdict(source)
        if source.source_id in store and store[source.source_id]['text'] != source.text:
            raise ValueError('Source ID has conflicting text')
        store[source.source_id] = record


def _margin(scorer, text, claim):
    if hasattr(scorer, 'margin'):
        return scorer.margin(text, claim['positive'], claim['negative'])
    return contrast_margin(lambda value: scorer(text, value), claim['positive'], claim['negative'])


def _same_context(value, expected, aliases=()):
    def normalize(text):
        return re.sub(r'\s+', ' ', str(text or '').strip().casefold()).rstrip('.')
    return normalize(value) in {normalize(text) for text in [expected, *aliases]}


def _article_key(source_id):
    return source_id.split(':')[0] if re.match(r'^PMC\d+:', source_id) else source_id


def _verify_claim(claim, unit, query, *, backend, retrieve, scorer, options, sources, context_units=()):
    records = retrieve(query + ' ' + claim['subject'] + ' opposite contradictory evidence')
    _store_sources(sources, records)
    chunks = chunk_sources(records, characters=options.chunk_characters, overlap=options.chunk_overlap)
    field = backend.complete(prompt('ava'), dict(claim=claim, source=unit,
                                              context_units=list(context_units), counters=chunks))
    by_unit = {c['unit_id']: c for c in chunks}
    counters = []
    for item in field.get('counter_units', []):
        if item['unit_id'] not in by_unit:
            raise ValueError('Counterevidence references an unknown source unit')
        counter = by_unit[item['unit_id']]
        counters.append(dict(
            source_id=counter['unit_id'], document_id=counter['source_id'],
            margin=_margin(scorer, counter['text'], claim),
            comparable=item['comparable'] is True,
        ))
    decision = verify_margin(
        claim['support_margin'], counters, options.tau,
        fields_pass=claim['fields_pass'] and field['supported'] is True,
    )
    claim.update(decision=decision, source_supported=decision == 'accepted',
                 counter_margins=counters, verification=field)

def build_evidence(task, *, backend, retrieve, scorer, options):
    if options.max_rounds < 1 or options.patience < 1:
        raise ValueError('Positive retrieval budgets required')
    if set(task['heads']) != {'D', 'S'}:
        raise ValueError('Both task heads required')
    seed_query = f"{task['disease']} {task['tissue']} pathways gene expression"
    initial_store = {}
    for query in [seed_query, f"{task['disease']} gene function pathway enzyme signaling"]:
        _store_sources(initial_store, retrieve(query))
    from .retrieval import Source
    initial = [Source(**record) for record in initial_store.values()]
    discovery_chunks = chunk_sources(initial,
        characters=options.chunk_characters,
        overlap=options.chunk_overlap)
    discovery_task = {key: task[key] for key in ['disease', 'tissue', 'species'] if key in task}
    discovery = backend.complete(prompt('depa'), dict(task=discovery_task, sources=discovery_chunks))
    source_ids = {s.source_id for s in initial}
    for edge in discovery['edges']:
        if edge.get('source') not in source_ids:
            raise ValueError('Unknown pathway edge source')
    modules = {}
    for m in discovery['modules']:
        if not set(m['source_ids']) or not set(m['source_ids']) <= source_ids:
            raise ValueError('Unknown module source')
        modules[m['module_id']] = m['genes']
    planner = PathwayPlanner(discovery['edges'], modules)
    all_claims = {}
    rejected_claims = []
    candidate_context = {}
    trace = []
    stop_reasons = {}
    sources = {s.source_id: asdict(s) for s in initial}
    for (head, context) in task['heads'].items():
        stop_reasons[head] = 'max_rounds'
        evidence = set()
        attempted = set()
        seen = set()
        empty = 0
        if not modules:
            stop_reasons[head] = 'no_supported_modules'
            continue
        for turn in range(options.max_rounds):
            query = f"{task['disease']} {task['tissue']} {context['comparison']} gene expression"
            if turn:
                gaps = {m: {g: ['expression_direction'] for g in genes if g not in evidence} for (m,
                    genes) in modules.items()}
                choice = planner.choose(evidence, gaps, attempted)
                if choice is None:
                    stop_reasons[head] = 'no_eligible_gap'
                    break
                attempted.add((choice['module'], choice['gene']))
                query += ' ' + choice['gene'] + ' ' + choice['module']
            records = retrieve(query)
            _store_sources(sources, records)
            chunks = rank_chunks(query,
                chunk_sources(records, characters=options.chunk_characters, overlap=options.chunk_overlap),
                top_k=options.top_k)
            by_unit = {c['unit_id']: c for c in chunks}
            structured = (backend.complete(prompt('esa'),
                dict(task=task, head=head, context=context, modules=modules, chunks=chunks))['claims']
                if chunks else [])
            round_ids = set()
            training_records = []
            for claim in structured:
                claim = dict(claim, head=head, task_key=context['task_key'])
                if (not isinstance(claim.get('module_id'), str)
                        or claim['module_id'] not in modules
                        or not all(isinstance(claim.get(k), str) and claim[k].strip()
                                   for k in ['subject', 'claim'])
                        or not all(isinstance(claim.get(k), list) and len(claim[k]) == 2
                                   and all(isinstance(p, str) and p.strip() for p in claim[k])
                                   for k in ['positive', 'negative'])):
                    rejected_claims.append(dict(claim=claim, reason='incomplete_or_unassigned_claim'))
                    continue
                unit = by_unit.get(claim.get('unit_id'))
                if unit is None:
                    raise ValueError('Claim references unknown source unit')
                if claim.get('source_id') != unit['source_id']:
                    raise ValueError('Claim source/unit mismatch')
                claim['unit_ids'] = [unit['unit_id']]
                identity = '|'.join([head,
                    context['task_key'],
                    claim['module_id'],
                    unit['unit_id'],
                    claim['subject'],
                    claim['claim']])
                claim['claim_id'] = hashlib.sha256(identity.encode()).hexdigest()[:24]
                bound = bool(claim.get('quote')) and claim['quote'] in unit['text']
                context_ids = claim.get('context_unit_ids', [])
                if isinstance(context_ids, list):
                    normalized = []
                    for cid in context_ids:
                        candidates = [key for key, value in by_unit.items() if value['source_id'] == cid]
                        normalized.append(candidates[0] if cid not in by_unit and len(candidates) == 1 else cid)
                    context_ids = normalized
                    claim['context_unit_ids'] = context_ids
                article = _article_key(unit['source_id'])
                context_valid = (isinstance(context_ids, list)
                                 and all(cid in by_unit and _article_key(by_unit[cid]['source_id']) == article
                                         for cid in context_ids))
                context_units = [by_unit[cid] for cid in context_ids] if context_valid else []
                claim['context_units'] = context_units
                fields = bound and claim.get('subject') in modules.get(claim.get('module_id'), [])
                fields = fields and context_valid
                if 'species' in task:
                    fields = (fields and bool(task['species']) and bool(claim.get('species'))
                              and _same_context(claim.get('species'), task['species'], task.get('species_aliases', [])))
                if claim.get('relation') == 'expression_change':
                    fields = (fields
                              and _same_context(claim.get('tissue'), task['tissue'], task.get('tissue_aliases', []))
                              and _same_context(claim.get('comparison'), context['comparison'], context.get('comparison_aliases', [])))
                if hasattr(scorer, 'update') and not fields:
                    rejected_claims.append(dict(claim=claim, reason='source_or_context_mismatch'))
                    continue
                claim['fields_pass'] = bool(fields)
                all_claims[claim['claim_id']] = claim
                candidate_context[claim['claim_id']] = (unit, query)
                round_ids.add(claim['claim_id'])
                if fields:
                    training_records.append(dict(source_id=unit['source_id'], source=unit['text'],
                        positive=claim['positive'], negative=claim['negative']))
            if training_records and hasattr(scorer, 'update'):
                options = replace(options, tau=scorer.update(training_records))
            for claim in all_claims.values():
                unit, _ = candidate_context[claim['claim_id']]
                margin = _margin(scorer, unit['text'], claim)
                decision = verify_margin(margin, [], options.tau, fields_pass=claim['fields_pass'])
                claim.update(support_margin=margin, decision=decision, source_supported=decision == 'accepted')
            accepted = [all_claims[cid] for cid in round_ids if all_claims[cid]['decision'] == 'accepted']
            evidence.clear()
            current_modules = compile_modules(
                list(all_claims.values()), tau=options.tau,
                task_keys={head: context['task_key']},
            )
            evidence.update(gene for module in current_modules for gene in module.genes)
            ids = {c['claim_id'] for c in accepted}
            new = len(ids - seen)
            seen |= ids
            empty = 0 if new else empty + 1
            trace.append(dict(head=head,
                round=turn + 1,
                tau=options.tau,
                query=query,
                new_claims=new,
                accepted=len(accepted), evidence_genes=sorted(evidence),
                remaining_gaps={m: sorted(set(genes) - evidence) for m, genes in modules.items()}))
            if empty >= options.patience:
                stop_reasons[head] = 'no_new_evidence'
                break
    claims = list(all_claims.values())
    for claim in claims:
        unit, query = candidate_context[claim['claim_id']]
        _verify_claim(claim, unit, query, backend=backend, retrieve=retrieve,
                      scorer=scorer, options=options, sources=sources,
                      context_units=claim.get('context_units', []))
    compiled = compile_modules(claims,
        tau=options.tau,
        task_keys={h: c['task_key'] for (h, c) in task['heads'].items()})
    return dict(claims=claims,
        task=task,
        discovery=discovery,
        rejected_claims=rejected_claims,
        modules=[m.to_dict() for m in compiled],
        sources=sources,
        trace=trace,
        stop_reasons=stop_reasons,
        scorer_training=getattr(scorer, 'history', []),
        options=asdict(options))
