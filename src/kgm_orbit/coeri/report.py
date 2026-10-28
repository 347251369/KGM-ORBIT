import copy
import re
import numpy as np
from ..evidence.workflow import prompt
from ..evidence.registry import module_inputs, validate_bundle, source_dependencies, proposition

def validate_statements(packet, response):
    if not isinstance(response, dict) or set(response) != {'risk_assessment', 'statements', 'conclusion'}:
        raise ValueError('Risk report requires risk_assessment, statements and conclusion')
    if not isinstance(response['statements'], list):
        raise ValueError('Report statements must be an array')
    selected = set(packet['selected_modules'])
    linked = {cid for m in packet['modules'] if m['module_id'] in selected
              for g in m['genes'] for cid in g['claim_ids'] if cid in packet['claims']}
    for key in ['risk_assessment', 'conclusion']:
        row = response[key]
        if not isinstance(row, dict) or set(row) != {'text', 'claim_ids'}:
            raise ValueError('Invalid report section: '+key)
        _validate_report_text(row)
        if not set(row['claim_ids']) <= linked:
            raise ValueError('Report section cites an unlinked claim')
    if not all(token in response['risk_assessment']['text'] for token in ['{{prediction}}', '{{risk}}']):
        raise ValueError('Risk assessment requires prediction and risk references')
    for row in response['statements']:
        _validate_report_text(row)
        if set(row) != {'module_id', 'text', 'claim_ids'}:
            raise ValueError('Invalid statement fields')
        if row['module_id'] not in selected or not row['claim_ids']:
            raise ValueError('Invalid statement module or claims')
        module = next((m for m in packet['modules'] if m['module_id'] == row['module_id']))
        allowed = {cid for g in module['genes'] for cid in g['claim_ids']}
        if not set(row['claim_ids']) <= allowed & set(packet['claims']):
            raise ValueError('Statement claim is not linked to module')
        for cid in row['claim_ids']:
            if packet['claims'][cid]['source_id'] not in packet['sources']:
                raise ValueError('Source not available')
    return response


def _validate_report_text(row):
    if (not isinstance(row, dict) or not isinstance(row.get('text'), str) or not row['text'].strip()
            or not isinstance(row.get('claim_ids'), list)
            or any(not isinstance(cid, str) for cid in row['claim_ids'])
            or len(row['claim_ids']) != len(set(row['claim_ids']))):
        raise ValueError('Report text and unique claim IDs required')


def _render_report_text(text, values):
    def substitute(match):
        key = match.group(1).strip()
        if key not in values:
            raise ValueError('Unknown report value: '+key)
        value = values[key]
        return str(value) if key == 'prediction' else f'{value:.8g}'
    result = re.sub(r'\{\{([^{}]+)\}\}', substitute, text)
    if '{{' in result or '}}' in result:
        raise ValueError('Invalid report value reference')
    return result

def generate_report(packet, backend):
    modules = []
    for module in packet['modules']:
        if module['module_id'] not in packet['selected_modules']:
            continue
        genes = []
        for gene in module['genes']:
            ids = [cid for cid in gene['claim_ids'] if cid in packet['claims']
                   and source_dependencies(packet['claims'][cid]) <= set(packet['sources'])]
            if ids:
                genes.append(dict(gene=gene['gene'], head=gene['head'],
                                  direction=gene['direction'], claim_ids=ids))
        if genes:
            modules.append(dict(module_id=module['module_id'], genes=genes,
                contribution=module['knowledge_contribution'], components=module.get('components', {})))
    used = {cid for m in modules for gene in m['genes'] for cid in gene['claim_ids']}
    claims = {cid: packet['claims'][cid] for cid in sorted(used)}
    values = {key:packet[key] for key in ['prediction', 'risk', 'baseline']}
    model_output = dict(**values,
        module_contributions=[dict(module_id=m['module_id'], contribution=m['knowledge_contribution'],
            components=m.get('components', {})) for m in packet['modules']])
    if 'expression_contributions' in packet:
        model_output['expression_contribution'] = sum(packet['expression_contributions'].values())
    payload = dict(model_output=model_output, task=packet.get('task', {}), modules=modules, claims=claims,
                   sources={sid: packet['sources'][sid] for c in claims.values() for sid in source_dependencies(c)})
    response = (validate_statements(packet, backend.complete(prompt('ria'), payload))
                if modules else None)
    def verify(row, module=None):
        evidence = [claims[c] for c in row['claim_ids']]
        references = dict(values)
        if module is not None:
            references['contribution'] = module['contribution']
            references.update(module['components'])
        rendered = dict(row, text=_render_report_text(row['text'], references))
        result = backend.complete(prompt('report_verification'),
            dict(statement=rendered, model_output=model_output, task=payload['task'],
                module=module, evidence=evidence,
                sources={sid: payload['sources'][sid] for c in evidence for sid in source_dependencies(c)}))
        return rendered if result['supported'] is True else None
    assessment = verify(response['risk_assessment']) if response else None
    checked = []
    for row in response['statements'] if response else []:
        module = next(m for m in modules if m['module_id'] == row['module_id'])
        verified = verify(row, module)
        if verified is not None:
            checked.append(verified)
    conclusion = verify(response['conclusion']) if response else None
    def cited_text(row):
        return row['text'] + (' [' + ', '.join(row['claim_ids']) + ']' if row['claim_ids'] else '')
    lines = ['# Individual Risk Report', '', f"Sample: {packet['sample_id']}", '',
        '## Risk Assessment', '',
        f"Baseline: {packet['baseline']:.8g}",
        f"Risk score: {packet['risk']:.8g}",
        f"Predicted state: {packet['prediction']}"]
    if assessment:
        lines.extend(['', cited_text(assessment)])
    lines.extend(['', '## Module Contributions', ''])
    for m in packet['modules']:
        lines.append(f"{m['module_id']}: contribution={m['knowledge_contribution']:.8g}")
        for name, value in m.get('components', {}).items():
            lines.append(f"  {name}: {value:.8g}")
    if 'expression_contributions' in packet:
        lines.append(f"Expression branch contribution: {sum(packet['expression_contributions'].values()):.8g}")
    if checked:
        lines.extend(['', '## Evidence-Based Interpretation', ''])
        for r in checked:
            lines.append(r['module_id'] + ': ' + cited_text(r))
    if conclusion:
        lines.extend(['', '## Conclusion', '', cited_text(conclusion)])
    sections = [row for row in [assessment, *checked, conclusion] if row is not None]
    cited = sorted({cid for row in sections for cid in row['claim_ids']})
    if cited:
        lines.extend(['', '## Sources', ''])
        for cid in cited:
            claim = packet['claims'][cid]
            source = packet['sources'][claim['source_id']]
            lines.append(f"{cid}: {claim['source_id']} — {source.get('locator', '')}")
    return dict(text='\n'.join(lines), risk_assessment=assessment, statements=checked,
        conclusion=conclusion, proposed_count=len(response['statements']) if response else 0)

def intervene_sources(packet, *, withdraw, replacements):
    result = copy.deepcopy(packet)
    for source_id in set(withdraw) | set(replacements):
        result['sources'].pop(source_id, None)
        result['claims'] = {k: v for (k, v) in result['claims'].items() if source_id not in source_dependencies(v)}
    for module in result['modules']:
        for gene in module['genes']:
            gene['claim_ids'] = [cid for cid in gene['claim_ids'] if cid in result['claims']]
    for (source_id, value) in replacements.items():
        result['sources'][source_id] = value['source']
        for claim in value['claims']:
            if claim['claim_id'] in result['claims'] and result['claims'][claim['claim_id']] != claim:
                raise ValueError('Replacement claim ID has conflicting content')
            if claim['source_id'] != source_id:
                raise ValueError('Replacement source mismatch')
            if claim.get('decision') != 'accepted':
                raise ValueError('Verify replacement claims first')
            if not source_dependencies(claim) <= set(result['sources']):
                raise ValueError('Replacement context source is unavailable')
            linked = False
            for module in result['modules']:
                if module['module_id'] != claim['module_id']:
                    continue
                for gene in module['genes']:
                    if gene['gene'] == claim['subject'] and gene['head'] == claim['head']:
                        if proposition(claim) not in gene.get('propositions', []):
                            raise ValueError('Replacement must preserve the modeled proposition and conditions')
                        if claim['claim_id'] not in gene['claim_ids']:
                            gene['claim_ids'].append(claim['claim_id'])
                        linked = True
            if not linked:
                raise ValueError('Replacement claim must match an existing module gene/head')
            result['claims'][claim['claim_id']] = claim
    return result


def explain_oof_sample(expression, evidence_bundle, oof, splits, *, sample_id, repeat,
                       nodes, module_count, tolerance, module_scores=None):
    from ..variants import load_model
    from ..pherm.framework import PHERM
    from ..data import read_expression
    rows = oof[(oof['sample_id'].astype(str) == str(sample_id)) & (oof['repeat'] == repeat)]
    if len(rows) != 1:
        raise ValueError('One OOF row required for sample and repeat')
    row = rows.iloc[0]
    matches = [s for s in splits if s['repeat'] == repeat and s['fold'] == row['fold']]
    if len(matches) != 1 or not matches[0].get('model_file'):
        raise ValueError('Saved outer-fold model required')
    split = matches[0]
    if str(sample_id) in map(str, split['train']) or str(sample_id) not in map(str, split['test']):
        raise ValueError('Sample is not exclusive to the outer test fold')
    sample = expression.loc[expression.index.astype(str) == str(sample_id)]
    fitted = load_model(split['model_file'])
    if module_scores is None and not isinstance(fitted, PHERM):
        if not split.get('module_scores_file'):
            raise ValueError('Held-out module scores from the fitted outer-fold transform required')
        module_scores = read_expression(split['module_scores_file'])
    scores = None if module_scores is None else module_scores.loc[module_scores.index.astype(str) == str(sample_id)]
    packet = explain_prediction(fitted, sample, scores, evidence_bundle, nodes=nodes,
                            module_count=module_count, tolerance=tolerance)
    if abs(packet['risk'] - float(row['score'])) > tolerance or packet['prediction'] != int(row['prediction']):
        raise ValueError('Saved model does not match OOF prediction')
    packet['evaluation'] = dict(kind='OOF', repeat=int(repeat), fold=int(row['fold']))
    return packet


def explain_prediction(predictor, expression, module_scores, evidence_bundle, *, nodes, module_count, tolerance):
    from ..pherm.framework import PHERM
    from ..pherm.combined import MixPHERM
    if isinstance(predictor, (PHERM, MixPHERM)):
        return explain_modules(predictor, expression, module_scores, evidence_bundle,
            nodes=nodes, module_count=module_count, tolerance=tolerance)
    from ..evidence.registry import Module
    if not predictor.module_definitions:
        raise ValueError('Fitted module definitions required for source-linked interpretation')
    definitions = [Module(**m) for m in predictor.module_definitions]
    claims = validate_bundle(evidence_bundle, definitions)
    selected = [m for m in definitions if m.head == 'S']
    if [m.module_id for m in selected] != predictor.knowledge['modules']:
        raise ValueError('Fitted module definitions and score columns disagree')
    if type(module_count) is not int or module_count < 1:
        raise ValueError('Positive module count required')
    attribution = predictor.attribute(expression, module_scores, nodes=nodes, tolerance=tolerance)
    prediction = predictor.predict(expression, module_scores)
    rows = []
    for m in selected:
        contribution = attribution['module_contributions'][m.module_id]
        genes = [dict(gene=g, head='S', direction=direction, weight=weight, claim_ids=list(ids),
                      propositions=[proposition(claims[cid]) for cid in ids])
                 for g, direction, weight, ids in zip(m.genes, m.directions, m.weights, m.claim_ids)]
        rows.append(dict(module_id=m.module_id, knowledge_contribution=contribution, net=contribution,
                         importance=abs(contribution), genes=genes))
    rows.sort(key=lambda row: (-row['importance'], row['module_id']))
    used = {cid for row in rows for gene in row['genes'] for cid in gene['claim_ids']}
    linked = {cid: copy.deepcopy(claims[cid]) for cid in sorted(used)}
    sources = {sid: copy.deepcopy(evidence_bundle['sources'][sid])
               for claim in linked.values() for sid in source_dependencies(claim)}
    return dict(**attribution, prediction=int(prediction['prediction'][0]), task=copy.deepcopy(evidence_bundle.get('task', {})), modules=rows,
                selected_modules=[row['module_id'] for row in rows[:module_count]], claims=linked, sources=sources)


def explain_modules(predictor, expression, module_scores, bundle, *, nodes, module_count, tolerance):
    from ..pherm.framework import PHERM
    if type(module_count) is not int or module_count < 1:
        raise ValueError('Positive module count required')
    standalone = isinstance(predictor, PHERM)
    framework = predictor if standalone else predictor.framework
    claims = validate_bundle(bundle, framework.modules)
    args = (expression,) if standalone else (expression, module_scores)
    attribution = predictor.attribute(*args, nodes=nodes, tolerance=tolerance)
    prediction = predictor.predict(*args)
    inputs = module_inputs(framework.standardizer, expression, framework.modules)
    z, observed = framework.standardizer.transform(expression)
    index = {g:i for i,g in enumerate(framework.standardizer.genes)}
    rows = {}
    for m in framework.modules:
        row = rows.setdefault(m.module_id, dict(module_id=m.module_id, components={}, genes=[], evidence={}))
        j = inputs[m.head]['module_ids'].index(m.module_id)
        if standalone:
            components = ['D'] if m.head == 'D' else ['S_via_S', 'S_via_lambda']
            row['components'].update({key:float(attribution[key][j]) for key in components})
        else:
            row['components']['framework_'+m.head] = attribution['framework_contributions'][m.head][m.module_id]
            if m.head == 'S':
                row['components']['probability_S'] = attribution['module_contributions'][m.module_id]
        row['evidence'][m.head] = dict(score=float(inputs[m.head]['scores'][0,j]),
            coverage=float(inputs[m.head]['coverage'][0,j]), confidence=float(inputs[m.head]['confidence'][j]))
        for gene, direction, weight, ids in zip(m.genes, m.directions, m.weights, m.claim_ids):
            if gene not in index or not observed[0,index[gene]]:
                continue
            row['genes'].append(dict(gene=gene, head=m.head, direction=direction, weight=weight,
                standardized_expression=float(z[0,index[gene]]), claim_ids=list(ids),
                propositions=[proposition(claims[cid]) for cid in ids]))
    for row in rows.values():
        row.update(net=sum(row['components'].values()),
            knowledge_contribution=sum(row['components'].values()),
            importance=sum(abs(v) for v in row['components'].values()))
    ordered = sorted(rows.values(), key=lambda r:(-r['importance'],r['module_id']))
    used = {cid for row in ordered for gene in row['genes'] for cid in gene['claim_ids']}
    linked = {cid:copy.deepcopy(claims[cid]) for cid in sorted(used)}
    sources = {sid:copy.deepcopy(bundle['sources'][sid]) for claim in linked.values() for sid in source_dependencies(claim)}
    return dict(**attribution, prediction=int(prediction['prediction'][0]), task=copy.deepcopy(bundle.get('task', {})), modules=ordered,
        selected_modules=[row['module_id'] for row in ordered[:module_count]], claims=linked, sources=sources)
