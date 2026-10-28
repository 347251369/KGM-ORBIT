def combine_verification(margin_decision, field_decision, *, field_conflict=False):
    allowed = {'accepted', 'conflict', 'refuted', 'rejected', 'uncertain', 'unresolved'}
    if margin_decision not in allowed or field_decision not in allowed:
        raise ValueError('Unknown decision')
    if field_conflict or margin_decision == 'conflict':
        return 'conflict'
    if field_decision in ('rejected', 'refuted') or margin_decision in ('rejected', 'refuted'):
        return 'refuted'
    if margin_decision == 'accepted' and field_decision == 'accepted':
        return 'accepted'
    return 'unresolved'

def source_gate(row):
    if row.get('schema_consistency') != 'consistent' or not row.get('score_fields_consistent', False):
        return 'unresolved'
    if row.get('field_classification') == 'correct':
        return 'accepted'
    if row.get('field_classification') == 'incorrect':
        return 'refuted'
    return 'unresolved'
