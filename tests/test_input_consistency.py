import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

import numpy as np

from test_pipeline import fixture, options, fitted_fixture
from kgm_orbit.data import read_expression, GeneStandardizer
from kgm_orbit.evidence.registry import validate_bundle
from kgm_orbit.evidence.retrieval import Source
from kgm_orbit.evidence.workflow import build_evidence, EvidenceOptions
from kgm_orbit.pherm.prediction import ProbPHERM
from kgm_orbit.evidence.registry import Module
from kgm_orbit.coeri.report import explain_prediction, intervene_sources, generate_report


class InputConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y, cls.modules, cls.bundle = fixture()
        cls.model, cls.scores = fitted_fixture(cls.x, cls.y, cls.bundle)

    def packet(self):
        return explain_prediction(self.model, self.x.iloc[[30]], self.scores.iloc[[30]], self.bundle,
                              nodes=32, module_count=1, tolerance=1e-9)

    def test_model_evidence_membership(self):
        for field, value in [('subject', 'OTHER'), ('direction', 'decreasing'),
                             ('decision', 'refuted'), ('head', 'S'), ('measurement', 'protein')]:
            wrong = copy.deepcopy(self.bundle)
            wrong['claims'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                explain_prediction(self.model, self.x.iloc[:1], self.scores.iloc[:1], wrong,
                               nodes=32, module_count=1, tolerance=1e-9)
        wrong = copy.deepcopy(self.bundle)
        wrong['modules'][0]['weights'] = [2, 2]
        with self.assertRaises(ValueError):
            validate_bundle(wrong, [Module(**m) for m in self.model.module_definitions])
        wrong = copy.deepcopy(self.bundle)
        wrong['claims'].append(wrong['claims'][0])
        with self.assertRaises(ValueError):
            validate_bundle(wrong)

    def test_withdraw_context(self):
        packet = self.packet()
        packet['sources']['context'] = dict(text='Comparison context', locator='test')
        packet['claims']['SA']['context_units'] = [dict(source_id='context', text='Comparison context')]
        changed = intervene_sources(packet, withdraw=['context'], replacements={})
        self.assertNotIn('SA', changed['claims'])
        self.assertEqual(packet['risk'], changed['risk'])
        class Recorder:
            def complete(self, prompt, payload):
                from test_pipeline import report_response
                self.payload = payload
                return dict(supported=True) if 'statement' in payload else report_response([])
        recorder = Recorder()
        generate_report(changed, recorder)
        self.assertNotIn('Comparison context', json.dumps(recorder.payload))
        self.assertIn('SA', packet['claims'])

    def test_replacement_conditions(self):
        packet = self.packet()
        for field, value in [('direction', 'decreasing'), ('measurement', 'protein'),
                             ('comparison', 'different groups'), ('species', 'Mus musculus')]:
            claim = dict(self.bundle['claims'][2], source_id='new', claim_id='new')
            claim[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                intervene_sources(packet, withdraw=['source'], replacements={
                    'new': dict(source=dict(text='Replacement'), claims=[claim])})

    def test_species_eligibility(self):
        class Backend:
            def __init__(self, species):
                self.species = species
            def complete(self, prompt, payload):
                if 'counters' in payload:
                    return dict(supported=True, counter_units=[])
                if 'chunks' in payload:
                    unit = payload['chunks'][0]
                    return dict(claims=[dict(subject='A', claim='A increases',
                        relation='expression_change', module_id='m', source_id='s',
                        unit_id=unit['unit_id'], quote='A increases', species=self.species,
                        tissue='blood', comparison=payload['context']['comparison'],
                        measurement='RNA', direction='increasing', positive=['yes']*2, negative=['no']*2)])
                return dict(modules=[dict(module_id='m', genes=['A'], source_ids=['s'])], edges=[])
        task = dict(disease='example', species='Homo sapiens', tissue='blood',
                    heads={h: dict(task_key=h, comparison=h) for h in ['D', 'S']})
        for species, expected in [('Mus musculus', 0), (None, 0), ('Homo sapiens', 2)]:
            result = build_evidence(task, backend=Backend(species),
                retrieve=lambda q: [Source('s', 'A increases', 'test')],
                scorer=lambda s, c: float(c == 'yes'), options=EvidenceOptions(1, 1, 100, 0, 2, .1))
            self.assertEqual(len(result['modules']), expected)

    def test_csv_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'expression.csv'
            path.write_text('sample,A\n001,1\n1,2\nNA,3\n')
            self.assertEqual(read_expression(path).index.tolist(), ['001', '1', 'NA'])
            path.write_text('sample,A,A\n001,1,2\n')
            with self.assertRaises(ValueError):
                read_expression(path)

    def test_saved_model_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            self.model.save(path)
            saved = json.loads(path.read_text())
            for field, value in [('bias', float('nan')), ('scale', [0]), ('reliability', [9])]:
                bad = copy.deepcopy(saved)
                bad['knowledge'][field] = value
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    ProbPHERM.load(path)
            for scale in [[0] * len(self.model.raw['genes']), [1], [np.inf] * len(self.model.raw['genes'])]:
                with self.assertRaises(ValueError):
                    GeneStandardizer.from_dict(dict(genes=self.model.raw['genes'], median=self.model.raw['median'], center=self.model.raw['center'], scale=scale))
