import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from kgm_orbit.data import GeneStandardizer, map_labels, map_probes
from kgm_orbit.evidence.core import ContrastiveHead, verify_margin
from kgm_orbit.evidence.registry import Module, compile_modules
from kgm_orbit.evidence.retrieval import Source, chunk_sources, rank_chunks
from kgm_orbit.evidence.scoring import choose_threshold
from kgm_orbit.evidence.workflow import EvidenceOptions, build_evidence
from kgm_orbit.training import nested_oof
from kgm_orbit.pherm.prediction import ProbPHERM, PredictionOptions
from kgm_orbit.coeri.report import explain_prediction, intervene_sources, generate_report
from kgm_orbit.evaluation import prediction_metrics, repeat_metrics

def report_response(statements):
    return dict(risk_assessment=dict(text='Predicted state {{prediction}} has score {{risk}} relative to baseline {{baseline}}.', claim_ids=[]),
        statements=statements,
        conclusion=dict(text='The module contributions explain the difference from baseline.', claim_ids=[]))


def fixture():
    rng = np.random.default_rng(19)
    y = np.repeat([0, 1, 2], 12)
    x = pd.DataFrame({'A': y + rng.normal(0, 0.3, len(y)), 'B': y + rng.normal(0, 0.3, len(y)), 'C': rng.normal(size=len(y))},
        index=[f's{i}' for i in range(len(y))])
    claims = []
    for head in ['D', 'S']:
        for gene in ['A', 'B']:
            claims.append(dict(claim_id=head + gene,
                module_id='module',
                head=head,
                subject=gene,
                relation='expression_change',
                measurement='RNA',
                direction='increasing',
                decision='accepted',
                support_margin=0.8,
                source_id='source',
                unit_ids=['unit'],
                task_key=head,
                claim=gene + ' increases'))
    modules = compile_modules(claims, tau=0.2, task_keys={'D': 'D', 'S': 'S'})
    bundle = dict(modules=[m.to_dict() for m in modules],
        claims=claims,
        sources={'source': dict(text='A and B increase', locator='synthetic')})
    return (x, y, modules, bundle)

def options():
    return PredictionOptions(feature_count=8, expression_C=1., expression_l1_ratio=.25,
        expression_max_iterations=4000, expression_tolerance=1e-7,
        knowledge_alpha=.2, knowledge_max_iterations=1500, minimum_reliability=.15,
        maximum_weight=2., variance_floor=.02, evidence_clip=5., knowledge_fraction=.5,
        low_state_position=.45, severity_threshold=.5, threshold_grid=(.3,.5,.7), epsilon=1e-12, seed=23)

class TestModules:
    def fit(self, x, y):
        self.center = x[['A', 'B']].mean()
        self.reliability = [.8]
        self.module_definitions = fixture()[3]['modules']
        return self
    def transform(self, x):
        return pd.DataFrame({'module': (x[['A', 'B']]-self.center).mean(axis=1)}, index=x.index)

def fitted_fixture(x, y, bundle):
    transform = TestModules().fit(x, y)
    scores = transform.transform(x)
    fitted = ProbPHERM(options()).fit(x, y, scores, [.8], d_threshold=.5,
                                         module_definitions=bundle['modules'])
    return fitted, scores

class PipelineTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        (cls.x, cls.y, cls.modules, cls.bundle) = fixture()
        cls.fit, cls.scores = fitted_fixture(cls.x, cls.y, cls.bundle)

    def test_fold_preprocessing_is_frozen(self):
        s = GeneStandardizer().fit(self.x.iloc[:20])
        before = s.to_dict()
        s.transform(self.x.iloc[20:] * 1000)
        self.assertEqual(before, s.to_dict())
        (z, mask) = s.transform(self.x.iloc[:1].drop(columns='A'))
        self.assertFalse(mask[0, s.genes.index('A')])
        self.assertTrue(np.isfinite(z).all())

    def test_mapping(self):
        self.assertEqual(map_labels(['H', 'L', 'M'], {'H': 0, 'L': 1, 'M': 2}).tolist(), [0, 1, 2])
        x = map_probes(pd.DataFrame({'p1': [1], 'p2': [3]}), {'p1': 'A', 'p2': 'A'}, aggregation='mean')
        self.assertEqual(x.A.iloc[0], 2)
        with self.assertRaises(ValueError):
            map_labels(['unknown'], {'H': 0})

    def test_evidence_conflicts(self):
        claims = self.bundle['claims'] + [dict(self.bundle['claims'][0],
            claim_id='reverse',
            direction='decreasing')]
        modules = compile_modules(claims, tau=0.2, task_keys={'D': 'D', 'S': 'S'})
        self.assertEqual(next((m for m in modules if m.head == 'D')).genes, ('B',))
        self.assertEqual(verify_margin(0.7, [dict(source_id='c', margin=-0.2, comparable=True)], 0.2),
            'unresolved')
        self.assertEqual(verify_margin(0.7, [dict(source_id='c', margin=-0.3, comparable=True)], 0.2),
            'conflict')

    def test_model_save_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'model.json'
            self.fit.save(p)
            loaded = ProbPHERM.load(p)
            np.testing.assert_array_equal(self.fit.predict(self.x, self.scores)['risk'],
                loaded.predict(self.x, self.scores)['risk'])

    def test_predictor_attribution(self):
        packet = explain_prediction(self.fit,
            self.x.iloc[[30]],
            self.scores.iloc[[30]],
            self.bundle,
            nodes=32,
            module_count=1,
            tolerance=1e-10)
        self.assertLess(abs(packet['completeness_residual']), 1e-10)
        self.assertAlmostEqual(packet['risk'],
            packet['baseline'] + sum((m['net'] for m in packet['modules'])) + sum(packet['expression_contributions'].values()))
        removed = intervene_sources(packet, withdraw=['source'], replacements={})
        self.assertEqual(removed['risk'], packet['risk'])
        self.assertFalse(removed['claims'])
        claim = dict(self.bundle['claims'][2], source_id='new', claim_id='new_claim')
        replaced = intervene_sources(removed,
            withdraw=[],
            replacements={'new': dict(source={'text': 'A increases'}, claims=[claim])})
        self.assertIn('new_claim', replaced['claims'])
        self.assertEqual(replaced['risk'], packet['risk'])

    def test_report_renderer(self):

        class Backend:

            def complete(self, prompt, payload):
                if 'statement' in payload:
                    return dict(supported=True, reason='matched')
                return report_response([dict(module_id='module', text='A increases.', claim_ids=['SA'])])
        packet = explain_prediction(self.fit,
            self.x.iloc[[30]],
            self.scores.iloc[[30]],
            self.bundle,
            nodes=32,
            module_count=1,
            tolerance=1e-10)
        report = generate_report(packet, Backend())
        self.assertIn('Risk score:', report['text'])
        self.assertEqual(len(report['statements']), 1)

    def test_nested_oof_boundaries(self):
        (oof,
            splits) = nested_oof(self.x,
            self.y,
            TestModules,
            options=options(),
            seeds=[7],
            outer_folds=3,
            inner_folds=2)
        self.assertEqual(len(oof), len(self.y))
        self.assertFalse(oof.sample_id.duplicated().any())
        self.assertTrue(all((not set(s['train']) & set(s['test']) for s in splits)))
        self.assertEqual(len(repeat_metrics(oof)), 1)

    def test_metrics(self):
        m = prediction_metrics([0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2])
        self.assertEqual(m['ordering_auc'], 1)
        self.assertEqual(m['macro_f1'], 1)
    def test_scoring_and_retrieval(self):
        head = ContrastiveHead(l2=.2).fit([[1, 0], [-1, 0], [2, 1], [-2, 1]], [[0, 1], [2, 3]])
        self.assertGreater(head.score([[1, 0]])[0], head.score([[-1, 0]])[0])
        threshold = choose_threshold([0, 0, 1, 1], [0, 0.1, 0.8, 0.9], nonnegative=True)
        self.assertTrue(0.1 <= threshold < 0.8)
        chunks = chunk_sources([Source('s', 'alpha beta gamma delta', 'locator')],
            characters=12,
            overlap=2)
        self.assertTrue(all((c['text'] == 'alpha beta gamma delta'[c['start']:c['end']] for c in chunks)))
        self.assertTrue(rank_chunks('alpha', chunks, top_k=1)[0]['text'].startswith('alpha'))

    def test_evidence_workflow(self):
        source = Source('s', 'A increases and B increases.', 'synthetic')

        class Backend:

            def complete(self, prompt, payload):
                if 'counterevidence' in prompt:
                    return dict(supported=True, reason='quote', counter_units=[])
                if 'chunks' in payload:
                    unit = payload['chunks'][0]
                    return {'claims': [dict(subject='A',
                        claim='A increases',
                        relation='expression_change',
                        object='RNA',
                        module_id='m',
                        source_id='s',
                        unit_id=unit['unit_id'],
                        quote='A increases',
                        tissue='blood',
                        comparison=payload['context']['comparison'],
                        measurement='RNA',
                        direction='increasing',
                        positive=['yes', 'yes'],
                        negative=['no', 'no'])]}
                return dict(modules=[dict(module_id='m', genes=['A', 'B'], source_ids=['s'])],
                    edges=[dict(gene_a='A', gene_b='B', source='s', relation='same_reaction')])
        task = dict(disease='synthetic',
            tissue='blood',
            heads={h: dict(task_key=h, comparison=h) for h in ['D', 'S']})
        result = build_evidence(task,
            backend=Backend(),
            retrieve=lambda q: [source],
            scorer=lambda source,
            claim: 1 if claim == 'yes' else 0,
            options=EvidenceOptions(max_rounds=2, patience=2, chunk_characters=100, chunk_overlap=0, top_k=2, tau=0.2))
        self.assertEqual(len(result['modules']), 2)
        self.assertEqual({m['head'] for m in result['modules']}, {'D', 'S'})
if __name__ == '__main__':
    unittest.main()
