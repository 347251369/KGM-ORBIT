import tempfile
import unittest
import numpy as np

from test_pipeline import fixture, options, TestModules, fitted_fixture
from kgm_orbit.evidence.scoring import fit_scorer
from kgm_orbit.evidence.workflow import build_evidence, EvidenceOptions
from kgm_orbit.evidence.retrieval import Source
from kgm_orbit.training import nested_oof
from kgm_orbit.coeri.report import (
    explain_prediction, explain_oof_sample, generate_report, intervene_sources,
)


class CoreWorkflowTests(unittest.TestCase):
    def test_contrastive_training_and_held_out_threshold(self):
        class Encoder:
            def encode(self, texts, normalize_embeddings=True):
                return np.array([[1., 0.] if 'yes' in text else [0., 1.] for text in texts])

        training = [dict(source_id='train', source='yes source', positive='yes claim', negatives=['no claim'])]
        validation = [
            dict(source_id='v1', source='yes source', positive=['yes', 'yes'], negative=['no', 'no'], label=1),
            dict(source_id='v2', source='yes source', positive=['no', 'no'], negative=['yes', 'yes'], label=0),
        ]
        scorer, tau = fit_scorer(Encoder(), training, validation, l2=.01)
        self.assertGreater(scorer.margin('yes source', ['yes'] * 2, ['no'] * 2), tau)
        self.assertLess(scorer.margin('yes source', ['no'] * 2, ['yes'] * 2), tau)
        validation[0]['source_id'] = 'train'
        with self.assertRaisesRegex(ValueError, 'overlap'):
            fit_scorer(Encoder(), training, validation, l2=.01)

    def test_module_identity_and_final_verification_order(self):
        events = []

        class Backend:
            def complete(self, prompt, payload):
                if 'counters' in payload:
                    events.append('AVA')
                    return dict(supported=True, counter_units=[])
                if 'chunks' in payload:
                    events.append('ESA')
                    unit = payload['chunks'][0]
                    return {'claims': [dict(
                        subject='A', claim='A increases', relation='expression_change',
                        module_id=m, source_id='s', unit_id=unit['unit_id'], quote='A increases',
                        tissue='blood', comparison=payload['context']['comparison'],
                        measurement='RNA', direction='increasing',
                        positive=['yes'] * 2, negative=['no'] * 2,
                    ) for m in ['m1', 'm2']]}
                return dict(modules=[dict(module_id=m, genes=['A'], source_ids=['s'])
                                     for m in ['m1', 'm2']], edges=[])

        task = dict(disease='example', tissue='blood',
                    heads={h: dict(task_key=h, comparison=h) for h in ['D', 'S']})
        bundle = build_evidence(
            task, backend=Backend(), retrieve=lambda q: [Source('s', 'A increases', 'source')],
            scorer=lambda s, c: float(c == 'yes'),
            options=EvidenceOptions(2, 2, 100, 0, 2, .2),
        )
        self.assertEqual(len(bundle['modules']), 4)
        self.assertEqual(len({c['claim_id'] for c in bundle['claims']}), 4)
        self.assertLess(max(i for i, e in enumerate(events) if e == 'ESA'), events.index('AVA'))
        self.assertEqual(bundle['stop_reasons']['D'], 'no_eligible_gap')

    def test_expression_chain_and_withdrawal_report(self):
        x, y, modules, bundle = fixture()
        fitted, scores = fitted_fixture(x, y, bundle)
        packet = explain_prediction(fitted, x.iloc[[30]], scores.iloc[[30]], bundle, nodes=32, module_count=1, tolerance=1e-9)
        self.assertAlmostEqual(packet['risk'], packet['baseline'] + sum(packet['expression_contributions'].values()) + sum(packet['module_contributions'].values()))
        removed = intervene_sources(packet, withdraw=['source'], replacements={})
        self.assertTrue(all(not g['claim_ids'] for m in removed['modules'] for g in m['genes']))

        class Backend:
            def complete(self, *args):
                raise AssertionError('No source-supported statements to generate')

        report = generate_report(removed, Backend())
        self.assertEqual(report['statements'], [])
        self.assertIn('Risk score:', report['text'])
        self.assertEqual(removed['risk'], packet['risk'])

    def test_saved_outer_fold_explanation(self):
        x, y, modules, bundle = fixture()
        with tempfile.TemporaryDirectory() as folder:
            oof, splits = nested_oof(x, y, TestModules, options=options(), seeds=[7],
                                     outer_folds=3, inner_folds=2, model_dir=folder)
            sample_id = oof.iloc[0]['sample_id']
            packet = explain_oof_sample(x, bundle, oof, splits, sample_id=sample_id, repeat=1,
                                        nodes=32, module_count=1, tolerance=1e-9)
            self.assertEqual(packet['evaluation']['kind'], 'OOF')
            self.assertAlmostEqual(packet['risk'], oof.iloc[0]['score'])
            changed = oof.copy()
            changed.loc[changed['sample_id'] == sample_id, 'score'] += 1
            with self.assertRaisesRegex(ValueError, 'does not match'):
                explain_oof_sample(x, bundle, changed, splits, sample_id=sample_id, repeat=1,
                                   nodes=32, module_count=1, tolerance=1e-9)


if __name__ == '__main__':
    unittest.main()
