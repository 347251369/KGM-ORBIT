import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy.special import expit
from kgm_orbit.pherm.prediction import ProbPHERM, PredictionOptions, combine_heads, select_d_threshold
from kgm_orbit.pipeline import fit_prediction_model
from test_pipeline import options


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(71)
        cls.y = np.repeat([0, 1, 2], 18)
        cls.x = pd.DataFrame(rng.normal(size=(54, 8)), columns=list('ABCDEFGH'))
        cls.x['A'] += cls.y
        cls.m = pd.DataFrame({'m1': cls.x.A, 'm2': cls.x.B})
        cls.options = options()
        cls.model = ProbPHERM(cls.options).fit(cls.x, cls.y, cls.m, [.8, .9], d_threshold=.5)

    def test_hierarchy_and_threshold(self):
        result = combine_heads([.2, .6, .8], [.1, .4, .9], [.3, .8, .7],
                               options=self.options, threshold=.5)
        np.testing.assert_allclose(result['probabilities'].sum(1), 1)
        np.testing.assert_allclose(result['risk'], .45*result['probabilities'][:, 1]+result['probabilities'][:, 2])
        np.testing.assert_array_equal(result['prediction'], [0, 2, 2])
        self.assertEqual(select_d_threshold([0, 1, 2], [.1, .9, .9], [.1, .1, .9], options=self.options), .5)

    def test_save_load_and_input_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.json'
            self.model.save(path)
            restored = ProbPHERM.load(path)
            np.testing.assert_array_equal(self.model.predict(self.x, self.m)['risk'], restored.predict(self.x, self.m)['risk'])
            data = json.loads(path.read_text())
            data['knowledge']['weight'][0] = -1
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                ProbPHERM.load(path)
        with self.assertRaises(ValueError):
            self.model.predict(self.x, self.m.iloc[::-1])

    def test_final_score_attribution(self):
        packet = self.model.attribute(self.x.iloc[[36]], self.m.iloc[[36]], tolerance=1e-10)
        self.assertLess(abs(packet['completeness_residual']), 1e-10)
        self.assertAlmostEqual(packet['risk'], packet['baseline']+sum(packet['expression_contributions'].values())
                               +sum(packet['module_contributions'].values()), places=10)
        self.assertEqual(set(packet['expression_contributions']), set(self.model.raw['genes']))

    def test_knowledge_constraints_and_gaussian_terms(self):
        k = self.model.knowledge
        self.assertTrue((np.asarray(k['weight']) >= 0).all())
        positive, negative = self.m.to_numpy()[self.y == 2], self.m.to_numpy()[self.y == 1]
        variance = np.maximum((positive.var(0, ddof=1)+negative.var(0, ddof=1))/2, self.options.variance_floor)
        np.testing.assert_allclose(k['coef'], (positive.mean(0)-negative.mean(0))/variance)
        _, evidence = self.model.inputs(self.x, self.m)
        np.testing.assert_allclose(self.model.predict(self.x, self.m)['knowledge_S'],
                                   expit(evidence@k['weight']+k['bias']))

    def test_module_refitting_is_inner_fold_local(self):
        seen = []
        class Modules:
            def fit(self, x, y):
                self.center = x.A.mean()
                self.reliability = [.8]
                seen.append(set(x.index))
                return self
            def transform(self, x):
                return pd.DataFrame({'m1': x.A-self.center}, index=x.index)
        model, transform, inner = fit_prediction_model(self.x, self.y, Modules, options=self.options,
                                                       inner_folds=3, inner_seed=7)
        self.assertEqual(len(seen), 4)
        for fold, train in enumerate(seen[:-1], 1):
            self.assertFalse(train & set(inner.index[inner.fold == fold]))
        self.assertEqual(model.d_threshold, select_d_threshold(inner.label, inner.D, inner.S, options=self.options))
        self.assertTrue(np.isfinite(model.predict(self.x, transform.transform(self.x))['risk']).all())

    def test_conditional_attribution_baseline_and_completeness(self):
        z, e = self.model.inputs(self.x.iloc[:1], self.m.iloc[:1])
        packet = self.model.conditional_module_attribution(z[0], e[0], sample_id='sample')
        k = self.model.knowledge
        zero_evidence = -np.asarray(k['center'])/np.asarray(k['scale'])
        expected = self.model.from_inputs(z[0], zero_evidence)
        self.assertAlmostEqual(packet['baseline'], float(expected['risk']))
        self.assertAlmostEqual(packet['risk'], packet['baseline'] + sum(packet['module_contributions'].values()))

    def test_gaussian_calibration_with_unequal_case_counts(self):
        keep = np.r_[np.arange(36), np.arange(36, 45)]
        fitted = ProbPHERM(self.options).fit(self.x.iloc[keep], self.y[keep], self.m.iloc[keep], [.8,.9], d_threshold=.5)
        positive = self.m.to_numpy()[keep][self.y[keep] == 2]
        negative = self.m.to_numpy()[keep][self.y[keep] == 1]
        variance = ((len(positive)-1)*positive.var(0,ddof=1)+(len(negative)-1)*negative.var(0,ddof=1))/(len(positive)+len(negative)-2)
        np.testing.assert_allclose(fitted.knowledge['coef'], (positive.mean(0)-negative.mean(0))/np.maximum(variance,.05))

