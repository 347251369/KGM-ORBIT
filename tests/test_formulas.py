import unittest
import numpy as np
from kgm_orbit.pherm.formulas import claim_weight, aggregate

class FormulaTests(unittest.TestCase):

    def test_margin_weight(self):
        self.assertAlmostEqual(claim_weight([0.6, 0.8], 0.2), 0.5)
        with self.assertRaises(ValueError):
            claim_weight([0.2], 0.2)

    def test_measured_denominator_and_coverage(self):
        a = aggregate([[2, np.nan]], [1, 3], [1, -1], np.array([[1, 0]], bool))
        np.testing.assert_allclose(a['raw'], [2])
        np.testing.assert_allclose(a['score'], [0.25 * np.tanh(2)])
        self.assertEqual(a['confidence'], 2)

    def test_empty_coverage_refuses(self):
        with self.assertRaises(ValueError):
            aggregate([[np.nan]], [1], [1], np.array([[0]], bool))

