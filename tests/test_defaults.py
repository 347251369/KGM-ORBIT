import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
from test_pipeline import fixture, TestModules
from kgm_orbit.cli import main, parser
from kgm_orbit.pherm.framework import PHERM, FrameworkOptions
from kgm_orbit.pherm.prediction import ProbPHERM
from kgm_orbit.pherm.combined import MixPHERM
from kgm_orbit.pipeline import fit_prediction_model
from kgm_orbit.training import nested_oof
from kgm_orbit.evidence.core import ContrastiveHead


class DefaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.x, cls.y, cls.modules, cls.bundle = fixture()

    def test_python_defaults_train_and_predict(self):
        model, transform, inner = fit_prediction_model(self.x,self.y,evidence_bundle=self.bundle)
        result = model.predict(self.x)
        self.assertTrue(np.isfinite(result['risk']).all())
        np.testing.assert_allclose(result['probabilities'].sum(1),1)
        self.assertTrue((result['probabilities']>=0).all())
        self.assertIs(transform,model.standardizer)
        self.assertIsNone(inner)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.json'
            model.save(path)
            np.testing.assert_array_equal(PHERM.load(path).predict(self.x)['risk'],result['risk'])

    def test_command_defaults_and_partial_override(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            self.x.to_csv(p/'x.csv')
            pd.DataFrame({'label':self.y},index=self.x.index).to_csv(p/'y.csv')
            (p/'bundle.json').write_text(json.dumps(self.bundle))
            args = ['train','--expression',str(p/'x.csv'),'--labels',str(p/'y.csv'),
                '--evidence',str(p/'bundle.json')]
            self.assertEqual(main(args+['--output',str(p/'default.json')]),0)
            self.assertEqual(main(['predict','--model',str(p/'default.json'),
                '--expression',str(p/'x.csv'),'--output',str(p/'predictions.csv')]),0)
            self.assertEqual(len(pd.read_csv(p/'predictions.csv')),len(self.x))
            (p/'override.json').write_text(json.dumps({'prediction_epochs':3,'seed':31}))
            self.assertEqual(main(args+['--parameters',str(p/'override.json'),'--output',str(p/'custom.json')]),0)
            fitted = PHERM.load(p/'custom.json')
            self.assertEqual(len(fitted.training_trace),3)
            self.assertEqual(fitted.options.seed,31)
            self.assertEqual(fitted.options.calibration_epochs,FrameworkOptions().calibration_epochs)

    def test_retained_model_defaults(self):
        scores = TestModules().fit(self.x,self.y).transform(self.x)
        probability = ProbPHERM().fit(self.x,self.y,scores,[.8],d_threshold=.5,
            module_definitions=self.bundle['modules'])
        combined = MixPHERM(self.modules).fit(self.x,self.y,scores,[.8],d_threshold=.5)
        for model in [probability,combined]:
            result = model.predict(self.x,scores)
            self.assertTrue(np.isfinite(result['risk']).all())
            np.testing.assert_allclose(result['probabilities'].sum(1),1)
        head = ContrastiveHead().fit([[1,0],[-1,0],[2,1],[-2,1]],[[0,1],[2,3]])
        self.assertTrue(np.isfinite(head.score([[1,0]])).all())

    def test_oof_defaults(self):
        oof,splits = nested_oof(self.x,self.y,evidence_bundle=self.bundle,seeds=[13],outer_folds=2)
        self.assertEqual(len(oof),len(self.x))
        self.assertTrue(oof['sample_id'].is_unique)
        self.assertTrue(np.isfinite(oof['score']).all())
        self.assertTrue(all(not set(s['train']) & set(s['test']) for s in splits))
