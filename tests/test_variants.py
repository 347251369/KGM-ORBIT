from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from test_pipeline import fixture, TestModules, options
from kgm_orbit.pherm.framework import PHERM, FrameworkOptions
from kgm_orbit.pherm.combined import MixPHERM
from kgm_orbit.pherm.prediction import ProbPHERM
from kgm_orbit.pipeline import fit_prediction_model
from kgm_orbit.variants import main, load_model


class VariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.x, cls.y, cls.modules, cls.bundle = fixture()
        cls.framework_options = FrameworkOptions(nodes=2, jaccard_threshold=.1,
            calibration_epochs=4, prediction_epochs=8, learning_rate=.03, weight_decay=.001,
            loss_d=1., loss_s=1., loss_ordinal=1., seed=23)
        cls.probability, cls.transform, cls.inner = fit_prediction_model(cls.x, cls.y, TestModules,
            options=options(), inner_folds=3, inner_seed=7)
        cls.scores = cls.transform.transform(cls.x)
        cls.framework = PHERM(cls.modules, cls.framework_options).fit(cls.x, cls.y)
        cls.combined = MixPHERM(cls.modules, framework_options=cls.framework_options,
            prediction_options=options(), framework_fraction=.4).fit(cls.x, cls.y, cls.scores, [.8],
            d_threshold=cls.probability.d_threshold)

    def test_framework_calibration_and_gate(self):
        calibration = self.framework.model.calibration_s
        with torch.no_grad():
            grid = torch.linspace(-2,2,100,dtype=torch.float64)[:,None]
            self.assertTrue((torch.diff(calibration(grid),dim=0)>0).all())
            np.testing.assert_allclose(calibration(torch.zeros((1,1),dtype=torch.float64)).numpy(),0)
        prediction = self.framework.predict(self.x)
        self.assertGreater(np.ptp(prediction['lambda_value']),0)
        np.testing.assert_allclose(prediction['probabilities'].sum(1),1)
        self.assertTrue((prediction['probabilities']>=0).all())
        packet = self.framework.attribute(self.x.iloc[:1])
        self.assertLess(abs(packet['residual']),1e-8)

    def test_sparse_mask_excludes_unconnected_module(self):
        from kgm_orbit.pherm._framework_network import SparseBranch
        branch = SparseBranch(np.eye(2,dtype=bool)).double()
        branch(torch.ones((1,2),dtype=torch.float64)).sum().backward()
        self.assertEqual(branch.weight.grad[0,1].item(),0)
        self.assertEqual(branch.weight.grad[1,0].item(),0)

    def test_combination_and_endpoints(self):
        a = self.combined.framework.predict(self.x)['probabilities']
        b = self.combined.probability.predict(self.x,self.scores)['probabilities']
        result = self.combined.predict(self.x,self.scores)
        np.testing.assert_allclose(result['probabilities'],.4*a+.6*b)
        np.testing.assert_allclose(result['risk'],.45*result['probabilities'][:,1]+result['probabilities'][:,2])
        old = self.combined.framework_fraction
        try:
            for fraction, expected in [(0,b),(1,a)]:
                self.combined.framework_fraction=fraction
                np.testing.assert_allclose(self.combined.predict(self.x,self.scores)['probabilities'],expected)
        finally:
            self.combined.framework_fraction=old
        packet = self.combined.attribute(self.x.iloc[:1],self.scores.iloc[:1])
        self.assertLess(abs(packet['completeness_residual']),1e-8)
        self.assertAlmostEqual(packet['risk'],result['risk'][0])

    def test_save_load_all_models(self):
        with tempfile.TemporaryDirectory() as directory:
            for name,model in [('framework',self.framework),('probability',self.probability),('combined',self.combined)]:
                path=Path(directory)/(name+'.json');model.save(path);loaded=load_model(path)
                args=(self.x,) if name=='framework' else (self.x,self.scores)
                np.testing.assert_array_equal(model.predict(*args)['risk'],loaded.predict(*args)['risk'])
                np.testing.assert_array_equal(model.predict(*args)['prediction'],loaded.predict(*args)['prediction'])

    def test_invalid_fraction_and_model_state(self):
        for value in [-.1,1.1,float('nan')]:
            with self.assertRaises(ValueError):
                MixPHERM(self.modules,framework_options=self.framework_options,
                    prediction_options=options(),framework_fraction=value)
        payload=self.framework.to_dict();payload['parameters']['cut_low']=float('nan')
        with self.assertRaises(ValueError):
            PHERM.from_dict(payload)

    def test_commands_for_three_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.x.to_csv(root/'x.csv')
            import pandas as pd
            pd.DataFrame({'label':self.y},index=self.x.index).to_csv(root/'y.csv')
            self.scores.to_csv(root/'scores.csv');self.inner.to_csv(root/'inner.csv')
            for name,value in [('bundle',self.bundle),('framework',asdict(self.framework_options)),
                               ('parameters',asdict(options())),('reliability',{'module':.8})]:
                (root/(name+'.json')).write_text(json.dumps(value))
            def run(command,**fields):
                argv=[command]
                for k,v in fields.items():argv.extend(['--'+k.replace('_','-'),str(v)])
                self.assertEqual(main(argv),0)
            for name in ['framework','probability','combined']:
                model=root/(name+'_model.json')
                fields=dict(architecture=name,expression=root/'x.csv',labels=root/'y.csv',
                    evidence=root/'bundle.json',output=model)
                if name!='probability':fields['framework_parameters']=root/'framework.json'
                if name!='framework':fields.update(module_scores=root/'scores.csv',inner_oof=root/'inner.csv',
                    reliability=root/'reliability.json',parameters=root/'parameters.json')
                if name=='combined':fields['framework_fraction']=.4
                run('train',**fields)
                prediction_fields=dict(model=model,expression=root/'x.csv')
                if name!='framework':prediction_fields['module_scores']=root/'scores.csv'
                run('predict',**prediction_fields,output=root/(name+'.csv'))
                run('attribute',**prediction_fields,sample=self.x.index[0],nodes=64,tolerance=1e-8,
                    output=root/(name+'_attribution.json'))
                self.assertEqual(len(pd.read_csv(root/(name+'.csv'))),len(self.x))
