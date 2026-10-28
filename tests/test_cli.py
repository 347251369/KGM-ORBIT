import json
from dataclasses import asdict
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch, Mock
import numpy as np
import pandas as pd
from test_pipeline import fixture, options, TestModules, report_response
from kgm_orbit.cli import main, load_labels
from kgm_orbit.pipeline import fit_prediction_model, interpret
from kgm_orbit.pherm.prediction import ProbPHERM
from kgm_orbit.training import fit_from_scores


class Backend:
    def complete(self, prompt, payload):
        return (dict(supported=True) if 'statement' in payload else
                report_response([dict(module_id='module', text='A increases.', claim_ids=['SA'])]))


class CommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y, _, cls.bundle = fixture()
        cls.fitted, cls.transform, cls.inner = fit_prediction_model(cls.x, cls.y, TestModules,
            options=options(), inner_folds=3, inner_seed=7)
        cls.scores = cls.transform.transform(cls.x)

    def test_training_prediction_explanation_and_report_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            self.x.to_csv(p/'expression.csv')
            pd.DataFrame({'label':self.y}, index=self.x.index).to_csv(p/'labels.csv')
            self.scores.to_csv(p/'scores.csv')
            self.inner.to_csv(p/'inner.csv')
            for name, value in [('evidence', self.bundle), ('parameters', asdict(options())), ('reliability', {'module':.8})]:
                (p/f'{name}.json').write_text(json.dumps(value))
            def run(command, **kwargs):
                argv = [command]
                for key, value in kwargs.items():
                    argv.extend(['--'+key.replace('_','-'), str(value)])
                self.assertEqual(main(argv), 0)
            run('train', architecture='probability', expression=p/'expression.csv', labels=p/'labels.csv', module_scores=p/'scores.csv',
                reliability=p/'reliability.json', inner_oof=p/'inner.csv', evidence=p/'evidence.json',
                parameters=p/'parameters.json', output=p/'model.json')
            run('predict', model=p/'model.json', expression=p/'expression.csv', module_scores=p/'scores.csv', output=p/'prediction.csv')
            predicted = pd.read_csv(p/'prediction.csv', index_col=0)
            np.testing.assert_allclose(predicted.risk, self.fitted.predict(self.x,self.scores)['risk'])
            np.testing.assert_allclose(predicted[['probability_0','probability_1','probability_2']].sum(1),1)
            run('explain', model=p/'model.json', expression=p/'expression.csv', module_scores=p/'scores.csv',
                evidence=p/'evidence.json', sample=self.x.index[30], nodes=64, module_count=1,
                tolerance=1e-8, output=p/'packet.json')
            packet = json.loads((p/'packet.json').read_text())
            self.assertAlmostEqual(packet['risk'], predicted.loc[self.x.index[30],'risk'])
            (p/'service.json').write_text('{}')
            with patch('kgm_orbit.cli.JSONBackend', return_value=Backend()):
                run('report', packet=p/'packet.json', service=p/'service.json', output=p/'report.txt')
            self.assertIn('A increases.', (p/'report.txt').read_text())
            original = (p/'packet.json').read_bytes()
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                run('explain', model=p/'model.json', expression=p/'expression.csv', module_scores=p/'scores.csv',
                    evidence=p/'evidence.json', sample=self.x.index[30], nodes=64, module_count=1,
                    tolerance=1e-8, output=p/'packet.json')
            self.assertEqual(error.exception.code,2)
            self.assertEqual(original,(p/'packet.json').read_bytes())

    def test_public_interpret_uses_current_predictor(self):
        packet, report = interpret(self.fitted, self.x.iloc[[30]], self.scores.iloc[[30]], self.bundle,
            backend=Backend(),nodes=64,module_count=1,tolerance=1e-8)
        self.assertIn('Expression branch contribution:', report['text'])
        self.assertEqual(packet['sample_id'],self.x.index[30])

    def test_invalid_inner_predictions(self):
        for field, value in [('label',9),('fold',float('inf')),('D',float('nan'))]:
            inner = self.inner.astype(float).copy();inner.iloc[0,inner.columns.get_loc(field)] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                fit_from_scores(self.x,self.y,self.scores,{'module':.8},inner,self.bundle,options=options())

    def test_label_identifiers_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'labels.csv'
            p.write_text('sample,label\n002,1\n001,0\n003,2\n')
            x=pd.DataFrame({'A':[1,2,3]},index=['001','002','003'])
            self.assertEqual(load_labels(p,x).tolist(),[0,1,2])

    def test_local_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            (p/'sources.json').write_text(json.dumps([dict(source_id='a',text='platelet evidence',locator='test')]))
            main(['retrieve','--query','platelet','--sources',str(p/'sources.json'),
                  '--limit','3','--timeout','10','--output',str(p/'result.json')])
            self.assertEqual(json.loads((p/'result.json').read_text())[0]['source_id'],'a')

    def test_evidence_command_passes_encoder_revision_and_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            for name, value in [('task', {'disease':'example'}), ('service', {}),
                ('options', dict(max_rounds=1,patience=1,chunk_characters=100,chunk_overlap=0,top_k=2,tau=.1)),
                ('sources', [dict(source_id='s',text='A increases',locator='test')])]:
                (p/f'{name}.json').write_text(json.dumps(value))
            encoder = ModuleType('sentence_transformers')
            encoder.SentenceTransformer = Mock()
            with patch.dict('sys.modules', {'sentence_transformers':encoder}), \
                 patch('kgm_orbit.cli.JSONBackend', return_value=Backend()), \
                 patch('kgm_orbit.evidence.core.ContrastiveHead.load'), \
                 patch('kgm_orbit.evidence.workflow.build_evidence',return_value=self.bundle) as build:
                main(['evidence','--task',str(p/'task.json'),'--service',str(p/'service.json'),
                      '--options',str(p/'options.json'),'--encoder','local-encoder','--encoder-revision','revision',
                      '--scorer',str(p/'esa.json'),'--sources',str(p/'sources.json'),
                      '--limit','2','--timeout','10','--output',str(p/'evidence.json')])
            encoder.SentenceTransformer.assert_called_once_with('local-encoder',revision='revision')
            self.assertEqual(build.call_args.args[0],{'disease':'example'})
            self.assertEqual(json.loads((p/'evidence.json').read_text()),json.loads(json.dumps(self.bundle)))

    def test_module_command_help_and_required_arguments(self):
        from kgm_orbit.cli import parser
        for command in ['retrieve','evidence','train','predict','explain','report']:
            with self.subTest(command=command):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as outcome:
                    parser().parse_args([command,'--help'])
                self.assertEqual(outcome.exception.code,0)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as outcome:
            parser().parse_args(['train'])
        self.assertEqual(outcome.exception.code,2)
