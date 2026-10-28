from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stderr
import numpy as np
import pandas as pd
import test_variants
from test_pipeline import report_response
from kgm_orbit.cli import main
from kgm_orbit.coeri.report import explain_prediction, generate_report, intervene_sources, explain_oof_sample
from kgm_orbit.evidence.scoring import IterativeScorer
from kgm_orbit.evidence.workflow import build_evidence, EvidenceOptions
from kgm_orbit.evidence.retrieval import Source, PubMedRetriever, PMCFullTextRetriever
from kgm_orbit.pipeline import fit_prediction_model
from kgm_orbit.training import nested_oof
from kgm_orbit.pherm.framework import PHERM


class Encoder:
    def encode(self, texts, **kwargs):
        return np.array([[float('yes' in t), float('no' in t), 1.] for t in texts])


class ReportBackend:
    def complete(self, prompt, payload):
        if 'statement' in payload:
            return dict(supported=True)
        return report_response([dict(module_id=m['module_id'], text='A increases.',
            claim_ids=[m['genes'][0]['claim_ids'][0]]) for m in payload['modules']])


class MethodFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_variants.VariantTests.setUpClass()
        cls.f = test_variants.VariantTests

    def packet(self, model, expression=None):
        return explain_prediction(model, self.f.x.iloc[:1] if expression is None else expression,
            self.f.scores.iloc[:1], self.f.bundle, nodes=64, module_count=1, tolerance=1e-8)

    def test_framework_report_components_and_source_intervention(self):
        packet = self.packet(self.f.framework)
        row = packet['modules'][0]
        self.assertEqual(set(row['components']), {'D','S_via_S','S_via_lambda'})
        self.assertAlmostEqual(row['importance'],sum(abs(v) for v in row['components'].values()))
        self.assertAlmostEqual(packet['risk'],packet['baseline']+sum(m['net'] for m in packet['modules']))
        self.assertEqual({g['head'] for g in row['genes']},{'D','S'})
        self.assertIn('S_via_lambda:',generate_report(packet,ReportBackend())['text'])
        withdrawn = intervene_sources(packet,withdraw=['source'],replacements={})
        self.assertEqual(generate_report(withdrawn,ReportBackend())['statements'],[])
        self.assertEqual(withdrawn['risk'],packet['risk'])

    def test_missing_gene_not_used_as_source_evidence(self):
        sample = self.f.x.iloc[:1].copy()
        sample['A'] = np.nan
        packet = self.packet(self.f.framework,sample)
        self.assertEqual({g['gene'] for m in packet['modules'] for g in m['genes']},{'B'})
        self.assertEqual(set(packet['claims']),{'DB','SB'})
        self.assertAlmostEqual(packet['modules'][0]['evidence']['D']['coverage'],.5)

    def test_component_ranking_does_not_cancel(self):
        original = self.f.framework.attribute(self.f.x.iloc[:1])
        original.update(D=[2.],S_via_S=[-3.],S_via_lambda=[1.])
        with patch.object(self.f.framework,'attribute',return_value=original):
            row = self.packet(self.f.framework)['modules'][0]
        self.assertEqual(row['net'],0)
        self.assertEqual(row['importance'],6)

    def test_combined_report_is_complete(self):
        packet = self.packet(self.f.combined)
        self.assertAlmostEqual(packet['risk'],packet['baseline']+
            sum(m['net'] for m in packet['modules'])+sum(packet['expression_contributions'].values()))
        self.assertTrue(generate_report(packet,ReportBackend())['statements'])

    def test_default_cli_and_saved_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            self.f.x.to_csv(p/'x.csv')
            pd.DataFrame({'label':self.f.y},index=self.f.x.index).to_csv(p/'y.csv')
            for name,value in [('bundle',self.f.bundle),('parameters',asdict(self.f.framework_options))]:
                (p/(name+'.json')).write_text(json.dumps(value))
            def run(command, **kwargs):
                argv = [command]
                for key,value in kwargs.items():
                    argv.extend(['--'+key.replace('_','-'),str(value)])
                self.assertEqual(main(argv),0)
            run('train',expression=p/'x.csv',labels=p/'y.csv',evidence=p/'bundle.json',
                parameters=p/'parameters.json',output=p/'model.json')
            self.assertEqual(json.loads((p/'model.json').read_text())['architecture'],'framework')
            run('predict',model=p/'model.json',expression=p/'x.csv',output=p/'predictions.csv')
            run('explain',model=p/'model.json',expression=p/'x.csv',evidence=p/'bundle.json',
                sample=self.f.x.index[0],nodes=64,module_count=1,tolerance=1e-8,output=p/'packet.json')
            packet = json.loads((p/'packet.json').read_text())
            self.assertTrue(packet['claims'])
            (p/'service.json').write_text('{}')
            with patch('kgm_orbit.cli.JSONBackend',return_value=ReportBackend()):
                run('report',packet=p/'packet.json',service=p/'service.json',output=p/'report.txt')
            train = self.f.x.iloc[1:]
            fitted = type(self.f.framework)(self.f.modules,self.f.framework_options).fit(train,self.f.y[1:])
            fitted.save(p/'fold.json')
            values = fitted.predict(self.f.x.iloc[:1])
            oof = pd.DataFrame([dict(sample_id=self.f.x.index[0],repeat=0,fold=0,
                score=values['risk'][0],prediction=values['prediction'][0])])
            splits = [dict(repeat=0,fold=0,train=list(train.index),test=[self.f.x.index[0]],model_file=str(p/'fold.json'))]
            result = explain_oof_sample(self.f.x,self.f.bundle,oof,splits,sample_id=self.f.x.index[0],
                repeat=0,nodes=64,module_count=1,tolerance=1e-8)
            self.assertEqual(result['evaluation']['kind'],'OOF')

    def test_iterative_esa_trains_before_feedback(self):
        validation = [dict(source_id='held'+str(label),source='A increases',positive=[pos]*2,
            negative=[neg]*2,label=label) for label,pos,neg in [(1,'yes','no'),(0,'no','yes')]]
        scorer = IterativeScorer(Encoder(),validation,l2=.01)
        class Backend:
            def complete(self,prompt,payload):
                if 'counters' in payload:
                    return dict(supported=True,counter_units=[])
                if 'chunks' in payload:
                    unit = payload['chunks'][0]
                    return dict(claims=[dict(module_id='m',subject='A',claim='A increases',
                        source_id=unit['source_id'],unit_id=unit['unit_id'],quote='A increases',
                        positive=['yes']*2,negative=['no']*2,relation='expression_change',
                        measurement='RNA',direction='increasing',tissue='blood',comparison=payload['context']['comparison'])])
                return dict(modules=[dict(module_id='m',genes=['A'],source_ids=['s'])],edges=[])
        bundle = build_evidence(dict(disease='d',tissue='blood',heads={h:dict(task_key=h,comparison=h) for h in ['D','S']}),
            backend=Backend(),retrieve=lambda q:[Source('s','A increases','synthetic')],scorer=scorer,
            options=EvidenceOptions(2,1,100,0,2,.1))
        self.assertEqual({m['head'] for m in bundle['modules']},{'D','S'})
        self.assertTrue(scorer.history)
        self.assertEqual(bundle['options']['tau'],scorer.tau)
        self.assertTrue(all(c['support_margin']>scorer.tau for c in bundle['claims']))
        self.assertTrue(bundle['trace'][0]['evidence_genes'])
        scorer.update([dict(source_id='new',source='B increases',positive=['yes']*2,negative=['no']*2)])
        self.assertEqual(len(scorer.history),2)
        blocked = IterativeScorer(Encoder(),[dict(r,source_id='PMC1:p') for r in validation],l2=.01)
        with self.assertRaisesRegex(ValueError,'outside validation'):
            blocked.update([dict(source_id='PMC1:q',source='A increases',positive=['yes']*2,negative=['no']*2)])

    def test_query_configuration_required_and_used(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p/'db.json').write_text(json.dumps({'opentargets':{'endpoint':'https://example.org/custom'}}))
            argv = ['retrieve','--query','d','--database','opentargets','--limit','2','--timeout','10','--output',str(p/'out.json')]
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                main(argv)
            with patch('kgm_orbit.evidence.retrieval._request_json',return_value={'data':{'search':{'hits':[]}}}) as request:
                self.assertEqual(main(argv+['--databases',str(p/'db.json')]),0)
            self.assertEqual(request.call_args.args[0],'https://example.org/custom')
        with self.assertRaises(TypeError):
            PubMedRetriever(limit=1,timeout=10)
        with self.assertRaises(ValueError):
            PubMedRetriever(endpoint='',limit=1,timeout=10)
        with patch('kgm_orbit.evidence.retrieval.urlopen',return_value=io.BytesIO(b'<article/>')) as request:
            PMCFullTextRetriever(['PMC1'],endpoint='https://example.org/fulltext/',limit=1,timeout=10).fetch()
        self.assertEqual(request.call_args.args[0],'https://example.org/fulltext/PMC1/fullTextXML')

    def test_framework_pipeline_entry(self):
        fitted, standardizer, inner = fit_prediction_model(self.f.x,self.f.y,
            evidence_bundle=self.f.bundle,options=self.f.framework_options)
        self.assertIsInstance(fitted,PHERM)
        self.assertIs(standardizer,fitted.standardizer)
        self.assertIsNone(inner)
        np.testing.assert_array_equal(fitted.predict(self.f.x)['risk'],self.f.framework.predict(self.f.x)['risk'])
        with self.assertRaisesRegex(ValueError,'Evidence bundle required'):
            fit_prediction_model(self.f.x,self.f.y,options=self.f.framework_options)

    def test_framework_oof_preprocessing_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            oof, splits = nested_oof(self.f.x,self.f.y,evidence_bundle=self.f.bundle,
                options=self.f.framework_options,seeds=[7,11],outer_folds=3,model_dir=directory)
            self.assertEqual(len(oof),2*len(self.f.x))
            self.assertTrue((oof.groupby(['repeat','sample_id']).size()==1).all())
            for split in splits:
                self.assertFalse(set(split['train']) & set(split['test']))
                fitted = PHERM.load(split['model_file'])
                expected = self.f.x.loc[split['train'],fitted.standardizer.genes].mean().to_numpy()
                np.testing.assert_allclose(fitted.standardizer.center,expected)
                self.assertIsNone(split['module_scores_file'])
                rows = oof[(oof['repeat']==split['repeat']) & (oof['fold']==split['fold'])].set_index('sample_id')
                values = fitted.predict(self.f.x.loc[split['test']])
                np.testing.assert_allclose(values['risk'],rows.loc[split['test'],'score'])
            row = oof.iloc[0]
            packet = explain_oof_sample(self.f.x,self.f.bundle,oof,splits,sample_id=row['sample_id'],
                repeat=int(row['repeat']),nodes=64,module_count=1,tolerance=1e-8)
            self.assertAlmostEqual(packet['risk'],row['score'])
