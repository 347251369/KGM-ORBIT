import json
import tempfile
import unittest
from pathlib import Path
from kgm_orbit.evidence.retrieval import sources_from_pmc_xml, Source
from kgm_orbit.evidence.workflow import build_evidence, EvidenceOptions, _same_context
from kgm_orbit.evidence.backend import JSONBackend
from kgm_orbit.cli import retrieval
from argparse import Namespace

class EvidenceInputTests(unittest.TestCase):
    def test_pmc_locations(self):
        xml='<article><front><article-id pub-id-type="pmc">123</article-id><article-title>Study</article-title><abstract id="a"><p>A increases.</p></abstract></front><body><sec><p id="p1">Gene <italic>A</italic> is higher.</p></sec></body></article>'
        rows=sources_from_pmc_xml(xml,'PMC123')
        self.assertEqual(rows[1].source_id,'PMC123:p1')
        self.assertEqual(rows[1].locator,'https://pmc.ncbi.nlm.nih.gov/articles/PMC123/#p1')
        self.assertEqual(rows[1].text,'Gene A is higher.')
        self.assertEqual(sources_from_pmc_xml('<article><body><p>Text</p></body></article>','PMC2')[0].title,'PMC2')

    def test_explicit_context_aliases(self):
        self.assertTrue(_same_context(' VT versus normal. ','VTE versus healthy controls',['VT versus normal']))
        self.assertFalse(_same_context('controls versus VTE','VTE versus controls'))
        self.assertFalse(_same_context('platelets','whole blood'))

    def test_local_source_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'sources.json'
            path.write_text(json.dumps([dict(source_id='s',text='A increases',locator='https://example.org')]))
            retrieve=retrieval(Namespace(limit=2,timeout=10,sources=str(path),database=[],pmc=None))
            self.assertEqual(retrieve('A')[0].source_id,'s')

    def run_workflow(self, empty=False, invalid=False):
        class Backend:
            def complete(self,prompt,payload):
                if 'counters' in payload:
                    return dict(supported=True,counter_units=[])
                if 'chunks' in payload:
                    unit=payload['chunks'][0]
                    return dict(claims=[dict(module_id=None if invalid else 'm',subject='A',
                      claim='A increases',positive=['yes']*2,negative=['no']*2,source_id='PMC1:p',
                      unit_id=unit['unit_id'],quote='A increases',relation='expression_change',
                      measurement='RNA',direction='increasing',tissue='blood',
                      comparison=payload['context']['comparison'],context_unit_ids=['PMC1:p'])])
                return dict(modules=[] if empty else [dict(module_id='m',genes=['A'],source_ids=['PMC1:p'])],edges=[])
        return build_evidence(dict(disease='d',tissue='blood',heads={h:dict(task_key=h,comparison=h) for h in ['D','S']}),
          backend=Backend(),retrieve=lambda q:[Source('PMC1:p','A increases','source')],
          scorer=lambda s,c:float(c=='yes'),options=EvidenceOptions(1,1,100,0,2,.1))

    def test_empty_discovery(self):
        result=self.run_workflow(empty=True)
        self.assertEqual(result['modules'],[])
        self.assertEqual(result['stop_reasons']['D'],'no_supported_modules')

    def test_null_module_is_recorded(self):
        result=self.run_workflow(invalid=True)
        self.assertEqual(len(result['rejected_claims']),2)
        self.assertEqual(result['modules'],[])

    def test_unambiguous_context_source_resolves(self):
        result=self.run_workflow()
        self.assertEqual(len(result['modules']),2)
        self.assertEqual(result['claims'][0]['context_unit_ids'],['PMC1:p:0:11'])

    def test_model_options_cannot_override_messages(self):
        with self.assertRaisesRegex(ValueError,'core request'):
            JSONBackend(endpoint='',model='',api_key_env='',temperature=0,max_tokens=1,timeout=1,
                        model_options={'messages':[]})
