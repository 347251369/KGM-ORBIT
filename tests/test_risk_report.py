import copy
import unittest
from test_pipeline import fixture, fitted_fixture, report_response
from kgm_orbit.coeri.report import explain_prediction, generate_report


class RiskReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x, y, _, bundle = fixture()
        fitted, scores = fitted_fixture(x,y,bundle)
        cls.packet = explain_prediction(fitted,x.iloc[:1],scores.iloc[:1],bundle,
            nodes=64,module_count=1,tolerance=1e-8)

    def backend(self, response=None, reject=None):
        class Backend:
            def __init__(self):
                self.generation = None
                self.verification = []
            def complete(self, prompt, payload):
                if 'statement' in payload:
                    self.verification.append(payload)
                    return dict(supported=payload['statement']['text'] != reject)
                self.generation = payload
                return response if response is not None else report_response([
                    dict(module_id='module',text='Contribution {{contribution}} is linked to the expression evidence.',claim_ids=['SA'])])
        return Backend()

    def test_complete_report_and_numerical_context(self):
        backend = self.backend()
        report = generate_report(self.packet,backend)
        for title in ['Risk Assessment','Module Contributions','Evidence-Based Interpretation','Conclusion','Sources']:
            self.assertIn('## '+title,report['text'])
        self.assertNotIn('{{',report['text'])
        self.assertIn(f"{self.packet['risk']:.8g}",report['risk_assessment']['text'])
        self.assertEqual(backend.generation['model_output']['prediction'],self.packet['prediction'])
        self.assertEqual(backend.generation['modules'][0]['contribution'],self.packet['modules'][0]['net'])
        self.assertEqual(len(backend.verification),3)
        self.assertTrue(all(v['model_output']['risk']==self.packet['risk'] for v in backend.verification))
        self.assertIn('Contribution ',report['statements'][0]['text'])

    def test_reject_unsupported_conclusion(self):
        response = report_response([])
        response['conclusion']['text'] = 'An unsupported clinical conclusion.'
        report = generate_report(self.packet,self.backend(response,reject=response['conclusion']['text']))
        self.assertIsNone(report['conclusion'])
        self.assertNotIn('unsupported clinical',report['text'])

    def test_unknown_values_and_unlinked_citations(self):
        for key,value in [('text','Unknown {{event_probability}}.'),('claim_ids',['missing'])]:
            response = report_response([])
            response['conclusion'][key] = value
            with self.subTest(key=key),self.assertRaises(ValueError):
                generate_report(self.packet,self.backend(response))
        response = report_response([])
        response['risk_assessment']['text'] = 'A score of 0.99.'
        with self.assertRaises(ValueError):
            generate_report(self.packet,self.backend(response))

    def test_conclusion_sources_are_rendered(self):
        response = report_response([])
        response['conclusion'] = dict(text='A increases in the source comparison.',claim_ids=['SA'])
        report = generate_report(self.packet,self.backend(response))
        self.assertIn('SA: source',report['text'])

    def test_report_requires_complete_sections(self):
        response = report_response([])
        for field in response:
            incomplete = copy.deepcopy(response)
            del incomplete[field]
            with self.subTest(field=field),self.assertRaises(ValueError):
                generate_report(self.packet,self.backend(incomplete))
