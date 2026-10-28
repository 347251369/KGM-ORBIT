import io
import json
import unittest
from unittest.mock import patch

from kgm_orbit.evidence.retrieval import (
    ReactomeRetriever, OpenTargetsRetriever, ExpressionAtlasRetriever,
    _request_json, merge_retrievers,
)


class DatabaseQueryTests(unittest.TestCase):
    @patch('kgm_orbit.evidence.retrieval._request_json')
    def test_reactome_preserves_summary_and_references(self, request):
        request.side_effect = [
            {'results': [{'entries': [{'stId': 'R-HSA-1'}, {'stId': 'R-HSA-1'}]}]},
            {'displayName': 'Platelet activation', 'summation': [{'text': 'A curated reaction.'}],
             'literatureReference': [{'pubMedIdentifier': 123}]},
            [{'displayName': 'Protein A'}, {'displayName': 'Protein B'}],
        ]
        records = ReactomeRetriever(endpoint='https://example.org/reactome', limit=2, timeout=10)('platelet')
        self.assertEqual(len(records), 1)
        self.assertIn('A curated reaction.', records[0].text)
        self.assertIn('Protein B', records[0].text)
        self.assertIn('123', records[0].text)
        self.assertTrue(records[0].locator.endswith('R-HSA-1'))

    @patch('kgm_orbit.evidence.retrieval._request_json')
    def test_open_targets_uses_selected_disease(self, request):
        request.return_value = {'data': {'disease': {
            'name': 'Disease A', 'associatedTargets': {'rows': [{
                'score': .7, 'datatypeScores': [{'id': 'genetic_association', 'score': .7}],
                'target': {'id': 'ENSG1', 'approvedSymbol': 'GENE1'},
            }]},
        }}}
        records = OpenTargetsRetriever(endpoint='https://example.org/graphql', limit=1, timeout=10, disease_id='EFO_1')('GENE1')
        self.assertEqual(records[0].source_id, 'OT:EFO_1:ENSG1')
        self.assertIn('genetic_association', records[0].text)
        self.assertEqual(request.call_args.args[2]['variables'], {'id': 'EFO_1', 'n': 1})
        self.assertEqual(request.call_count, 1)

    @patch('kgm_orbit.evidence.retrieval._request_json')
    def test_open_targets_search_and_empty_results(self, request):
        request.return_value = {'data': {'search': {'hits': []}}}
        self.assertEqual(OpenTargetsRetriever(endpoint='https://example.org/graphql', limit=1, timeout=10)('unknown'), [])
        request.side_effect = [
            {'data': {'search': {'hits': [{'id': 'EFO_2', 'name': 'Disease B'}]}}},
            {'data': {'disease': None}},
        ]
        self.assertEqual(OpenTargetsRetriever(endpoint='https://example.org/graphql', limit=1, timeout=10)('Disease B'), [])
        self.assertEqual(request.call_args.args[2]['variables']['id'], 'EFO_2')

    @patch('kgm_orbit.evidence.retrieval._request_json')
    def test_atlas_filters_species_and_unrelated_index_entries(self, request):
        request.return_value = {'experiments': [
            {'experimentAccession': 'E-1', 'species': 'Homo sapiens',
             'experimentDescription': 'Sepsis blood study',
             'experimentalFactors': ['disease', 'sampling time']},
            {'experimentAccession': 'E-2', 'species': 'Mus musculus',
             'experimentDescription': 'Sepsis study'},
            {'experimentAccession': 'E-3', 'species': 'Homo sapiens',
             'experimentDescription': 'Healthy liver study'},
        ]}
        records = ExpressionAtlasRetriever(endpoint='https://example.org/atlas', limit=5, timeout=10)('sepsis')
        self.assertEqual([s.source_id for s in records], ['GXA:E-1'])
        self.assertIn('sampling time', records[0].text)
        self.assertTrue(records[0].locator.endswith('/E-1/Results'))

    @patch('kgm_orbit.evidence.retrieval.urlopen')
    def test_transport_posts_json_and_reports_graphql_errors(self, open_url):
        open_url.return_value = io.BytesIO(b'{"data": {}}')
        self.assertEqual(_request_json('https://example.org/query', 10, {'query': 'query X{}'}), {'data': {}})
        request = open_url.call_args.args[0]
        self.assertEqual(request.get_method(), 'POST')
        self.assertEqual(json.loads(request.data), {'query': 'query X{}'})
        open_url.return_value = io.BytesIO(b'{"errors": [{"message": "Unknown field"}]}')
        with self.assertRaisesRegex(ValueError, 'Unknown field'):
            _request_json('https://example.org/query', 10)

    def test_invalid_query_settings(self):
        for retriever in [ReactomeRetriever, OpenTargetsRetriever, ExpressionAtlasRetriever]:
            with self.assertRaises(ValueError):
                retriever(endpoint='https://example.org/query', limit=0, timeout=10)
            with self.assertRaises(ValueError):
                retriever(endpoint='https://example.org/query', limit=1, timeout=float('nan'))


if __name__ == '__main__':
    unittest.main()
