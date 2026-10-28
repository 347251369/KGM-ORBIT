from dataclasses import dataclass, asdict
from urllib.parse import urlencode, quote, urlsplit
from urllib.request import Request, urlopen
from html import unescape
import re
import json
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass(frozen=True)
class Source:
    source_id: str
    text: str
    locator: str
    title: str = ''

def chunk_sources(sources, *, characters, overlap):
    if characters <= 0 or not 0 <= overlap < characters:
        raise ValueError('Invalid chunk sizes')
    result = []
    for source in sources:
        for start in range(0, len(source.text), characters - overlap):
            end = min(start + characters, len(source.text))
            result.append(dict(source_id=source.source_id,
                unit_id=f'{source.source_id}:{start}:{end}',
                text=source.text[start:end],
                locator=source.locator,
                start=start,
                end=end))
            if end == len(source.text):
                break
    return result

def rank_chunks(query, chunks, *, top_k, semantic_encoder=None, semantic_weight=0):
    if top_k < 1 or not 0 <= semantic_weight <= 1:
        raise ValueError('Invalid ranking settings')
    if not chunks:
        return []
    if semantic_weight and semantic_encoder is None:
        raise ValueError('Semantic encoder required')
    texts = [c['text'] for c in chunks]
    vec = TfidfVectorizer(ngram_range=(1, 2), token_pattern='(?u)\\b\\w+\\b')
    try:
        matrix = vec.fit_transform(texts)
        lexical = (matrix @ vec.transform([query]).T).toarray().ravel()
    except ValueError:
        lexical = np.zeros(len(texts))
    score = (1 - semantic_weight) * lexical
    if semantic_weight:
        embeddings = np.asarray(semantic_encoder.encode([query] + texts, normalize_embeddings=True))
        score += semantic_weight * (embeddings[1:] @ embeddings[0])
    order = sorted(range(len(chunks)), key=lambda i: (-score[i], chunks[i]['unit_id']))[:top_k]
    return [dict(chunks[i], retrieval_score=float(score[i])) for i in order]

class LocalCorpus:

    def __init__(self, sources, limit):
        self.sources = list(sources)
        self.limit = limit

    def __call__(self, query):
        chunks = [dict(source_id=s.source_id, unit_id=s.source_id, text=s.text) for s in self.sources]
        ranked = rank_chunks(query, chunks, top_k=self.limit)
        by_id = {s.source_id: s for s in self.sources}
        return [by_id[r['source_id']] for r in ranked]


def sources_from_pmc_xml(content, identifier):
    root = ET.fromstring(content)
    title_node = root.find('.//article-title')
    title = ''.join(title_node.itertext()) if title_node is not None else identifier
    pmc = root.findtext('.//article-id[@pub-id-type="pmc"]') or identifier
    pmc = pmc if pmc.startswith('PMC') else 'PMC' + pmc
    nodes = root.findall('.//abstract') + root.findall('.//body//p') + root.findall('.//body//table-wrap')
    sources = []
    for index, node in enumerate(nodes):
        text = ' '.join(' '.join(node.itertext()).split())
        if not text:
            continue
        location = node.get('id') or f'{node.tag}_{index}'
        locator = f'https://pmc.ncbi.nlm.nih.gov/articles/{pmc}/'
        locator += '#' + location if node.get('id') else '; ' + location
        sources.append(Source(f'{pmc}:{location}', text, locator, title))
    return sources


class PMCFullTextRetriever:

    def __init__(self, article_ids, *, endpoint, limit, timeout, provider='europepmc'):
        _query_settings(limit, timeout)
        self.endpoint = _endpoint(endpoint)
        self.article_ids = list(dict.fromkeys(article_ids))
        if not self.article_ids or any(not re.fullmatch(r'(?:PMC)?\d+', value) for value in self.article_ids):
            raise ValueError('Provide numeric PMID or PMC identifiers')
        self.limit, self.timeout = limit, timeout
        if provider not in {'europepmc', 'bioc'}:
            raise ValueError('Choose europepmc or bioc')
        if provider == 'europepmc' and any(not value.startswith('PMC') for value in self.article_ids):
            raise ValueError('Europe PMC full text requires PMC identifiers')
        self.provider = provider
        self._sources = None

    def fetch(self):
        result = []
        for identifier in self.article_ids:
            if self.provider == 'europepmc':
                url = self.endpoint + f'/{identifier}/fullTextXML'
                with urlopen(url, timeout=self.timeout) as response:
                    result.extend(sources_from_pmc_xml(response.read(), identifier))
                continue
            url = self.endpoint + f'/BioC_json/{identifier}/unicode'
            payload = _request_json(url, self.timeout)
            collections = payload if isinstance(payload, list) else [payload]
            for collection in collections:
                for document in collection['documents']:
                    passages = document.get('passages', [])
                    pmcid = str(document.get('id') or identifier)
                    if not pmcid.startswith('PMC'):
                        pmcid = 'PMC' + pmcid
                    title = next((p['text'] for p in passages
                                  if p.get('infons', {}).get('type') == 'front' and p.get('text')), pmcid)
                    for index, passage in enumerate(passages):
                        text = passage.get('text', '').strip()
                        if not text or passage.get('infons', {}).get('section_type') == 'REF':
                            continue
                        offset = passage.get('offset', index)
                        result.append(Source(
                            f'{pmcid}:passage:{offset}', text,
                            f'https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/; BioC offset {offset}', title,
                        ))
        self._sources = result
        return result

    def __call__(self, query):
        sources = self.fetch() if self._sources is None else self._sources
        return LocalCorpus(sources, self.limit)(query)

class PubMedRetriever:

    def __init__(self, *, endpoint, limit, timeout):
        _query_settings(limit, timeout)
        self.endpoint = _endpoint(endpoint)
        self.limit = limit
        self.timeout = timeout

    def __call__(self, query):
        base = self.endpoint + '/'
        with urlopen(base + 'esearch.fcgi?' + urlencode(dict(db='pubmed', term=query, retmode='json', retmax=self.limit)),
            timeout=self.timeout) as r:
            ids = json.load(r)['esearchresult']['idlist']
        if not ids:
            return []
        with urlopen(base + 'efetch.fcgi?' + urlencode(dict(db='pubmed', id=','.join(ids), retmode='xml')),
            timeout=self.timeout) as r:
            root = ET.fromstring(r.read())
        result = []
        for article in root.findall('.//PubmedArticle'):
            pmid = article.findtext('.//MedlineCitation/PMID')
            title = ''.join(article.find('.//ArticleTitle').itertext())
            text = '\n'.join((''.join(p.itertext()) for p in article.findall('.//AbstractText')))
            if text:
                result.append(Source(f'PMID:{pmid}',
                    text,
                    f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
                    title))
        return result

def merge_retrievers(retrievers):

    def retrieve(query):
        sources = {}
        for retriever in retrievers:
            for s in retriever(query):
                if s.source_id in sources and sources[s.source_id].text != s.text:
                    raise ValueError('Source ID has conflicting content')
                sources[s.source_id] = s
        return list(sources.values())
    return retrieve


def _request_json(url, timeout, payload=None):
    body = None if payload is None else json.dumps(payload).encode('utf-8')
    request = Request(url, data=body, headers={
        'Accept': 'application/json', 'Content-Type': 'application/json',
        'User-Agent': 'KGM-ORBIT',
    })
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if isinstance(result, dict) and result.get('errors'):
        raise ValueError(f'Database query failed: {result["errors"]}')
    return result


def _query_settings(limit, timeout):
    if type(limit) is not int or limit < 1 or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError('Positive record limit and timeout required')


def _endpoint(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Database endpoint required')
    value = value.strip().rstrip('/')
    parsed = urlsplit(value)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.query or parsed.fragment:
        raise ValueError('Database endpoint must be an HTTP(S) URL without query or fragment')
    return value


def _text(value):
    if isinstance(value, list):
        return '; '.join(filter(None, (_text(item) for item in value)))
    if isinstance(value, dict):
        return _text(value.get('text') or value.get('displayName') or value.get('name') or '')
    return unescape(re.sub(r'<[^>]+>', '', str(value or ''))).strip()


class ReactomeRetriever:

    def __init__(self, *, endpoint, limit, timeout, species='Homo sapiens'):
        _query_settings(limit, timeout)
        self.limit, self.timeout, self.species = limit, timeout, species
        self.endpoint = _endpoint(endpoint)

    def __call__(self, query):
        url = self.endpoint + '/search/query?' + urlencode({
            'query': query, 'species': self.species,
            'types': 'Pathway,Reaction', 'cluster': 'true',
        })
        payload = _request_json(url, self.timeout)
        groups = payload['results']
        entries = [item for group in groups for item in group.get('entries', [])]
        result, seen = [], set()
        for item in entries:
            stable_id = str(item.get('stId') or item.get('id') or '')
            if not stable_id or stable_id in seen:
                continue
            seen.add(stable_id)
            identifier = quote(stable_id, safe='')
            detail = _request_json(self.endpoint + '/data/query/' + identifier, self.timeout)
            participants = _request_json(
                self.endpoint + '/data/participants/' + identifier, self.timeout,
            )
            title = _text(detail.get('displayName') or item.get('name') or stable_id)
            summary = _text(detail.get('summation') or item.get('summation'))
            names = sorted({_text(p) for p in participants if _text(p)})
            references = [str(r.get('pubMedIdentifier') or r.get('doi') or _text(r))
                          for r in detail.get('literatureReference', [])]
            parts = [title, summary]
            if names:
                parts.append('Participants: ' + '; '.join(names))
            if references:
                parts.append('References: ' + '; '.join(references))
            result.append(Source(
                'REACTOME:' + stable_id, '\n'.join(filter(None, parts)),
                'https://reactome.org/content/detail/' + identifier, title,
            ))
            if len(result) >= self.limit:
                break
        return result


class OpenTargetsRetriever:

    def __init__(self, *, endpoint, limit, timeout, disease_id=None):
        _query_settings(limit, timeout)
        self.limit, self.timeout, self.disease_id = limit, timeout, disease_id
        self.endpoint = _endpoint(endpoint)

    def __call__(self, query):
        disease_id = self.disease_id
        if disease_id is None:
            search = _request_json(self.endpoint, self.timeout, {
                'query': 'query Search($q:String!){search(queryString:$q,'
                         'entityNames:["disease"],page:{index:0,size:1}){hits{id name}}}',
                'variables': {'q': query},
            })
            hits = search['data']['search']['hits']
            if not hits:
                return []
            disease_id = hits[0]['id']
        payload = _request_json(self.endpoint, self.timeout, {
            'query': 'query Assoc($id:String!,$n:Int!){disease(efoId:$id){id name '
                     'associatedTargets(page:{index:0,size:$n}){rows{score '
                     'datatypeScores{id score} target{id approvedSymbol approvedName}}}}}',
            'variables': {'id': disease_id, 'n': self.limit},
        })
        disease = payload['data']['disease']
        if disease is None:
            return []
        result = []
        for row in disease['associatedTargets']['rows'][:self.limit]:
            target = row['target']
            symbol = target.get('approvedSymbol') or target['id']
            title = f"{disease['name']} — {symbol} association"
            types = ', '.join(t['id'] for t in row.get('datatypeScores', []))
            text = (f"{title}. Target: {target['id']}. "
                    f"Association ranking score: {row['score']}. Evidence types: {types}.")
            result.append(Source(
                f"OT:{disease_id}:{target['id']}", text,
                f'https://platform.opentargets.org/evidence/{quote(target["id"], safe="")}/'
                f'{quote(disease_id, safe="")}', title,
            ))
        return result


class ExpressionAtlasRetriever:

    def __init__(self, *, endpoint, limit, timeout, species='Homo sapiens'):
        _query_settings(limit, timeout)
        self.limit, self.timeout, self.species = limit, timeout, species
        self.endpoint = _endpoint(endpoint)

    def __call__(self, query):
        url = self.endpoint + '?' + urlencode({'query': query, 'species': self.species})
        payload = _request_json(url, self.timeout)
        if isinstance(payload, list):
            experiments = payload
        elif 'experiments' in payload:
            experiments = payload['experiments']
        else:
            experiments = payload['results']
        terms = set(re.findall(r'[\w-]+', query.casefold()))
        terms -= {'and', 'or', 'the', 'gene', 'genes', 'expression', 'pathways'}
        result, seen = [], set()
        for item in experiments:
            species = item.get('species')
            if species and self.species.casefold() not in {
                str(s).casefold() for s in (species if isinstance(species, list) else [species])
            }:
                continue
            accession = item.get('experimentAccession') or item.get('accession')
            if not accession or accession in seen:
                continue
            title = _text(item.get('experimentDescription') or item.get('description')
                          or item.get('title') or accession)
            factors = item.get('experimentalFactors') or []
            factor_text = json.dumps(factors, ensure_ascii=False)
            searchable = (title + ' ' + factor_text + ' ' + accession).casefold()
            if terms and not any(term in searchable for term in terms):
                continue
            seen.add(accession)
            kind = item.get('experimentType') or item.get('rawExperimentType') or ''
            text = (f'{title}\nAccession: {accession}\nSpecies: {species or ""}'
                    f'\nExperiment type: {kind}\nExperimental factors: {factor_text}')
            result.append(Source(
                'GXA:' + accession, text,
                f'https://www.ebi.ac.uk/gxa/experiments/{quote(accession, safe="")}/Results',
                title,
            ))
            if len(result) >= self.limit:
                break
        return result
