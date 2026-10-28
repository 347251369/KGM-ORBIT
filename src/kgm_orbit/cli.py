"""Command-line evidence construction, PHERM prediction and interpretation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pandas as pd
import numpy as np
from .data import read_expression, read_table
from .pherm.prediction import ProbPHERM, PredictionOptions
from .pherm.framework import PHERM, FrameworkOptions
from .training import fit_from_scores
from .coeri.report import explain_prediction, generate_report
from .evidence.backend import JSONBackend
from .evidence.registry import Module, validate_bundle
from .evidence.retrieval import (Source, LocalCorpus, PubMedRetriever, ReactomeRetriever,
    OpenTargetsRetriever, ExpressionAtlasRetriever, PMCFullTextRetriever, merge_retrievers)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def load_labels(path, expression):
    frame = read_table(path)
    if frame.shape[1] != 1 or frame.index.has_duplicates or set(frame.index) != set(expression.index):
        raise ValueError('Labels require one column and the expression sample IDs')
    labels = pd.to_numeric(frame.iloc[:, 0].reindex(expression.index))
    if not labels.isin([0, 1, 2]).all():
        raise ValueError('Labels must be 0, 1 or 2')
    return labels.astype(int)


def retrieval(args):
    common = dict(limit=args.limit, timeout=args.timeout)
    config = read_json(args.databases) if getattr(args, 'databases', None) else {}
    def settings(name):
        if name not in config:
            raise ValueError('Database configuration required: '+name)
        return dict(common, **config[name])
    choices = dict(pubmed=lambda: PubMedRetriever(**settings('pubmed')),
        reactome=lambda: ReactomeRetriever(**settings('reactome'), species=args.species),
        opentargets=lambda: OpenTargetsRetriever(**settings('opentargets'), disease_id=args.disease_id),
        atlas=lambda: ExpressionAtlasRetriever(**settings('atlas'), species=args.species))
    retrievers = [choices[name]() for name in args.database]
    if args.sources:
        retrievers.append(LocalCorpus([Source(**r) for r in read_json(args.sources)], args.limit))
    if args.pmc:
        retrievers.append(PMCFullTextRetriever(args.pmc, **settings('pmc')))
    if not retrievers:
        raise ValueError('Select a database, PMC IDs or a local source file')
    return merge_retrievers(retrievers)


def run(args):
    if args.command == 'retrieve':
        write_json(args.output, [asdict(s) for s in retrieval(args)(args.query)])
    elif args.command == 'evidence':
        from sentence_transformers import SentenceTransformer
        from .evidence.core import ContrastiveHead
        from .evidence.scoring import PairScorer, IterativeScorer
        from .evidence.workflow import build_evidence, EvidenceOptions
        encoder = SentenceTransformer(args.encoder, **({'revision': args.encoder_revision} if args.encoder_revision else {}))
        if args.validation:
            scorer = IterativeScorer(encoder, read_json(args.validation), l2=args.l2)
        else:
            scorer = PairScorer(encoder, ContrastiveHead.load(args.scorer))
        bundle = build_evidence(read_json(args.task), backend=JSONBackend(**read_json(args.service)),
            retrieve=retrieval(args), scorer=scorer,
            options=EvidenceOptions(**read_json(args.options)))
        write_json(args.output, bundle)
    elif args.command == 'train':
        expression = read_expression(args.expression)
        bundle = read_json(args.evidence)
        validate_bundle(bundle)
        labels = load_labels(args.labels, expression)
        modules = [Module(**m) for m in bundle['modules']]
        if args.architecture == 'framework':
            model = PHERM(modules, FrameworkOptions(**(read_json(args.parameters) if args.parameters else {}))).fit(expression, labels)
        else:
            for key in ['module_scores', 'reliability', 'inner_oof']:
                if not getattr(args, key):
                    raise ValueError('--'+key.replace('_','-')+' required')
            model = fit_from_scores(expression, labels,
                read_expression(args.module_scores), read_json(args.reliability),
                read_table(args.inner_oof), bundle,
                options=PredictionOptions(**(read_json(args.parameters) if args.parameters else {})))
            if args.architecture == 'combined':
                from .pherm.combined import MixPHERM
                combined = MixPHERM(modules,
                    framework_options=FrameworkOptions(**(read_json(args.framework_parameters) if args.framework_parameters else {})),
                    prediction_options=model.options, framework_fraction=args.framework_fraction)
                combined.probability = model
                combined.framework.fit(expression, labels)
                model = combined
        model.save(args.output)
    elif args.command in {'predict', 'explain'}:
        from .variants import load_model
        model = load_model(args.model)
        expression = read_expression(args.expression)
        scores = None
        if not isinstance(model, PHERM):
            if not args.module_scores:
                raise ValueError('--module-scores required')
            scores = read_expression(args.module_scores)
        if args.command == 'predict':
            values = model.predict(expression) if scores is None else model.predict(expression, scores)
            frame = pd.DataFrame({k:v for k,v in values.items() if np.asarray(v).ndim == 1}, index=expression.index)
            for j in range(3):
                frame[f'probability_{j}'] = values['probabilities'][:, j]
            frame.to_csv(args.output, index_label='sample_id')
        else:
            if args.sample not in expression.index or (scores is not None and args.sample not in scores.index):
                raise ValueError('Sample ID is absent from expression or module scores')
            bundle = read_json(args.evidence)
            validate_bundle(bundle)
            packet = explain_prediction(model, expression.loc[[args.sample]],
                None if scores is None else scores.loc[[args.sample]], bundle,
                nodes=args.nodes, module_count=args.module_count, tolerance=args.tolerance)
            write_json(args.output, packet)
    elif args.command == 'report':
        result = generate_report(read_json(args.packet), JSONBackend(**read_json(args.service)))
        Path(args.output).write_text(result['text'], encoding='utf-8')


def parser():
    root = argparse.ArgumentParser(prog='kgm-orbit', description=__doc__)
    sub = root.add_subparsers(dest='command', required=True)
    def required(p, *names):
        for name in names:
            p.add_argument('--'+name, required=True)
    def sources(p):
        p.add_argument('--databases', help='Database endpoint configuration JSON')
        p.add_argument('--database', choices=['pubmed', 'reactome', 'opentargets', 'atlas'], action='append', default=[])
        p.add_argument('--sources', help='Local source-record JSON')
        p.add_argument('--pmc', nargs='+', help='PMC article IDs')
        p.add_argument('--species', default='Homo sapiens')
        p.add_argument('--disease-id')
        p.add_argument('--limit', type=int, required=True)
        p.add_argument('--timeout', type=float, required=True)
    p = sub.add_parser('retrieve', help='Query databases or source records')
    required(p, 'query', 'output'); sources(p)
    p = sub.add_parser('evidence', help='Build DEPA/ESA/AVA evidence')
    required(p, 'task', 'service', 'options', 'encoder', 'output'); sources(p)
    scoring = p.add_mutually_exclusive_group(required=True)
    scoring.add_argument('--validation', help='Verified ESA validation claims JSON')
    scoring.add_argument('--scorer', help='Fitted ESA head JSON')
    p.add_argument('--l2', type=float, default=0.1)
    p.add_argument('--encoder-revision')
    p = sub.add_parser('train', help='Fit PHERM from expression and evidence')
    required(p, 'expression', 'labels', 'evidence', 'output')
    p.add_argument('--parameters', help='Optional training parameter overrides')
    p.add_argument('--architecture', choices=['framework', 'probability', 'combined'], default='framework')
    for name in ['module-scores', 'reliability', 'inner-oof', 'framework-parameters']:
        p.add_argument('--'+name)
    p.add_argument('--framework-fraction', type=float, default=0.5)
    p = sub.add_parser('predict', help='Predict with a saved PHERM model')
    required(p, 'model', 'expression', 'output')
    p.add_argument('--module-scores')
    p = sub.add_parser('explain', help='Calculate a source-linked attribution packet')
    required(p, 'model', 'expression', 'evidence', 'sample', 'output')
    p.add_argument('--module-scores')
    p.add_argument('--nodes', type=int, required=True)
    p.add_argument('--module-count', type=int, required=True)
    p.add_argument('--tolerance', type=float, required=True)
    p = sub.add_parser('report', help='Generate text from an attribution packet')
    required(p, 'packet', 'service', 'output')
    return root


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    try:
        if Path(args.output).exists():
            raise FileExistsError('Output already exists: '+args.output)
        run(args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        command.exit(2, f'kgm-orbit: {exc}\n')
    return 0
