import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .cli import read_json, write_json, load_labels
from .data import read_expression
from .data import read_table
from .evidence.registry import Module, validate_bundle
from .pherm.framework import PHERM, FrameworkOptions
from .pherm.prediction import ProbPHERM, PredictionOptions
from .pherm.combined import MixPHERM
from .training import fit_from_scores


def load_model(path):
    architecture = read_json(path).get('architecture', 'probability')
    models = dict(framework=PHERM, probability=ProbPHERM, combined=MixPHERM)
    if architecture not in models:
        raise ValueError('Unknown model architecture')
    return models[architecture].load(path)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m kgm_orbit.variants')
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train')
    train.add_argument('--architecture', choices=['framework', 'probability', 'combined'], required=True)
    for name in ['expression', 'labels', 'evidence', 'output']:
        train.add_argument('--'+name, required=True)
    for name in ['module-scores', 'reliability', 'inner-oof', 'parameters', 'framework-parameters']:
        train.add_argument('--'+name)
    train.add_argument('--framework-fraction', type=float, default=0.5)
    for name in ['predict', 'attribute']:
        command = commands.add_parser(name)
        for field in ['model', 'expression', 'output']:
            command.add_argument('--'+field, required=True)
        command.add_argument('--module-scores')
        if name == 'attribute':
            command.add_argument('--sample', required=True)
            command.add_argument('--nodes', type=int, required=True)
            command.add_argument('--tolerance', type=float, required=True)
    args = parser.parse_args(argv)
    try:
        if Path(args.output).exists():
            raise FileExistsError('Output already exists: '+args.output)
        expression = read_expression(args.expression)
        if args.command == 'train':
            bundle = read_json(args.evidence)
            validate_bundle(bundle)
            modules = [Module(**m) for m in bundle['modules']]
            labels = load_labels(args.labels, expression)
            if args.architecture in {'framework', 'combined'}:
                framework_options = FrameworkOptions(**(read_json(args.framework_parameters) if args.framework_parameters else {}))
            if args.architecture == 'framework':
                model = PHERM(modules, framework_options).fit(expression, labels)
            else:
                for key in ['module_scores', 'reliability', 'inner_oof']:
                    if not getattr(args, key):
                        raise ValueError('--'+key.replace('_', '-')+' required')
                if args.architecture == 'combined' and (args.framework_fraction is None
                        or not np.isfinite(args.framework_fraction) or not 0 <= args.framework_fraction <= 1):
                    raise ValueError('--framework-fraction must be in [0,1]')
                prediction_options = PredictionOptions(**(read_json(args.parameters) if args.parameters else {}))
                probability = fit_from_scores(expression, labels, read_expression(args.module_scores),
                    read_json(args.reliability), read_table(args.inner_oof), bundle, options=prediction_options)
                if args.architecture == 'probability':
                    model = probability
                else:
                    model = MixPHERM(modules, framework_options=framework_options,
                        prediction_options=prediction_options, framework_fraction=args.framework_fraction)
                    model.probability = probability
                    model.framework.fit(expression, labels)
            model.save(args.output)
        else:
            model = load_model(args.model)
            scores = None
            if not isinstance(model, PHERM):
                if not args.module_scores:
                    raise ValueError('--module-scores required')
                scores = read_expression(args.module_scores)
            if args.command == 'predict':
                values = model.predict(expression) if scores is None else model.predict(expression, scores)
                frame = pd.DataFrame({k:v for k,v in values.items() if np.asarray(v).ndim == 1}, index=expression.index)
                for i in range(3):
                    frame[f'probability_{i}'] = values['probabilities'][:, i]
                frame.to_csv(args.output, index_label='sample_id')
            else:
                sample = expression.loc[[args.sample]]
                kwargs = dict(nodes=args.nodes, tolerance=args.tolerance)
                packet = model.attribute(sample, **kwargs) if scores is None else model.attribute(sample, scores.loc[[args.sample]], **kwargs)
                write_json(args.output, packet)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f'kgm-orbit: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
