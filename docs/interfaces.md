# Interface Reference

## Evidence Construction

`build_evidence` runs DEPA, ESA, and AVA and returns claims, modules, sources, and retrieval records. The task supplies disease, species, tissue, and ordered D/S comparisons. Only accepted RNA expression claims with matching conditions compile into directional model inputs.

`IterativeScorer` trains ESA on accumulated claim pairs and selects the verification threshold using source-disjoint verified validation records. Existing claims are rescored before retrieval feedback and final AVA verification. Sources retain their identifiers and locations.

## PHERM Training and Prediction

`PHERM(modules, options).fit(expression, labels)` fits expression preprocessing, confidence-weighted monotone calibration, adjacency-constrained D/S branches, the sample-specific lambda gate, and classification cutpoints. Expression uses unique sample IDs and gene columns; labels are aligned 0/1/2 states.

`pipeline.fit_prediction_model` validates the evidence bundle and fits PHERM:

```python
model, standardizer, _ = fit_prediction_model(
    expression, labels, evidence_bundle=bundle, options=options,
)
```

`predict(expression)` returns D, S, lambda_value, risk, state probabilities, and prediction. `save` and `load` retain preprocessing, modules, learned parameters, and training options.

The `train` command accepts expression and label CSVs and an evidence bundle. It uses the code defaults unless a parameter JSON is supplied. `PHERM` and `fit_prediction_model` also accept omitted training options. Prediction uses the saved model and expression CSV. See the [configuration reference](configuration_reference.md) for parameter fields and the [command reference](commands.md) for invocation examples.

## Interpretation and Risk Reports

`PHERM.attribute` decomposes the zero-evidence-to-sample score difference into D, S-via-S, and S-via-lambda contributions. Doubling the quadrature nodes checks convergence and contribution completeness.

`explain_prediction(model, expression, None, evidence_bundle, nodes=..., module_count=..., tolerance=...)` validates module definitions, ranks modules by the sum of absolute component contributions, and traces measured genes to directional claims and original sources. Each module retains its D/S memberships and component values.

`generate_report(packet, backend)` asks RIA for risk assessment, module interpretations, and a conclusion. It substitutes numerical references, verifies each section against model outputs and sources, and returns the rendered report and verified sections.

`pipeline.interpret(model, expression, None, evidence_bundle, backend=..., nodes=..., module_count=..., tolerance=...)` combines attribution and report generation for one sample.

`intervene_sources` removes claims depending on withdrawn primary or contextual sources while preserving numerical predictions. Replacement claims must retain the modeled proposition and conditions.

## Evaluation

`training.nested_oof(expression, labels, evidence_bundle=bundle, options=options, seeds=..., outer_folds=..., model_dir=...)` returns an OOF table and split records. PHERM is fitted independently on each outer training split with the supplied training settings. With `model_dir`, each fitted fold model is saved.

Pool held-out predictions within each repeat. `repeat_metrics` calculates ordering AUC, macro Recall, and macro F1 from sample_id, repeat, fold, label, score, and prediction.

`explain_oof_sample` loads the saved model for the requested sample and repeat, checks held-out membership and prediction agreement, and constructs its source-linked explanation from expression and the evidence bundle.
