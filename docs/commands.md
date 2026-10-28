# Command Reference

Install the package from the project directory with `python -m pip install ".[semantic]"`. Use `kgm-orbit` or `python -m kgm_orbit`; each subcommand has `--help`. Paths refer to your prepared study inputs. Output files must not already exist.

## Retrieve Sources

```bash
kgm-orbit retrieve --query "venous thromboembolism" --databases databases.json --database pubmed --database reactome --database opentargets --database atlas --species "Homo sapiens" --limit 10 --timeout 30 --output sources.json
```

Select databases with repeated `--database` arguments. `--disease-id` selects an Open Targets disease explicitly. `--sources records.json` adds local source records; `--pmc PMC123 PMC456` adds full-text passages. The result is a JSON source array. Limit 10 and timeout 30 above are usage examples.

## Construct Evidence

```bash
kgm-orbit evidence --task task.json --service service.json --options evidence_options.json --encoder sentence-transformers/all-MiniLM-L6-v2 --encoder-revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf --validation esa_validation.json --sources sources.json --limit 10 --timeout 30 --output evidence.json
```

`task.json` contains disease, species, tissue, and `heads`, with D/S `task_key` and ordered `comparison` values. `evidence_options.json` contains `max_rounds`, `patience`, `chunk_characters`, `chunk_overlap`, `top_k`, and `tau`. ESA trains on accumulated extracted claim pairs before each retrieval update and selects tau on the supplied verified validation claims. Use `--l2` to override regularization, or `--scorer esa.json` instead of `--validation` to use a fitted scorer. Local encoder directories are also accepted.

`service.json` supplies `endpoint`, `model`, `api_key_env`, `temperature`, `max_tokens`, and `timeout`; optional fields are `model_options`, `cache_dir`, and `replay`. Store the API key in the named environment variable. Configuration values and input fields are described in the [configuration reference](configuration_reference.md).

## Train and Predict

```bash
kgm-orbit train --expression expression.csv --labels labels.csv --evidence evidence.json --output pherm.json
kgm-orbit predict --model pherm.json --expression test_expression.csv --output predictions.csv
```

Expression and labels use sample IDs in the first CSV column. Labels have one column of 0/1/2 values. The evidence bundle supplies D/S modules. Training uses the defaults in `FrameworkOptions`. An optional `--parameters parameters.json` overrides individual fields listed in the [configuration reference](configuration_reference.md). Preprocessing is fitted on the training samples and saved with the model.

Prediction CSV includes D, S, lambda_value, risk, predicted state, and three state probabilities. Saved model files determine the prediction and explanation implementation automatically.

## Explain and Generate a Report

```bash
kgm-orbit explain --model pherm.json --expression test_expression.csv --evidence evidence.json --sample SAMPLE_ID --nodes 64 --module-count 5 --tolerance 1e-8 --output explanation.json
kgm-orbit report --packet explanation.json --service service.json --output report.txt
```

`explain` calculates the full-score attribution locally and links modules to their retained evidence. The example uses 64 initial quadrature nodes, five selected modules and a tolerance of 1e-8. `report` supplies computed outputs and linked evidence to RIA for risk assessment, module interpretation, and a conclusion. Each section is verified against model outputs and sources. Numerical references are filled from the packet, and the report includes contributions and citations.

