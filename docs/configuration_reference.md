# Configuration and Getting Started

## Model and Service Settings

| Item | Configuration |
|---|---|
| Generation model | `deepseek-v4-flash-0731` |
| Service endpoint | `https://api.deepseek.com/v1/chat/completions` |
| API key environment variable | `DEEPSEEK_API_KEY` |
| temperature | `0` |
| Response format | JSON object, set by JSONBackend |
| ESA text encoder | `sentence-transformers/all-MiniLM-L6-v2`, with frozen parameters |
| Encoder revision | `c9745ed1d9f207416be6d2e6f8de32d1f16199bf` |
| ESA contrastive head | L2 regularization with a code default; L-BFGS-B optimization |

Set max_tokens and timeout explicitly for each agent call.

## Database Queries

| Source | Query endpoint | Role in KGM-ORBIT | Required inputs |
|---|---|---|---|
| PubMed | `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`, ESearch / EFetch | Literature retrieval, titles, and abstracts | Record limit and request timeout |
| PubTator3 | `https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson` | Publication passages and entity information by PMID | PMID, retained text, and source locations |
| Reactome | `https://reactome.org/ContentService/`, search/query, data/query, data/participants | Pathways, reactions, participating genes, and relationships | Disease or pathway terms, species, and record IDs |
| Open Targets | `https://api.platform.opentargets.org/api/v4/graphql` | Disease–gene associations | Disease identifier and query fields |
| Expression Atlas | `https://www.ebi.ac.uk/gxa/json/experiments` | Expression studies and comparison context | Query terms, species, experiment accession, and relevant expression results |

Sources can be queried online or supplied as downloaded records; no local database service is required. Disease associations, pathway membership, and expression directions retain their distinct evidence meanings. The Expression Atlas experiment index locates studies; directional claims require the corresponding expression results.

Query addresses are supplied explicitly; retrievers have no default endpoint. Create a local `databases.json` with an `endpoint` for each selected database:

```json
{
  "pubmed": {"endpoint": "YOUR_PUBMED_EUTILS_BASE_URL"},
  "reactome": {"endpoint": "YOUR_REACTOME_CONTENT_SERVICE_BASE_URL"},
  "opentargets": {"endpoint": "YOUR_OPEN_TARGETS_GRAPHQL_URL"},
  "atlas": {"endpoint": "YOUR_ATLAS_EXPERIMENTS_JSON_URL"},
  "pmc": {"endpoint": "YOUR_FULL_TEXT_BASE_URL", "provider": "europepmc"}
}
```

Replace the placeholders with HTTP(S) addresses compatible with the APIs above. Pass `--databases databases.json` to `retrieve` or `evidence` when selecting online sources. Only selected entries are required. Local `--sources` needs no endpoint. PubMed appends `esearch.fcgi` and `efetch.fcgi`; Reactome appends its search and data paths. Open Targets and Atlas use the supplied endpoint directly. PMC appends `{PMC_ID}/fullTextXML` for `europepmc`, or `BioC_json/{ID}/unicode` for `bioc`. Source record links identify evidence and are separate from query endpoints.

PubMedRetriever, ReactomeRetriever, OpenTargetsRetriever, and ExpressionAtlasRetriever query the four sources directly, and merge_retrievers combines their results. PubTator3 passages and other downloaded material can be supplied through LocalCorpus. All retrievers return Source objects containing source_id, text, locator, and an optional title. locator identifies the publication, database record, or passage. Retain download dates and record identifiers with the source material.

Reactome returns pathway or reaction descriptions, participants, and references. Open Targets returns disease–gene associations. disease_id fixes the target disease; otherwise, the first disease returned by the query is used. Expression Atlas returns experiment descriptions, species, and experimental factors, filtered by species and query terms to locate expression studies.

API documentation: [Reactome Content Service](https://reactome.org/dev/content-service), [Open Targets GraphQL](https://platform-docs.opentargets.org/data-access/graphql-api), and [Expression Atlas](https://www.ebi.ac.uk/gxa/home).

## Required Inputs

| Category | Inputs to prepare |
|---|---|
| Python environment | Python 3.10+; run `pip install ".[semantic]"` in the project directory |
| Model service | Accessible endpoint, available model, API key in an environment variable, max_tokens, and timeout |
| Text encoding | The encoder above or its local directory, verified ESA validation claims and contrastive L2, or a fitted ESA head |
| Retrieval | Record limits and timeouts; species for Reactome/Atlas; optional Open Targets disease_id |
| Study task | disease, species, tissue, and a task_key and comparison for each D/S head |
| Expression and labels | Sample-by-gene expression matrix, aligned three-state labels, and probe mapping where needed |
| PHERM | D/S evidence modules and FrameworkOptions listed below |
| CoERI | Quadrature nodes, error tolerance, number of selected modules, and interpretation backend |

Set the API key in the local `DEEPSEEK_API_KEY` environment variable, rather than a configuration file. Specify local model and source directories as needed.

The service example uses caller-supplied output limits and timeouts:

```python
from kgm_orbit.evidence.backend import JSONBackend
from kgm_orbit.evidence.retrieval import (
    PubMedRetriever, ReactomeRetriever, OpenTargetsRetriever,
    ExpressionAtlasRetriever, merge_retrievers,
)

backend = JSONBackend(
    endpoint="https://api.deepseek.com/v1/chat/completions",
    model="deepseek-v4-flash-0731",
    api_key_env="DEEPSEEK_API_KEY",
    temperature=0,
    max_tokens=output_token_limit,
    timeout=model_timeout_seconds,
)
retrieve = merge_retrievers([
    PubMedRetriever(endpoint=pubmed_endpoint, limit=publication_limit, timeout=retrieval_timeout_seconds),
    ReactomeRetriever(
        endpoint=reactome_endpoint,
        limit=pathway_limit, timeout=retrieval_timeout_seconds, species=species,
    ),
    OpenTargetsRetriever(
        endpoint=opentargets_endpoint,
        limit=association_limit, timeout=retrieval_timeout_seconds,
        disease_id=disease_id,
    ),
    ExpressionAtlasRetriever(
        endpoint=atlas_endpoint,
        limit=experiment_limit, timeout=retrieval_timeout_seconds, species=species,
    ),
])
```

For example, species may be `Homo sapiens`; disease_id uses an Open Targets disease identifier. Set the record limits and timeouts for the run. The example leaves the disease and experiment parameters to the caller.

`IterativeScorer` trains on accumulated extracted positive/negative claims before retrieval feedback and selects tau on verified validation records. Each validation record contains `source_id`, `source`, two `positive` paraphrases, two `negative` paraphrases, and a binary `label` indicating support. Validation articles are excluded from training. `PairScorer` applies an already fitted head. Pass either scorer to `build_evidence` with the task, retriever, backend, and EvidenceOptions.

## Parameter Reference

Runtime settings are supplied through function arguments, `EvidenceOptions` and `FrameworkOptions`.

task defines the study question and EvidenceOptions controls evidence construction. Generated modules and evidence bundles are passed between stages as outputs. The tables below describe the parameters used by these interfaces.

## Evidence Construction

| Parameter | Meaning |
|---|---|
| disease, tissue, heads | Disease, tissue, and D/S comparisons |
| max_rounds, patience | Maximum retrieval rounds and consecutive rounds without new evidence |
| chunk_characters, chunk_overlap, top_k | Chunk length, overlap, and number of retained chunks |
| encoder | Frozen text encoder; ESA uses MiniLM representations |
| ContrastiveHead.l2 | Contrastive-learning L2; optionally overridden with --l2 |
| Contrastive-head optimization | L-BFGS-B |
| tau | Verification threshold selected on validation data |
| endpoint, model | Service endpoint and model identifier |
| api_key_env | Name of the environment variable containing the API key |
| temperature, max_tokens, timeout | Generation temperature, output limit, and timeout |
| cache_dir, replay | Response-cache location and replay mode |

Prompts are stored in `src/kgm_orbit/resources/prompts/`. Modules, gene directions, and evidence weights are generated by the workflow.

Use `--pmc` with space-separated PMC IDs to retrieve paragraphs through Europe PMC. `--sources` supplies a JSON array with source_id, text, locator, and an optional title. Omit `--database` to use only local or PMC sources. Retrieval and evidence options control record limits, chunking, and ranking.

model_options accepts service-specific parameters. For DeepSeek, `{"thinking":{"type":"disabled"}}` can be supplied in the service JSON. Core request fields are set through endpoint, model, and the other service parameters.

Each D/S task_key identifies the comparison to which a claim applies. comparison specifies the ordered groups. Direction describes the expression change under that comparison; tissue is checked against the claim's source context.

task may contain species_aliases and tissue_aliases, and each head may contain comparison_aliases, limited to semantically equivalent names. Species must match the specified study species; missing or mismatched species remain unresolved. Same-article context is linked through context_unit_ids. source_tissue and source_comparison retain the original wording.

The encoder uses the same representation for ESA training and evidence scoring. tau is selected on a validation set with verification labels and is also used to calculate evidence weights. Service parameters are passed to JSONBackend; api_key_env specifies only the environment variable name. Caching stores and replays responses to identical requests.

## PHERM and Evaluation

Training defaults are defined in `FrameworkOptions`. The optional `--parameters` JSON overrides any of these fields:

| Field | Meaning |
|---|---|
| nodes | Number of Softplus terms per module calibration |
| jaccard_threshold | Module adjacency threshold in [0,1] |
| calibration_epochs, prediction_epochs | Calibration and prediction training epochs |
| learning_rate, weight_decay | Adam learning rate and weight decay |
| loss_d, loss_s, loss_ordinal | Weights of disease, case-severity, and ordinal objectives |
| seed | Training random seed |

Omit `--parameters` to use the code defaults, or supply only the fields to override. The attribution `--nodes` argument instead controls numerical integration.

Expression preprocessing, module calibration, prediction branches, and classification cutpoints are fitted on the training samples. For repeated evaluation, supply the split seeds and fold count, and fit these steps separately within each training fold. Keep the disease evidence bundle fixed across folds when it is constructed independently of the expression data used for evaluation.

## CoERI

`PHERM.attribute` accepts one expression row, quadrature `nodes`, and numerical `tolerance`. The baseline is zero D/S evidence. It returns D, S-via-S, and S-via-lambda contributions. Integration is repeated with twice the requested nodes and checked for completeness.

`explain_prediction` also takes the evidence bundle and `module_count`. It ranks modules by the sum of absolute component contributions and links measured member genes to claims and sources. RIA receives the computed outputs and linked evidence and returns risk assessment, module interpretations, and a conclusion. `generate_report` substitutes numerical references, verifies each section against model outputs and sources, and renders the risk report. `intervene_sources` withdraws or replaces sources without changing the numerical prediction.
