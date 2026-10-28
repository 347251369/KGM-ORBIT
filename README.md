# KGM-ORBIT

KGM-ORBIT integrates disease knowledge with transcriptomic data for ordinal risk assessment and traceable interpretation. A pathway-guided multiagent system comprising DEPA, ESA, and AVA iteratively retrieves, structures, and verifies source evidence to construct a disease evidence set. The Pathway-based Hierarchical Evidence Risk Model (PHERM) combines this evidence with gene expression to construct and hierarchically integrate disease risk evidence and risk severity evidence into an ordinal risk score. Chain-of-Evidence-based Risk Interpretation (CoERI) verifies attribution consistency and links module contributions to biological interpretations and original sources, guiding agents to generate traceable reports consistent with individual predictions.

## Requirements

Python 3.10+, NumPy, pandas, SciPy, scikit-learn, sentence-transformers, and PyTorch. Dependency ranges are specified in [pyproject.toml](pyproject.toml).

1. Enter the project directory and create an environment:

   ```bash
   cd KGM-ORBIT
   conda create -n kgm-orbit python=3.10 -y
   conda activate kgm-orbit
   ```

2. Install the package and text-encoding dependencies:

   ```bash
   python -m pip install ".[semantic]"
   ```

3. Configure the model service and retrieval sources using the [configuration reference](docs/configuration_reference.md). Set the API key in `DEEPSEEK_API_KEY`. Evidence construction uses a text encoder and verified validation claims to train ESA and select its verification threshold.

The model service uses `deepseek-v4-flash-0731`. Retrieval supports PubMed, Reactome, Open Targets, Expression Atlas.

## Data Downloads

Download the original data from GEO:

| Dataset | GEO record | Download files |
|---|---|---|
| VTE | [GSE48000](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE48000) | [GEO files](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE48nnn/GSE48000/) |
| MASLD | [GSE126848](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126848) | [GEO files](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE126nnn/GSE126848/) |
| AD | [GSE63060](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63060) | [GEO files](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63060/) |
| Sepsis | [GSE154918](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE154918) | [GEO files](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE154nnn/GSE154918/) |

## Data Preparation

Prepare the following inputs for your study:

| Input | Format |
|---|---|
| Gene expression | CSV with sample IDs in the first column and genes in the remaining columns |
| Clinical-state labels | CSV with sample IDs and one label column containing `0`, `1`, or `2` |
| Evidence task | Disease, species, tissue, and the D/S task keys and ordered comparison groups |
| Source records | Retrieved passages or a JSON array containing `source_id`, `text`, and `locator` |

Align expression and label sample IDs, and define the three-state label mapping for the study. Disease modules, gene directions, and evidence weights are obtained through evidence construction.

## Software Tests

Run the code tests with:

```bash
python -m unittest discover -s tests -v
```

## Usage

Use the command line for retrieval, evidence construction, training, prediction, attribution, and report generation:

```bash
kgm-orbit --help
kgm-orbit train --help
```

`python -m kgm_orbit` is an equivalent entry point. See the [command reference](docs/commands.md) for required inputs and examples, the [configuration reference](docs/configuration_reference.md) for training parameters, and the [interface reference](docs/interfaces.md) for Python usage.
