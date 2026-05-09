# IPF-RAG: Temporal-Aware Retrieval-Augmented Generation for Idiopathic Pulmonary Fibrosis Care

This repository contains the cleaned, publication-ready code release for the paper **IPF-RAG: Temporal-Aware Retrieval-Augmented Generation for Idiopathic Pulmonary Fibrosis Care**.

## Overview
IPF-RAG is a retrieval-augmented clinical generation framework for idiopathic pulmonary fibrosis care. The released code focuses on the core methodological components used in the paper:

- temporal-aware retrieval over longitudinal IPF clinical records;
- case-structured evidence aggregation;
- medication evidence augmentation;
- standard RAG baseline;
- ablation experiment entrypoints.

This repository is intentionally curated for academic release. Exploratory code, temporary scripts, private datasets, local indexes, and secrets have been excluded.

## Repository Structure

```text
ipf_rag_github_release/
  config/
    config.example.yaml
    private_data_placeholder.md
  examples/
    test_questions.txt
  scripts/
    run_ablation.py
    run_generation.py
  src/
    ipf_rag/
      core/
      experiments/
      pipelines/
      utils/
  .env.example
  .gitignore
  LICENSE
  pyproject.toml
  README.md
  requirements.txt
```

## Included Components
- `src/ipf_rag/pipelines/rag_pipeline.py`:
  core pipeline implementation for Standard RAG and IPF-RAG.
- `src/ipf_rag/core/prompts.py`:
  prompt templates for the main framework and ablation variants.
- `src/ipf_rag/experiments/generation.py`:
  generation entry logic for the main IPF-RAG experiment.
- `src/ipf_rag/experiments/ablation.py`:
  ablation experiment entry logic.
- `scripts/run_generation.py`:
  script entrypoint for main generation.
- `scripts/run_ablation.py`:
  script entrypoint for ablation studies.

## Excluded Components
The following are deliberately not included in the public release:

- private IPF knowledge base;
- local/private FAISS indexes;
- private medication corpus files;
- MIMIC-derived private data files;
- raw experiment outputs and logs;
- API keys, private endpoints, and local personal paths;
- temporary, testing, or exploratory scripts not required for reproducing the method.

## Private Data Layout
Because the underlying IPF knowledge base is private, users should prepare the following local directory structure by themselves:

```text
private_data/
  ipf_knowledge_base/
  medication_information/
    Specific_information.txt
    drug_dict_cache.json
  indexes/
    ipf_index/
    medication_index/
models/
  bge-small-zh-v1.5/
examples/
  test_questions.txt
```

Notes:
- `private_data/` must remain untracked.
- `Specific_information.txt` is expected by the medication evidence loader.
- FAISS indexes should be locally prepared from private resources.
- The local embedding model path currently assumes `models/bge-small-zh-v1.5/`.

## Installation
### Option 1: install from `pyproject.toml`

```bash
pip install -e .
```

### Option 2: install from `requirements.txt`

```bash
pip install -r requirements.txt
```

Recommended Python version: `3.10+`.

## Environment Variables
Copy `.env.example` to `.env` and fill in your API key:

```bash
IPF_RAG_API_KEY=your_real_key
```

The current release reads the API key from the environment variable `IPF_RAG_API_KEY`.

## How to Run
### Main IPF-RAG generation

```bash
python scripts/run_generation.py
```

### Ablation study

```bash
python scripts/run_ablation.py
```

Generated outputs will be written to the local `outputs/` directory.

## Methodological Notes
This release preserves the main logic of the paper implementation:

1. retrieve candidate longitudinal IPF records;
2. split records by temporal sequence;
3. align medication orders with temporally relevant disease-course segments;
4. aggregate retrieved case evidence into a structured context block;
5. augment generation with medication knowledge;
6. generate the final recommendation with a case-structured prompt.

The codebase was refactored for clarity and public release readiness, but the core methodological logic was kept consistent with the original experimental implementation.

## Reproducibility Statement
Due to privacy restrictions, the original IPF knowledge base and local knowledge base are not publicly released. Therefore, exact end-to-end numerical replication requires authorized local access to equivalent private data resources.

This public repository is intended to provide:
- the full method implementation logic;
- the main retrieval and prompting pipeline;
- experiment entrypoints for generation and ablation;
- a clean, reusable code structure suitable for academic inspection and secondary development.

## Citation
If you use this repository in academic work, please cite the corresponding paper.

```text
@article{ipf_rag_2026,
  title={IPF-RAG: Temporal-Aware Retrieval-Augmented Generation for Idiopathic Pulmonary Fibrosis Care},
  author={Anonymous},
  year={2026}
}
```

## License
This project is released under the MIT License.
