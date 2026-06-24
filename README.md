# MycoExtract

MycoExtract is a domain-aware, multi-LLM framework for curating kinetic and degradation evidence for enzymatic mycotoxin detoxification from scientific literature.

This repository contains the extraction workflow code, prompts, configuration files, deterministic quality-tier rules, and validation utilities used for the MycoExtract manuscript. Curated data tables, original/source data, web-resource files, manuscript figure-generation assets, and modelling release materials are not included in this code-only release.

## Manuscript Configuration

The manuscript-reported extraction experiments used the frozen v8 configuration in:

```text
config/extraction_config_v8.yaml
```

The LLM roles in the manuscript workflow are:

| Role | Model identifier in release config | Provider |
|---|---|---|
| Text extractor 1 | `kimi-k2-0905-preview` | Moonshot/Kimi |
| Text extractor 2 | `deepseek-chat` | DeepSeek |
| Text extractor 3 | `MiniMax-M2.7` | MiniMax |
| Table-image branch | `mimo-v2.5` | MiMo |
| Aggregation / review | `mimo-v2.5-pro` | MiMo |

Older configuration files are retained for provenance and earlier development runs, but `config/extraction_config_v8.yaml` is the manuscript-facing configuration.

## Repository Layout

```text
config/                  Frozen extraction configurations
prompts/                 Extraction, review, and sequence-detective prompts
scripts/                 Pipeline entry points and validation utilities
src/                     MycoExtract Python package
tests/                   Unit and regression tests
docs/                    Architecture and run documentation
examples/                Minimal usage example
```

Key code-release files:

```text
config/extraction_config_v8.yaml
prompts/prompts_extract_from_text_v8.txt
prompts/prompts_extract_from_table_v8.txt
scripts/run_all_papers_full_extraction.py
src/
```

## Installation

Python 3.8 or newer is required.

```bash
pip install -r requirements.txt
```

For model calls, create a local `.env` file from `.env.example` and fill in your own API keys. Do not commit `.env`.

```bash
cp .env.example .env
```

## Running Extraction

The full all-paper workflow expects parsed paper folders containing `full.md` and MinerU-style structured content files.

```bash
python scripts/run_all_papers_full_extraction.py \
  --config config/extraction_config_v8.yaml \
  --input-dir data/papers \
  --output-root analysis_outputs/mycoextract_run
```

For a lightweight inventory check that does not call LLM APIs:

```bash
python scripts/run_all_papers_full_extraction.py --inventory-only
```

## Validation

The repository includes validation utilities and regression tests for the extraction pipeline. Manuscript source data, generated figures, web-resource files, and modelling datasets are intentionally excluded from this code-only release.

## Secret Handling

The release contains only placeholder API-key examples and environment-variable names. Real API keys should be supplied through a local `.env` file or the host environment. The `.gitignore` excludes `.env`, parsed PDF corpora, run outputs, logs, caches, and large local artifacts.

## License

This project is released under the MIT License. See `LICENSE`.
