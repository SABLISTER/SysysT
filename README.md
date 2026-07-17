# SysysT

SysysT is a desktop application for building reproducible, LLM-assisted systematic literature reviews. It combines academic-database search, relevance screening, full-text retrieval, structured claim and statistical extraction, evidence synthesis, meta-analysis, and report generation in a PySide6 interface.

The application is designed to keep intermediate artifacts inspectable. Search results, scoring decisions, extracted claims, validation records, plots, and reports are written to local JSON, Markdown, and image files instead of being hidden behind a single generated answer.

> [!IMPORTANT]
> SysysT is research software. LLM output, extracted statistics, citations, and generated reports require human review before they are used in scholarship, clinical work, or policy decisions.

## Highlights

- Search PubMed, Semantic Scholar, OpenAlex, Elicit, Scopus, and Web of Science.
- Use local models through Ollama or LM Studio, or cloud models through OpenAI, Anthropic, and LongCat.
- Resume interrupted relevance-screening and claim-extraction runs from checkpoints.
- Retrieve open-access full text and retain auditable source metadata.
- Extract claims, methods, coordinates, and statistical results into structured artifacts.
- Run random-effects meta-analysis, sensitivity checks, publication-bias diagnostics, GRADE assessment, and PRISMA reporting.
- Compare saved snapshots for living-review updates.
- Inspect the workflow through a Win98-inspired PySide6 desktop interface.

## Pipeline

SysysT exposes 11 main stages plus optional full-text and statistical-enrichment steps:

| Stage | Purpose | Primary artifact |
|---|---|---|
| 1. Search | Query and deduplicate academic sources | `data/corpus/corpus.json` |
| 2. Score | Score and filter papers for relevance | `data/claims/relevant.json` |
| 2b. Full text | Retrieve and index available full text | local full-text store |
| 2c. Statistics | Extract candidate statistical evidence | `data/claims/statistical_extractions.json` |
| 3. Extract | Extract structured claims | `data/claims/claims.json` |
| 4. Validate | Review methods and validate supporting spans | `data/method_review/method_profiles_validated.json` |
| 5. Audit | Apply quality and span checks | `claims_filtered.json` |
| 6. Synthesize | Cluster evidence and draft a synthesis | `clusters.json`, `abstract_draft.md` |
| 7. Full-text metrics | Extract metrics and run quantitative analyses | `data/fulltext/auto_extracted_metrics.json` |
| 8. Interrater | Measure reviewer agreement | `interrater_report.md` |
| 9. Robustness | Run sensitivity and robustness checks | `data/validation/robustness_results.json` |
| 10. Verify | Check the draft against extracted evidence | `verification_report.md` |
| 11. Compare | Compare saved review snapshots | `report.md` |

Artifacts are routed into named run folders when a run name is configured.

## Requirements

- Python 3.11 or newer
- A supported operating system for PySide6
- At least one LLM backend for LLM-assisted stages
- API credentials for any credentialed literature sources you choose to use

Local Ollama and LM Studio workflows can be used without sending prompts to a cloud LLM provider. Literature-provider terms and rate limits still apply.

## Installation

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/SABLISTER/SysysT.git
cd SysysT
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

If you use [uv](https://docs.astral.sh/uv/), the equivalent full development setup is:

```bash
uv sync --extra rag --extra dev
```

## Configuration

The easiest setup path is **File → Settings** in the application. Each LLM-backed stage can also override the global provider and model.

For file-based configuration, copy the key-free example:

```bash
cp pipeline_config.example.json pipeline_config.json
```

`pipeline_config.json` is ignored by Git because it can contain credentials and local paths. Common credentials can instead be provided through environment variables:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export LONGCAT_API_KEY="..."
export S2_API_KEY="..."
export PUBMED_API_KEY="..."
export PUBMED_EMAIL="you@example.org"
export ELICIT_API_KEY="..."
export SCOPUS_API_KEY="..."
export WOS_API_KEY="..."
```

The Settings dialog includes a **Remember Keys** option. Leave it disabled on shared machines; when enabled, keys are stored through the operating system's Qt settings mechanism.

Research questions, search domains, relevance criteria, and synthesis settings live in `research_config.yaml`. Reusable examples are available in `templates/`.

## Run the application

```bash
python main.py
```

Start with the Search tab, choose the literature providers to query, and save the project once you have selected a working directory. Later stages consume the artifacts produced by earlier stages, but existing JSON artifacts can also be loaded to resume or inspect a review.

## Data and privacy

Generated review data is stored under `data/`, `output/`, and `snapshots/`; these paths are ignored by Git. They can contain copyrighted full text, unpublished analyses, search histories, and sensitive research questions. Review them before sharing a project or report bundle.

API keys should be entered through Settings, environment variables, or an ignored `pipeline_config.json`. Never add live keys to `pipeline_config.example.json` or a research template.

## Tests

Run the automated test suite from the repository root:

```bash
python -m pytest -q
```

For a quick syntax check across the source tree:

```bash
python -m compileall -q acquire analysis core gui hard_mode process stages
```

## Repository layout

```text
acquire/     Academic-source clients, full-text retrieval, and deduplication
analysis/    Statistical and scoring helpers
core/        Configuration, models, checkpoints, and snapshots
gui/         PySide6 interface, workers, and pipeline orchestration
hard_mode/   Experimental advanced-review workflow
process/     Extraction, synthesis, meta-analysis, and reporting
stages/      Pipeline stage entry points
templates/   Reusable research configurations
tests/       Automated regression tests
```

## Current limitations

- `hard_mode/` is experimental and may not match the maturity of the main workflow.
- External search coverage depends on provider availability, credentials, and rate limits.
- Full-text retrieval is limited by publisher access and open-access availability.
- Model behavior is provider- and prompt-dependent; deterministic checks reduce but do not remove the need for expert review.

## License

SysysT is licensed under the [Mozilla Public License 2.0](LICENSE). You may
use it commercially and combine it with a larger project. Modifications to
MPL-covered source files must remain available under the MPL when distributed,
and the source's license and copyright notices may not be removed.

If you use SysysT in research, please cite it using the metadata in
[CITATION.cff](CITATION.cff). Citation is requested as a scholarly norm; it
is separate from the MPL's legal requirements.
