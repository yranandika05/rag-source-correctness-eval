# RAG Source Correctness Evaluation

Experimental retrieval evaluation code for the paper:

**Evaluating Source Correctness in Retrieval-Augmented Generation over Similar Technical Documentation Sources**

This repository investigates a narrow but important RAG failure mode: retrieving from the wrong source when two documentation collections are semantically and stylistically similar. The current experiment compares GitHub Docs and GitLab Docs.

## Overview

This is not a production chatbot. The main goal is to evaluate whether retrieval returns chunks from the intended documentation source before any answer generation step is added.

The prototype uses Haystack 2.30 for:

- `Document` objects
- document metadata
- indexing pipelines
- chunking
- in-memory document storage
- BM25 and dense retrieval pipelines

The evaluated documentation collections are:

- GitHub Docs in `data/github_docs/`
- GitLab Docs in `data/gitlab_docs/`

Generation is intentionally optional because generation can hide retrieval errors. The retrieval output is saved as CSV so each retrieved chunk can be inspected directly.

## Research Question

When a RAG system searches across two similar technical documentation sources, how often does it retrieve chunks from the intended source?

This project focuses on source correctness rather than answer fluency. For example, if a question asks about GitHub but the retriever returns a GitLab chunk, that is counted as a source-level retrieval error even if the text sounds plausible.

## Project Structure

- `load_documents.py` loads local Markdown/text files as Haystack `Document` objects.
- `indexing_pipeline.py` splits, embeds, and stores documents using a Haystack indexing pipeline.
- `retrievers.py` contains BM25, dense, hybrid, and metadata-aware retrieval functions.
- `evaluate.py` reads questions, saves detailed retrieval results, and computes source metrics.
- `run_experiment.py` orchestrates the full experiment.
- `evaluation_questions.csv` contains the evaluation set.
- `results/` is created when you run the experiment and is ignored by Git.

## Setup

Use a clean virtual environment. If your environment already has `farm-haystack` installed, remove it first because that is Haystack 1.x and uses different imports.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip uninstall -y farm-haystack haystack-ai
pip install --upgrade -r requirements.txt
```

The main dependency is pinned to:

```text
haystack-ai==2.30.0
pydantic>=2.7,<3
```

As of June 8, 2026, PyPI lists `haystack-ai` 2.30.0 as a stable release published on June 3, 2026.

## Data

Add `.md` or `.txt` files under:

```text
data/github_docs/
data/gitlab_docs/
```

Each file becomes a Haystack `Document` with metadata:

- `source`: `GitHub` or `GitLab`
- `file_path`: local file path
- `section_title`: first Markdown heading, when available

The indexing pipeline then splits those documents into chunks. Haystack copies the metadata onto the chunks and adds its own split metadata such as `source_id` and `split_id`.

## Evaluation Questions

`evaluation_questions.csv` must contain:

```csv
question_id,question,intended_source,category
```

Example:

```csv
1,How do you create a new repository in GitHub?,GitHub,repository management
```

## Run

```bash
python run_experiment.py
```

Optional settings:

```bash
python run_experiment.py \
  --top-k 5 \
  --split-length 250 \
  --split-overlap 50 \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

The script writes:

- `results/retrieval_results.csv`
- `results/source_metrics.csv`

## Retrieval Methods

- `bm25`: keyword retrieval using `InMemoryBM25Retriever`.
- `dense`: embedding retrieval using `SentenceTransformersTextEmbedder` and `InMemoryEmbeddingRetriever`.
- `hybrid`: retrieves with both BM25 and dense retrieval, normalizes scores per method, then combines them.
- `metadata_aware`: if a query explicitly contains `GitHub` or `GitLab`, it applies a source metadata filter before running hybrid retrieval.

## Metrics

`Source Accuracy@k` answers:

> For each question, did at least one of the top-k retrieved chunks come from the intended source?

`Wrong Source Rate@k` answers:

> Across all top-k retrieved chunks, what fraction came from the wrong source?

These are intentionally source-level metrics. They do not yet evaluate whether the exact paragraph is correct, only whether retrieval points to the intended documentation source.

## Why This Design

The experiment keeps the pipeline small and inspectable:

- local Markdown/text documents only
- no UI
- no agents
- no authentication
- no deployment layer
- no required LLM generation

By saving every retrieved chunk with source, rank, score, and text preview, you can inspect whether failures come from lexical ambiguity, semantic similarity between GitHub/GitLab docs, chunking choices, or source metadata handling.

## Suggested Repository Name

Recommended GitHub repository name:

```text
rag-source-correctness-eval
```

Other reasonable options:

- `source-correctness-rag`
- `technical-docs-rag-evaluation`
- `github-gitlab-rag-source-eval`

## Status

This is an experimental university NLP project and a research prototype for portfolio use. The code prioritizes clarity and reproducibility over production features.
