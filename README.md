# RAG Source Correctness Evaluation

This repository contains the retrieval experiment used to study source
correctness across GitHub Docs and GitLab Docs. The main question is
whether retrieval returns chunks from the intended documentation source
when both corpora are indexed together.

The experiment focuses only on retrieval. It does not generate RAG
answers or evaluate answer correctness.

## Research Question

How do different retrieval strategies affect source correctness in RAG-based question answering over similar technical documentation sources?

The main failure mode of interest is source confusion. For example, if a question is intended for GitHub Docs but the top retrieved chunks come from GitLab Docs, this is treated as a source-level retrieval error even if the retrieved text is topically related.

## Experimental Setup

## Experimental Setup

The code expects local documentation files under:

```text
data/github_docs/content/
data/gitlab_docs/doc/

`load_documents.py` loads `.md`, `.mdx`, and `.txt` files from those directories as Haystack `Document` objects. Each loaded document receives metadata for `source`, `file_path`, and `section_title`. The source is inferred from the file path as `GitHub`, `GitLab`, or `Unknown`.

The indexing pipeline uses Haystack's `DocumentSplitter` with word-based chunking. The default settings in `run_experiment.py` are:

```text
split_by: word
split_length: 250
split_overlap: 50
embedding_model: sentence-transformers/all-MiniLM-L6-v2
top_k: 5
```

The in-memory document store is created with cosine embedding similarity. In the recorded `baseline_50q_analysis` run, the indexed corpus contains 42,662 chunks:

```text
GitHub: 13,813 chunks
GitLab: 28,849 chunks
```

The raw documentation directories and generated index cache are ignored by Git.

## Retrieval Methods

The experiment compares four retrieval methods defined in `retrievers.py`.

`bm25` is the lexical baseline. It uses Haystack's `InMemoryBM25Retriever` without custom BM25 scoring changes.

`dense` embeds the query with `SentenceTransformersTextEmbedder` and retrieves from pre-embedded document chunks using Haystack's `InMemoryEmbeddingRetriever`. The default model is `sentence-transformers/all-MiniLM-L6-v2`, and the document store uses cosine similarity.

`hybrid` retrieves candidates from both BM25 and dense retrieval. It requests twice the final retrieval depth from each retriever, min-max normalizes BM25 and dense scores separately, combines the normalized scores with equal weights, and reranks the combined candidate set by the resulting score. This is not Reciprocal Rank Fusion.

`metadata_aware` is a source-constrained hybrid variant. It detects literal occurrences of `GitHub` or `GitLab` in the query. If either platform name is present, it applies a hard filter on `meta.source` before running hybrid retrieval. If neither name is present, it falls back to ordinary unfiltered hybrid retrieval.


## Evaluation

The evaluation set is stored in `evaluation_questions.csv` and contains
50 manually constructed questions:

- 15 explicit-source queries
- 15 terminology-oriented queries
- 10 paraphrased queries
- 10 ambiguous cross-source queries

Forty questions have an intended source:

- GitHub: 19
- GitLab: 21

The remaining ten questions have no single intended source and are marked
as `Ambiguous`.

`Source Accuracy@k` measures whether at least one chunk from the intended
source appears within the top-k results.

`Wrong Source Rate@k` measures the fraction of retrieved top-k chunks
that originate from the other documentation source.

Ambiguous questions are excluded from these two metrics and are analyzed
separately through retrieved-source distributions.

These are source-level retrieval metrics. They do not measure the
semantic relevance of individual chunks or the correctness of a
generated answer.

## Repository Structure

```text
load_documents.py          Load local Markdown/text documentation as Haystack Documents.
indexing_pipeline.py       Split, embed, and write documents into an in-memory store.
retrievers.py              BM25, dense, hybrid, and metadata-aware retrieval functions.
evaluate.py                Load questions, save retrieval results, and compute metrics.
reporting.py               Create run folders, save config, and generate question views.
cache_utils.py             Save and load cached indexed Haystack Documents.
run_experiment.py          Orchestrate indexing, retrieval, evaluation, and reporting.
evaluation_questions.csv   The 50-question evaluation set.
requirements.txt           Python dependencies.
scripts/fetch_data.sh      Sparse-clone helper for GitHub Docs and GitLab Docs.
```

Generated directories such as `results/`, `storage/`, and the local documentation directories under `data/` are ignored by Git.

## Setup

Use a clean Python environment. If an old Haystack 1.x package is installed, remove it before installing this project's dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip uninstall -y farm-haystack haystack-ai
pip install --upgrade -r requirements.txt
```

The dependency file currently pins:

```text
haystack-ai==2.30.0
sentence-transformers==5.5.1
pandas==3.0.3
tqdm==4.67.1
pydantic>=2.7,<3
```

## Data

You can place compatible documentation files manually under the expected directories, or use the helper script:

```bash
bash scripts/fetch_data.sh
```

The script creates `data/`, sparse-clones the upstream repositories, and checks out the documentation subdirectories expected by the loader:

```text
https://github.com/github/docs.git       -> data/github_docs/content/
https://gitlab.com/gitlab-org/gitlab.git -> data/gitlab_docs/doc/
```

The repository does not pin upstream documentation commit hashes, so exact corpus-version reproducibility depends on the local data snapshot used for a run.

## Running the Experiment

Run the experiment with:

```bash
python run_experiment.py
```

Without a run name, outputs are written to incrementing directories such as 
```bash
results/run_001/ and results/run_002/.
```
A named run can be created with:
```bash
python run_experiment.py --run-name baseline_50q
```
To intentionally replace an existing named run:
```bash
python run_experiment.py --run-name baseline_50q --overwrite-run
```
Additional options can be inspected with:
```bash
python run_experiment.py --help
```
Use ```--rebuild-index``` when the underlying documentation files have
changed and the cached index should not be reused.



## Notes and Limitations

This is a research prototype for retrieval evaluation. The document store is in memory, the corpus is loaded from local files, and generated outputs are not committed. The evaluation checks whether retrieval points to the intended documentation source; it does not verify whether the retrieved paragraph is the best supporting evidence for an answer.
