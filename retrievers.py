from collections.abc import Iterable

from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever


def source_filter(source: str) -> dict:
    """Build a Haystack metadata filter for one documentation source."""
    return {"field": "meta.source", "operator": "==", "value": source}


def detect_explicit_source(query: str) -> str | None:
    lowered = query.lower()
    if "github" in lowered:
        return "GitHub"
    if "gitlab" in lowered:
        return "GitLab"
    return None


def build_bm25_pipeline(document_store) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
    return pipeline


def build_dense_pipeline(document_store, embedding_model: str) -> Pipeline:
    text_embedder = SentenceTransformersTextEmbedder(model=embedding_model)
    text_embedder.warm_up()

    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    return pipeline


def bm25_retrieve(pipeline: Pipeline, query: str, top_k: int = 5, filters: dict | None = None) -> list[Document]:
    result = pipeline.run({"retriever": {"query": query, "top_k": top_k, "filters": filters}})
    return result["retriever"]["documents"]


def dense_retrieve(pipeline: Pipeline, query: str, top_k: int = 5, filters: dict | None = None) -> list[Document]:
    result = pipeline.run(
        {
            "text_embedder": {"text": query},
            "retriever": {"top_k": top_k, "filters": filters},
        }
    )
    return result["retriever"]["documents"]


def _normalize_scores(documents: Iterable[Document]) -> dict[str, float]:
    docs = list(documents)
    if not docs:
        return {}

    scores = [float(doc.score or 0.0) for doc in docs]
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return {doc.id: 1.0 for doc in docs}

    return {
        doc.id: (float(doc.score or 0.0) - min_score) / (max_score - min_score)
        for doc in docs
    }


def hybrid_retrieve(
    bm25_pipeline: Pipeline,
    dense_pipeline: Pipeline,
    query: str,
    top_k: int = 5,
    filters: dict | None = None,
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5,
) -> list[Document]:
    bm25_docs = bm25_retrieve(bm25_pipeline, query=query, top_k=top_k * 2, filters=filters)
    dense_docs = dense_retrieve(dense_pipeline, query=query, top_k=top_k * 2, filters=filters)

    bm25_scores = _normalize_scores(bm25_docs)
    dense_scores = _normalize_scores(dense_docs)

    by_id: dict[str, Document] = {}
    for doc in [*bm25_docs, *dense_docs]:
        by_id[doc.id] = doc

    scored_docs = []
    for doc_id, doc in by_id.items():
        combined_score = (bm25_weight * bm25_scores.get(doc_id, 0.0)) + (
            dense_weight * dense_scores.get(doc_id, 0.0)
        )
        doc.score = combined_score
        doc.meta["bm25_normalized_score"] = bm25_scores.get(doc_id, 0.0)
        doc.meta["dense_normalized_score"] = dense_scores.get(doc_id, 0.0)
        scored_docs.append(doc)

    scored_docs.sort(key=lambda doc: float(doc.score or 0.0), reverse=True)
    return scored_docs[:top_k]


def metadata_aware_retrieve(
    bm25_pipeline: Pipeline,
    dense_pipeline: Pipeline,
    query: str,
    top_k: int = 5,
) -> list[Document]:
    """Run a source-filtered hybrid retrieval upper-bound variant.

    This method filters, not merely prioritizes. If the query explicitly names
    GitHub or GitLab, retrieval is restricted to chunks whose `meta.source`
    matches that source. This is useful as an upper-bound comparison: it shows
    how retrieval behaves when source intent is detected and enforced.

    If no explicit source mention is detected, it falls back to unfiltered
    hybrid retrieval. The returned list contains up to `top_k` documents,
    depending on how many matching chunks exist.
    """
    explicit_source = detect_explicit_source(query)
    filters = source_filter(explicit_source) if explicit_source else None
    return hybrid_retrieve(
        bm25_pipeline=bm25_pipeline,
        dense_pipeline=dense_pipeline,
        query=query,
        top_k=top_k,
        filters=filters,
    )
