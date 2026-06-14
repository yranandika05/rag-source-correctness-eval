import argparse
from pathlib import Path

from tqdm import tqdm

from evaluate import (
    compute_ambiguous_source_report,
    compute_source_metrics,
    load_evaluation_questions,
    save_results,
)
from indexing_pipeline import create_document_store, index_documents
from load_documents import load_local_documents
from retrievers import (
    bm25_retrieve,
    build_bm25_pipeline,
    build_dense_pipeline,
    dense_retrieve,
    hybrid_retrieve,
    metadata_aware_retrieve,
)


DATA_DIRS = ["data/github_docs", "data/gitlab_docs"]
EVALUATION_PATH = "evaluation_questions.csv"
RESULTS_PATH = "results/retrieval_results.csv"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
SPLIT_LENGTH = 250
SPLIT_OVERLAP = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG retrieval source-correctness evaluation.")
    parser.add_argument("--evaluation-path", default=EVALUATION_PATH)
    parser.add_argument("--results-path", default=RESULTS_PATH)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--split-length", type=int, default=SPLIT_LENGTH)
    parser.add_argument("--split-overlap", type=int, default=SPLIT_OVERLAP)
    parser.add_argument(
        "--max-files-per-source",
        type=int,
        default=None,
        help="Optional cap on loaded files per source for small early experiments.",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.split_length < 1:
        parser.error("--split-length must be at least 1")
    if args.split_overlap < 0:
        parser.error("--split-overlap must be 0 or greater")
    if args.split_overlap >= args.split_length:
        parser.error("--split-overlap must be smaller than --split-length")
    if args.max_files_per_source is not None and args.max_files_per_source < 1:
        parser.error("--max-files-per-source must be at least 1 when provided")
    return args


def build_method_map(bm25_pipeline, dense_pipeline, top_k: int) -> dict:
    return {
        "bm25": lambda query: bm25_retrieve(bm25_pipeline, query=query, top_k=top_k),
        "dense": lambda query: dense_retrieve(dense_pipeline, query=query, top_k=top_k),
        "hybrid": lambda query: hybrid_retrieve(
            bm25_pipeline=bm25_pipeline,
            dense_pipeline=dense_pipeline,
            query=query,
            top_k=top_k,
        ),
        "metadata_aware": lambda query: metadata_aware_retrieve(
            bm25_pipeline=bm25_pipeline,
            dense_pipeline=dense_pipeline,
            query=query,
            top_k=top_k,
        ),
    }


def serialize_hit(question: dict, method: str, rank: int, document) -> dict:
    return {
        "question_id": question["question_id"],
        "question": question["question"],
        "intended_source": question["intended_source"],
        "category": question["category"],
        "method": method,
        "rank": rank,
        "chunk_id": document.id,
        "retrieved_source": document.meta.get("source", ""),
        "score": document.score,
        "text_preview": (document.content or "")[:280].replace("\n", " "),
    }


def main() -> None:
    args = parse_args()

    print("Loading local documentation files...")
    documents = load_local_documents(DATA_DIRS, max_files_per_source=args.max_files_per_source)
    if not documents:
        raise SystemExit("No documents found. Add .md, .mdx, or .txt files under data/github_docs/ and data/gitlab_docs/.")
    print(f"Loaded {len(documents)} source documents.")

    print("Creating Haystack InMemoryDocumentStore and running indexing pipeline...")
    document_store = create_document_store()
    index_documents(
        document_store=document_store,
        documents=documents,
        embedding_model=args.embedding_model,
        split_length=args.split_length,
        split_overlap=args.split_overlap,
    )
    print(f"Indexed {document_store.count_documents()} chunks.")

    print("Loading evaluation questions...")
    questions = load_evaluation_questions(args.evaluation_path)
    if not questions:
        raise SystemExit("No evaluation questions found.")

    print("Building retrieval pipelines...")
    bm25_pipeline = build_bm25_pipeline(document_store)
    dense_pipeline = build_dense_pipeline(document_store, args.embedding_model)
    method_map = build_method_map(bm25_pipeline, dense_pipeline, args.top_k)

    rows = []
    for method, retriever_fn in method_map.items():
        print(f"Running method: {method}")
        for question in tqdm(questions, desc=method):
            hits = retriever_fn(question["question"])
            for rank, document in enumerate(hits, start=1):
                rows.append(serialize_hit(question, method, rank, document))

    save_results(args.results_path, rows)
    print(f"Saved detailed retrieval results to {args.results_path}")

    metrics = compute_source_metrics(args.results_path, top_k=args.top_k)
    metrics_path = Path(args.results_path).with_name("source_metrics.csv")
    metrics.to_csv(metrics_path, index=False)

    print(f"Saved source metrics to {metrics_path}")
    if not metrics.empty:
        print("\nSource evaluation metrics")
        print(metrics.to_string(index=False))
    else:
        print("No non-ambiguous source metrics were computed.")

    ambiguous_report = compute_ambiguous_source_report(args.results_path, top_k=args.top_k)
    ambiguous_report_path = Path(args.results_path).with_name("ambiguous_source_report.csv")
    ambiguous_report.to_csv(ambiguous_report_path, index=False)

    print(f"Saved ambiguous source report to {ambiguous_report_path}")
    if not ambiguous_report.empty:
        print("\nAmbiguous query retrieved-source distribution")
        print(ambiguous_report.to_string(index=False))


if __name__ == "__main__":
    main()
