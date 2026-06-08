from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore


def create_document_store() -> InMemoryDocumentStore:
    """Create a simple in-memory store suitable for small experiments."""
    return InMemoryDocumentStore(embedding_similarity_function="cosine")


def build_indexing_pipeline(
    document_store: InMemoryDocumentStore,
    embedding_model: str,
    split_length: int = 250,
    split_overlap: int = 50,
) -> Pipeline:
    """Build the Haystack pipeline used to split, embed, and store documents."""
    document_embedder = SentenceTransformersDocumentEmbedder(model=embedding_model)
    document_embedder.warm_up()

    pipeline = Pipeline()
    pipeline.add_component(
        "splitter",
        DocumentSplitter(split_by="word", split_length=split_length, split_overlap=split_overlap),
    )
    pipeline.add_component("document_embedder", document_embedder)
    pipeline.add_component("writer", DocumentWriter(document_store=document_store))

    pipeline.connect("splitter.documents", "document_embedder.documents")
    pipeline.connect("document_embedder.documents", "writer.documents")
    return pipeline


def index_documents(
    document_store: InMemoryDocumentStore,
    documents: list,
    embedding_model: str,
    split_length: int = 250,
    split_overlap: int = 50,
) -> None:
    """Run the indexing pipeline over loaded Haystack Documents."""
    pipeline = build_indexing_pipeline(
        document_store=document_store,
        embedding_model=embedding_model,
        split_length=split_length,
        split_overlap=split_overlap,
    )
    pipeline.run({"splitter": {"documents": documents}})
