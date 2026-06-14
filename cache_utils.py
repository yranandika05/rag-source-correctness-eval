import hashlib
import pickle
from pathlib import Path
from typing import Any

from haystack import Document


CACHE_DIR = Path("storage")


def build_cache_key(
    embedding_model: str,
    split_length: int,
    split_overlap: int,
    max_files_per_source: int | None = None,
) -> str:
    """Build a stable cache key from settings that affect indexed chunks."""
    key_parts: dict[str, Any] = {
        "embedding_model": embedding_model,
        "split_length": split_length,
        "split_overlap": split_overlap,
        "max_files_per_source": max_files_per_source,
    }
    raw_key = repr(sorted(key_parts.items())).encode("utf-8")
    return hashlib.sha256(raw_key).hexdigest()[:16]


def cache_path_for_key(cache_key: str, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"index_cache_{cache_key}.pkl"


def cache_exists(cache_path: str | Path) -> bool:
    return Path(cache_path).exists()


def save_cached_documents(cache_path: str | Path, documents: list[Document]) -> None:
    """Persist chunked Haystack Documents, including metadata and embeddings."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(documents, file)


def load_cached_documents(cache_path: str | Path) -> list[Document]:
    """Load cached chunked Haystack Documents with embeddings."""
    with Path(cache_path).open("rb") as file:
        documents = pickle.load(file)

    if not isinstance(documents, list):
        raise ValueError(f"Cache file does not contain a document list: {cache_path}")
    return documents
