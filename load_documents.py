import re
from pathlib import Path

from haystack import Document


TEXT_EXTENSIONS = {".md", ".txt"}
SOURCE_DIRS = {
    "github_docs": "GitHub",
    "gitlab_docs": "GitLab",
}


def infer_source_from_path(path: Path) -> str:
    """Infer the documentation source from a directory or file path."""
    lowered_parts = [part.lower() for part in path.parts]
    if any("github" in part for part in lowered_parts):
        return "GitHub"
    if any("gitlab" in part for part in lowered_parts):
        return "GitLab"
    return "Unknown"


def extract_section_title(text: str) -> str:
    """Use the first Markdown heading as a lightweight section title."""
    heading = re.search(r"^(#{1,6})\s+(.+)$", text, flags=re.MULTILINE)
    return heading.group(2).strip() if heading else ""


def load_local_documents(base_dirs: list[str]) -> list[Document]:
    """Load local Markdown/text files as Haystack Documents.

    Splitting happens later in the indexing pipeline so that chunking remains a
    Haystack concern. Metadata added here is copied onto every chunk.
    """
    documents: list[Document] = []

    for base_dir in base_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            print(f"Skipping missing directory: {base_path}")
            continue

        for file_path in sorted(base_path.rglob("*")):
            if file_path.suffix.lower() not in TEXT_EXTENSIONS:
                continue

            text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue

            documents.append(
                Document(
                    content=text,
                    meta={
                        "source": infer_source_from_path(file_path),
                        "file_path": str(file_path),
                        "section_title": extract_section_title(text),
                    },
                )
            )

    return documents
