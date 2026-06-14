import re
from pathlib import Path

from haystack import Document


TEXT_EXTENSIONS = {".md", ".mdx", ".txt"}
EXCLUDED_DIRS = {".git", "node_modules", ".github", "assets", "images", "public", "static"}


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


def iter_text_files(base_path: Path) -> list[Path]:
    """Return supported text files while skipping irrelevant documentation assets."""
    files: list[Path] = []
    pending = [base_path]

    while pending:
        current = pending.pop()
        for child in sorted(current.iterdir()):
            if child.is_dir():
                if child.name in EXCLUDED_DIRS:
                    continue
                pending.append(child)
            elif child.suffix.lower() in TEXT_EXTENSIONS:
                files.append(child)

    return sorted(files)


def load_local_documents(base_dirs: list[str], max_files_per_source: int | None = None) -> list[Document]:
    """Load local Markdown/text files as Haystack Documents.

    Splitting happens later in the indexing pipeline so that chunking remains a
    Haystack concern. Metadata added here is copied onto every chunk.
    """
    documents: list[Document] = []
    files_seen_by_source: dict[str, int] = {}

    for base_dir in base_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            print(f"Skipping missing directory: {base_path}")
            continue

        source = infer_source_from_path(base_path)
        files_seen_by_source.setdefault(source, 0)

        for file_path in iter_text_files(base_path):
            if max_files_per_source is not None and files_seen_by_source[source] >= max_files_per_source:
                break

            text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue

            files_seen_by_source[source] += 1
            documents.append(
                Document(
                    content=text,
                    meta={
                        "source": source,
                        "file_path": str(file_path),
                        "section_title": extract_section_title(text),
                    },
                )
            )

    return documents
