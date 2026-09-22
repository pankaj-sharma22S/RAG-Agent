"""Load local files and URLs with LangChain's community document loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Type
from urllib.parse import urlparse

from bs4 import SoupStrainer
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    WebBaseLoader,
    YoutubeLoader,
)
from langchain_core.documents import Document

try:
    from guardrails.pii import global_pii_guard
    from guardrails.prompt_injection import global_injection_guard
except ModuleNotFoundError:
    from app.guardrails.pii import global_pii_guard
    from app.guardrails.prompt_injection import global_injection_guard


LoaderClass = Type[object]


class UniversalLoader:
    """Dispatch local files and URLs to standard LangChain document loaders."""

    LOADER_MAPPING: dict[str, LoaderClass] = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".ppt": UnstructuredPowerPointLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".xls": UnstructuredExcelLoader,
        ".csv": CSVLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".py": TextLoader,
        ".json": TextLoader,
        ".js": TextLoader,
        ".ts": TextLoader,
        ".html": TextLoader,
        ".log": TextLoader,
    }

    _MAX_FILE_SIZE = 50 * 1024 * 1024

    @staticmethod
    def _is_url(source: str) -> bool:
        parsed = urlparse(source)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _is_youtube_url(source: str) -> bool:
        hostname = (urlparse(source).hostname or "").lower()
        return hostname in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}

    @classmethod
    def _loader_for_url(cls, source: str):
        if cls._is_youtube_url(source):
            return YoutubeLoader.from_youtube_url(source, add_video_info=True)
        return WebBaseLoader(
            web_paths=[source],
            bs_kwargs={
                "parse_only": SoupStrainer(["article", "main", "body", "title"]),
            },
        )

    @classmethod
    def _loader_for_file(cls, source: str):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")
        if not path.is_file():
            raise ValueError(f"Source is not a file: {source}")
        if path.stat().st_size > cls._MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds maximum allowed limit of 50 MB: {source}")

        loader_class = cls.LOADER_MAPPING.get(path.suffix.lower())
        if loader_class is None:
            supported = ", ".join(sorted(cls.LOADER_MAPPING))
            raise ValueError(f"Unsupported file type '{path.suffix.lower()}'. Supported extensions: {supported}")
        if loader_class is TextLoader:
            return loader_class(source, encoding="utf-8", autodetect_encoding=True)
        return loader_class(source)

    @classmethod
    def _load_raw(cls, source: str) -> tuple[list[Document], str, str]:
        if cls._is_url(source):
            loader = cls._loader_for_url(source)
            source_type = "youtube" if cls._is_youtube_url(source) else "web"
            return loader.load(), source_type, ""
        path = Path(source)
        return cls._loader_for_file(source).load(), "file", path.suffix.lower()

    @staticmethod
    def _standardize_metadata(
        documents: list[Document], source: str, source_type: str, file_extension: str
    ) -> list[Document]:
        display_name = Path(source).name if source_type == "file" else source
        standardized: list[Document] = []
        for document in documents:
            pii_result = global_pii_guard.scan_and_mask(document.page_content)
            document.page_content = global_injection_guard.wrap_untrusted_content(
                pii_result.sanitized_text,
                source_name=display_name,
                content_type=f"{source_type}_{file_extension.lstrip('.') or 'url'}",
            )
            document.metadata = {
                **document.metadata,
                "source": source,
                "source_id": source,
                "source_type": source_type,
                "file_extension": file_extension,
            }
            standardized.append(document)
        return standardized

    @classmethod
    def load(cls, source: str) -> list[Document]:
        """Load ``source`` and return all documents produced by its loader."""
        if not isinstance(source, str) or not source.strip():
            raise RuntimeError(f"Failed loading {source!r}: source must be a non-empty path or URL")
        source = source.strip()
        try:
            documents, source_type, file_extension = cls._load_raw(source)
            return cls._standardize_metadata(documents, source, source_type, file_extension)
        except Exception as error:
            raise RuntimeError(f"Failed loading {source}: {error}") from error


def load_document(source: str) -> list[Document]:
    """Load a local file or URL into standardized LangChain documents."""
    return UniversalLoader.load(source)
