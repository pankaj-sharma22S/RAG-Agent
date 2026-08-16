"""Unified multimodal document loader with pre-indexing content security and untrusted boundary protection."""

from __future__ import annotations

from pathlib import Path
from core.document import create_document

try:
    from guardrails.pii import global_pii_guard
    from guardrails.prompt_injection import global_injection_guard
except ModuleNotFoundError:
    from app.guardrails.pii import global_pii_guard
    from app.guardrails.prompt_injection import global_injection_guard


def load_document(file_path: str):
    """Load supported file types, scan/redact sensitive credentials, and encapsulate in untrusted boundary envelopes."""
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check size limit (max 50 MB)
    if path_obj.stat().st_size > 50 * 1024 * 1024:
        raise ValueError(f"File size exceeds maximum allowed limit of 50 MB: {file_path}")

    extension = path_obj.suffix.lower()

    try:
        if extension == ".pdf":
            from ingestion.pdf_loader import load_pdf
            doc = load_pdf(file_path)

        elif extension == ".docx":
            from ingestion.docx_loader import load_docx
            doc = load_docx(file_path)

        elif extension == ".pptx":
            from ingestion.pptx_loader import load_pptx
            doc = load_pptx(file_path)

        elif extension in [".xlsx", ".xls"]:
            from ingestion.excel_loader import load_excel
            doc = load_excel(file_path)

        elif extension in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]:
            from ingestion.image_loader import load_image
            doc = load_image(file_path)

        elif extension in [".mp3", ".wav", ".m4a"]:
            from ingestion.audio_loader import load_audio
            doc = load_audio(file_path)

        elif extension in [".mp4", ".mkv", ".avi", ".mov"]:
            from ingestion.video_loader import load_video
            doc = load_video(file_path)

        elif extension in [".txt", ".csv", ".md", ".json", ".py", ".js", ".ts", ".html", ".log"]:
            # Plain text / code / data loader
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            doc = create_document(
                content,
                {"source": file_path, "type": extension.lstrip("."), "filename": path_obj.name},
            )

        else:
            raise ValueError(f"Unsupported file type: '{extension}'. Supported types: PDF, DOCX, PPTX, XLSX, TXT, CSV, MD, JSON, Images, Audio, Video.")

        # 1. Content Security Scan: Redact passwords, API keys, private keys, connection strings, PII
        raw_text = doc.page_content
        pii_res = global_pii_guard.scan_and_mask(raw_text)
        sanitized_raw_text = pii_res.sanitized_text

        # 2. Encapsulate all loaded document content in untrusted security boundary envelope
        wrapped_text = global_injection_guard.wrap_untrusted_content(
            sanitized_raw_text, source_name=path_obj.name, content_type=f"file_{extension.lstrip('.')}"
        )
        doc.page_content = wrapped_text
        return doc

    except Exception as e:
        raise RuntimeError(f"Failed loading {file_path}: {e}") from e
