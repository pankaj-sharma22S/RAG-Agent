"""Multimodal PDF Document Loader extracting text, page numbers, and embedded images."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List

import fitz  # PyMuPDF
from core.document import create_document
from langsmith import traceable

try:
    from ingestion.image_loader import embed_image
except ImportError:
    try:
        from app.ingestion.image_loader import embed_image
    except ImportError:
        embed_image = lambda path: []

logger = logging.getLogger("PDFLoader")

EXTRACTED_IMAGES_DIR = Path("data/extracted_images")
EXTRACTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


@traceable(
    metadata={
        "module": "loader",
        "tool": "pymupdf_multimodal",
    }
)
def load_pdf(file_path: str) -> List[Any]:
    """Extract text per page and extract/link embedded images with page-level metadata."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    doc = fitz.open(file_path)
    total_pages = len(doc)
    file_stem = path.stem
    documents: List[Any] = []

    EXTRACTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for page_idx in range(total_pages):
        page = doc[page_idx]
        page_num = page_idx + 1
        page_text = page.get_text("text").strip()

        # Fallback to blocks if text string was empty
        if not page_text:
            blocks = page.get_text("blocks")
            block_texts = [b[4].strip() for b in blocks if len(b) > 4 and b[4].strip()]
            if block_texts:
                page_text = "\n\n".join(block_texts).strip()

        # Extract embedded images on this page
        extracted_page_images: list[dict[str, Any]] = []
        try:
            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image")
                image_ext = base_image.get("ext", "png")
                
                if image_bytes:
                    img_filename = f"{file_stem}_p{page_num}_img{img_idx + 1}.{image_ext}"
                    img_path = EXTRACTED_IMAGES_DIR / img_filename
                    
                    if not img_path.exists():
                        with open(img_path, "wb") as f:
                            f.write(image_bytes)

                    image_url = f"/api/v1/images/{img_filename}"
                    extracted_page_images.append({
                        "image_path": str(img_path),
                        "image_url": image_url,
                        "image_filename": img_filename,
                        "page": page_num,
                        "source": file_path,
                        "xref": xref,
                    })
        except Exception as img_err:
            logger.warning("Could not extract images from PDF %s page %d: %s", file_path, page_num, img_err)

        # If no raster images found, render page pixmap so diagrams/slides are extracted as visual assets
        if not extracted_page_images:
            try:
                pix = page.get_pixmap(dpi=150)
                img_filename = f"{file_stem}_p{page_num}_visual.png"
                img_path = EXTRACTED_IMAGES_DIR / img_filename
                if not img_path.exists():
                    pix.save(str(img_path))

                image_url = f"/api/v1/images/{img_filename}"
                extracted_page_images.append({
                    "image_path": str(img_path),
                    "image_url": image_url,
                    "image_filename": img_filename,
                    "page": page_num,
                    "source": file_path,
                })
            except Exception as pix_err:
                logger.debug("Page pixmap rendering non-fatal warning: %s", pix_err)

        has_images = len(extracted_page_images) > 0
        image_urls = [img["image_url"] for img in extracted_page_images]
        image_paths = [img["image_path"] for img in extracted_page_images]
        primary_image_url = image_urls[0] if image_urls else ""

        # Format page content - never drop a page silently
        if page_text:
            content = page_text
        elif has_images:
            content = f"[Page {page_num} of {path.name}: Contains {len(extracted_page_images)} figures/diagrams]"
        else:
            content = f"[Page {page_num} of {path.name}: Document section without plain text stream]"

        doc_meta = {
            "source": file_path,
            "source_id": file_path,
            "filename": path.name,
            "source_type": "pdf",
            "type": "pdf",
            "page": page_num,
            "total_pages": total_pages,
            "has_images": has_images,
            "image_url": primary_image_url,
            "image_count": len(extracted_page_images),
        }

        if image_urls:
            doc_meta["image_urls"] = image_urls
        if image_paths:
            doc_meta["image_paths"] = image_paths

        documents.append(create_document(content, doc_meta))

    doc.close()

    if not documents:
        fallback_text = f"Scanned or empty PDF: {path.name}."
        return [create_document(fallback_text, {"source": file_path, "type": "pdf", "page": 1, "total_pages": 1})]

    return documents
