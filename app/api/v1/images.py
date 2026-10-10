"""Static image serving and vision retrieval endpoints for Multimodal RAG."""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

logger = logging.getLogger("ImagesAPI")
router = APIRouter(prefix="/images", tags=["Multimodal Images"])

IMAGE_SEARCH_DIRS = [
    Path("data/extracted_images"),
    Path("app/data/extracted_images"),
    Path("data/uploads"),
    Path("app/data/uploads"),
    Path("./data"),
]


@router.get("/{filename}", summary="Serve extracted or uploaded multimodal image")
def get_image(filename: str):
    """Serve image files for frontend rendering in chat citations and documents."""
    # Sanitize filename
    clean_name = Path(filename).name
    if not clean_name or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid image filename")

    for directory in IMAGE_SEARCH_DIRS:
        candidate = directory / clean_name
        if candidate.exists() and candidate.is_file():
            media_type, _ = mimetypes.guess_type(str(candidate))
            return FileResponse(
                path=candidate,
                media_type=media_type or "image/png",
                filename=clean_name,
            )

    raise HTTPException(status_code=404, detail=f"Image '{clean_name}' not found")
