"""Image OCR and local HuggingFace vision embeddings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from core.document import create_document
from langsmith import traceable

try:
    from paddleocr import PaddleOCR
except ImportError:  # OCR is optional for image embedding.
    PaddleOCR = None

logger = logging.getLogger(__name__)

_ocr = None
_image_processor = None
_image_model = None

IMAGE_EMBEDDING_MODEL = "google/siglip-base-patch16-224"
IMAGE_EMBEDDING_DIMENSION = 768


def get_ocr():
    """Load PaddleOCR only for an image-ingestion request."""
    global _ocr
    if _ocr is None:
        if PaddleOCR is None:
            raise RuntimeError("PaddleOCR is not installed")
        # PaddleOCR enables oneDNN (MKL-DNN) by default.  On Windows its
        # oneDNN executor can fail on some OCR models with a PIR attribute
        # conversion error, preventing otherwise valid photos from uploading.
        _ocr = PaddleOCR(lang="en", device="cpu", enable_mkldnn=False)
    return _ocr


def _get_image_embedding_model():
    """Load the processor and model only when an image embedding is requested."""
    global _image_processor, _image_model

    if _image_processor is None or _image_model is None:
        from transformers import AutoModel, AutoProcessor

        _image_processor = AutoProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL)
        _image_model = AutoModel.from_pretrained(IMAGE_EMBEDDING_MODEL)
        _image_model.eval()

    return _image_processor, _image_model


def embed_image(image_path: str) -> List[float]:
    """Return a normalized 768-dimensional SigLIP embedding for an image.

    Embedding is deliberately optional: missing ML dependencies, model download
    failures, invalid images, and inference errors return an empty vector so
    image ingestion can continue through its OCR path.
    """
    try:
        import torch
        import torch.nn.functional as F
        from PIL import Image

        path = Path(image_path)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("supported image formats are .png, .jpg, .jpeg, and .webp")
        if not path.is_file():
            raise FileNotFoundError(f"image file not found: {image_path}")

        processor, model = _get_image_embedding_model()
        with Image.open(path) as image:
            inputs = processor(images=image.convert("RGB"), return_tensors="pt")

        with torch.no_grad():
            features = model.get_image_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            if not isinstance(features, torch.Tensor):
                features = features[0]
            features = F.normalize(features, p=2, dim=-1)

        embedding = features[0].detach().cpu().tolist()
        if len(embedding) != IMAGE_EMBEDDING_DIMENSION:
            raise ValueError(
                f"model returned {len(embedding)} dimensions; expected {IMAGE_EMBEDDING_DIMENSION}"
            )
        return [float(value) for value in embedding]
    except Exception as error:
        logger.warning("Image embedding unavailable for %s: %s", image_path, error)
        return []

@traceable(
    metadata={
        "module": "loader",
        "tool":"youtube_loader"
    }
)
def load_image(file_path: str):

    try:

        ocr = get_ocr()
        # PaddleOCR 3.x uses ``predict`` while 2.x uses ``ocr``.
        result = ocr.predict(file_path) if hasattr(ocr, "predict") else ocr.ocr(file_path, cls=True)

        text = ""


        for page in result:
            if hasattr(page, "get"):
                text += "\n".join(page.get("rec_texts", [])) + "\n"
            elif page:
                text += "\n".join(line[1][0] for line in page if len(line) > 1) + "\n"


        if not text.strip():

            text = f"Image: {Path(file_path).name}. No readable text was detected in this image."


        
        return create_document(
            
                    text,
            
                    {
                        "source": file_path,
                        "type": "image"
                    }
            
                )


    except Exception as e:

        raise ValueError(
            f"Image OCR failed: {e}"
        )
