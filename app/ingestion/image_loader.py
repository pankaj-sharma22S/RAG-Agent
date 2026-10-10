"""Lightweight image text extractor and OCR wrapper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from core.document import create_document
from langsmith import traceable

logger = logging.getLogger(__name__)

_siglip_model = None
_siglip_processor = None
_siglip_failed = False


def get_ocr():
    """Load OCR tool lazily if installed."""
    global _ocr
    if _ocr is None:
        try:
            from paddleocr import PaddleOCR
            _ocr = PaddleOCR(lang="en", device="cpu", enable_mkldnn=False)
        except Exception:
            _ocr = None
    return _ocr


def _get_siglip_vision():
    """Lazily load SigLIP vision model & image processor on CPU with isolated error handling."""
    global _siglip_model, _siglip_processor, _siglip_failed
    if _siglip_failed:
        return None, None
    if _siglip_model is None or _siglip_processor is None:
        try:
            import torch
            from transformers import AutoModel, AutoImageProcessor
            model_name = "google/siglip-base-patch16-224"
            _siglip_processor = AutoImageProcessor.from_pretrained(model_name)
            _siglip_model = AutoModel.from_pretrained(model_name).to("cpu")
            _siglip_model.eval()
        except Exception as e:
            logger.warning("SigLIP vision model initialization failed (vision embeddings fallback): %s", e)
            _siglip_failed = True
            return None, None
    return _siglip_model, _siglip_processor



def embed_image(image_path: str) -> List[float]:
    """Generate normalized image embedding using SigLIP on CPU (768 dimensions)."""
    model, processor = _get_siglip_vision()
    if model is None or processor is None:
        return []
    try:
        import torch
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                tensor = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                tensor = outputs.last_hidden_state[:, 0]
            elif isinstance(outputs, torch.Tensor):
                tensor = outputs
            elif isinstance(outputs, (tuple, list)):
                tensor = outputs[0]
            else:
                tensor = outputs

            normalized = tensor / tensor.norm(p=2, dim=-1, keepdim=True)
            return normalized.squeeze(0).cpu().tolist()
    except Exception as e:
        logger.warning("Failed to compute image embedding for %s: %s", image_path, e)
        return []


@traceable(
    metadata={
        "module": "loader",
        "tool": "image_loader",
    }
)
def load_image(file_path: str):
    """Load readable text from an image file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    ocr = get_ocr()
    if ocr is not None:
        try:
            result = ocr.predict(file_path) if hasattr(ocr, "predict") else ocr.ocr(file_path, cls=True)
            text_lines = []
            for page in result or []:
                if hasattr(page, "get"):
                    text_lines.extend(page.get("rec_texts", []))
                elif page:
                    text_lines.extend(line[1][0] for line in page if len(line) > 1)

            extracted = "\n".join(text_lines).strip()
            if extracted:
                return create_document(extracted, {"source": file_path, "type": "image"})
        except Exception as e:
            logger.warning("OCR processing error: %s", e)

    fallback_text = f"Image: {path.name}. Content indexed for metadata search."
    return create_document(fallback_text, {"source": file_path, "type": "image"})
