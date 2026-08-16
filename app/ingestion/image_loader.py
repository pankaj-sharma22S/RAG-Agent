from paddleocr import PaddleOCR
from pathlib import Path
from core.document import create_document
from langsmith import traceable

_ocr = None


def get_ocr():
    """Load PaddleOCR only for an image-ingestion request."""
    global _ocr
    if _ocr is None:
        # PaddleOCR enables oneDNN (MKL-DNN) by default.  On Windows its
        # oneDNN executor can fail on some OCR models with a PIR attribute
        # conversion error, preventing otherwise valid photos from uploading.
        _ocr = PaddleOCR(lang="en", device="cpu", enable_mkldnn=False)
    return _ocr

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
