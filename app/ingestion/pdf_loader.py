import fitz

from core.document import create_document
from langsmith import traceable

@traceable(
    metadata={
        "module": "loader",
        "tool":"pymupdf"
    }
)
def load_pdf(file_path: str):
    """
    Extract text from PDF.
    """

    document = fitz.open(file_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    if not text.strip():
        raise ValueError(
            "No selectable text found in PDF. Scanned PDFs require OCR support."
        )

   
    return create_document(

        text,

        {
            "source": file_path,
            "type": "pdf"
        }

    )
