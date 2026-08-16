from docx import Document
from core.document import create_document
from langsmith import traceable
@traceable(
    metadata={
        "module": "loader",
        "tool":"docs"
    }
)
def load_docx(file_path: str):

    doc = Document(file_path)

    text = ""

    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"


  
    return create_document(
            
                    text,
            
                    {
                        "source": file_path,
                        "type": "docx"
                    }
            
                )
    