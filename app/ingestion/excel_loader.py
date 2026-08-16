import pandas as pd
from core.document import create_document
from langsmith import traceable
@traceable(
    metadata={
        "module": "loader",
        "tool":"excel"
    }
)
def load_excel(file_path: str):

    excel = pd.ExcelFile(file_path)

    text = ""


    for sheet in excel.sheet_names:

        df = pd.read_excel(
            file_path,
            sheet_name=sheet
        )

        text += f"\nSheet: {sheet}\n"

        text += df.to_string()


   
    return create_document(
        
                text,
        
                {
                    "source": file_path,
                    "type": "excel"
                }
        
            )