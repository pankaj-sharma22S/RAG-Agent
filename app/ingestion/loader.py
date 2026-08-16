from pathlib import Path
from langsmith import traceable

@traceable
def load_document(file_path: str):
    try: 
        extension = Path(file_path).suffix.lower()


        if extension == ".pdf":
            from ingestion.pdf_loader import load_pdf
            return load_pdf(file_path)


        elif extension == ".docx":
            from ingestion.docx_loader import load_docx
            return load_docx(file_path)


        elif extension == ".pptx":
            from ingestion.pptx_loader import load_pptx
            return load_pptx(file_path)


        elif extension in [".xlsx", ".xls"]:
            from ingestion.excel_loader import load_excel
            return load_excel(file_path)

        elif extension in [
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
            ".tif",
            ".tiff",
        ]:
            from ingestion.image_loader import load_image
            return load_image(file_path)
        
        elif extension in [
                 ".mp3",
                 ".wav",
                 ".m4a"
                ]:
            from ingestion.audio_loader import load_audio
            return load_audio(file_path)
        elif extension in [
    ".mp4",
    ".mkv",
    ".avi",
    ".mov"
]:
            from ingestion.video_loader import load_video
            return load_video(file_path)

        else:
            raise ValueError(
                f"Unsupported file type: {extension}"
        )


    
    except Exception as e:
        raise RuntimeError(f"Failed loading {file_path}: {e}") from e
