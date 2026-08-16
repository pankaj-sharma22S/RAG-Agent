from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field, HttpUrl
import os
from pathlib import Path
from ingest import ingest_file, ingest_url
router = APIRouter()


class URLRequest(BaseModel):
    url: HttpUrl
    thread_id: str = Field(default="api_user", min_length=1, max_length=100)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    thread_id: str = Field(default="api_user", min_length=1, max_length=100)


UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


@router.post(
    "/upload",
    summary="Upload and index a document or photo",
    description=(
        "Supported files: PDF, DOCX, PPTX, XLS/XLSX; photos in PNG, JPG, "
        "JPEG, WEBP, BMP, or TIFF; and MP3, WAV, M4A, MP4, MKV, AVI, MOV."
    ),
)
async def upload_file(
    file: UploadFile = File(...),
    thread_id: str = "api_user",
):
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Please choose a file before uploading.")

    file_path = UPLOAD_DIR / filename
    try:
        with open(file_path, "wb") as buffer:
            total_bytes = 0
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    buffer.close()
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail="File is too large. The maximum upload size is 25 MB.",
                    )
                buffer.write(chunk)
        ingest_file(str(file_path), session_id=thread_id)
    except HTTPException:
        raise
    except Exception as error:
        # Loader failures used to escape as opaque 500 responses.
        raise HTTPException(status_code=400, detail=f"Upload ingestion failed: {error}") from error
    finally:
        await file.close()

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "location": str(file_path),
        "status": "uploaded"
    }


@router.post("/url")
async def ingest_public_url(request: URLRequest):
    """Index a YouTube transcript or readable text from a public web link."""
    try:
        chunks = ingest_url(str(request.url), session_id=request.thread_id)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Link ingestion failed: {error}") from error

    return {
        "url": str(request.url),
        "chunks_indexed": len(chunks),
        "status": "indexed",
    }


@router.post("/query")
async def query_documents(request: QueryRequest):
    """Answer a question from documents stored in the persistent Chroma index."""
    try:
        # Import lazily so starting the upload API does not initialize the LLM.
        from rag_workflow import graph_app
        from langchain_core.messages import HumanMessage

        result = graph_app.invoke(
            {
                "question": request.question,
                "messages": [HumanMessage(content=request.question)],
                "force_document_retrieval": True,
                "session_id": request.thread_id,
            },
            config={"configurable": {"thread_id": request.thread_id}},
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Document query failed: {error}") from error

    documents = result.get("documents", [])
    return {
        "answer": str(result["messages"][-1].content),
        "retrieved_chunks": len(documents),
        "route": result.get("route"),
        "sources": [document.metadata for document in documents],
    }
