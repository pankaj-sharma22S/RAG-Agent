import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.upload import router as upload_router
from api.github import router as github_router

logger = logging.getLogger("AppLifespan")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm embedding model and vector store on startup to eliminate cold-start latency."""
    logger.info("Pre-warming embedding model and vector store on server startup...")
    try:
        from embedding.embedding_model import get_embedding_model
        from vectorstore.croma_store import get_vectorstore
        # Warm up embedding singleton and Chroma collection in memory
        embed_model = get_embedding_model()
        embed_model.embed_query("warmup text for zero cold start latency")
        get_vectorstore()
        logger.info("Embedding model and vector store pre-warmed successfully (Ready for sub-second ingestion).")
    except Exception as err:
        logger.warning(f"Lifespan pre-warming warning: {err}")
    yield


app = FastAPI(
    title="Knowledge AI Backend",
    lifespan=lifespan,
)

# The upload page is commonly served by a separate local frontend during
# development.  Without CORS, browsers block its multipart upload before the
# API receives the selected file.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    upload_router,
    prefix="/api"
)

app.include_router(
    github_router,
    prefix="/api"
)


@app.get("/")
def home():
    return {
        "message": "Backend Running"
    }
