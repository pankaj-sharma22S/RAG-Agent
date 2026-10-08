"""Production FastAPI application entrypoint with structured routing, auth, middleware, and observability."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.chat_routes import router as legacy_chat_router
from api.github import router as legacy_github_router
from api.upload import router as legacy_upload_router
from api.v1.router import api_v1_router
from core.config import settings
from core.error_handlers import register_exception_handlers
from core.middleware import RequestLoggingMiddleware

# Configure root application logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AppLifespan")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm embedding model, vector store, and SQLite databases on startup."""
    logger.info("Starting up %s v%s in %s mode...", settings.APP_NAME, settings.APP_VERSION, settings.ENVIRONMENT)
    try:
        from embedding.embedding_model import get_embedding_model
        from vectorstore.croma_store import get_vectorstore

        # Pre-warm embedding singleton and Chroma collection to eliminate cold start latency
        embed_model = get_embedding_model()
        embed_model.embed_query("warmup query for zero cold start")
        get_vectorstore()
        logger.info("Embedding singleton and vector store pre-warmed successfully.")
    except Exception as err:
        logger.warning("Startup pre-warming non-fatal warning: %s", err)

    yield

    logger.info("Shutting down %s...", settings.APP_NAME)


# Initialize FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise Adaptive Multimodal RAG & Agentic Knowledge Backend.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# 1. Register Global Exception Handlers
register_exception_handlers(app)

# 2. Register Request Correlation & Timing Middleware
app.add_middleware(RequestLoggingMiddleware)

# 3. Register CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# 4. Mount Versioned API (v1)
app.include_router(api_v1_router)

# 5. Mount Legacy /api Routes for Backward Compatibility
app.include_router(legacy_upload_router, prefix="/api")
app.include_router(legacy_github_router, prefix="/api")
app.include_router(legacy_chat_router, prefix="/api")


@app.get("/", summary="Root status check")
def home():
    """Simple status check."""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "status": "online",
        "docs": "/docs",
        "api_v1": settings.API_V1_PREFIX,
    }


@app.get("/health", summary="Root health redirect")
def root_health():
    """Root health endpoint redirecting to /api/v1/health."""
    from api.v1.health import health_check
    return health_check()
