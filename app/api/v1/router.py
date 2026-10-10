"""Aggregation of all API v1 endpoints under a unified router."""

from __future__ import annotations

from fastapi import APIRouter

from .auth import router as auth_router
from .chat import router as chat_router
from .github import router as github_router
from .health import router as health_router
from .images import router as images_router
from .threads import router as threads_router
from .upload import router as upload_router

api_v1_router = APIRouter(prefix="/api/v1")

# Mount modular sub-routers
api_v1_router.include_router(health_router)
api_v1_router.include_router(auth_router)
api_v1_router.include_router(chat_router)
api_v1_router.include_router(upload_router)
api_v1_router.include_router(github_router)
api_v1_router.include_router(threads_router)
api_v1_router.include_router(images_router)

