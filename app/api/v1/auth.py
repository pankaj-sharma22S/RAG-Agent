"""Authentication and current user identity endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from core.auth import AuthenticatedUser, get_current_user
from memory.service import global_memory_service
from schemas.api_schemas import CurrentUserResponse, UserProfileResponse

router = APIRouter(prefix="/auth", tags=["Authentication & User Identity"])


@router.get("/me", response_model=CurrentUserResponse, summary="Get current authenticated user identity")
async def get_me(current_user: AuthenticatedUser = Depends(get_current_user)) -> CurrentUserResponse:
    """Return the cryptographically verified user identity extracted from the Supabase JWT."""
    # Retrieve semantic memory profile if available
    profile_data = None
    try:
        profile = global_memory_service.store.get_user_profile(current_user.user_id)
        profile_data = UserProfileResponse(
            user_id=profile.user_id,
            name=profile.name,
            profession=profile.profession,
            technical_stack=profile.technical_stack,
            custom_rules=profile.custom_rules,
            last_updated=profile.last_updated.isoformat(),
        )
    except Exception:
        pass

    return CurrentUserResponse(
        user_id=current_user.user_id,
        email=current_user.email,
        role=current_user.role,
        is_anonymous=current_user.is_anonymous,
        profile=profile_data,
    )
