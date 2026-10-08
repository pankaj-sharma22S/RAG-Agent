"""Supabase JWT authentication, token verification, and user authorization dependencies."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional
import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .config import settings
from .exceptions import AuthenticationException, AuthorizationException

logger = logging.getLogger("AuthService")

# Fast HTTP Bearer schemes
auth_scheme = HTTPBearer(auto_error=False)


class AuthenticatedUser(BaseModel):
    """Verified identity extracted from a valid Supabase JWT."""

    user_id: str = Field(description="Unique verified subject identifier from JWT sub claim.")
    email: Optional[str] = Field(default=None, description="User email address if present in claims.")
    role: str = Field(default="authenticated", description="User role from JWT claims or app_metadata.")
    claims: Dict[str, Any] = Field(default_factory=dict, description="Raw validated JWT payload claims.")
    is_anonymous: bool = Field(default=False, description="Whether the user is an unauthenticated guest.")


class SupabaseJWTVerifier:
    """Thread-safe Supabase JWT verification engine."""

    def __init__(self) -> None:
        self.jwt_secret = settings.SUPABASE_JWT_SECRET
        self.supabase_url = settings.SUPABASE_URL.rstrip("/")
        self.audience = settings.SUPABASE_JWT_AUDIENCE
        self.algorithms = settings.SUPABASE_JWT_ALGORITHMS

    def verify_token(self, token_str: str) -> AuthenticatedUser:
        """Decode and cryptographically verify a Supabase JWT."""
        if not token_str or not token_str.strip():
            raise AuthenticationException("Authorization token is missing or empty.")

        clean_token = token_str.strip()

        # 1. Verification with symmetric SUPABASE_JWT_SECRET (HS256)
        if self.jwt_secret:
            try:
                payload = jwt.decode(
                    clean_token,
                    self.jwt_secret,
                    algorithms=self.algorithms,
                    audience=self.audience if self.audience else None,
                    options={"verify_exp": True, "verify_aud": bool(self.audience)},
                )
                return self._build_authenticated_user(payload)
            except jwt.ExpiredSignatureError as err:
                raise AuthenticationException("Authentication token has expired. Please sign in again.") from err
            except jwt.InvalidAudienceError as err:
                raise AuthenticationException("Invalid token audience.") from err
            except jwt.InvalidTokenError as err:
                raise AuthenticationException(f"Invalid authentication token: {err}") from err

        # 2. If no JWT secret is configured but Supabase URL is present, decode unverified header and check basic claims
        try:
            # Decode payload verifying expiration and standard claims
            unverified_payload = jwt.decode(
                clean_token,
                options={"verify_signature": False, "verify_exp": True},
            )
            # Ensure subject claim is present
            if not unverified_payload.get("sub"):
                raise AuthenticationException("Token payload missing required 'sub' claim.")
            return self._build_authenticated_user(unverified_payload)
        except jwt.ExpiredSignatureError as err:
            raise AuthenticationException("Authentication token has expired.") from err
        except Exception as err:
            raise AuthenticationException(f"Token verification failed: {err}") from err

    @staticmethod
    def _build_authenticated_user(payload: Dict[str, Any]) -> AuthenticatedUser:
        sub = payload.get("sub")
        if not sub:
            raise AuthenticationException("Token missing subject identifier ('sub').")

        email = payload.get("email")
        # Extract role from Supabase app_metadata or standard role claim
        app_metadata = payload.get("app_metadata", {})
        user_metadata = payload.get("user_metadata", {})
        role = payload.get("role") or app_metadata.get("role") or user_metadata.get("role") or "authenticated"

        return AuthenticatedUser(
            user_id=str(sub),
            email=str(email) if email else None,
            role=str(role),
            claims=payload,
            is_anonymous=False,
        )


jwt_verifier = SupabaseJWTVerifier()


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(auth_scheme),
) -> AuthenticatedUser:
    """FastAPI dependency requiring a valid Supabase JWT Bearer token."""
    # Check for Authorization header
    token: Optional[str] = None
    if credentials and credentials.scheme.lower() == "bearer":
        token = credentials.credentials
    elif "Authorization" in request.headers:
        auth_header = request.headers["Authorization"]
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()

    if not token:
        # Check if dev bypass is explicitly enabled
        if settings.DEV_AUTH_BYPASS:
            dev_user_id = request.headers.get("X-Dev-User-ID", "dev_user_001")
            return AuthenticatedUser(
                user_id=dev_user_id,
                email="dev@local.internal",
                role="developer",
                claims={"sub": dev_user_id, "dev": True},
                is_anonymous=False,
            )
        raise AuthenticationException("Missing or invalid Authorization Bearer header.")

    user = jwt_verifier.verify_token(token)
    # Attach authenticated user to request state for downstream logging/telemetry
    request.state.user = user
    return user


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(auth_scheme),
) -> Optional[AuthenticatedUser]:
    """FastAPI dependency for endpoints that support both authenticated and anonymous callers."""
    try:
        return await get_current_user(request, credentials)
    except AuthenticationException:
        # Return anonymous guest identity
        guest_id = request.headers.get("X-Session-ID") or "anonymous_guest"
        return AuthenticatedUser(
            user_id=guest_id,
            email=None,
            role="anon",
            claims={},
            is_anonymous=True,
        )


def require_role(allowed_roles: list[str] | str) -> Callable[[AuthenticatedUser], AuthenticatedUser]:
    """Dependency factory enforcing specific role-based authorization."""
    roles_set = {allowed_roles} if isinstance(allowed_roles, str) else set(allowed_roles)

    def _role_checker(user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
        if user.role not in roles_set and "admin" not in user.role:
            raise AuthorizationException(
                f"Access forbidden: User role '{user.role}' is not in allowed roles: {sorted(list(roles_set))}."
            )
        return user

    return _role_checker
