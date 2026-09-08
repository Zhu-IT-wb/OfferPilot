from dataclasses import dataclass
from typing import Annotated, Any, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.services.leetcode_dashboard_auth import (
    DashboardTokenError,
    DashboardTokenSigner,
)


AGENT_API_ACTOR_PURPOSE = "agent_api_actor"
_bearer_scheme = HTTPBearer(auto_error=False, scheme_name="AgentActorToken")


@dataclass(frozen=True)
class AgentApiActor:
    user_id: str
    source: str = "api"


def issue_agent_api_actor_token(
    secret: str,
    user_id: str,
    *,
    ttl_seconds: int = 3_600,
    now: Optional[int] = None,
) -> str:
    """Issue a short-lived token for a trusted HTTP API caller."""

    normalized_user_id = _validated_user_id(user_id)
    normalized_secret = str(secret or "").strip()
    if ttl_seconds <= 0:
        raise ValueError("Agent API token TTL must be positive.")
    return DashboardTokenSigner(normalized_secret).issue(
        purpose=AGENT_API_ACTOR_PURPOSE,
        claims={"sub": normalized_user_id},
        ttl_seconds=ttl_seconds,
        now=now,
    )


async def require_agent_api_actor(
    request: Request,
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials],
        Depends(_bearer_scheme),
    ],
) -> AgentApiActor:
    """Resolve the Agent actor exclusively from a server-verifiable token."""

    app_settings = request.app.state.settings
    secret = str(getattr(app_settings, "agent_api_signing_secret", "") or "").strip()
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent API authentication is not configured.",
        )
    if credentials is None or credentials.scheme.lower() != "bearer":
        _raise_unauthorized()

    try:
        payload = DashboardTokenSigner(secret).verify(
            credentials.credentials,
            purpose=AGENT_API_ACTOR_PURPOSE,
        )
        user_id = _validated_user_id(payload.get("sub"))
    except (AttributeError, DashboardTokenError, TypeError, ValueError):
        _raise_unauthorized()
    return AgentApiActor(user_id=user_id)


def validate_actor_hints(
    actor: AgentApiActor,
    *,
    user_id: Optional[str],
    source: Optional[str],
) -> None:
    """Reject legacy identity hints that disagree with the authenticated actor."""

    if user_id is not None and user_id != actor.user_id:
        _raise_actor_mismatch()
    if source is not None and source != actor.source:
        _raise_actor_mismatch()


def _validated_user_id(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Agent API token subject is missing.")
    normalized = value.strip()
    if (
        not normalized
        or normalized != value
        or len(normalized) > 128
        or any(character in normalized for character in ("\r", "\n", "\0"))
    ):
        raise ValueError("Agent API token subject is invalid.")
    return normalized


def _raise_unauthorized() -> None:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Agent API actor token is missing, invalid, or expired.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _raise_actor_mismatch() -> None:
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Request identity does not match the authenticated actor.",
    )
