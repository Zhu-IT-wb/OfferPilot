from typing import Any, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.agents.runtime import (
    AgentActorMismatch,
    AgentInteractionError,
    AgentRuntimeConflict,
    AgentThreadNotFound,
)
from app.schemas.agent import (
    AgentResumeRequest,
    AgentRunRequest,
    AgentRunPublicResponse,
    AgentRunResponse,
)
from app.services.llm_service import LLMConfigurationError, LLMRequestError
from app.services.agent_api_auth import (
    AgentApiActor,
    require_agent_api_actor,
    validate_actor_hints,
)
from app.tools.tool_presentation import redact_tool_identifiers

router = APIRouter(prefix="/agent")


def get_agent_runtime(request: Request) -> Any:
    runtime = getattr(request.app.state, "agent_runtime", None)
    if runtime is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent runtime is not available.",
        )
    return runtime


def _raise_runtime_http_error(exc: Exception) -> NoReturn:
    if isinstance(exc, AgentThreadNotFound):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent thread was not found.",
        ) from exc
    if isinstance(exc, AgentActorMismatch):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=redact_tool_identifiers(str(exc)),
        ) from exc
    if isinstance(exc, (AgentRuntimeConflict, AgentInteractionError)):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=redact_tool_identifiers(str(exc)),
        ) from exc
    if isinstance(exc, LLMConfigurationError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is not configured.",
        ) from exc
    if isinstance(exc, LLMRequestError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM service request failed; please retry later.",
        ) from exc
    raise exc


@router.post(
    "/runs",
    response_model=AgentRunPublicResponse,
    response_model_exclude_none=True,
)
async def start_agent_run(
    payload: AgentRunRequest,
    actor: AgentApiActor = Depends(require_agent_api_actor),
    runtime: Any = Depends(get_agent_runtime),
) -> AgentRunResponse:
    validate_actor_hints(
        actor,
        user_id=payload.user_id,
        source=payload.source,
    )
    try:
        return await runtime.start(
            message=payload.message,
            user_id=actor.user_id,
            source=actor.source,
            conversation_scope=payload.conversation_scope,
            request_id=payload.request_id,
        )
    except (
        AgentActorMismatch,
        AgentRuntimeConflict,
        AgentInteractionError,
        AgentThreadNotFound,
        LLMConfigurationError,
        LLMRequestError,
    ) as exc:
        _raise_runtime_http_error(exc)


@router.post(
    "/runs/{thread_id}/resume",
    response_model=AgentRunPublicResponse,
    response_model_exclude_none=True,
)
async def resume_agent_run(
    thread_id: str,
    payload: AgentResumeRequest,
    actor: AgentApiActor = Depends(require_agent_api_actor),
    runtime: Any = Depends(get_agent_runtime),
) -> AgentRunResponse:
    validate_actor_hints(
        actor,
        user_id=payload.user_id,
        source=payload.source,
    )
    try:
        return await runtime.resume(
            thread_id=thread_id,
            interaction_id=payload.interaction_id,
            decision=payload.decision.value,
            value=payload.value,
            text=payload.text,
            user_id=actor.user_id,
            source=actor.source,
            request_id=payload.request_id,
        )
    except (
        AgentActorMismatch,
        AgentRuntimeConflict,
        AgentInteractionError,
        AgentThreadNotFound,
        LLMConfigurationError,
        LLMRequestError,
    ) as exc:
        _raise_runtime_http_error(exc)


@router.get(
    "/runs/{thread_id}",
    response_model=AgentRunPublicResponse,
    response_model_exclude_none=True,
)
async def get_agent_run(
    thread_id: str,
    user_id: str | None = Query(default=None, min_length=1, max_length=128),
    source: str | None = Query(default=None, min_length=1, max_length=64),
    actor: AgentApiActor = Depends(require_agent_api_actor),
    runtime: Any = Depends(get_agent_runtime),
) -> AgentRunResponse:
    validate_actor_hints(actor, user_id=user_id, source=source)
    try:
        return await runtime.get_state(
            thread_id=thread_id,
            user_id=actor.user_id,
            source=actor.source,
        )
    except (AgentActorMismatch, AgentRuntimeConflict, AgentThreadNotFound) as exc:
        _raise_runtime_http_error(exc)
