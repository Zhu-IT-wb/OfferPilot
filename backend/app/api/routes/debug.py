from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.routes.agent import _raise_runtime_http_error, get_agent_runtime
from app.agents.runtime import (
    AgentActorMismatch,
    AgentInteractionError,
    AgentRuntimeConflict,
    AgentThreadNotFound,
)
from app.schemas.agent import (
    AgentRunPublicResponse,
    AgentRunResponse,
    DebugAgentRequest,
)
from app.schemas.llm import DebugLLMRequest, DebugLLMResponse
from app.services.llm_service import (
    LLMConfigurationError,
    LLMRequestError,
    LLMService,
)

router = APIRouter(prefix="/debug")


# 调试大模型调用链路。
@router.post("/llm", response_model=DebugLLMResponse)
async def debug_llm(request: DebugLLMRequest) -> DebugLLMResponse:
    service = LLMService()

    try:
        result = await service.generate_text(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except LLMRequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return DebugLLMResponse(
        provider=result.provider,
        model=result.model,
        content=result.content,
    )


# 调试 Agent 编排后的响应。
@router.post(
    "/agent",
    response_model=AgentRunPublicResponse,
    response_model_exclude_none=True,
)
async def debug_agent(
    request: DebugAgentRequest,
    runtime: Any = Depends(get_agent_runtime),
) -> AgentRunResponse:
    try:
        return await runtime.start(
            message=request.message,
            user_id=request.user_id,
            source="debug",
            conversation_scope=request.conversation_scope,
            request_id=request.request_id,
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
