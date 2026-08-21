from fastapi import APIRouter, HTTPException, status

from app.agents.intent_classifier import IntentClassifier
from app.agents.orchestrator import AgentOrchestrator
from app.schemas.agent import AgentResponse, DebugAgentRequest
from app.schemas.intent import DebugIntentRequest, IntentClassification
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


# 调试自然语言意图识别结果。
@router.post("/intent", response_model=IntentClassification)
async def debug_intent(request: DebugIntentRequest) -> IntentClassification:
    classifier = IntentClassifier()

    try:
        return await classifier.classify(request.message)
    except LLMRequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc


# 调试 Agent 编排后的响应。
@router.post("/agent", response_model=AgentResponse, response_model_exclude_none=True)
async def debug_agent(request: DebugAgentRequest) -> AgentResponse:
    orchestrator = AgentOrchestrator()

    try:
        return await orchestrator.handle_message(
            message=request.message,
            confirmed=request.confirmed,
            user_id=request.user_id,
            source=request.source,
            conversation_scope=request.conversation_scope,
        )
    except LLMRequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
