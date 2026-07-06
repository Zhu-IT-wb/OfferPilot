from fastapi import APIRouter, HTTPException, status

from app.agents.orchestrator import AgentOrchestrator
from app.schemas.agent import AgentMessageRequest, AgentResponse
from app.services.llm_service import LLMRequestError

router = APIRouter(prefix="/agent")


# 处理外部直接调用 Agent 的消息请求。
@router.post("/message", response_model=AgentResponse, response_model_exclude_none=True)
async def handle_agent_message(request: AgentMessageRequest) -> AgentResponse:
    orchestrator = AgentOrchestrator()

    try:
        return await orchestrator.handle_message(
            message=request.message,
            confirmed=request.confirmed,
            user_id=request.user_id,
            source=request.source,
        )
    except LLMRequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
