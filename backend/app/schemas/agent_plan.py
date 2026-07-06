from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.agent import AgentActionName, AgentResponse
from app.schemas.intent import IntentName


# 描述计划中准备调用的一步工具动作。
class AgentPlanStep(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


# 描述 Agent Planner 生成的结构化执行计划。
class AgentPlan(BaseModel):
    intent: IntentName
    confidence: float = Field(..., ge=0.0, le=1.0)
    action: AgentActionName
    reply: str
    need_confirmation: bool = False
    slots: Dict[str, Any] = Field(default_factory=dict)
    missing_slots: List[str] = Field(default_factory=list)
    steps: List[AgentPlanStep] = Field(default_factory=list)
    reason: str = ""

    # 将计划转换为现有 API 使用的 AgentResponse。
    def to_response(self) -> AgentResponse:
        return AgentResponse(
            intent=self.intent,
            confidence=self.confidence,
            action=self.action,
            reply=self.reply,
            need_confirmation=self.need_confirmation,
            slots=self.slots,
            missing_slots=self.missing_slots,
        )
