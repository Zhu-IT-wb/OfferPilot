from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.schemas.agent import AgentActionName, AgentResponse
from app.schemas.intent import IntentName


# 保存等待用户确认或补槽的 Agent 动作。
@dataclass(frozen=True)
class PendingAgentAction:
    intent: IntentName
    confidence: float
    action: AgentActionName
    reply: str
    slots: Dict[str, Any]
    missing_slots: List[str]
    need_confirmation: bool
    original_message: str

    # 根据 Agent 响应创建待确认动作对象。
    @classmethod
    def from_response(cls, response: AgentResponse, original_message: str) -> "PendingAgentAction":
        return cls(
            intent=response.intent,
            confidence=response.confidence,
            action=response.action,
            reply=response.reply,
            slots=response.slots.copy(),
            missing_slots=list(response.missing_slots),
            need_confirmation=response.need_confirmation,
            original_message=original_message,
        )


# 保存最近一轮 Agent 回复，供下一轮追问使用。
@dataclass(frozen=True)
class RecentAgentContext:
    original_message: str
    intent: IntentName
    action: AgentActionName
    reply: str
    slots: Dict[str, Any]
    tool_result: Optional[Dict[str, Any]]

    # 从 AgentResponse 中提取可序列化的短期上下文。
    @classmethod
    def from_response(cls, response: AgentResponse, original_message: str) -> "RecentAgentContext":
        tool_result = None
        if response.tool_result is not None:
            if hasattr(response.tool_result, "model_dump"):
                tool_result = response.tool_result.model_dump()
            else:
                tool_result = response.tool_result.dict()

        return cls(
            original_message=original_message,
            intent=response.intent,
            action=response.action,
            reply=response.reply,
            slots=response.slots.copy(),
            tool_result=tool_result,
        )

    # 转成 LLM Planner 可读的简洁上下文。
    def to_prompt_context(self) -> Dict[str, Any]:
        return {
            "original_message": self.original_message,
            "intent": self.intent.value,
            "action": self.action.value,
            "reply": self.reply,
            "slots": self.slots,
            "tool_result": self.tool_result,
        }


# 在内存中保存每个会话的待确认动作。
class InMemoryConversationStore:
    # 初始化当前组件所需的依赖和配置。
    def __init__(self, max_sessions: int = 1000) -> None:
        self.max_sessions = max_sessions
        self._pending_actions: "OrderedDict[str, PendingAgentAction]" = OrderedDict()
        self._recent_contexts: "OrderedDict[str, RecentAgentContext]" = OrderedDict()

    # 获取 pending action。
    def get_pending_action(self, conversation_id: str) -> Optional[PendingAgentAction]:
        pending = self._pending_actions.get(conversation_id)
        if pending is not None:
            self._pending_actions.move_to_end(conversation_id)
        return pending

    # 保存或更新 pending action。
    def set_pending_action(self, conversation_id: str, pending: PendingAgentAction) -> None:
        self._pending_actions[conversation_id] = pending
        self._pending_actions.move_to_end(conversation_id)
        while len(self._pending_actions) > self.max_sessions:
            self._pending_actions.popitem(last=False)

    # 处理 clear_pending_action 相关逻辑。
    def clear_pending_action(self, conversation_id: str) -> None:
        self._pending_actions.pop(conversation_id, None)

    # 获取最近一轮 Agent 上下文。
    def get_recent_context(self, conversation_id: str) -> Optional[RecentAgentContext]:
        recent_context = self._recent_contexts.get(conversation_id)
        if recent_context is not None:
            self._recent_contexts.move_to_end(conversation_id)
        return recent_context

    # 保存最近一轮 Agent 上下文。
    def set_recent_context(self, conversation_id: str, recent_context: RecentAgentContext) -> None:
        self._recent_contexts[conversation_id] = recent_context
        self._recent_contexts.move_to_end(conversation_id)
        while len(self._recent_contexts) > self.max_sessions:
            self._recent_contexts.popitem(last=False)

    # 处理 clear 相关逻辑。
    def clear(self) -> None:
        self._pending_actions.clear()
        self._recent_contexts.clear()


_default_conversation_store = InMemoryConversationStore()


# 获取 default conversation store。
def get_default_conversation_store() -> InMemoryConversationStore:
    return _default_conversation_store
