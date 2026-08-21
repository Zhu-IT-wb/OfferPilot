from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Sequence

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "action": self.action.value,
            "reply": self.reply,
            "slots": deepcopy(self.slots),
            "missing_slots": list(self.missing_slots),
            "need_confirmation": self.need_confirmation,
            "original_message": self.original_message,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "PendingAgentAction":
        return cls(
            intent=IntentName(value["intent"]),
            confidence=float(value["confidence"]),
            action=AgentActionName(value["action"]),
            reply=str(value.get("reply", "")),
            slots=deepcopy(value.get("slots") or {}),
            missing_slots=list(value.get("missing_slots") or []),
            need_confirmation=bool(value.get("need_confirmation", False)),
            original_message=str(value.get("original_message", "")),
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
    application_focus: Dict[str, Any]

    # 从 AgentResponse 中提取可序列化的短期上下文。
    @classmethod
    def from_response(
        cls,
        response: AgentResponse,
        original_message: str,
        previous: Optional["RecentAgentContext"] = None,
    ) -> "RecentAgentContext":
        tool_result = None
        if response.tool_result is not None:
            if hasattr(response.tool_result, "model_dump"):
                tool_result = response.tool_result.model_dump()
            else:
                tool_result = response.tool_result.dict()

        application_focus = previous.application_focus.copy() if previous is not None else {}
        if response.action in {
            AgentActionName.CREATE_APPLICATION,
            AgentActionName.QUERY_APPLICATION,
            AgentActionName.UPDATE_APPLICATION,
            AgentActionName.RESCHEDULE_INTERVIEW,
            AgentActionName.CANCEL_INTERVIEW,
        }:
            for key in ("application_id", "company", "role"):
                value = response.slots.get(key)
                if value:
                    application_focus[key] = value

            tool_data = tool_result.get("data") if tool_result else None
            application = tool_data.get("application") if isinstance(tool_data, dict) else None
            if isinstance(application, dict):
                for source_key, focus_key in (
                    ("id", "application_id"),
                    ("company", "company"),
                    ("role", "role"),
                ):
                    value = application.get(source_key)
                    if value:
                        application_focus[focus_key] = value

        return cls(
            original_message=original_message,
            intent=response.intent,
            action=response.action,
            reply=response.reply,
            slots=response.slots.copy(),
            tool_result=tool_result,
            application_focus=application_focus,
        )

    # 转成 LLM Planner 可读的简洁上下文。
    def to_prompt_context(self) -> Dict[str, Any]:
        return {
            "original_message": self.original_message,
            "intent": self.intent.value,
            "action": self.action.value,
            "reply": self.reply,
            "slots": deepcopy(self.slots),
            "tool_result": deepcopy(self.tool_result),
            "application_focus": deepcopy(self.application_focus),
        }

    def to_dict(self) -> Dict[str, Any]:
        return deepcopy(self.to_prompt_context())

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "RecentAgentContext":
        return cls(
            original_message=str(value.get("original_message", "")),
            intent=IntentName(value["intent"]),
            action=AgentActionName(value["action"]),
            reply=str(value.get("reply", "")),
            slots=deepcopy(value.get("slots") or {}),
            tool_result=deepcopy(value.get("tool_result")),
            application_focus=deepcopy(value.get("application_focus") or {}),
        )


@dataclass(frozen=True)
class ConversationEvent:
    event_type: str
    role: str
    payload: Dict[str, Any]
    turn_id: str
    sequence: int = 0
    created_at: str = ""

    def to_prompt_context(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_type": self.event_type,
            "role": self.role,
            "payload": deepcopy(self.payload),
        }


@dataclass(frozen=True)
class ConversationSummary:
    user_goals: List[str]
    target_roles: List[str]
    active_applications: List[Dict[str, Any]]
    upcoming_interviews: List[Dict[str, Any]]
    training_progress: List[Dict[str, Any]]
    confirmed_facts: List[str]
    decisions: List[str]
    completed_actions: List[str]
    open_loops: List[str]
    tool_evidence: List[Dict[str, Any]]
    summary_upto_sequence: int = 0
    compaction_count: int = 0
    schema_version: int = 1

    @classmethod
    def empty(cls) -> "ConversationSummary":
        return cls(
            user_goals=[],
            target_roles=[],
            active_applications=[],
            upcoming_interviews=[],
            training_progress=[],
            confirmed_facts=[],
            decisions=[],
            completed_actions=[],
            open_loops=[],
            tool_evidence=[],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "summary_upto_sequence": self.summary_upto_sequence,
            "compaction_count": self.compaction_count,
            "user_goals": list(self.user_goals),
            "target_roles": list(self.target_roles),
            "active_applications": deepcopy(self.active_applications),
            "upcoming_interviews": deepcopy(self.upcoming_interviews),
            "training_progress": deepcopy(self.training_progress),
            "confirmed_facts": list(self.confirmed_facts),
            "decisions": list(self.decisions),
            "completed_actions": list(self.completed_actions),
            "open_loops": list(self.open_loops),
            "tool_evidence": deepcopy(self.tool_evidence),
        }

    def to_prompt_context(self) -> Dict[str, Any]:
        return self.to_dict()

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ConversationSummary":
        return cls(
            schema_version=int(value.get("schema_version", 1)),
            summary_upto_sequence=max(0, int(value.get("summary_upto_sequence", 0))),
            compaction_count=max(0, int(value.get("compaction_count", 0))),
            user_goals=_string_list(value.get("user_goals")),
            target_roles=_string_list(value.get("target_roles")),
            active_applications=_dict_list(value.get("active_applications")),
            upcoming_interviews=_dict_list(value.get("upcoming_interviews")),
            training_progress=_dict_list(value.get("training_progress")),
            confirmed_facts=_string_list(value.get("confirmed_facts")),
            decisions=_string_list(value.get("decisions")),
            completed_actions=_string_list(value.get("completed_actions")),
            open_loops=_string_list(value.get("open_loops")),
            tool_evidence=_dict_list(value.get("tool_evidence")),
        )


class ConversationStore(Protocol):
    def get_pending_action(self, conversation_id: str) -> Optional[PendingAgentAction]: ...

    def set_pending_action(self, conversation_id: str, pending: PendingAgentAction) -> None: ...

    def clear_pending_action(self, conversation_id: str) -> None: ...

    def get_recent_context(self, conversation_id: str) -> Optional[RecentAgentContext]: ...

    def set_recent_context(
        self,
        conversation_id: str,
        recent_context: RecentAgentContext,
    ) -> None: ...

    def append_events(
        self,
        conversation_id: str,
        events: Sequence[ConversationEvent],
    ) -> List[ConversationEvent]: ...

    def get_recent_events(
        self,
        conversation_id: str,
        limit: int = 40,
    ) -> List[ConversationEvent]: ...

    def get_events_after(
        self,
        conversation_id: str,
        sequence: int,
    ) -> List[ConversationEvent]: ...

    def get_conversation_summary(
        self,
        conversation_id: str,
    ) -> Optional[ConversationSummary]: ...

    def save_conversation_summary(
        self,
        conversation_id: str,
        summary: ConversationSummary,
        expected_upto_sequence: int,
    ) -> bool: ...


# 在内存中保存每个会话的待确认动作。
class InMemoryConversationStore:
    # 初始化当前组件所需的依赖和配置。
    def __init__(self, max_sessions: int = 1000, max_events_per_session: int = 200) -> None:
        self.max_sessions = max_sessions
        self.max_events_per_session = max_events_per_session
        self._pending_actions: "OrderedDict[str, PendingAgentAction]" = OrderedDict()
        self._recent_contexts: "OrderedDict[str, RecentAgentContext]" = OrderedDict()
        self._events: "OrderedDict[str, List[ConversationEvent]]" = OrderedDict()
        self._summaries: "OrderedDict[str, ConversationSummary]" = OrderedDict()

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

    def append_events(
        self,
        conversation_id: str,
        events: Sequence[ConversationEvent],
    ) -> List[ConversationEvent]:
        history = self._events.setdefault(conversation_id, [])
        next_sequence = history[-1].sequence + 1 if history else 1
        persisted = []
        for event in events:
            stored = replace(
                event,
                sequence=next_sequence,
                created_at=event.created_at or _utc_now(),
                payload=deepcopy(event.payload),
            )
            history.append(stored)
            persisted.append(stored)
            next_sequence += 1

        if len(history) > self.max_events_per_session:
            del history[: len(history) - self.max_events_per_session]
        self._events.move_to_end(conversation_id)
        while len(self._events) > self.max_sessions:
            self._events.popitem(last=False)
        return persisted

    def get_recent_events(
        self,
        conversation_id: str,
        limit: int = 40,
    ) -> List[ConversationEvent]:
        if limit <= 0:
            return []
        history = self._events.get(conversation_id)
        if history is None:
            return []
        self._events.move_to_end(conversation_id)
        return [replace(event, payload=deepcopy(event.payload)) for event in history[-limit:]]

    def get_events_after(
        self,
        conversation_id: str,
        sequence: int,
    ) -> List[ConversationEvent]:
        history = self._events.get(conversation_id)
        if history is None:
            return []
        self._events.move_to_end(conversation_id)
        return [
            replace(event, payload=deepcopy(event.payload))
            for event in history
            if event.sequence > sequence
        ]

    def get_conversation_summary(
        self,
        conversation_id: str,
    ) -> Optional[ConversationSummary]:
        summary = self._summaries.get(conversation_id)
        if summary is None:
            return None
        self._summaries.move_to_end(conversation_id)
        return ConversationSummary.from_dict(summary.to_dict())

    def save_conversation_summary(
        self,
        conversation_id: str,
        summary: ConversationSummary,
        expected_upto_sequence: int,
    ) -> bool:
        existing = self._summaries.get(conversation_id)
        current_sequence = existing.summary_upto_sequence if existing is not None else 0
        if current_sequence != expected_upto_sequence:
            return False
        self._summaries[conversation_id] = ConversationSummary.from_dict(summary.to_dict())
        self._summaries.move_to_end(conversation_id)
        while len(self._summaries) > self.max_sessions:
            self._summaries.popitem(last=False)
        return True

    # 处理 clear 相关逻辑。
    def clear(self) -> None:
        self._pending_actions.clear()
        self._recent_contexts.clear()
        self._events.clear()
        self._summaries.clear()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _dict_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [deepcopy(item) for item in value if isinstance(item, dict)]
