from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.schemas.agent import AgentActionName, AgentResponse
from app.schemas.intent import IntentName


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


class InMemoryConversationStore:
    def __init__(self, max_sessions: int = 1000) -> None:
        self.max_sessions = max_sessions
        self._pending_actions: "OrderedDict[str, PendingAgentAction]" = OrderedDict()

    def get_pending_action(self, conversation_id: str) -> Optional[PendingAgentAction]:
        pending = self._pending_actions.get(conversation_id)
        if pending is not None:
            self._pending_actions.move_to_end(conversation_id)
        return pending

    def set_pending_action(self, conversation_id: str, pending: PendingAgentAction) -> None:
        self._pending_actions[conversation_id] = pending
        self._pending_actions.move_to_end(conversation_id)
        while len(self._pending_actions) > self.max_sessions:
            self._pending_actions.popitem(last=False)

    def clear_pending_action(self, conversation_id: str) -> None:
        self._pending_actions.pop(conversation_id, None)

    def clear(self) -> None:
        self._pending_actions.clear()


_default_conversation_store = InMemoryConversationStore()


def get_default_conversation_store() -> InMemoryConversationStore:
    return _default_conversation_store
