import threading
from dataclasses import dataclass, field
from typing import Any

from app.services.agent_runtime_store import AgentRuntimeStore


@dataclass
class FeishuDeliveryAttempt:
    event_id: str
    state: dict[str, Any] = field(default_factory=dict)
    completed: bool = False


class FeishuTextDeliveryCoordinator:
    """Persist delivery separately from immutable Agent execution receipts.

    The active claim is process-local, matching the single-worker runtime. A
    processing receipt without an active claim is recoverable after a restart.
    """

    source = "feishu:delivery"

    def __init__(self, store: AgentRuntimeStore) -> None:
        self.store = store
        self._lock = threading.Lock()
        self._active: set[str] = set()

    def claim(
        self, event_id: str, fingerprint: str, thread_id: str
    ) -> FeishuDeliveryAttempt | None:
        with self._lock:
            record = self.store.get_receipt_record(self.source, event_id)
            if record is None:
                self.store.begin_receipt(
                    self.source, event_id, fingerprint=fingerprint, thread_id=thread_id
                )
                record = self.store.get_receipt_record(self.source, event_id)
            if record is None:
                raise RuntimeError("Feishu delivery receipt could not be created.")
            if record.fingerprint != fingerprint or record.thread_id != thread_id:
                raise ValueError("Feishu event ID was reused with a different message.")
            if record.status == "completed":
                return FeishuDeliveryAttempt(event_id, record.response or {}, True)
            if event_id in self._active:
                return None
            self._active.add(event_id)
            return FeishuDeliveryAttempt(event_id, record.response or {})

    def stage(self, attempt: FeishuDeliveryAttempt, **values: Any) -> None:
        attempt.state.update(values)
        self.store.stage_receipt_response(self.source, attempt.event_id, attempt.state)

    def finish(self, attempt: FeishuDeliveryAttempt, result: dict[str, Any]) -> None:
        attempt.state["result"] = result
        self.store.finish_receipt(self.source, attempt.event_id, attempt.state)

    def release(self, event_id: str) -> None:
        with self._lock:
            self._active.discard(event_id)
