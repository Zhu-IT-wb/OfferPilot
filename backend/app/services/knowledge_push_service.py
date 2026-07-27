import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

from app.models.interview_knowledge import (
    KnowledgeAssignmentStatus,
    KnowledgeDeliveryType,
)
from app.repositories.interview_knowledge_repository import InterviewKnowledgeRepository
from app.services.feishu_service import (
    FeishuConfigurationError,
    FeishuMessageService,
    FeishuRequestError,
)
from app.services.knowledge_messages import build_knowledge_reminder_card
from app.services.knowledge_recommendation import KnowledgeRecommendationWorkflow


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KnowledgePushSummary:
    morning_sent: int = 0
    noon_sent: int = 0
    evening_sent: int = 0
    failed: int = 0


class KnowledgePushService:
    def __init__(
        self,
        repository: InterviewKnowledgeRepository,
        message_service: Optional[FeishuMessageService] = None,
        interval_seconds: int = 60,
        dashboard_url: Optional[str] = None,
    ) -> None:
        self.repository = repository
        self.message_service = message_service or FeishuMessageService()
        self.interval_seconds = max(interval_seconds, 5)
        self.dashboard_url = dashboard_url
        self._stop_event: Optional[asyncio.Event] = None
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task is not None:
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is not None:
            await self._task
            self._task = None

    async def _run(self) -> None:
        if self._stop_event is None:
            return
        while not self._stop_event.is_set():
            try:
                await self.run_due(datetime.now(ZoneInfo("Asia/Shanghai")))
            except Exception as exc:
                logger.warning("Knowledge push cycle failed: %s", exc)
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.interval_seconds
                )
            except asyncio.TimeoutError:
                continue

    async def run_due(self, now: datetime) -> KnowledgePushSummary:
        counts = {item: 0 for item in KnowledgeDeliveryType}
        failed = 0
        for subscription in self.repository.list_enabled_subscriptions():
            local_now = now.astimezone(ZoneInfo(subscription.timezone))
            delivery_type = _current_delivery_type(
                local_now.strftime("%H:%M"),
                subscription.morning_time,
                subscription.noon_time,
                subscription.evening_time,
            )
            if delivery_type is None:
                continue
            today = local_now.date()
            if self.repository.has_delivery(subscription.owner_id, today, delivery_type):
                continue
            recommendations = KnowledgeRecommendationWorkflow(self.repository).get_today(
                subscription.owner_id, today
            )
            if delivery_type != KnowledgeDeliveryType.MORNING:
                recommendations = [
                    item for item in recommendations
                    if item.assignment.status == KnowledgeAssignmentStatus.PENDING
                ]
                if not recommendations:
                    self.repository.record_delivery(
                        subscription.owner_id, today, delivery_type
                    )
                    continue
            try:
                await self.message_service.send_interactive_message(
                    receive_id=subscription.feishu_open_id,
                    card=build_knowledge_reminder_card(
                        recommendations, delivery_type, self.dashboard_url
                    ),
                    idempotency_key=_delivery_idempotency_key(
                        subscription.owner_id, today, delivery_type
                    ),
                )
            except (FeishuConfigurationError, FeishuRequestError) as exc:
                failed += 1
                logger.warning("Knowledge push failed: %s", exc)
                continue
            self.repository.record_delivery(subscription.owner_id, today, delivery_type)
            counts[delivery_type] += 1
        return KnowledgePushSummary(
            morning_sent=counts[KnowledgeDeliveryType.MORNING],
            noon_sent=counts[KnowledgeDeliveryType.NOON],
            evening_sent=counts[KnowledgeDeliveryType.EVENING],
            failed=failed,
        )


def _current_delivery_type(
    current_time: str, morning_time: str, noon_time: str, evening_time: str
) -> Optional[KnowledgeDeliveryType]:
    if current_time >= evening_time:
        return KnowledgeDeliveryType.EVENING
    if current_time >= noon_time:
        return KnowledgeDeliveryType.NOON
    if current_time >= morning_time:
        return KnowledgeDeliveryType.MORNING
    return None


def _delivery_idempotency_key(
    owner_id: str, delivery_on: date, delivery_type: KnowledgeDeliveryType
) -> str:
    source = f"offerpilot://knowledge/{owner_id}/{delivery_on}/{delivery_type.value}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, source))
