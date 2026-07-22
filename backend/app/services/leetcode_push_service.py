import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from app.models.leetcode import (
    LeetCodeAssignmentStatus,
    LeetCodeDeliveryType,
    LeetCodeRecommendation,
)
from app.repositories.leetcode_repository import LeetCodeRepository
from app.services.feishu_service import (
    FeishuConfigurationError,
    FeishuMessageService,
    FeishuRequestError,
)
from app.services.leetcode_messages import build_leetcode_card
from app.services.leetcode_recommendation import LeetCodeRecommendationWorkflow


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeetCodePushSummary:
    morning_sent: int = 0
    noon_sent: int = 0
    evening_sent: int = 0
    failed: int = 0


class LeetCodePushService:
    def __init__(
        self,
        repository: LeetCodeRepository,
        message_service: Optional[FeishuMessageService] = None,
        interval_seconds: int = 60,
        max_attempts: int = 3,
    ) -> None:
        self.repository = repository
        self.message_service = message_service or FeishuMessageService()
        self.interval_seconds = max(interval_seconds, 5)
        self.max_attempts = max(max_attempts, 1)
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
                logger.warning("LeetCode push cycle failed: %s", exc)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                continue

    async def run_due(self, now: datetime) -> LeetCodePushSummary:
        morning_sent = 0
        noon_sent = 0
        evening_sent = 0
        failed = 0
        for subscription in self.repository.list_enabled_subscriptions():
            local_now = now.astimezone(ZoneInfo(subscription.timezone))
            today = local_now.date()
            current_time = local_now.strftime("%H:%M")
            delivery_type = _current_delivery_type(
                current_time=current_time,
                morning_time=subscription.morning_time,
                noon_time=subscription.noon_time,
                evening_time=subscription.evening_time,
            )
            if delivery_type is None:
                continue
            workflow = LeetCodeRecommendationWorkflow(self.repository)
            recommendations = workflow.get_today(subscription.owner_id, today)
            if self.repository.has_delivery(subscription.owner_id, today, delivery_type):
                continue
            if (
                self.repository.get_delivery_attempts(
                    subscription.owner_id, today, delivery_type
                )
                >= self.max_attempts
            ):
                continue
            selected = recommendations
            if delivery_type != LeetCodeDeliveryType.MORNING:
                selected = [
                    item
                    for item in recommendations
                    if item.assignment.status == LeetCodeAssignmentStatus.PENDING
                ]
                if not selected:
                    self.repository.record_delivery(
                        subscription.owner_id, today, delivery_type
                    )
                    continue
            sent = await self._send_with_retry(
                owner_id=subscription.owner_id,
                delivery_on=today,
                delivery_type=delivery_type,
                receive_id=subscription.feishu_open_id,
                card=build_leetcode_card(selected, delivery_type),
            )
            if not sent:
                failed += 1
                continue
            if delivery_type != LeetCodeDeliveryType.MORNING:
                for item in selected:
                    item.assignment.feedback_reminded_at = local_now
                    self.repository.save_assignment(item.assignment)
            self.repository.record_delivery(subscription.owner_id, today, delivery_type)
            if delivery_type == LeetCodeDeliveryType.MORNING:
                morning_sent += 1
            elif delivery_type == LeetCodeDeliveryType.NOON:
                noon_sent += 1
            else:
                evening_sent += 1
        return LeetCodePushSummary(
            morning_sent=morning_sent,
            noon_sent=noon_sent,
            evening_sent=evening_sent,
            failed=failed,
        )

    async def _send_with_retry(
        self,
        owner_id: str,
        delivery_on: date,
        delivery_type: LeetCodeDeliveryType,
        receive_id: str,
        card: Dict[str, Any],
    ) -> bool:
        previous_attempts = self.repository.get_delivery_attempts(
            owner_id, delivery_on, delivery_type
        )
        remaining_attempts = self.max_attempts - previous_attempts
        idempotency_key = _delivery_idempotency_key(owner_id, delivery_on, delivery_type)
        for attempt in range(remaining_attempts):
            try:
                await self.message_service.send_interactive_message(
                    receive_id=receive_id,
                    card=card,
                    idempotency_key=idempotency_key,
                )
                return True
            except (FeishuConfigurationError, FeishuRequestError) as exc:
                self.repository.record_delivery_attempt(owner_id, delivery_on, delivery_type)
                logger.warning(
                    "LeetCode push failed: receive_id_present=%s attempt=%s error=%s",
                    bool(receive_id),
                    previous_attempts + attempt + 1,
                    exc,
                )
        return False


def _delivery_idempotency_key(
    owner_id: str,
    delivery_on: date,
    delivery_type: LeetCodeDeliveryType,
) -> str:
    source = (
        f"offerpilot://leetcode/{owner_id}/{delivery_on.isoformat()}/{delivery_type.value}"
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, source))


def _current_delivery_type(
    current_time: str,
    morning_time: str,
    noon_time: str,
    evening_time: str,
) -> Optional[LeetCodeDeliveryType]:
    if current_time >= evening_time:
        return LeetCodeDeliveryType.EVENING
    if current_time >= noon_time:
        return LeetCodeDeliveryType.NOON
    if current_time >= morning_time:
        return LeetCodeDeliveryType.MORNING
    return None
