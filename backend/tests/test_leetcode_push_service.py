import asyncio
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from app.models.leetcode import (
    LeetCodeDeliveryType,
    LeetCodePracticeResult,
    LeetCodeSubscription,
)
from app.repositories.leetcode_repository import InMemoryLeetCodeRepository
from app.services.leetcode_catalog import load_hot100_snapshot
from app.services.leetcode_push_service import LeetCodePushService
from app.services.leetcode_recommendation import LeetCodeRecommendationWorkflow
from app.services.feishu_service import FeishuRequestError


def test_leetcode_pushes_morning_once_and_reminds_only_pending_at_night() -> None:
    class FakeMessageService:
        def __init__(self):
            self.messages = []

        async def send_text_message(
            self, receive_id, text, receive_id_type="open_id", idempotency_key=None
        ):
            self.messages.append((receive_id, text, idempotency_key))

    repository = InMemoryLeetCodeRepository(problems=load_hot100_snapshot().problems)
    repository.save_subscription(
        LeetCodeSubscription(owner_id="feishu:ou_1", feishu_open_id="ou_1")
    )
    messages = FakeMessageService()
    service = LeetCodePushService(repository=repository, message_service=messages)
    timezone = ZoneInfo("Asia/Shanghai")

    morning = datetime(2026, 7, 22, 9, 0, tzinfo=timezone)
    asyncio.run(service.run_due(morning))
    asyncio.run(service.run_due(morning))
    recommendations = LeetCodeRecommendationWorkflow(repository).get_today(
        "feishu:ou_1", morning.date()
    )
    LeetCodeRecommendationWorkflow(repository).record_result(
        owner_id="feishu:ou_1",
        assignment_id=recommendations[0].assignment.id,
        result=LeetCodePracticeResult.INDEPENDENT,
        practiced_on=morning.date(),
    )
    asyncio.run(service.run_due(datetime(2026, 7, 22, 21, 0, tzinfo=timezone)))
    asyncio.run(service.run_due(datetime(2026, 7, 22, 21, 1, tzinfo=timezone)))

    assert len(messages.messages) == 2
    assert "今日 LeetCode" in messages.messages[0][1]
    assert recommendations[0].problem.title_zh not in messages.messages[1][1]
    assert recommendations[1].problem.title_zh in messages.messages[1][1]
    assert recommendations[2].problem.title_zh in messages.messages[1][1]
    assert f"2. {recommendations[1].problem.frontend_id}." in messages.messages[1][1]
    assert f"3. {recommendations[2].problem.frontend_id}." in messages.messages[1][1]
    assert uuid.UUID(messages.messages[0][2]).version == 5
    assert uuid.UUID(messages.messages[1][2]).version == 5
    assert messages.messages[0][2] != messages.messages[1][2]


def test_leetcode_push_stops_retrying_after_three_persisted_failures() -> None:
    class FailingMessageService:
        def __init__(self):
            self.attempts = 0

        async def send_text_message(
            self, receive_id, text, receive_id_type="open_id", idempotency_key=None
        ):
            self.attempts += 1
            raise FeishuRequestError("temporary failure")

    repository = InMemoryLeetCodeRepository(problems=load_hot100_snapshot().problems)
    repository.save_subscription(
        LeetCodeSubscription(owner_id="feishu:ou_1", feishu_open_id="ou_1")
    )
    messages = FailingMessageService()
    service = LeetCodePushService(repository=repository, message_service=messages)
    now = datetime(2026, 7, 22, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    first = asyncio.run(service.run_due(now))
    second = asyncio.run(service.run_due(now))

    assert first.failed == 1
    assert second.failed == 0
    assert messages.attempts == 3
    assert repository.get_delivery_attempts(
        "feishu:ou_1", now.date(), LeetCodeDeliveryType.MORNING
    ) == 3
    assert repository.has_delivery(
        "feishu:ou_1", now.date(), LeetCodeDeliveryType.MORNING
    ) is False
