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


def test_leetcode_pushes_cards_at_eight_noon_and_six_for_pending_problems() -> None:
    class FakeMessageService:
        def __init__(self):
            self.cards = []

        async def send_interactive_message(
            self, receive_id, card, receive_id_type="open_id", idempotency_key=None
        ):
            self.cards.append((receive_id, card, idempotency_key))

    repository = InMemoryLeetCodeRepository(problems=load_hot100_snapshot().problems)
    repository.save_subscription(
        LeetCodeSubscription(owner_id="feishu:ou_1", feishu_open_id="ou_1")
    )
    messages = FakeMessageService()
    service = LeetCodePushService(repository=repository, message_service=messages)
    timezone = ZoneInfo("Asia/Shanghai")

    morning = datetime(2026, 7, 22, 8, 0, tzinfo=timezone)
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
    asyncio.run(service.run_due(datetime(2026, 7, 22, 12, 0, tzinfo=timezone)))
    asyncio.run(service.run_due(datetime(2026, 7, 22, 12, 1, tzinfo=timezone)))
    asyncio.run(service.run_due(datetime(2026, 7, 22, 18, 0, tzinfo=timezone)))
    asyncio.run(service.run_due(datetime(2026, 7, 22, 18, 1, tzinfo=timezone)))

    assert len(messages.cards) == 3
    assert messages.cards[0][1]["header"]["title"]["content"] == "🎯 今日 LeetCode · 3 题"
    assert messages.cards[1][1]["header"]["title"]["content"] == "⏰ 12:00 刷题进度提醒"
    assert messages.cards[2][1]["header"]["title"]["content"] == "🔥 18:00 今日最后提醒"
    noon_content = str(messages.cards[1][1])
    evening_content = str(messages.cards[2][1])
    assert recommendations[0].problem.title_zh not in noon_content
    assert recommendations[1].problem.title_zh in noon_content
    assert recommendations[2].problem.title_zh in noon_content
    assert "明天会自动顺延" in evening_content
    assert all(uuid.UUID(item[2]).version == 5 for item in messages.cards)
    assert len({item[2] for item in messages.cards}) == 3
    assert repository.has_delivery(
        "feishu:ou_1", morning.date(), LeetCodeDeliveryType.NOON
    ) is True


def test_leetcode_push_stops_retrying_after_three_persisted_failures() -> None:
    class FailingMessageService:
        def __init__(self):
            self.attempts = 0

        async def send_interactive_message(
            self, receive_id, card, receive_id_type="open_id", idempotency_key=None
        ):
            self.attempts += 1
            raise FeishuRequestError("temporary failure")

    repository = InMemoryLeetCodeRepository(problems=load_hot100_snapshot().problems)
    repository.save_subscription(
        LeetCodeSubscription(owner_id="feishu:ou_1", feishu_open_id="ou_1")
    )
    messages = FailingMessageService()
    service = LeetCodePushService(repository=repository, message_service=messages)
    now = datetime(2026, 7, 22, 8, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

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
