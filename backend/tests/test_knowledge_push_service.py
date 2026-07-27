import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from app.models.interview_knowledge import KnowledgeSubscription
from app.repositories.interview_knowledge_repository import (
    InMemoryInterviewKnowledgeRepository,
)
from app.services.knowledge_catalog import load_knowledge_catalog
from app.services.knowledge_push_service import KnowledgePushService


class FakeMessageService:
    def __init__(self) -> None:
        self.messages = []

    async def send_interactive_message(self, **kwargs) -> None:
        self.messages.append(kwargs)


def test_morning_push_is_idempotent_and_links_to_knowledge_practice() -> None:
    repository = InMemoryInterviewKnowledgeRepository(
        questions=load_knowledge_catalog().questions
    )
    repository.save_subscription(
        KnowledgeSubscription(
            owner_id="feishu:ou_owner",
            feishu_open_id="ou_owner",
        )
    )
    messages = FakeMessageService()
    service = KnowledgePushService(
        repository=repository,
        message_service=messages,
        dashboard_url="https://offerpilot.example/study/knowledge",
    )
    now = datetime(2026, 7, 27, 8, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    first = asyncio.run(service.run_due(now))
    second = asyncio.run(service.run_due(now))

    assert first.morning_sent == 1
    assert second.morning_sent == 0
    assert len(messages.messages) == 1
    card_text = str(messages.messages[0]["card"])
    assert "今日八股" in card_text
    assert "5 题" in card_text
    assert "https://offerpilot.example/study/knowledge" in card_text
    assert "参考答案" not in card_text
