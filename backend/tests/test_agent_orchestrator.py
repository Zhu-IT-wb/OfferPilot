import asyncio

from fastapi.testclient import TestClient

from app.agents.orchestrator import AgentOrchestrator
from app.api.routes import debug
from app.main import app
from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository
from app.schemas.agent import AgentActionName, AgentResponse
from app.schemas.intent import IntentClassification, IntentName
from app.tools.offerpilot_tools import build_offerpilot_tool_registry


def test_orchestrator_plans_create_application_with_confirmation() -> None:
    class FakeIntentClassifier:
        async def classify(self, message):
            assert message == "新增投递深信服开发实习，明天下午三点一面"
            return IntentClassification(
                intent=IntentName.ADD_APPLICATION,
                confidence=0.86,
                slots={
                    "company": "深信服",
                    "role": "开发实习",
                    "interview_time": "明天下午三点",
                    "round": "一面",
                },
            )

    orchestrator = AgentOrchestrator(intent_classifier=FakeIntentClassifier())

    result = asyncio.run(orchestrator.handle_message("新增投递深信服开发实习，明天下午三点一面"))

    assert result.intent == IntentName.ADD_APPLICATION
    assert result.action == AgentActionName.CREATE_APPLICATION
    assert result.need_confirmation is True
    assert result.missing_slots == []
    assert result.slots["company"] == "深信服"
    assert "写入投递表" in result.reply


def test_orchestrator_asks_clarification_when_application_slots_missing() -> None:
    class FakeIntentClassifier:
        async def classify(self, message):
            assert message == "新增投递深信服"
            return IntentClassification(
                intent=IntentName.ADD_APPLICATION,
                confidence=0.62,
                slots={"company": "深信服"},
            )

    orchestrator = AgentOrchestrator(intent_classifier=FakeIntentClassifier())

    result = asyncio.run(orchestrator.handle_message("新增投递深信服"))

    assert result.action == AgentActionName.ASK_CLARIFICATION
    assert result.need_confirmation is False
    assert result.missing_slots == ["role"]
    assert "岗位" in result.reply


def test_orchestrator_plans_today_tasks_read_action() -> None:
    class FakeIntentClassifier:
        async def classify(self, message):
            return IntentClassification(
                intent=IntentName.GET_TODAY_TASKS,
                confidence=0.72,
                slots={},
            )

    orchestrator = AgentOrchestrator(intent_classifier=FakeIntentClassifier())

    result = asyncio.run(orchestrator.handle_message("今天任务是什么？"))

    assert result.action == AgentActionName.LIST_TODAY_TASKS
    assert result.need_confirmation is False
    assert "今天的任务" in result.reply
    assert result.tool_result is not None
    assert result.tool_result.success is True
    assert len(result.tool_result.data["tasks"]) == 3


def test_orchestrator_does_not_execute_mutating_action_without_confirmation() -> None:
    class FakeIntentClassifier:
        async def classify(self, message):
            return IntentClassification(
                intent=IntentName.ADD_APPLICATION,
                confidence=0.86,
                slots={"company": "深信服", "role": "开发实习"},
            )

    repository = InMemoryOfferPilotRepository()
    registry = build_offerpilot_tool_registry(repository)
    orchestrator = AgentOrchestrator(
        intent_classifier=FakeIntentClassifier(),
        tool_registry=registry,
    )

    result = asyncio.run(orchestrator.handle_message("新增投递深信服开发实习"))

    assert result.action == AgentActionName.CREATE_APPLICATION
    assert result.need_confirmation is True
    assert result.tool_result is None
    assert repository.applications == []


def test_orchestrator_executes_create_application_when_confirmed() -> None:
    class FakeIntentClassifier:
        async def classify(self, message):
            return IntentClassification(
                intent=IntentName.ADD_APPLICATION,
                confidence=0.86,
                slots={
                    "company": "深信服",
                    "role": "开发实习",
                    "interview_time": "明天下午三点",
                    "round": "一面",
                },
            )

    repository = InMemoryOfferPilotRepository()
    registry = build_offerpilot_tool_registry(repository)
    orchestrator = AgentOrchestrator(
        intent_classifier=FakeIntentClassifier(),
        tool_registry=registry,
    )

    result = asyncio.run(
        orchestrator.handle_message(
            "新增投递深信服开发实习，明天下午三点一面",
            confirmed=True,
        )
    )

    assert result.action == AgentActionName.CREATE_APPLICATION
    assert result.need_confirmation is False
    assert result.tool_result is not None
    assert result.tool_result.success is True
    assert result.tool_result.data["application"]["id"] == "app_1"
    assert result.tool_result.data["application"]["status"] == "interview_1"
    assert len(repository.applications) == 1


def test_debug_agent_route_returns_orchestrated_response(monkeypatch) -> None:
    class FakeAgentOrchestrator:
        async def handle_message(self, message, confirmed=False):
            assert message == "今天任务是什么？"
            assert confirmed is False
            return AgentResponse(
                intent=IntentName.GET_TODAY_TASKS,
                confidence=0.9,
                action=AgentActionName.LIST_TODAY_TASKS,
                reply="我识别到你想查看今日任务。",
                slots={},
            )

    monkeypatch.setattr(debug, "AgentOrchestrator", FakeAgentOrchestrator)
    client = TestClient(app)

    response = client.post(
        "/api/debug/agent",
        json={"message": "今天任务是什么？"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "intent": "get_today_tasks",
        "confidence": 0.9,
        "action": "list_today_tasks",
        "reply": "我识别到你想查看今日任务。",
        "need_confirmation": False,
        "slots": {},
        "missing_slots": [],
    }
