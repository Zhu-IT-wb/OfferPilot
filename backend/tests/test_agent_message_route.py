from fastapi.testclient import TestClient

from app.api.routes import agent as agent_route
from app.core.config import Settings
from app.main import create_app
from app.schemas.agent import AgentActionName, AgentResponse
from app.schemas.intent import IntentName


def test_agent_message_route_returns_orchestrated_response(monkeypatch) -> None:
    class FakeAgentOrchestrator:
        async def handle_message(
            self,
            message,
            confirmed=False,
            user_id="local_user",
            source="api",
            conversation_scope=None,
        ):
            assert message == "今天任务是什么？"
            assert confirmed is False
            assert user_id == "local_user"
            assert source == "api"
            assert conversation_scope is None
            return AgentResponse(
                intent=IntentName.GET_TODAY_TASKS,
                confidence=0.9,
                action=AgentActionName.LIST_TODAY_TASKS,
                reply="今天的任务：1. LeetCode 206. 反转链表",
                slots={},
            )

    monkeypatch.setattr(agent_route, "AgentOrchestrator", FakeAgentOrchestrator)
    app = create_app(Settings(debug_routes_enabled=False))
    client = TestClient(app)

    response = client.post(
        "/api/agent/message",
        json={"message": "今天任务是什么？"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "intent": "get_today_tasks",
        "confidence": 0.9,
        "action": "list_today_tasks",
        "reply": "今天的任务：1. LeetCode 206. 反转链表",
        "need_confirmation": False,
        "slots": {},
        "missing_slots": [],
    }


def test_agent_message_route_passes_confirmation(monkeypatch) -> None:
    class FakeAgentOrchestrator:
        async def handle_message(
            self,
            message,
            confirmed=False,
            user_id="local_user",
            source="api",
            conversation_scope=None,
        ):
            assert message == "新增投递深信服开发实习"
            assert confirmed is True
            assert user_id == "will"
            assert source == "feishu"
            assert conversation_scope == "chat-will"
            return AgentResponse(
                intent=IntentName.ADD_APPLICATION,
                confidence=0.86,
                action=AgentActionName.CREATE_APPLICATION,
                reply="已创建投递记录：深信服 - 开发实习。",
                slots={"company": "深信服", "role": "开发实习"},
            )

    monkeypatch.setattr(agent_route, "AgentOrchestrator", FakeAgentOrchestrator)
    app = create_app(Settings(debug_routes_enabled=False))
    client = TestClient(app)

    response = client.post(
        "/api/agent/message",
        json={
            "message": "新增投递深信服开发实习",
            "confirmed": True,
            "source": "feishu",
            "user_id": "will",
            "conversation_scope": "chat-will",
        },
    )

    assert response.status_code == 200
    assert response.json()["action"] == "create_application"


def test_agent_message_route_stays_enabled_when_debug_routes_disabled(monkeypatch) -> None:
    class FakeAgentOrchestrator:
        async def handle_message(
            self,
            message,
            confirmed=False,
            user_id="local_user",
            source="api",
            conversation_scope=None,
        ):
            return AgentResponse(
                intent=IntentName.GET_TODAY_TASKS,
                confidence=0.9,
                action=AgentActionName.LIST_TODAY_TASKS,
                reply="ok",
                slots={},
            )

    monkeypatch.setattr(agent_route, "AgentOrchestrator", FakeAgentOrchestrator)
    app = create_app(Settings(environment="production", debug_routes_enabled=False))
    client = TestClient(app)

    agent_response = client.post("/api/agent/message", json={"message": "今天任务是什么？"})
    debug_response = client.post("/api/debug/agent", json={"message": "今天任务是什么？"})

    assert agent_response.status_code == 200
    assert debug_response.status_code == 404
