from fastapi.testclient import TestClient

from app.api.routes import feishu
from app.core.config import Settings
from app.main import create_app
from app.schemas.agent import AgentActionName, AgentResponse
from app.schemas.intent import IntentName
from app.services.feishu_service import FeishuMessageResult, FeishuRequestError


def _disable_feishu_token(monkeypatch) -> None:
    monkeypatch.setattr(
        feishu,
        "settings",
        Settings(feishu_verification_token=""),
    )


def test_feishu_event_returns_challenge(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    app = create_app(Settings(debug_routes_enabled=False))
    client = TestClient(app)

    response = client.post(
        "/api/feishu/events",
        json={
            "type": "url_verification",
            "challenge": "challenge_token",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"challenge": "challenge_token"}


def test_feishu_event_extracts_text_and_calls_agent(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)

    class FakeAgentOrchestrator:
        async def handle_message(self, message, confirmed=False, user_id="local_user", source="api"):
            assert message == "今天任务是什么？"
            assert confirmed is False
            assert user_id == "ou_test"
            assert source == "feishu"
            return AgentResponse(
                intent=IntentName.GET_TODAY_TASKS,
                confidence=0.9,
                action=AgentActionName.LIST_TODAY_TASKS,
                reply="今天的任务：1. LeetCode 206. 反转链表",
                slots={},
            )

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def send_text_message(self, receive_id, text):
            assert receive_id == "ou_test"
            assert text == "今天的任务：1. LeetCode 206. 反转链表"
            return FeishuMessageResult(
                message_id="om_test",
                raw_response={"code": 0, "data": {"message_id": "om_test"}},
            )

    monkeypatch.setattr(feishu, "AgentOrchestrator", FakeAgentOrchestrator)
    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    app = create_app(Settings(debug_routes_enabled=False))
    client = TestClient(app)

    response = client.post(
        "/api/feishu/events",
        json={
            "schema": "2.0",
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou_test"}},
                "message": {
                    "message_type": "text",
                    "content": "{\"text\":\"今天任务是什么？\"}",
                },
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "handled": True,
        "event_type": "im.message.receive_v1",
        "message": "今天任务是什么？",
        "user_id": "ou_test",
        "agent_response": {
            "intent": "get_today_tasks",
            "confidence": 0.9,
            "action": "list_today_tasks",
            "reply": "今天的任务：1. LeetCode 206. 反转链表",
            "need_confirmation": False,
            "slots": {},
            "missing_slots": [],
        },
        "reply_sent": True,
        "reply_message_id": "om_test",
    }


def test_feishu_event_ignores_duplicate_event_id(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    feishu._processed_event_ids.clear()
    calls = {"agent": 0, "send": 0}

    class FakeAgentOrchestrator:
        async def handle_message(self, message, confirmed=False, user_id="local_user", source="api"):
            calls["agent"] += 1
            return AgentResponse(
                intent=IntentName.GET_TODAY_TASKS,
                confidence=0.9,
                action=AgentActionName.LIST_TODAY_TASKS,
                reply="ok",
                slots={},
            )

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def send_text_message(self, receive_id, text):
            calls["send"] += 1
            return FeishuMessageResult(
                message_id="om_test",
                raw_response={"code": 0, "data": {"message_id": "om_test"}},
            )

    monkeypatch.setattr(feishu, "AgentOrchestrator", FakeAgentOrchestrator)
    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    app = create_app(Settings(debug_routes_enabled=False))
    client = TestClient(app)
    payload = {
        "schema": "2.0",
        "header": {
            "event_type": "im.message.receive_v1",
            "event_id": "event_test_1",
        },
        "event": {
            "sender": {"sender_id": {"open_id": "ou_test"}},
            "message": {
                "message_type": "text",
                "content": {"text": "今天任务是什么？"},
            },
        },
    }

    first = client.post("/api/feishu/events", json=payload)
    second = client.post("/api/feishu/events", json=payload)

    assert first.status_code == 200
    assert first.json()["handled"] is True
    assert second.status_code == 200
    assert second.json() == {
        "handled": False,
        "event_type": "im.message.receive_v1",
        "message": "重复事件已忽略。",
    }
    assert calls == {"agent": 1, "send": 1}


def test_feishu_event_ignores_non_text_message(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    app = create_app(Settings(debug_routes_enabled=False))
    client = TestClient(app)

    response = client.post(
        "/api/feishu/events",
        json={
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou_test"}},
                "message": {
                    "message_type": "image",
                    "content": "{}",
                },
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "handled": False,
        "event_type": "im.message.receive_v1",
        "message": "当前只处理文本消息事件。",
        "user_id": "ou_test",
    }


def test_feishu_event_route_stays_enabled_when_debug_routes_disabled(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)

    class FakeAgentOrchestrator:
        async def handle_message(self, message, confirmed=False, user_id="local_user", source="api"):
            return AgentResponse(
                intent=IntentName.GET_TODAY_TASKS,
                confidence=0.9,
                action=AgentActionName.LIST_TODAY_TASKS,
                reply="ok",
                slots={},
            )

    class FakeFeishuMessageService:
        def is_configured(self):
            return False

    monkeypatch.setattr(feishu, "AgentOrchestrator", FakeAgentOrchestrator)
    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    app = create_app(Settings(environment="production", debug_routes_enabled=False))
    client = TestClient(app)

    feishu_response = client.post(
        "/api/feishu/events",
        json={
            "event": {
                "message": {
                    "message_type": "text",
                    "content": {"text": "今天任务是什么？"},
                }
            }
        },
    )
    debug_response = client.post("/api/debug/agent", json={"message": "今天任务是什么？"})

    assert feishu_response.status_code == 200
    assert debug_response.status_code == 404


def test_feishu_event_accepts_valid_top_level_token(monkeypatch) -> None:
    monkeypatch.setattr(
        feishu,
        "settings",
        Settings(feishu_verification_token="verify_token"),
    )
    app = create_app(Settings(debug_routes_enabled=False))
    client = TestClient(app)

    response = client.post(
        "/api/feishu/events",
        json={
            "type": "url_verification",
            "token": "verify_token",
            "challenge": "challenge_token",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"challenge": "challenge_token"}


def test_feishu_event_accepts_valid_header_token(monkeypatch) -> None:
    class FakeAgentOrchestrator:
        async def handle_message(self, message, confirmed=False, user_id="local_user", source="api"):
            assert message == "今天任务是什么？"
            assert user_id == "ou_test"
            assert source == "feishu"
            return AgentResponse(
                intent=IntentName.GET_TODAY_TASKS,
                confidence=0.9,
                action=AgentActionName.LIST_TODAY_TASKS,
                reply="ok",
                slots={},
            )

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def send_text_message(self, receive_id, text):
            return FeishuMessageResult(
                message_id="om_test",
                raw_response={"code": 0, "data": {"message_id": "om_test"}},
            )

    monkeypatch.setattr(feishu, "AgentOrchestrator", FakeAgentOrchestrator)
    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    monkeypatch.setattr(
        feishu,
        "settings",
        Settings(feishu_verification_token="verify_token"),
    )
    app = create_app(Settings(debug_routes_enabled=False))
    client = TestClient(app)

    response = client.post(
        "/api/feishu/events",
        json={
            "schema": "2.0",
            "header": {
                "event_type": "im.message.receive_v1",
                "token": "verify_token",
            },
            "event": {
                "sender": {"sender_id": {"open_id": "ou_test"}},
                "message": {
                    "message_type": "text",
                    "content": {"text": "今天任务是什么？"},
                }
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["handled"] is True
    assert response.json()["reply_sent"] is True


def test_feishu_event_rejects_invalid_token(monkeypatch) -> None:
    monkeypatch.setattr(
        feishu,
        "settings",
        Settings(feishu_verification_token="verify_token"),
    )
    app = create_app(Settings(debug_routes_enabled=False))
    client = TestClient(app)

    response = client.post(
        "/api/feishu/events",
        json={
            "type": "url_verification",
            "token": "wrong_token",
            "challenge": "challenge_token",
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid Feishu verification token."


def test_feishu_event_rejects_missing_token_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(
        feishu,
        "settings",
        Settings(feishu_verification_token="verify_token"),
    )
    app = create_app(Settings(debug_routes_enabled=False))
    client = TestClient(app)

    response = client.post(
        "/api/feishu/events",
        json={
            "type": "url_verification",
            "challenge": "challenge_token",
        },
    )

    assert response.status_code == 403


def test_feishu_event_reports_unsent_reply_when_credentials_missing(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)

    class FakeAgentOrchestrator:
        async def handle_message(self, message, confirmed=False, user_id="local_user", source="api"):
            return AgentResponse(
                intent=IntentName.GET_TODAY_TASKS,
                confidence=0.9,
                action=AgentActionName.LIST_TODAY_TASKS,
                reply="ok",
                slots={},
            )

    class FakeFeishuMessageService:
        def is_configured(self):
            return False

    monkeypatch.setattr(feishu, "AgentOrchestrator", FakeAgentOrchestrator)
    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    app = create_app(Settings(debug_routes_enabled=False))
    client = TestClient(app)

    response = client.post(
        "/api/feishu/events",
        json={
            "event": {
                "sender": {"sender_id": {"open_id": "ou_test"}},
                "message": {
                    "message_type": "text",
                    "content": {"text": "今天任务是什么？"},
                },
            }
        },
    )

    assert response.status_code == 200
    assert response.json()["reply_sent"] is False
    assert "credentials" in response.json()["reply_error"]


def test_feishu_event_reports_reply_send_error(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)

    class FakeAgentOrchestrator:
        async def handle_message(self, message, confirmed=False, user_id="local_user", source="api"):
            return AgentResponse(
                intent=IntentName.GET_TODAY_TASKS,
                confidence=0.9,
                action=AgentActionName.LIST_TODAY_TASKS,
                reply="ok",
                slots={},
            )

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def send_text_message(self, receive_id, text):
            raise FeishuRequestError("send failed")

    monkeypatch.setattr(feishu, "AgentOrchestrator", FakeAgentOrchestrator)
    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    app = create_app(Settings(debug_routes_enabled=False))
    client = TestClient(app)

    response = client.post(
        "/api/feishu/events",
        json={
            "event": {
                "sender": {"sender_id": {"open_id": "ou_test"}},
                "message": {
                    "message_type": "text",
                    "content": {"text": "今天任务是什么？"},
                },
            }
        },
    )

    assert response.status_code == 200
    assert response.json()["reply_sent"] is False
    assert response.json()["reply_error"] == "send failed"
