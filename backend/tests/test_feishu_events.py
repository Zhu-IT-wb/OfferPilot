from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from app.api.routes import feishu
from app.core.config import Settings
from app.main import create_app
from app.models.leetcode import LeetCodePracticeResult
from app.repositories.leetcode_repository import InMemoryLeetCodeRepository
from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository
from app.schemas.agent import AgentActionName, AgentResponse
from app.schemas.intent import IntentName
from app.services.feishu_service import FeishuBitableRecordResult
from app.services.feishu_service import FeishuMessageResult, FeishuRequestError
from app.services.leetcode_catalog import load_hot100_snapshot
from app.services.leetcode_recommendation import LeetCodeRecommendationWorkflow


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


def test_feishu_today_leetcode_reply_is_an_interactive_card(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    repository = InMemoryLeetCodeRepository(problems=load_hot100_snapshot().problems)
    cards = []

    class FakeAgentOrchestrator:
        async def handle_message(self, message, confirmed=False, user_id="local_user", source="api"):
            return AgentResponse(
                intent=IntentName.GET_TODAY_LEETCODE,
                confidence=1.0,
                action=AgentActionName.GET_TODAY_LEETCODE,
                reply="text fallback",
                slots={},
            )

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def send_interactive_message(self, receive_id, card):
            cards.append((receive_id, card))
            return FeishuMessageResult(message_id="om_card", raw_response={"code": 0})

    monkeypatch.setattr(feishu, "AgentOrchestrator", FakeAgentOrchestrator)
    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    monkeypatch.setattr(feishu, "get_default_leetcode_repository", lambda: repository)
    client = TestClient(create_app(Settings(debug_routes_enabled=False)))

    response = client.post(
        "/api/feishu/events",
        json={
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou_card"}},
                "message": {"message_type": "text", "content": {"text": "今天刷什么"}},
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["reply_sent"] is True
    assert cards[0][0] == "ou_card"
    assert cards[0][1]["header"]["title"]["content"] == "🎯 今日 LeetCode · 3 题"


def test_feishu_card_button_records_leetcode_result(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    repository = InMemoryLeetCodeRepository(problems=load_hot100_snapshot().problems)
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    recommendation = LeetCodeRecommendationWorkflow(repository).get_today(
        "feishu:ou_card", today
    )[0]
    monkeypatch.setattr(feishu, "get_default_leetcode_repository", lambda: repository)
    client = TestClient(create_app(Settings(debug_routes_enabled=False)))

    response = client.post(
        "/api/feishu/events",
        json={
            "schema": "2.0",
            "header": {
                "event_type": "card.action.trigger",
                "event_id": "card_action_1",
            },
            "event": {
                "operator": {"operator_id": {"open_id": "ou_card"}},
                "action": {
                    "tag": "button",
                    "value": {
                        "action": "leetcode_result",
                        "assignment_id": recommendation.assignment.id,
                        "result": "independent",
                    },
                },
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["toast"]["type"] == "success"
    assert "独立完成" in response.json()["toast"]["content"]
    assignment = repository.list_assignments("feishu:ou_card", today)[0]
    assert assignment.result == LeetCodePracticeResult.INDEPENDENT
    progress = repository.get_progress("feishu:ou_card", recommendation.problem.id)
    assert progress is not None
    assert progress.next_review_on == today + timedelta(days=7)


def test_feishu_rejects_a_stale_card_after_problem_rolls_over(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    repository = InMemoryLeetCodeRepository(problems=load_hot100_snapshot().problems)
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    yesterday = today - timedelta(days=1)
    workflow = LeetCodeRecommendationWorkflow(repository)
    stale = workflow.get_today("feishu:ou_card", yesterday)[0]
    current = workflow.get_today("feishu:ou_card", today)
    monkeypatch.setattr(feishu, "get_default_leetcode_repository", lambda: repository)
    client = TestClient(create_app(Settings(debug_routes_enabled=False)))

    response = client.post(
        "/api/feishu/events",
        json={
            "header": {
                "event_type": "card.action.trigger",
                "event_id": "card_action_stale",
            },
            "event": {
                "operator": {"operator_id": {"open_id": "ou_card"}},
                "action": {
                    "tag": "button",
                    "value": {
                        "action": "leetcode_result",
                        "assignment_id": stale.assignment.id,
                        "result": "independent",
                    },
                },
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["toast"]["type"] == "error"
    assert stale.assignment.status.value == "skipped"
    carried = next(item for item in current if item.problem.id == stale.problem.id)
    assert carried.assignment.status.value == "pending"


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


def test_feishu_event_syncs_bitable_record_change(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="腾讯", role="Java 后端实习")
    repository.set_runtime_setting("feishu.offerpilot_bitable_app_token", "bascn_offerpilot")
    repository.set_runtime_setting("feishu.offerpilot_bitable_table_id", "tbl_applications")

    class FakeBitableService:
        app_token = ""
        table_id = ""

        def is_bitable_sync_enabled(self):
            return True

        def get_record(self, app_token, table_id, record_id):
            assert app_token == "bascn_offerpilot"
            assert table_id == "tbl_applications"
            assert record_id == "rec_app_1"
            return FeishuBitableRecordResult(
                record_id="rec_app_1",
                raw_response={"code": 0},
                fields={
                    "OfferPilot记录ID": application.id,
                    "公司": "腾讯",
                    "岗位": "AI 应用开发",
                    "投递状态": "二面阶段",
                    "面试轮次": "二面",
                },
            )

    monkeypatch.setattr(feishu, "get_default_offerpilot_repository", lambda: repository)
    monkeypatch.setattr(feishu, "FeishuBitableService", FakeBitableService)
    app = create_app(Settings(debug_routes_enabled=False))
    client = TestClient(app)

    response = client.post(
        "/api/feishu/events",
        json={
            "schema": "2.0",
            "header": {
                "event_type": "drive.file.bitable_record_changed_v1",
                "event_id": "event_bitable_1",
            },
            "event": {
                "app_token": "bascn_offerpilot",
                "table_id": "tbl_applications",
                "record_id": "rec_app_1",
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "handled": True,
        "event_type": "drive.file.bitable_record_changed_v1",
        "message": "多维表格记录已回写数据库：app_1",
    }
    assert repository.applications[0].role == "AI 应用开发"
    assert repository.applications[0].status.value == "interview_2"
    assert repository.applications[0].round == "二面"


def test_feishu_event_syncs_bitable_record_change_with_camel_case_payload(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    repository = InMemoryOfferPilotRepository()
    repository.create_application(company="小红书", role="Java 后端实习")
    repository.set_runtime_setting("feishu.offerpilot_bitable_app_token", "bascn_offerpilot")
    repository.set_runtime_setting("feishu.offerpilot_bitable_table_id", "tbl_applications")
    repository.set_runtime_setting(
        "feishu.offerpilot_bitable_record_id.tbl_applications.app_1",
        "recvoblCdubyrv",
    )

    class FakeBitableService:
        app_token = ""
        table_id = ""

        def is_bitable_sync_enabled(self):
            return True

        def get_record(self, app_token, table_id, record_id):
            assert app_token == "bascn_offerpilot"
            assert table_id == "tbl_applications"
            assert record_id == "recvoblCdubyrv"
            return FeishuBitableRecordResult(
                record_id="recvoblCdubyrv",
                raw_response={"code": 0},
                fields={
                    "公司": "小红书",
                    "岗位": "Java 后端开发实习",
                    "投递状态": "已投递",
                },
            )

    monkeypatch.setattr(feishu, "get_default_offerpilot_repository", lambda: repository)
    monkeypatch.setattr(feishu, "FeishuBitableService", FakeBitableService)
    app = create_app(Settings(debug_routes_enabled=False))
    client = TestClient(app)

    response = client.post(
        "/api/feishu/events",
        json={
            "schema": "2.0",
            "header": {
                "event_type": "drive.file.bitable_record_changed_v1",
                "event_id": "event_bitable_2",
            },
            "event": {
                "file_token": "bascn_offerpilot",
                "tableId": "tbl_applications",
                "actionList": [
                    {
                        "recordId": "recvoblCdubyrv",
                        "action": "record_updated",
                    }
                ],
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["handled"] is True
    assert repository.applications[0].role == "Java 后端开发实习"
    assert repository.applications[0].status.value == "submitted"


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
