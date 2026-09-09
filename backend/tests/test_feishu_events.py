import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Event
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from app.api.routes import feishu
from app.core.config import Settings
from app.main import create_app
from app.models.leetcode import LeetCodePracticeResult
from app.repositories.leetcode_repository import InMemoryLeetCodeRepository
from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository
from app.schemas.agent import (
    AgentArtifact,
    AgentInteraction,
    AgentInteractionType,
    AgentResumeDecision,
    AgentRunResponse,
    AgentRunStatus,
)
from app.services.feishu_service import FeishuBitableRecordResult
from app.services.feishu_service import FeishuMessageResult, FeishuRequestError
from app.services.agent_runtime_store import (
    InMemoryAgentRuntimeStore,
    SQLiteAgentRuntimeStore,
)
from app.services.bitable_sync_service import remember_bitable_record_mapping
from app.services.bitable_tenancy import save_owner_bitable_resource
from app.services.leetcode_catalog import load_hot100_snapshot
from app.services.leetcode_recommendation import LeetCodeRecommendationWorkflow


def _disable_feishu_token(monkeypatch) -> None:
    monkeypatch.setattr(
        feishu,
        "settings",
        Settings(
            feishu_verification_token="",
            feishu_allow_unverified_events=True,
        ),
    )


def _agent_run_response(
    *,
    reply: str = "ok",
    status: AgentRunStatus = AgentRunStatus.COMPLETED,
    interaction: AgentInteraction | None = None,
    artifacts: list[AgentArtifact] | None = None,
) -> AgentRunResponse:
    return AgentRunResponse(
        thread_id="thread_feishu",
        run_id="run_feishu",
        status=status,
        reply=reply,
        interaction=interaction,
        artifacts=artifacts or [],
    )


class FakeAgentRuntime:
    def __init__(
        self, response=None, state=None, replay_response=None, runtime_store=None
    ) -> None:
        self.response = response or _agent_run_response()
        self.state = state
        self.replay_response = replay_response
        self.calls = []
        self.replay_calls = []
        self.runtime_store = runtime_store or InMemoryAgentRuntimeStore()

    @staticmethod
    def thread_id_for(user_id, source, conversation_scope=None):
        return f"thread:{source}:{conversation_scope or user_id}"

    async def get_state(self, **kwargs):
        self.calls.append(("get_state", kwargs))
        if self.state is None:
            raise feishu.AgentThreadNotFound(kwargs["thread_id"])
        return self.state

    async def replay_request(self, **kwargs):
        self.replay_calls.append(kwargs)
        return self.replay_response

    async def start(self, **kwargs):
        self.calls.append(("start", kwargs))
        return self.response

    async def resume(self, **kwargs):
        self.calls.append(("resume", kwargs))
        return self.response

    async def resume_message(self, **kwargs):
        self.calls.append(("resume_message", kwargs))
        return self.response


def _app_with_runtime(runtime=None, **settings_kwargs):
    settings_kwargs.setdefault("feishu_verification_token", "")
    settings_kwargs.setdefault("feishu_allow_unverified_events", True)
    app = create_app(Settings(debug_routes_enabled=False, **settings_kwargs))
    app.state.agent_runtime = runtime or FakeAgentRuntime()
    return app


def test_feishu_event_returns_challenge(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    app = create_app(
        Settings(
            debug_routes_enabled=False,
            feishu_verification_token="",
            feishu_allow_unverified_events=True,
        )
    )
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


def test_feishu_conversation_scope_isolated_by_chat_type() -> None:
    p2p_payload = {
        "header": {"tenant_key": "tenant-a"},
        "event": {
            "sender": {"sender_id": {"open_id": "ou-a"}},
            "message": {"chat_type": "p2p", "chat_id": "chat-a", "message_id": "om-a"},
        },
    }
    group_root_payload = {
        "header": {"tenant_key": "tenant-a"},
        "event": {
            "message": {"chat_type": "group", "chat_id": "chat-g", "message_id": "om-root"},
        },
    }
    group_reply_payload = {
        "header": {"tenant_key": "tenant-a"},
        "event": {
            "message": {
                "chat_type": "group",
                "chat_id": "chat-g",
                "message_id": "om-child",
                "root_id": "om-root",
            },
        },
    }

    assert feishu._extract_conversation_scope(p2p_payload) == "tenant-a:chat-a:ou-a"
    assert feishu._extract_conversation_scope(group_root_payload) == "tenant-a:chat-g:om-root"
    assert feishu._extract_conversation_scope(group_reply_payload) == "tenant-a:chat-g:om-root"


def test_feishu_agent_identity_requires_open_id() -> None:
    payload = {
        "event": {
            "sender": {
                "sender_id": {
                    "union_id": "on-not-an-open-id",
                    "user_id": "user-not-an-open-id",
                }
            }
        }
    }

    assert feishu._extract_user_id(payload) is None


def test_feishu_event_starts_shared_runtime_and_replies_to_source_message(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    runtime = FakeAgentRuntime(
        response=_agent_run_response(reply="今天的任务：1. LeetCode 206. 反转链表")
    )
    sent = []

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def reply_text_message(self, **kwargs):
            sent.append(kwargs)
            return FeishuMessageResult(message_id="om_reply", raw_response={"code": 0})

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    app = _app_with_runtime(runtime)
    client = TestClient(app)
    response = client.post(
        "/api/feishu/events",
        json={
            "header": {
                "event_type": "im.message.receive_v1",
                "event_id": "event-start-1",
                "tenant_key": "tenant-a",
            },
            "event": {
                "sender": {"sender_id": {"open_id": "ou-test"}},
                "message": {
                    "chat_type": "p2p",
                    "chat_id": "chat-a",
                    "message_id": "om-source",
                    "message_type": "text",
                    "content": {"text": "今天任务是什么？"},
                },
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["agent_response"]["status"] == "completed"
    assert runtime.calls[0] == (
        "get_state",
        {
            "thread_id": "thread:feishu:tenant-a:chat-a:ou-test",
            "user_id": "ou-test",
            "source": "feishu",
        },
    )
    assert runtime.calls[1][0] == "start"
    assert runtime.calls[1][1]["conversation_scope"] == "tenant-a:chat-a:ou-test"
    assert runtime.calls[1][1]["request_id"] == "event-start-1"
    assert sent[0]["message_id"] == "om-source"
    assert sent[0]["text"] == "今天的任务：1. LeetCode 206. 反转链表"
    assert sent[0]["idempotency_key"] == feishu._stable_reply_uuid(
        "event-start-1", {}, "text"
    )
    assert sent[0]["reply_in_thread"] is False


def test_feishu_progress_card_precedes_blocked_runtime_and_updates_in_group_thread(
    monkeypatch,
) -> None:
    _disable_feishu_token(monkeypatch)
    runtime_started = Event()
    release_runtime = Event()
    message_calls = []

    class BlockingRuntime(FakeAgentRuntime):
        async def start(self, **kwargs):
            self.calls.append(("start", kwargs))
            runtime_started.set()
            await asyncio.to_thread(release_runtime.wait, 5)
            return _agent_run_response(reply="投递记录已整理完成。")

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def reply_interactive_message(self, **kwargs):
            message_calls.append(("progress", kwargs))
            return FeishuMessageResult(message_id="om-progress", raw_response={"code": 0})

        async def update_interactive_message(self, **kwargs):
            message_calls.append(("update", kwargs))
            return FeishuMessageResult(message_id="om-progress", raw_response={"code": 0})

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    client = TestClient(_app_with_runtime(BlockingRuntime()))
    payload = {
        "header": {"event_id": "event-progress", "tenant_key": "tenant-a"},
        "event": {
            "sender": {"sender_id": {"open_id": "ou-owner"}},
            "message": {
                "chat_type": "group",
                "chat_id": "chat-group",
                "message_id": "om-root",
                "message_type": "text",
                "content": {"text": "整理我的投递记录"},
            },
        },
    }

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(client.post, "/api/feishu/events", json=payload)
            assert runtime_started.wait(5)
            assert len(message_calls) == 1
            progress_kwargs = message_calls[0][1]
            assert progress_kwargs["message_id"] == "om-root"
            assert progress_kwargs["reply_in_thread"] is True
            assert progress_kwargs["card"]["config"] == {
                "wide_screen_mode": True,
                "update_multi": True,
            }
            progress_text = str(progress_kwargs["card"])
            assert "正在处理你的请求" in progress_text
            assert "tool" not in progress_text.lower()
            assert "参数" not in progress_text
            release_runtime.set()
            response = pending.result(timeout=5)
    finally:
        release_runtime.set()

    assert response.status_code == 200
    assert [kind for kind, _ in message_calls] == ["progress", "update"]
    assert message_calls[1][1]["message_id"] == "om-progress"
    assert "投递记录已整理完成" in str(message_calls[1][1]["card"])
    assert message_calls[1][1]["card"]["config"]["update_multi"] is True
    assert response.json()["reply_message_id"] == "om-progress"


def test_feishu_progress_update_failure_sends_stable_separate_final_reply(
    monkeypatch,
) -> None:
    _disable_feishu_token(monkeypatch)
    reply_calls = []
    runtime = FakeAgentRuntime(response=_agent_run_response(reply="任务已完成。"))

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def reply_interactive_message(self, **kwargs):
            reply_calls.append(kwargs)
            message_id = "om-progress" if len(reply_calls) == 1 else "om-final"
            return FeishuMessageResult(message_id=message_id, raw_response={"code": 0})

        async def update_interactive_message(self, **kwargs):
            raise FeishuRequestError("missing im:message:update")

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    payload = {
        "header": {"event_id": "event-update-fallback"},
        "event": {
            "sender": {"sender_id": {"open_id": "ou-test"}},
            "message": {
                "chat_type": "p2p",
                "chat_id": "chat-test",
                "message_id": "om-source",
                "message_type": "text",
                "content": {"text": "查询任务"},
            },
        },
    }

    response = TestClient(_app_with_runtime(runtime)).post(
        "/api/feishu/events",
        json=payload,
    )

    assert response.status_code == 200
    assert len(reply_calls) == 2
    assert reply_calls[0]["idempotency_key"] == feishu._stable_reply_uuid(
        "event-update-fallback", payload, "progress"
    )
    assert reply_calls[1]["idempotency_key"] == feishu._stable_reply_uuid(
        "event-update-fallback", payload, "interactive"
    )
    assert reply_calls[0]["idempotency_key"] != reply_calls[1]["idempotency_key"]
    assert "任务已完成" in str(reply_calls[1]["card"])
    assert reply_calls[1]["card"]["config"]["update_multi"] is True


def test_feishu_progress_failure_does_not_block_runtime(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    runtime = FakeAgentRuntime(response=_agent_run_response(reply="仍然完成了请求。"))
    final_replies = []

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def reply_interactive_message(self, **kwargs):
            raise FeishuRequestError("progress unavailable")

        async def update_interactive_message(self, **kwargs):
            raise AssertionError("no progress message exists")

        async def reply_text_message(self, **kwargs):
            final_replies.append(kwargs)
            return FeishuMessageResult(message_id="om-final", raw_response={"code": 0})

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    response = TestClient(_app_with_runtime(runtime)).post(
        "/api/feishu/events",
        json={
            "header": {"event_id": "event-progress-failed"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou-test"}},
                "message": {
                    "message_id": "om-source",
                    "message_type": "text",
                    "content": {"text": "查询任务"},
                },
            },
        },
    )

    assert response.status_code == 200
    assert any(call[0] == "start" for call in runtime.calls)
    assert final_replies[0]["text"] == "仍然完成了请求。"


def test_feishu_runtime_exception_replaces_progress_with_friendly_failure(
    monkeypatch,
) -> None:
    _disable_feishu_token(monkeypatch)
    updated_cards = []

    class FailingRuntime(FakeAgentRuntime):
        async def start(self, **kwargs):
            raise RuntimeError("private provider trace")

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def reply_interactive_message(self, **kwargs):
            return FeishuMessageResult(message_id="om-progress", raw_response={"code": 0})

        async def update_interactive_message(self, **kwargs):
            updated_cards.append(kwargs["card"])
            return FeishuMessageResult(message_id="om-progress", raw_response={"code": 0})

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    response = TestClient(_app_with_runtime(FailingRuntime())).post(
        "/api/feishu/events",
        json={
            "header": {"event_id": "event-runtime-failed"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou-test"}},
                "message": {
                    "message_id": "om-source",
                    "message_type": "text",
                    "content": {"text": "执行任务"},
                },
            },
        },
    )

    assert response.status_code == 200
    body = response.json()["agent_response"]
    assert body["status"] == "failed"
    assert "请稍后重试" in body["reply"]
    assert "private provider trace" not in str(response.json())
    assert "请稍后重试" in str(updated_cards[0])
    assert "private provider trace" not in str(updated_cards[0])
    assert updated_cards[0]["config"]["update_multi"] is True


def test_feishu_progress_is_replaced_by_explicit_artifact_card(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    final_card = {
        "header": {"title": {"tag": "plain_text", "content": "本周复习计划"}},
        "elements": [],
    }
    runtime = FakeAgentRuntime(
        response=_agent_run_response(
            reply="计划已生成。",
            artifacts=[AgentArtifact(type="study_plan", data={"feishu_card": final_card})],
        )
    )
    updated_cards = []

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def reply_interactive_message(self, **kwargs):
            return FeishuMessageResult(message_id="om-progress", raw_response={"code": 0})

        async def update_interactive_message(self, **kwargs):
            updated_cards.append(kwargs["card"])
            return FeishuMessageResult(message_id="om-progress", raw_response={"code": 0})

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    response = TestClient(_app_with_runtime(runtime)).post(
        "/api/feishu/events",
        json={
            "header": {"event_id": "event-artifact-update"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou-test"}},
                "message": {
                    "message_id": "om-source",
                    "message_type": "text",
                    "content": {"text": "生成复习计划"},
                },
            },
        },
    )

    assert response.status_code == 200
    assert updated_cards[0]["header"] == final_card["header"]
    assert updated_cards[0]["elements"] == final_card["elements"]
    assert updated_cards[0]["config"] == {
        "wide_screen_mode": True,
        "update_multi": True,
    }
    assert "config" not in final_card


def test_feishu_reply_boundary_redacts_internal_tool_names(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    runtime = FakeAgentRuntime(
        response=_agent_run_response(
            reply="准备执行 create_application，是否确认？"
        )
    )
    sent = []

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def reply_text_message(self, **kwargs):
            sent.append(kwargs)
            return FeishuMessageResult(message_id="om_reply", raw_response={"code": 0})

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    response = TestClient(_app_with_runtime(runtime)).post(
        "/api/feishu/events",
        json={
            "header": {"event_id": "event-redaction", "tenant_key": "tenant-a"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou-test"}},
                "message": {
                    "chat_type": "p2p",
                    "chat_id": "chat-a",
                    "message_id": "om-source",
                    "message_type": "text",
                    "content": {"text": "记录一条投递"},
                },
            },
        },
    )

    assert response.status_code == 200
    assert sent[0]["text"] == "准备执行 新增投递记录，是否确认？"
    assert "create_application" not in sent[0]["text"]


def test_feishu_first_group_reply_starts_the_stable_message_thread(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    runtime = FakeAgentRuntime(response=_agent_run_response(reply="请确认同步。"))
    sent = []

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def reply_text_message(self, **kwargs):
            sent.append(kwargs)
            return FeishuMessageResult(message_id="om_reply", raw_response={"code": 0})

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    response = TestClient(_app_with_runtime(runtime)).post(
        "/api/feishu/events",
        json={
            "header": {"event_id": "event-group", "tenant_key": "tenant-a"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou-owner"}},
                "message": {
                    "chat_type": "group",
                    "chat_id": "chat-group",
                    "message_id": "om-root",
                    "message_type": "text",
                    "content": {"text": "生成复习计划"},
                },
            },
        },
    )

    assert response.status_code == 200
    assert runtime.calls[1][1]["conversation_scope"] == "tenant-a:chat-group:om-root"
    assert sent[0]["message_id"] == "om-root"
    assert sent[0]["reply_in_thread"] is True


def test_feishu_text_without_sender_identity_never_accesses_runtime(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    runtime = FakeAgentRuntime()
    response = TestClient(_app_with_runtime(runtime)).post(
        "/api/feishu/events",
        json={
            "header": {"event_type": "im.message.receive_v1", "event_id": "no-user"},
            "event": {
                "message": {
                    "chat_type": "group",
                    "chat_id": "chat-no-user",
                    "message_id": "om-no-user",
                    "message_type": "text",
                    "content": {"text": "查询我的面试"},
                }
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["handled"] is False
    assert runtime.calls == []
    assert "未执行任何用户数据操作" in response.json()["message"]


def test_feishu_actor_conflict_returns_readable_degraded_response(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)

    class ActorConflictRuntime(FakeAgentRuntime):
        async def start(self, **kwargs):
            raise feishu.AgentActorMismatch("different actor")

    class FakeFeishuMessageService:
        def is_configured(self):
            return False

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    response = TestClient(_app_with_runtime(ActorConflictRuntime())).post(
        "/api/feishu/events",
        json={
            "header": {"event_type": "im.message.receive_v1", "event_id": "actor-conflict"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou-other"}},
                "message": {
                    "chat_type": "group",
                    "chat_id": "chat-shared",
                    "message_id": "om-conflict",
                    "message_type": "text",
                    "content": {"text": "继续同步"},
                },
            },
        },
    )

    assert response.status_code == 200
    body = response.json()["agent_response"]
    assert body["status"] == "degraded"
    assert body["warnings"] == ["当前会话身份不匹配"]
    assert "发起人" in body["reply"]


def test_feishu_artifact_drives_interactive_reply(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    card = {
        "header": {"title": {"tag": "plain_text", "content": "本周复习计划"}},
        "elements": [],
    }
    runtime = FakeAgentRuntime(
        response=_agent_run_response(
            reply="计划已生成。",
            artifacts=[
                AgentArtifact(type="study_plan", data={"feishu_card": card})
            ],
        )
    )
    sent = []

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def reply_interactive_message(self, **kwargs):
            sent.append(kwargs)
            return FeishuMessageResult(message_id="om-card", raw_response={"code": 0})

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    client = TestClient(_app_with_runtime(runtime))
    response = client.post(
        "/api/feishu/events",
        json={
            "header": {"event_id": "event-card"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou-card"}},
                "message": {
                    "message_id": "om-source",
                    "message_type": "text",
                    "content": {"text": "生成复习计划"},
                },
            },
        },
    )

    assert response.status_code == 200
    assert sent[0]["card"]["header"] == card["header"]
    assert sent[0]["card"]["elements"] == card["elements"]
    assert sent[0]["card"]["config"]["update_multi"] is True
    assert sent[0]["message_id"] == "om-source"


def test_feishu_study_plan_artifact_builds_summary_card() -> None:
    card = feishu._build_artifact_card(
        artifacts=[
            AgentArtifact(
                type="study_plan",
                title="后端薄弱点复习",
                data={
                    "sessions": [
                        {
                            "topic": "JVM GC",
                            "start_at": "2026-09-03T19:00:00+08:00",
                            "duration_minutes": 45,
                        }
                    ],
                    "unscheduled": [{"topic": "Redis"}],
                },
            )
        ],
        fallback_text="已根据面试薄弱点生成计划。",
        user_id="ou-test",
    )

    assert card is not None
    assert card["config"]["update_multi"] is True
    assert card["header"]["title"]["content"] == "后端薄弱点复习"
    assert "JVM GC" in card["elements"][0]["content"]
    assert "1 项暂未排入日历" in card["elements"][0]["content"]


def test_feishu_bitable_artifact_builds_the_application_table_button() -> None:
    table_url = "https://example.feishu.cn/base/app_table?table=applications"
    card = feishu._build_artifact_card(
        artifacts=[
            AgentArtifact(
                type="feishu_bitable",
                id="applications_table",
                title="投递记录",
                url=table_url,
                data={"button_text": "打开多维表格"},
            )
        ],
        fallback_text="投递记录已经同步。",
        user_id="ou-test",
    )

    assert card is not None
    assert card["config"]["update_multi"] is True
    assert card["header"]["title"]["content"] == "投递记录"
    actions = card["elements"][-1]["actions"]
    assert actions == [
        {
            "tag": "button",
            "type": "primary",
            "text": {"tag": "plain_text", "content": "打开多维表格"},
            "url": table_url,
        }
    ]


def test_feishu_project_candidates_are_rendered_from_artifact_data() -> None:
    card = feishu._build_artifact_card(
        artifacts=[
            AgentArtifact(
                type="project_selection",
                title="选择训练项目",
                data={
                    "launch_url": "https://offerpilot.example.com/study/projects",
                    "candidates": [
                        {"id": "project/order", "name": "订单系统"},
                        {"id": "project_agent", "name": "AI Agent"},
                    ],
                },
            )
        ],
        fallback_text="请选择一个项目。",
        user_id="ou-test",
    )

    assert card is not None
    assert card["config"]["update_multi"] is True
    actions = card["elements"][-1]["actions"]
    assert [item["text"]["content"] for item in actions] == ["订单系统", "AI Agent"]
    assert actions[0]["url"].endswith("start_project_id=project%2Forder")


@pytest.mark.parametrize(
    "current_kind", ["executing", "completed", "replaced", "new_run", "missing"]
)
def test_feishu_old_waiting_receipt_never_redisplays_resolved_or_replaced_confirmation(
    monkeypatch, current_kind,
) -> None:
    _disable_feishu_token(monkeypatch)
    interaction = AgentInteraction(
        id="interaction-create",
        type=AgentInteractionType.APPROVAL,
        prompt="请确认新增华为软件开发岗位投递。",
        allowed_actions=[AgentResumeDecision.APPROVE, AgentResumeDecision.CANCEL],
    )
    original_response = _agent_run_response(
        reply=interaction.prompt,
        status=AgentRunStatus.WAITING_FOR_INPUT,
        interaction=interaction,
    )
    current = None
    if current_kind == "executing":
        current = _agent_run_response(
            reply="确认正在执行。", status=AgentRunStatus.PARTIAL
        )
    elif current_kind == "completed":
        current = _agent_run_response(reply="投递记录已创建。")
    elif current_kind == "replaced":
        current = _agent_run_response(
            reply="请确认更新后的工作地点。",
            status=AgentRunStatus.WAITING_FOR_INPUT,
            interaction=interaction.model_copy(update={"id": "interaction-new"}),
        )
    elif current_kind == "new_run":
        current = original_response.model_copy(update={"run_id": "run_new"})
    runtime = FakeAgentRuntime(state=current, replay_response=original_response)
    message_calls = []

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def reply_text_message(self, **kwargs):
            message_calls.append(("text", kwargs))
            return FeishuMessageResult(
                message_id="om-confirm-final", raw_response={"code": 0}
            )

        async def reply_interactive_message(self, **kwargs):
            message_calls.append(("progress", kwargs))
            return FeishuMessageResult(
                message_id="om-initial-progress",
                raw_response={"code": 0},
            )

        async def update_interactive_message(self, **kwargs):
            message_calls.append(("update", kwargs))
            return FeishuMessageResult(
                message_id="om-initial-progress",
                raw_response={"code": 0},
            )

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    client = TestClient(_app_with_runtime(runtime))
    payload = {
        "header": {"event_id": "event-create-initial", "tenant_key": "tenant-a"},
        "event": {
            "sender": {"sender_id": {"open_id": "ou-test"}},
            "message": {
                "chat_type": "p2p",
                "chat_id": "chat-test",
                "message_id": "om-create-initial",
                "message_type": "text",
                "content": {"text": "我投递了华为的软件开发岗位"},
            },
        },
    }

    first_retry = client.post("/api/feishu/events", json=payload)
    later_retry = client.post("/api/feishu/events", json=payload)

    assert first_retry.status_code == 200
    assert later_retry.status_code == 200
    assert first_retry.json()["reply_sent"] is False
    assert later_retry.json()["reply_sent"] is False
    assert not first_retry.json().get("agent_response", {}).get("interaction")
    assert message_calls == []
    assert all(kind == "get_state" for kind, _ in runtime.calls)
    assert runtime.replay_calls == [
        {
            "thread_id": "thread:feishu:tenant-a:chat-test:ou-test",
            "user_id": "ou-test",
            "source": "feishu",
            "request_id": "event-create-initial",
        }
    ]


def test_feishu_duplicate_confirmation_replays_success_without_conflict_card(
    monkeypatch,
) -> None:
    _disable_feishu_token(monkeypatch)
    completed = _agent_run_response(reply="华为软件开发岗位已记录并同步到多维表格。")

    class ReplayedConfirmationRuntime(FakeAgentRuntime):
        async def start(self, **kwargs):
            raise feishu.AgentRuntimeConflict("duplicate was routed as a new task")

    runtime = ReplayedConfirmationRuntime(
        state=_agent_run_response(reply="华为软件开发岗位已记录。"),
        replay_response=completed,
    )
    message_calls = []

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def reply_text_message(self, **kwargs):
            message_calls.append(("text", kwargs))
            return FeishuMessageResult(
                message_id="om-confirm-final", raw_response={"code": 0}
            )

        async def reply_interactive_message(self, **kwargs):
            message_calls.append(("progress", kwargs))
            return FeishuMessageResult(
                message_id="om-confirm-progress",
                raw_response={"code": 0},
            )

        async def update_interactive_message(self, **kwargs):
            message_calls.append(("update", kwargs))
            return FeishuMessageResult(
                message_id="om-confirm-progress",
                raw_response={"code": 0},
            )

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    client = TestClient(_app_with_runtime(runtime))
    payload = {
        "header": {"event_id": "event-confirm", "tenant_key": "tenant-a"},
        "event": {
            "sender": {"sender_id": {"open_id": "ou-test"}},
            "message": {
                "chat_type": "p2p",
                "chat_id": "chat-test",
                "message_id": "om-confirm",
                "message_type": "text",
                "content": {"text": "嗯嗯"},
            },
        },
    }

    first_retry = client.post("/api/feishu/events", json=payload)
    later_retry = client.post("/api/feishu/events", json=payload)

    assert first_retry.status_code == 200
    assert later_retry.status_code == 200
    assert first_retry.json()["agent_response"]["status"] == "completed"
    assert later_retry.json()["agent_response"] == first_retry.json()["agent_response"]
    assert runtime.calls == []
    assert len(runtime.replay_calls) == 1
    assert [kind for kind, _ in message_calls] == ["text"]
    assert message_calls[0][1]["idempotency_key"] == feishu._stable_reply_uuid(
        "event-confirm", {}, "text"
    )
    assert "已记录并同步到多维表格" in message_calls[0][1]["text"]
    assert "确认已失效" not in message_calls[0][1]["text"]


def _text_delivery_payload(event_id="event-delivery"):
    return {
        "header": {
            "event_type": "im.message.receive_v1",
            "event_id": event_id,
            "tenant_key": "tenant-a",
        },
        "event": {
            "sender": {"sender_id": {"open_id": "ou-test"}},
            "message": {
                "chat_type": "p2p",
                "chat_id": "chat-test",
                "message_id": "om-source",
                "message_type": "text",
                "content": {"text": "记录我的投递"},
            },
        },
    }


def test_feishu_legacy_waiting_receipt_can_deliver_the_current_confirmation_once(
    monkeypatch,
) -> None:
    _disable_feishu_token(monkeypatch)
    interaction = AgentInteraction(
        id="interaction-still-current",
        type=AgentInteractionType.APPROVAL,
        prompt="请确认新增投递记录。",
        allowed_actions=[AgentResumeDecision.APPROVE, AgentResumeDecision.CANCEL],
    )
    pending = _agent_run_response(
        reply=interaction.prompt,
        status=AgentRunStatus.WAITING_FOR_INPUT,
        interaction=interaction,
    )
    runtime = FakeAgentRuntime(state=pending, replay_response=pending)
    sent = []

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def reply_text_message(self, **kwargs):
            sent.append(kwargs)
            return FeishuMessageResult(message_id="om-confirm", raw_response={"code": 0})

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    client = TestClient(_app_with_runtime(runtime))
    payload = _text_delivery_payload("event-legacy-current")
    first = client.post("/api/feishu/events", json=payload)
    duplicate = client.post("/api/feishu/events", json=payload)

    assert first.status_code == 200
    assert first.json()["agent_response"]["interaction"]["id"] == interaction.id
    assert first.json()["reply_sent"] is True
    assert duplicate.json() == first.json()
    assert len(sent) == 1
    assert sent[0]["text"] == interaction.prompt
    assert all(kind == "get_state" for kind, _ in runtime.calls)
    assert len(runtime.replay_calls) == 1


def test_feishu_delivered_approval_stays_deduplicated_after_restart(
    monkeypatch, tmp_path,
) -> None:
    _disable_feishu_token(monkeypatch)
    database_path = str(tmp_path / "delivery.db")
    interaction = AgentInteraction(
        id="interaction-delivered",
        type=AgentInteractionType.APPROVAL,
        prompt="请确认新增联想投递记录。",
        allowed_actions=[AgentResumeDecision.APPROVE, AgentResumeDecision.CANCEL],
    )
    pending_response = _agent_run_response(
        reply=interaction.prompt,
        status=AgentRunStatus.WAITING_FOR_INPUT,
        interaction=interaction,
    )
    runtime = FakeAgentRuntime(
        response=pending_response,
        runtime_store=SQLiteAgentRuntimeStore(database_path),
    )
    message_calls = []

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def reply_interactive_message(self, **kwargs):
            message_calls.append(("progress", kwargs))
            return FeishuMessageResult(message_id="om-progress", raw_response={"code": 0})

        async def update_interactive_message(self, **kwargs):
            message_calls.append(("update", kwargs))
            return FeishuMessageResult(message_id="om-progress", raw_response={"code": 0})

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    payload = _text_delivery_payload("event-approved-before-restart")
    first = TestClient(_app_with_runtime(runtime)).post("/api/feishu/events", json=payload)
    assert first.status_code == 200
    assert first.json()["agent_response"]["status"] == "waiting_for_input"

    restarted_runtime = FakeAgentRuntime(
        state=_agent_run_response(reply="联想投递已保存。"),
        replay_response=pending_response,
        runtime_store=SQLiteAgentRuntimeStore(database_path),
    )
    duplicate = TestClient(_app_with_runtime(restarted_runtime)).post(
        "/api/feishu/events", json=payload,
    )

    assert duplicate.status_code == 200
    assert duplicate.json() == first.json()
    assert [kind for kind, _ in message_calls] == ["progress", "update"]
    assert restarted_runtime.calls == []
    assert restarted_runtime.replay_calls == []


def test_feishu_recovered_confirmation_is_rechecked_immediately_before_delivery(
    monkeypatch,
) -> None:
    _disable_feishu_token(monkeypatch)
    interaction = AgentInteraction(
        id="interaction-racing-confirmation",
        type=AgentInteractionType.APPROVAL,
        prompt="请确认新增投递记录。",
        allowed_actions=[AgentResumeDecision.APPROVE, AgentResumeDecision.CANCEL],
    )
    pending = _agent_run_response(
        reply=interaction.prompt,
        status=AgentRunStatus.WAITING_FOR_INPUT,
        interaction=interaction,
    )

    class AdvancingRuntime(FakeAgentRuntime):
        async def get_state(self, **kwargs):
            self.calls.append(("get_state", kwargs))
            if len(self.calls) == 1:
                return pending
            return _agent_run_response(reply="投递记录已保存。")

    runtime = AdvancingRuntime(replay_response=pending)
    message_calls = []

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def reply_interactive_message(self, **kwargs):
            message_calls.append(("progress", kwargs))
            return FeishuMessageResult(message_id="om-progress", raw_response={"code": 0})

        async def reply_text_message(self, **kwargs):
            message_calls.append(("text", kwargs))
            return FeishuMessageResult(message_id="om-reply", raw_response={"code": 0})

        async def update_interactive_message(self, **kwargs):
            message_calls.append(("update", kwargs))
            return FeishuMessageResult(message_id="om-progress", raw_response={"code": 0})

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    client = TestClient(_app_with_runtime(runtime))
    payload = _text_delivery_payload("event-confirmed-between-checks")
    first = client.post("/api/feishu/events", json=payload)
    duplicate = client.post("/api/feishu/events", json=payload)

    assert first.status_code == 200
    assert first.json()["reply_sent"] is False
    assert not first.json().get("agent_response", {}).get("interaction")
    assert duplicate.json() == first.json()
    assert message_calls == []
    assert [kind for kind, _ in runtime.calls] == ["get_state", "get_state"]
    assert len(runtime.replay_calls) == 1


def test_feishu_confirmation_state_read_failure_can_retry_without_sending_stale_card(
    monkeypatch,
) -> None:
    _disable_feishu_token(monkeypatch)
    interaction = AgentInteraction(
        id="interaction-checkpoint-unavailable",
        type=AgentInteractionType.APPROVAL,
        prompt="请确认新增投递记录。",
        allowed_actions=[AgentResumeDecision.APPROVE, AgentResumeDecision.CANCEL],
    )
    pending = _agent_run_response(
        reply=interaction.prompt,
        status=AgentRunStatus.WAITING_FOR_INPUT,
        interaction=interaction,
    )

    class TemporarilyUnavailableStateRuntime(FakeAgentRuntime):
        async def get_state(self, **kwargs):
            self.calls.append(("get_state", kwargs))
            if len(self.calls) == 1:
                raise RuntimeError("checkpoint temporarily unavailable")
            return pending

    runtime = TemporarilyUnavailableStateRuntime(replay_response=pending)
    message_calls = []

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def reply_interactive_message(self, **kwargs):
            message_calls.append(("progress", kwargs))
            return FeishuMessageResult(message_id="om-progress", raw_response={"code": 0})

        async def reply_text_message(self, **kwargs):
            message_calls.append(("text", kwargs))
            return FeishuMessageResult(message_id="om-reply", raw_response={"code": 0})

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    client = TestClient(_app_with_runtime(runtime))
    payload = _text_delivery_payload("event-temporary-state-read-failure")
    first = client.post("/api/feishu/events", json=payload)

    assert first.status_code == 503
    assert message_calls == []

    retry = client.post("/api/feishu/events", json=payload)
    duplicate = client.post("/api/feishu/events", json=payload)

    assert retry.status_code == 200
    assert retry.json()["agent_response"]["interaction"]["id"] == interaction.id
    assert duplicate.json() == retry.json()
    assert [kind for kind, _ in message_calls] == ["text"]
    assert message_calls[0][1]["text"] == interaction.prompt
    assert all(kind == "get_state" for kind, _ in runtime.calls)


def test_feishu_failed_confirmation_delivery_does_not_resurrect_resolved_interaction(
    monkeypatch,
) -> None:
    _disable_feishu_token(monkeypatch)
    interaction = AgentInteraction(
        id="interaction-before-delivery-failure",
        type=AgentInteractionType.APPROVAL,
        prompt="请确认关闭每日刷题提醒。",
        allowed_actions=[AgentResumeDecision.APPROVE, AgentResumeDecision.CANCEL],
    )
    runtime = FakeAgentRuntime(
        response=_agent_run_response(
            reply=interaction.prompt,
            status=AgentRunStatus.WAITING_FOR_INPUT,
            interaction=interaction,
        )
    )
    message_calls = []

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def reply_interactive_message(self, **kwargs):
            if not message_calls:
                message_calls.append(("progress", kwargs))
                return FeishuMessageResult(message_id="om-progress", raw_response={"code": 0})
            message_calls.append(("fallback", kwargs))
            raise FeishuRequestError("temporary final delivery failure")

        async def update_interactive_message(self, **kwargs):
            message_calls.append(("update", kwargs))
            raise FeishuRequestError("temporary update failure")

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    client = TestClient(_app_with_runtime(runtime))
    payload = _text_delivery_payload("event-staged-obsolete-confirmation")
    first = client.post("/api/feishu/events", json=payload)
    assert first.status_code == 502
    assert [kind for kind, _ in message_calls] == ["progress", "update", "fallback"]
    runtime.state = _agent_run_response(reply="已关闭每日刷题提醒。")

    retry = client.post("/api/feishu/events", json=payload)

    assert retry.status_code == 200
    assert retry.json()["reply_sent"] is False
    assert not retry.json().get("agent_response", {}).get("interaction")
    assert [kind for kind, _ in message_calls] == ["progress", "update", "fallback", "update"]
    assert message_calls[-1][1]["message_id"] == "om-progress"
    assert "无需再次回复" in str(message_calls[-1][1]["card"])
    assert interaction.prompt not in str(message_calls[-1][1]["card"])
    assert len([call for call in runtime.calls if call[0] == "start"]) == 1


def test_feishu_active_duplicate_is_acknowledged_without_repeating_progress_or_work(
    monkeypatch,
) -> None:
    _disable_feishu_token(monkeypatch)
    runtime_started = Event()
    release_runtime = Event()
    message_calls = []

    class BlockingRuntime(FakeAgentRuntime):
        async def start(self, **kwargs):
            self.calls.append(("start", kwargs))
            runtime_started.set()
            await asyncio.to_thread(release_runtime.wait, 5)
            return _agent_run_response(reply="操作已完成。")

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def reply_interactive_message(self, **kwargs):
            message_calls.append(("progress", kwargs))
            return FeishuMessageResult(message_id="om-progress", raw_response={"code": 0})

        async def update_interactive_message(self, **kwargs):
            message_calls.append(("update", kwargs))
            return FeishuMessageResult(message_id="om-progress", raw_response={"code": 0})

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    runtime = BlockingRuntime()
    client = TestClient(_app_with_runtime(runtime))
    payload = _text_delivery_payload("event-concurrent-delivery")

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            original = pool.submit(client.post, "/api/feishu/events", json=payload)
            assert runtime_started.wait(5)
            retry = pool.submit(client.post, "/api/feishu/events", json=payload)
            try:
                duplicate = retry.result(timeout=2)
                assert duplicate.status_code == 200
                assert duplicate.json()["handled"] is True
                assert duplicate.json()["reply_sent"] is False
                assert [kind for kind, _ in message_calls] == ["progress"]
                assert len([call for call in runtime.calls if call[0] == "start"]) == 1
            finally:
                release_runtime.set()
            first = original.result(timeout=5)
    finally:
        release_runtime.set()

    assert first.status_code == 200
    assert [kind for kind, _ in message_calls] == ["progress", "update"]
    assert client.post("/api/feishu/events", json=payload).json() == first.json()
    assert len(message_calls) == 2


def test_feishu_delivery_retry_after_restart_reuses_progress_and_saved_result(
    monkeypatch, tmp_path,
) -> None:
    _disable_feishu_token(monkeypatch)
    database_path = str(tmp_path / "delivery-retry.db")
    runtime = FakeAgentRuntime(
        response=_agent_run_response(reply="投递已保存并同步。"),
        runtime_store=SQLiteAgentRuntimeStore(database_path),
    )
    message_calls = []
    update_attempts = []

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def reply_interactive_message(self, **kwargs):
            if kwargs["idempotency_key"] == feishu._stable_reply_uuid(
                "event-restart-retry", {}, "progress"
            ):
                message_calls.append(("progress", kwargs))
                return FeishuMessageResult(message_id="om-progress", raw_response={"code": 0})
            message_calls.append(("fallback", kwargs))
            raise FeishuRequestError("temporary final delivery failure")

        async def update_interactive_message(self, **kwargs):
            message_calls.append(("update", kwargs))
            update_attempts.append(kwargs)
            if len(update_attempts) == 1:
                raise FeishuRequestError("temporary update failure")
            return FeishuMessageResult(message_id="om-progress", raw_response={"code": 0})

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    payload = _text_delivery_payload("event-restart-retry")
    first = TestClient(_app_with_runtime(runtime)).post("/api/feishu/events", json=payload)
    assert first.status_code == 502

    restarted_runtime = FakeAgentRuntime(runtime_store=SQLiteAgentRuntimeStore(database_path))
    restarted_client = TestClient(_app_with_runtime(restarted_runtime))
    retried = restarted_client.post("/api/feishu/events", json=payload)
    duplicate = restarted_client.post("/api/feishu/events", json=payload)

    assert retried.status_code == 200
    assert retried.json()["reply_message_id"] == "om-progress"
    assert retried.json()["agent_response"]["reply"] == "投递已保存并同步。"
    assert duplicate.json() == retried.json()
    assert [kind for kind, _ in message_calls] == ["progress", "update", "fallback", "update"]
    assert all(call["message_id"] == "om-progress" for call in update_attempts)
    assert all("投递已保存并同步" in str(call["card"]) for call in update_attempts)
    assert len([call for call in runtime.calls if call[0] == "start"]) == 1
    assert restarted_runtime.calls == []
    assert restarted_runtime.replay_calls == []


@pytest.mark.parametrize("changed_field", ["text", "owner"])
def test_feishu_duplicate_event_id_with_changed_payload_is_rejected(
    monkeypatch, changed_field,
) -> None:
    _disable_feishu_token(monkeypatch)
    runtime = FakeAgentRuntime()
    sent = []

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def reply_text_message(self, **kwargs):
            sent.append(kwargs)
            return FeishuMessageResult(message_id="om-final", raw_response={"code": 0})

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    client = TestClient(_app_with_runtime(runtime))
    payload = _text_delivery_payload("event-delivery-conflict")
    assert client.post("/api/feishu/events", json=payload).status_code == 200
    if changed_field == "text":
        payload["event"]["message"]["content"]["text"] = "改成另一条任务"
    else:
        payload["event"]["sender"]["sender_id"]["open_id"] = "ou-other"

    conflict = client.post("/api/feishu/events", json=payload)

    assert conflict.status_code == 409
    assert len(sent) == 1
    assert len([call for call in runtime.calls if call[0] == "start"]) == 1


def test_feishu_pending_plain_confirmation_is_forwarded_to_resume_message(
    monkeypatch,
) -> None:
    _disable_feishu_token(monkeypatch)
    interaction = AgentInteraction(
        id="interaction-1",
        type=AgentInteractionType.APPROVAL,
        prompt="即将把 3 个复习时段写入飞书日历，是否确认？",
        allowed_actions=[
            AgentResumeDecision.APPROVE,
            AgentResumeDecision.REJECT,
            AgentResumeDecision.REVISE,
            AgentResumeDecision.CANCEL,
        ],
    )
    runtime = FakeAgentRuntime(
        state=_agent_run_response(
            reply=interaction.prompt,
            status=AgentRunStatus.WAITING_FOR_INPUT,
            interaction=interaction,
        ),
        response=_agent_run_response(reply="已同步到飞书日历。"),
    )

    class FakeFeishuMessageService:
        def is_configured(self):
            return False

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    client = TestClient(_app_with_runtime(runtime))
    response = client.post(
        "/api/feishu/events",
        json={
            "header": {"event_id": "event-approve"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou-test"}},
                "message": {
                    "chat_type": "p2p",
                    "chat_id": "chat-test",
                    "message_id": "om-confirm",
                    "message_type": "text",
                    "content": {"text": "确认"},
                },
            },
        },
    )

    assert response.status_code == 200
    assert [name for name, _ in runtime.calls] == ["get_state", "resume_message"]
    resume_call = runtime.calls[1][1]
    assert resume_call == {
        "thread_id": "thread_feishu",
        "interaction_id": "interaction-1",
        "message": "确认",
        "user_id": "ou-test",
        "source": "feishu",
        "request_id": "event-approve",
    }


def test_feishu_pending_compound_reply_is_forwarded_unchanged(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    interaction = AgentInteraction(
        id="interaction-1",
        type=AgentInteractionType.APPROVAL,
        prompt="是否创建日历事件？",
        allowed_actions=[AgentResumeDecision.APPROVE, AgentResumeDecision.CANCEL],
    )
    runtime = FakeAgentRuntime(
        state=_agent_run_response(
            reply=interaction.prompt,
            status=AgentRunStatus.WAITING_FOR_INPUT,
            interaction=interaction,
        )
    )

    class FakeFeishuMessageService:
        def is_configured(self):
            return False

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    response = TestClient(_app_with_runtime(runtime)).post(
        "/api/feishu/events",
        json={
            "header": {"event_id": "event-compound-reply"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou-test"}},
                "message": {
                    "chat_type": "p2p",
                    "chat_id": "chat-test",
                    "message_id": "om-new-task",
                    "message_type": "text",
                    "content": {"text": "确认，然后base上海"},
                },
            },
        },
    )

    assert response.status_code == 200
    assert [name for name, _ in runtime.calls] == ["get_state", "resume_message"]
    assert runtime.calls[1][1] == {
        "thread_id": "thread_feishu",
        "interaction_id": "interaction-1",
        "message": "确认，然后base上海",
        "user_id": "ou-test",
        "source": "feishu",
        "request_id": "event-compound-reply",
    }


def test_feishu_pending_form_forwards_free_text_to_resume_message(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    interaction = AgentInteraction(
        id="interaction-form",
        type=AgentInteractionType.PREFERENCE_FORM,
        prompt="请补充复习时区和每天可用时段。",
        allowed_actions=[AgentResumeDecision.ANSWER, AgentResumeDecision.CANCEL],
        required_fields=["timezone", "weekday_windows"],
    )
    runtime = FakeAgentRuntime(
        state=_agent_run_response(
            status=AgentRunStatus.WAITING_FOR_INPUT,
            interaction=interaction,
        )
    )

    class FakeFeishuMessageService:
        def is_configured(self):
            return False

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    response = TestClient(_app_with_runtime(runtime)).post(
        "/api/feishu/events",
        json={
            "event": {
                "sender": {"sender_id": {"open_id": "ou-test"}},
                "message": {
                    "chat_type": "p2p",
                    "chat_id": "chat-test",
                    "message_id": "om-unrelated",
                    "message_type": "text",
                    "content": {"text": "再帮我查询本周面试"},
                },
            },
        },
    )

    assert response.status_code == 200
    assert [name for name, _ in runtime.calls] == ["get_state", "resume_message"]
    assert runtime.calls[1][1]["message"] == "再帮我查询本周面试"


def test_feishu_pending_status_question_is_forwarded_to_resume_message(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    interaction = AgentInteraction(
        id="interaction-status",
        type=AgentInteractionType.APPROVAL,
        prompt="是否同步日历？",
        allowed_actions=[AgentResumeDecision.APPROVE, AgentResumeDecision.CANCEL],
    )
    runtime = FakeAgentRuntime(
        state=_agent_run_response(
            status=AgentRunStatus.WAITING_FOR_INPUT,
            interaction=interaction,
        )
    )

    class FakeFeishuMessageService:
        def is_configured(self):
            return False

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    response = TestClient(_app_with_runtime(runtime)).post(
        "/api/feishu/events",
        json={
            "event": {
                "sender": {"sender_id": {"open_id": "ou-test"}},
                "message": {
                    "chat_type": "p2p",
                    "chat_id": "chat-test",
                    "message_id": "om-status",
                    "message_type": "text",
                    "content": {"text": "现在什么状态？"},
                },
            },
        },
    )

    assert response.status_code == 200
    assert [name for name, _ in runtime.calls] == ["get_state", "resume_message"]
    assert runtime.calls[1][1]["message"] == "现在什么状态？"


def test_feishu_status_question_without_pending_only_reads_runtime_state(
    monkeypatch,
) -> None:
    _disable_feishu_token(monkeypatch)
    runtime = FakeAgentRuntime(
        state=_agent_run_response(reply="投递记录查询完成。"),
    )

    class FakeFeishuMessageService:
        def is_configured(self):
            return False

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    response = TestClient(_app_with_runtime(runtime)).post(
        "/api/feishu/events",
        json={
            "event": {
                "sender": {"sender_id": {"open_id": "ou-test"}},
                "message": {
                    "chat_type": "p2p",
                    "chat_id": "chat-test",
                    "message_id": "om-status-complete",
                    "message_type": "text",
                    "content": {"text": "现在什么状态？"},
                },
            },
        },
    )

    assert response.status_code == 200
    assert [name for name, _ in runtime.calls] == ["get_state"]
    assert "当前任务状态：已完成" in response.json()["agent_response"]["reply"]


def test_feishu_status_question_does_not_start_missing_thread(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    runtime = FakeAgentRuntime()

    class FakeFeishuMessageService:
        def is_configured(self):
            return False

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    response = TestClient(_app_with_runtime(runtime)).post(
        "/api/feishu/events",
        json={
            "event": {
                "sender": {"sender_id": {"open_id": "ou-test"}},
                "message": {
                    "chat_type": "p2p",
                    "chat_id": "chat-test",
                    "message_id": "om-status-empty",
                    "message_type": "text",
                    "content": {"text": "执行进度？"},
                },
            },
        },
    )

    assert response.status_code == 200
    assert [name for name, _ in runtime.calls] == ["get_state"]
    assert response.json()["agent_response"]["status"] == "degraded"
    assert "还没有可查询的任务" in response.json()["agent_response"]["reply"]


def test_feishu_card_button_records_leetcode_result(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    repository = InMemoryLeetCodeRepository(problems=load_hot100_snapshot().problems)
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    recommendation = LeetCodeRecommendationWorkflow(repository).get_today(
        "feishu:ou_card", today
    )[0]
    monkeypatch.setattr(feishu, "get_default_leetcode_repository", lambda: repository)
    client = TestClient(
        create_app(
            Settings(
                debug_routes_enabled=False,
                feishu_verification_token="",
                feishu_allow_unverified_events=True,
            )
        )
    )

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


def test_feishu_card_event_receipt_returns_cached_result_and_rejects_reuse(
    monkeypatch,
) -> None:
    _disable_feishu_token(monkeypatch)
    calls = []

    def handle_card(*, payload, action, repository=None):
        calls.append((payload, action, repository))
        return {"toast": {"type": "success", "content": "recorded once"}}

    monkeypatch.setattr(feishu, "_handle_leetcode_card_action", handle_card)
    runtime = FakeAgentRuntime()
    runtime.runtime_store = InMemoryAgentRuntimeStore()
    client = TestClient(_app_with_runtime(runtime))
    payload = {
        "header": {"event_type": "card.action.trigger", "event_id": "card_once"},
        "event": {
            "operator": {"operator_id": {"open_id": "ou_card"}},
            "action": {
                "value": {
                    "action": "leetcode_result",
                    "assignment_id": "assignment_1",
                    "result": "independent",
                }
            },
        },
    }

    first = client.post("/api/feishu/events", json=payload)
    duplicate = client.post("/api/feishu/events", json=payload)
    changed = {
        **payload,
        "event": {
            **payload["event"],
            "action": {
                "value": {
                    **payload["event"]["action"]["value"],
                    "result": "failed",
                }
            },
        },
    }
    conflict = client.post("/api/feishu/events", json=changed)

    assert first.status_code == 200
    assert duplicate.json() == first.json()
    assert len(calls) == 1
    assert conflict.status_code == 409
    assert "different payload" in conflict.json()["detail"]


def test_feishu_card_action_without_event_id_is_rejected_before_write(
    monkeypatch,
) -> None:
    _disable_feishu_token(monkeypatch)
    calls = []

    def handle_card(*, payload, action, repository=None):
        calls.append((payload, action, repository))
        return {"toast": {"type": "success", "content": "should not run"}}

    monkeypatch.setattr(feishu, "_handle_leetcode_card_action", handle_card)
    response = TestClient(_app_with_runtime()).post(
        "/api/feishu/events",
        json={
            "header": {"event_type": "card.action.trigger"},
            "event": {
                "operator": {"operator_id": {"open_id": "ou_card"}},
                "action": {
                    "value": {
                        "action": "leetcode_result",
                        "assignment_id": "assignment_1",
                        "result": "independent",
                    }
                },
            },
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Feishu event id is required for idempotent processing."
    }
    assert calls == []


def test_feishu_rejects_a_stale_card_after_problem_rolls_over(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    repository = InMemoryLeetCodeRepository(problems=load_hot100_snapshot().problems)
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    yesterday = today - timedelta(days=1)
    workflow = LeetCodeRecommendationWorkflow(repository)
    stale = workflow.get_today("feishu:ou_card", yesterday)[0]
    current = workflow.get_today("feishu:ou_card", today)
    monkeypatch.setattr(feishu, "get_default_leetcode_repository", lambda: repository)
    client = TestClient(
        create_app(
            Settings(
                debug_routes_enabled=False,
                feishu_verification_token="",
                feishu_allow_unverified_events=True,
            )
        )
    )

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


def test_feishu_retry_after_send_failure_reuses_request_id_and_reply_uuid(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    runtime = FakeAgentRuntime()
    reply_uuids = []

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def reply_text_message(self, **kwargs):
            reply_uuids.append(kwargs["idempotency_key"])
            if len(reply_uuids) == 1:
                raise FeishuRequestError("temporary send failure")
            return FeishuMessageResult(
                message_id="om_test",
                raw_response={"code": 0, "data": {"message_id": "om_test"}},
            )

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    app = _app_with_runtime(runtime)
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
                "chat_type": "p2p",
                "chat_id": "chat_test",
                "message_id": "om_source",
                "message_type": "text",
                "content": {"text": "今天任务是什么？"},
            },
        },
    }

    first = client.post("/api/feishu/events", json=payload)
    second = client.post("/api/feishu/events", json=payload)

    assert first.status_code == 502
    assert first.json() == {
        "detail": "Feishu reply delivery failed; the event can be retried."
    }
    assert second.status_code == 200
    assert second.json()["handled"] is True
    assert second.json()["reply_sent"] is True
    start_calls = [call for call in runtime.calls if call[0] == "start"]
    assert [call[1]["request_id"] for call in start_calls] == [
        "event_test_1",
    ]
    assert len(reply_uuids) == 2
    assert reply_uuids[0] == reply_uuids[1]


def test_feishu_event_ignores_non_text_message(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    app = create_app(
        Settings(
            debug_routes_enabled=False,
            feishu_verification_token="",
            feishu_allow_unverified_events=True,
        )
    )
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
    to_thread_calls = []
    real_to_thread = feishu.asyncio.to_thread

    async def tracking_to_thread(func, /, *args, **kwargs):
        to_thread_calls.append(func)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(feishu.asyncio, "to_thread", tracking_to_thread)
    app = create_app(
        Settings(
            debug_routes_enabled=False,
            feishu_verification_token="",
            feishu_allow_unverified_events=True,
        )
    )
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
    assert to_thread_calls == [feishu._handle_bitable_record_event]


def test_feishu_bitable_event_receipt_prevents_duplicate_sync(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    calls = []

    def handle_bitable(*, event_type, context, repository=None, bitable_service=None):
        calls.append((event_type, context, repository, bitable_service))
        return feishu.FeishuEventProcessResponse(
            handled=True,
            event_type=event_type,
            message="synced once",
        )

    monkeypatch.setattr(feishu, "_handle_bitable_record_event", handle_bitable)
    runtime = FakeAgentRuntime()
    runtime.runtime_store = InMemoryAgentRuntimeStore()
    client = TestClient(_app_with_runtime(runtime))
    payload = {
        "header": {
            "event_type": "drive.file.bitable_record_changed_v1",
            "event_id": "bitable_once",
        },
        "event": {
            "app_token": "bascn_offerpilot",
            "table_id": "tbl_applications",
            "record_id": "rec_application_1",
        },
    }

    first = client.post("/api/feishu/events", json=payload)
    duplicate = client.post("/api/feishu/events", json=payload)

    assert first.status_code == 200
    assert duplicate.json() == first.json()
    assert len(calls) == 1


@pytest.fixture
def bitable_deletion_context(monkeypatch, tmp_path):
    _disable_feishu_token(monkeypatch)
    audit_path = tmp_path / "feishu_events.log"
    monkeypatch.setattr(feishu, "_feishu_event_audit_path", lambda: audit_path)
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(
        company="Example", role="Backend", owner_id="feishu:ou_test"
    )
    save_owner_bitable_resource(
        repository,
        owner_id="feishu:ou_test",
        app_token="bascn_offerpilot",
        table_id="tbl_applications",
    )
    remember_bitable_record_mapping(
        repository,
        app_token="bascn_offerpilot",
        table_id="tbl_applications",
        record_id="rec_application_1",
        application_id=application.id,
        owner_id="feishu:ou_test",
    )

    class FakeBitableService:
        app_token = ""
        table_id = ""

        def __init__(self):
            self.records = {}
            self.get_record_calls = []
            self.error = FeishuRequestError("RecordIdNotFound", code=1254043)

        def get_record(self, app_token, table_id, record_id):
            self.get_record_calls.append((app_token, table_id, record_id))
            if record_id not in self.records:
                raise self.error
            return FeishuBitableRecordResult(
                record_id=record_id,
                raw_response={"code": 0},
                fields=self.records[record_id],
            )

    service = FakeBitableService()
    runtime = FakeAgentRuntime()
    runtime.runtime_store = InMemoryAgentRuntimeStore()
    runtime.offerpilot_repository = repository
    runtime.feishu_bitable_service = service
    return TestClient(_app_with_runtime(runtime)), repository, service, application, audit_path


def _bitable_action_event(
    *actions,
    event_id="bitable_delete",
    table_id="tbl_applications",
    operator_open_id="ou_test",
):
    return {
        "schema": "2.0",
        "header": {
            "event_type": "drive.file.bitable_record_changed_v1",
            "event_id": event_id,
        },
        "event": {
            "file_token": "bascn_offerpilot",
            "file_type": "bitable",
            "operator_id": {"open_id": operator_open_id},
            "table_id": table_id,
            "action_list": [
                {"action": action, "record_id": record_id}
                for action, record_id in actions
            ],
        },
    }


def test_feishu_deleted_record_event_soft_deletes_once(bitable_deletion_context) -> None:
    client, repository, service, application, audit_path = bitable_deletion_context
    payload = _bitable_action_event(("record_deleted", "rec_application_1"))

    first = client.post("/api/feishu/events", json=payload)
    duplicate = client.post("/api/feishu/events", json=payload)

    assert first.status_code == 200
    assert first.json() == {
        "handled": True,
        "event_type": "drive.file.bitable_record_changed_v1",
        "message": f"多维表格记录已同步删除：{application.id}",
    }
    assert duplicate.json() == first.json()
    assert repository.list_applications(owner_id="feishu:ou_test") == []
    assert repository.is_application_deleted(application.id, "feishu:ou_test")
    assert len(service.get_record_calls) == 1
    audit_rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    sync_rows = [row for row in audit_rows if row.get("phase") == "bitable_sync_result"]
    assert len(sync_rows) == 1
    assert sync_rows[0]["results"][0]["status"] == "deleted"
    assert sync_rows[0]["results"][0]["application_id"] == application.id


def test_feishu_bitable_mixed_edit_delete_batch(bitable_deletion_context) -> None:
    client, repository, service, deleted_application, _ = bitable_deletion_context
    updated_application = repository.create_application(
        company="Second", role="Backend", owner_id="feishu:ou_test"
    )
    remember_bitable_record_mapping(
        repository,
        app_token="bascn_offerpilot",
        table_id="tbl_applications",
        record_id="rec_application_2",
        application_id=updated_application.id,
        owner_id="feishu:ou_test",
    )
    service.records["rec_application_2"] = {
        "OfferPilot记录ID": updated_application.id,
        "OfferPilot用户ID": "feishu:ou_test",
        "公司": "Second",
        "岗位": "AI Engineer",
        "投递状态": "已投递",
    }
    payload = _bitable_action_event(
        ("record_edited", "rec_application_2"),
        ("record_deleted", "rec_application_1"),
        ("record_deleted", "rec_application_1"),
    )

    response = client.post("/api/feishu/events", json=payload)

    assert response.status_code == 200
    assert response.json()["message"] == (
        f"多维表格记录已回写数据库：{updated_application.id}；"
        f"已同步删除：{deleted_application.id}"
    )
    applications = repository.list_applications(owner_id="feishu:ou_test")
    assert [application.id for application in applications] == [updated_application.id]
    assert applications[0].role == "AI Engineer"
    assert applications[0].status.value == "submitted"
    assert repository.is_application_deleted(deleted_application.id, "feishu:ou_test")
    assert {call[2] for call in service.get_record_calls} == {
        "rec_application_1", "rec_application_2"
    }
    assert len(service.get_record_calls) == 2


@pytest.mark.parametrize(
    ("event_overrides", "expected_status"),
    [
        ({"table_id": "tbl_foreign"}, "ignored_foreign_table"),
        ({"operator_open_id": "ou_foreign"}, "owner_conflict"),
    ],
)
def test_feishu_deleted_record_event_rejects_foreign_scope(
    bitable_deletion_context, event_overrides, expected_status
) -> None:
    client, repository, service, application, _ = bitable_deletion_context

    response = client.post(
        "/api/feishu/events",
        json=_bitable_action_event(
            ("record_deleted", "rec_application_1"), **event_overrides
        ),
    )

    assert response.status_code == 200
    assert response.json()["handled"] is False
    assert expected_status in response.json()["message"]
    assert not repository.is_application_deleted(application.id, "feishu:ou_test")
    assert len(repository.list_applications(owner_id="feishu:ou_test")) == 1
    assert service.get_record_calls == []


def test_feishu_delayed_delete_event_keeps_current_live_record(bitable_deletion_context) -> None:
    client, repository, service, application, _ = bitable_deletion_context
    service.records["rec_application_1"] = {
        "OfferPilot记录ID": application.id,
        "OfferPilot用户ID": "feishu:ou_test",
        "公司": "Example",
        "岗位": "Current Role",
        "投递状态": "已投递",
    }

    response = client.post(
        "/api/feishu/events",
        json=_bitable_action_event(("record_deleted", "rec_application_1")),
    )

    assert response.status_code == 200
    assert response.json()["message"] == f"多维表格记录已回写数据库：{application.id}"
    assert not repository.is_application_deleted(application.id, "feishu:ou_test")
    assert repository.list_applications(owner_id="feishu:ou_test")[0].role == "Current Role"
    assert len(service.get_record_calls) == 1


@pytest.mark.parametrize("error_code", [None, 99991672])
def test_feishu_delete_event_read_failure_keeps_local_record(
    bitable_deletion_context, error_code
) -> None:
    client, repository, service, application, _ = bitable_deletion_context
    service.error = FeishuRequestError("request failed", code=error_code)

    response = client.post(
        "/api/feishu/events",
        json=_bitable_action_event(("record_deleted", "rec_application_1")),
    )

    assert response.status_code == 200
    assert "同步失败" in response.json()["message"]
    assert not repository.is_application_deleted(application.id, "feishu:ou_test")
    assert len(repository.list_applications(owner_id="feishu:ou_test")) == 1


def test_feishu_bitable_event_without_event_id_is_rejected_before_write(
    monkeypatch,
) -> None:
    _disable_feishu_token(monkeypatch)
    calls = []

    def handle_bitable(*, event_type, context, repository=None, bitable_service=None):
        calls.append((event_type, context, repository, bitable_service))
        return feishu.FeishuEventProcessResponse(
            handled=True,
            event_type=event_type,
            message="should not run",
        )

    monkeypatch.setattr(feishu, "_handle_bitable_record_event", handle_bitable)
    response = TestClient(_app_with_runtime()).post(
        "/api/feishu/events",
        json={
            "header": {"event_type": "drive.file.bitable_record_changed_v1"},
            "event": {
                "app_token": "bascn_offerpilot",
                "table_id": "tbl_applications",
                "record_id": "rec_application_1",
            },
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Feishu event id is required for idempotent processing."
    }
    assert calls == []


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
    app = create_app(
        Settings(
            debug_routes_enabled=False,
            feishu_verification_token="",
            feishu_allow_unverified_events=True,
        )
    )
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


def test_feishu_event_creates_manual_bitable_row_for_operator_owner(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)
    repository = InMemoryOfferPilotRepository()
    repository.set_runtime_setting("feishu.offerpilot_bitable_app_token", "bascn_offerpilot")
    repository.set_runtime_setting("feishu.offerpilot_bitable_table_id", "tbl_applications")

    class FakeBitableService:
        app_token = ""
        table_id = ""
        update_record_calls = []

        def is_bitable_sync_enabled(self):
            return True

        def get_record(self, app_token, table_id, record_id):
            assert app_token == "bascn_offerpilot"
            assert table_id == "tbl_applications"
            assert record_id == "rec_manual_owner"
            return FeishuBitableRecordResult(
                record_id=record_id,
                raw_response={"code": 0},
                fields={
                    "公司": "美团",
                    "岗位": "AI 全栈工程师",
                    "工作地点": "深圳",
                    "投递状态": "已投递",
                },
            )

        def update_record(self, app_token, table_id, record_id, fields):
            self.update_record_calls.append(
                {
                    "app_token": app_token,
                    "table_id": table_id,
                    "record_id": record_id,
                    "fields": fields,
                }
            )
            return FeishuBitableRecordResult(
                record_id=record_id,
                raw_response={"code": 0},
            )

    monkeypatch.setattr(feishu, "get_default_offerpilot_repository", lambda: repository)
    monkeypatch.setattr(feishu, "FeishuBitableService", FakeBitableService)
    client = TestClient(
        create_app(
            Settings(
                debug_routes_enabled=False,
                feishu_verification_token="",
                feishu_allow_unverified_events=True,
            )
        )
    )

    response = client.post(
        "/api/feishu/events",
        json={
            "schema": "2.0",
            "header": {
                "event_type": "drive.file.bitable_record_changed_v1",
                "event_id": "event_manual_owner",
            },
            "event": {
                "file_token": "bascn_offerpilot",
                "file_type": "bitable",
                "operator_id": {"open_id": "ou_test"},
                "table_id": "tbl_applications",
                "action_list": [
                    {
                        "action": "record_added",
                        "record_id": "rec_manual_owner",
                    }
                ],
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "handled": True,
        "event_type": "drive.file.bitable_record_changed_v1",
        "message": "多维表格记录已回写数据库：app_1",
    }
    applications = repository.list_applications(owner_id="feishu:ou_test")
    assert len(applications) == 1
    assert applications[0].company == "美团"
    assert repository.list_applications(owner_id="local_user") == []
    assert FakeBitableService.update_record_calls == [
        {
            "app_token": "bascn_offerpilot",
            "table_id": "tbl_applications",
            "record_id": "rec_manual_owner",
            "fields": {
                "OfferPilot记录ID": applications[0].id,
                "OfferPilot用户ID": "feishu:ou_test",
            },
        }
    ]


def test_feishu_event_route_stays_enabled_when_debug_routes_disabled(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)

    class FakeFeishuMessageService:
        def is_configured(self):
            return False

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    app = create_app(
        Settings(
            environment="production",
            debug_routes_enabled=False,
            feishu_verification_token="verify_token",
        )
    )
    app.state.agent_runtime = FakeAgentRuntime()
    client = TestClient(app)

    feishu_response = client.post(
        "/api/feishu/events",
        json={
            "token": "verify_token",
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
        Settings(feishu_verification_token="wrong_global_token"),
    )
    app = create_app(
        Settings(
            debug_routes_enabled=False,
            feishu_verification_token="verify_token",
        )
    )
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
    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def reply_text_message(self, **kwargs):
            return FeishuMessageResult(
                message_id="om_test",
                raw_response={"code": 0, "data": {"message_id": "om_test"}},
            )

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    app = _app_with_runtime(feishu_verification_token="verify_token")
    client = TestClient(app)

    response = client.post(
        "/api/feishu/events",
        json={
            "schema": "2.0",
            "header": {
                "event_type": "im.message.receive_v1",
                "event_id": "event-valid-header-token",
                "token": "verify_token",
            },
            "event": {
                "sender": {"sender_id": {"open_id": "ou_test"}},
                "message": {
                    "message_id": "om-valid-header-token",
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
    app = create_app(
        Settings(
            debug_routes_enabled=False,
            feishu_verification_token="verify_token",
        )
    )
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
    app = create_app(
        Settings(
            debug_routes_enabled=False,
            feishu_verification_token="verify_token",
        )
    )
    client = TestClient(app)

    response = client.post(
        "/api/feishu/events",
        json={
            "type": "url_verification",
            "challenge": "challenge_token",
        },
    )

    assert response.status_code == 403


def test_feishu_event_rejects_unconfigured_verification_in_production() -> None:
    app = create_app(
        Settings(
            environment="production",
            debug_routes_enabled=False,
            feishu_verification_token="",
        )
    )
    client = TestClient(app)

    response = client.post(
        "/api/feishu/events",
        json={
            "type": "url_verification",
            "challenge": "challenge_token",
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Feishu verification token is not configured."
    }


def test_feishu_event_rejects_unconfigured_verification_in_local_by_default() -> None:
    app = create_app(
        Settings(
            environment="local",
            debug_routes_enabled=False,
            feishu_verification_token="",
            feishu_allow_unverified_events=False,
        )
    )

    response = TestClient(app).post(
        "/api/feishu/events",
        json={
            "type": "url_verification",
            "challenge": "challenge_token",
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Feishu verification token is not configured."
    }


def test_feishu_event_ignores_unverified_override_in_production() -> None:
    app = create_app(
        Settings(
            environment="production",
            debug_routes_enabled=False,
            feishu_verification_token="",
            feishu_allow_unverified_events=True,
        )
    )

    response = TestClient(app).post(
        "/api/feishu/events",
        json={
            "type": "url_verification",
            "challenge": "challenge_token",
        },
    )

    assert response.status_code == 503


def test_feishu_text_event_without_event_id_is_rejected_before_agent() -> None:
    runtime = FakeAgentRuntime()
    response = TestClient(_app_with_runtime(runtime)).post(
        "/api/feishu/events",
        json={
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou_test"}},
                "message": {
                    "message_type": "text",
                    "content": {"text": "新增一条投递"},
                },
            },
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Feishu event id is required for idempotent processing."
    }
    assert runtime.calls == []


def test_feishu_event_reports_unsent_reply_when_credentials_missing(monkeypatch) -> None:
    _disable_feishu_token(monkeypatch)

    class FakeFeishuMessageService:
        def is_configured(self):
            return False

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    app = _app_with_runtime()
    client = TestClient(app)

    response = client.post(
        "/api/feishu/events",
        json={
            "event": {
                "sender": {"sender_id": {"open_id": "ou_test"}},
                "message": {
                    "message_id": "om-missing-credentials",
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

    class FakeFeishuMessageService:
        def is_configured(self):
            return True

        async def reply_text_message(self, **kwargs):
            raise FeishuRequestError("send failed")

    monkeypatch.setattr(feishu, "FeishuMessageService", FakeFeishuMessageService)
    app = _app_with_runtime()
    client = TestClient(app)

    response = client.post(
        "/api/feishu/events",
        json={
            "event": {
                "sender": {"sender_id": {"open_id": "ou_test"}},
                "message": {
                    "message_id": "om-reply-send-error",
                    "message_type": "text",
                    "content": {"text": "今天任务是什么？"},
                },
            }
        },
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Feishu reply delivery failed; the event can be retried."
    }
