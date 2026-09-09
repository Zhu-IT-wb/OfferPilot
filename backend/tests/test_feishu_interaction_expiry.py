import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.runtime import AgentRuntime
from app.api.routes import feishu
from app.core.config import Settings
from app.main import create_app
from app.models.tool_calling import ModelToolCall, ModelTurn, TokenUsage
from app.schemas.agent import AgentInteraction, AgentRunResponse, AgentRunStatus
from app.services.agent_runtime_store import InMemoryAgentRuntimeStore
from app.tools.agent_tool import (
    AgentToolDefinition,
    FunctionAgentTool,
    ToolApproval,
    ToolEffect,
)
from app.tools.agent_tool_registry import AgentToolRegistry


NOW = datetime(2026, 9, 9, 14, 19, tzinfo=timezone.utc)


class FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz is not None else NOW.replace(tzinfo=None)


def _waiting_response(expires_at):
    return AgentRunResponse(
        thread_id="thread-expiry",
        run_id="run-expiry",
        status="waiting_for_input",
        reply="Please confirm the pending operation.",
        interaction=AgentInteraction(
            id="interaction-expiry",
            type="approval",
            prompt="Please confirm the pending operation.",
            allowed_actions=["approve", "cancel"],
            expires_at=expires_at,
        ),
    )


class ReplayRuntime:
    def __init__(self, response):
        self.response = response
        self.runtime_store = InMemoryAgentRuntimeStore()
        self.reads = 0

    @staticmethod
    def thread_id_for(**kwargs):
        return "thread-expiry"

    async def replay_request(self, **kwargs):
        return self.response

    async def get_state(self, **kwargs):
        self.reads += 1
        return self.response


@pytest.mark.parametrize(
    ("expires_at", "stale"),
    [
        (NOW - timedelta(seconds=1), True),
        (NOW, True),
        ((NOW - timedelta(seconds=1)).replace(tzinfo=None), True),
        (NOW.astimezone(timezone(timedelta(hours=8))), True),
        (NOW + timedelta(seconds=1), False),
        (None, False),
    ],
)
def test_recovered_interaction_expiry_is_checked_without_mutating_runtime(
    monkeypatch, expires_at, stale,
):
    monkeypatch.setattr(feishu, "datetime", FrozenDateTime)
    pending = _waiting_response(expires_at)
    runtime = ReplayRuntime(pending)

    actual = asyncio.run(feishu._is_stale_interaction_reply(
        runtime=runtime,
        response=pending,
        thread_id=pending.thread_id,
        user_id="ou-owner",
    ))

    assert actual is stale
    assert runtime.reads == 1
    assert runtime.response.interaction == pending.interaction


def test_expired_same_interaction_receipt_never_sends_a_card(monkeypatch):
    monkeypatch.setattr(feishu, "datetime", FrozenDateTime)
    pending = _waiting_response(NOW - timedelta(hours=3))
    runtime = ReplayRuntime(pending)
    app = create_app(Settings(
        debug_routes_enabled=False,
        feishu_verification_token="",
        feishu_allow_unverified_events=True,
    ))
    app.state.agent_runtime = runtime

    def unexpected_message_service():
        raise AssertionError("An expired replay must not send progress or confirmation.")

    monkeypatch.setattr(feishu, "FeishuMessageService", unexpected_message_service)
    payload = {
        "header": {"event_id": "expired-original", "tenant_key": "tenant-test"},
        "event": {
            "sender": {"sender_id": {"open_id": "ou-owner"}},
            "message": {
                "chat_type": "p2p",
                "chat_id": "chat-test",
                "message_id": "om-expired",
                "message_type": "text",
                "content": {"text": "Old task"},
            },
        },
    }
    client = TestClient(app)
    first = client.post("/api/feishu/events", json=payload)
    duplicate = client.post("/api/feishu/events", json=payload)

    assert first.status_code == duplicate.status_code == 200
    assert first.json()["reply_sent"] is False
    assert first.json() == duplicate.json()
    assert "agent_response" not in first.json()
    assert runtime.reads == 1


@pytest.mark.parametrize("interaction_kind", ["approval", "clarification"])
def test_new_feishu_message_after_expiry_reaches_a_fresh_runtime_turn(interaction_kind):
    clock = [NOW - timedelta(hours=4)]
    name = "write_record" if interaction_kind == "approval" else "request_user_input"
    arguments = {} if interaction_kind == "approval" else {
        "kind": "clarification", "prompt": "Which previous option do you want?",
    }
    model_calls = []
    new_message = "LangChain与LangGraph的了解"

    class Model:
        async def complete(self, messages, tools, options):
            model_calls.append(messages)
            if len(model_calls) == 1:
                call = ModelToolCall(id="old-call", name=name, arguments=arguments)
                assistant = {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": call.id, "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }],
                }
                calls, content, reason = [call], "", "tool_calls"
            else:
                assert len(model_calls) == 2
                content, calls, reason = "Here is an overview of LangChain and LangGraph.", [], "stop"
                assistant = {"role": "assistant", "content": content}
            return ModelTurn(
                assistant_message=assistant, tool_calls=calls, content=content,
                reasoning_content=None, finish_reason=reason,
                usage=TokenUsage(8, 3, 11), provider="fake", model="scripted",
            )

    async def never_execute(arguments):
        raise AssertionError("The expired operation must never execute.")

    definition = AgentToolDefinition(
        name=name, description="Test pending operation",
        parameters={
            "type": "object",
            "properties": {} if interaction_kind == "approval" else {
                "kind": {"type": "string"}, "prompt": {"type": "string"},
            },
            "additionalProperties": False,
        },
        effect=ToolEffect.EXTERNAL_WRITE if interaction_kind == "approval" else ToolEffect.READ,
        approval=ToolApproval.ALWAYS if interaction_kind == "approval" else ToolApproval.NEVER,
        parallel_safe=False,
        control=interaction_kind != "approval",
    )
    runtime = AgentRuntime(
        model=Model(),
        tool_registry=AgentToolRegistry([FunctionAgentTool(definition, never_execute)]),
        checkpointer=InMemorySaver(),
        settings=Settings(agent_max_verifier_passes=0),
        clock=lambda: clock[0],
    )

    async def scenario():
        paused = await runtime.start(
            "Old task", user_id="ou-owner", source="feishu",
            conversation_scope="tenant:chat:ou-owner", request_id="old-event",
        )
        assert paused.status == AgentRunStatus.WAITING_FOR_INPUT
        clock[0] = NOW
        result = await feishu._dispatch_agent_message(
            runtime=runtime, message=new_message, user_id="ou-owner",
            conversation_scope="tenant:chat:ou-owner", thread_id=paused.thread_id,
            request_id="new-event",
        )
        assert result.thread_id == paused.thread_id
        assert result.run_id != paused.run_id
        assert result.status == AgentRunStatus.COMPLETED
        assert result.interaction is None
        assert "LangGraph" in result.reply
        assert any(
            item.get("role") == "user" and item.get("content") == new_message
            for item in model_calls[-1]
        )

    asyncio.run(scenario())
