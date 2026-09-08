import json
import sqlite3
from dataclasses import replace
from typing import Any, Dict, Iterable, List

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.runtime import AgentRuntime
from app.api.routes import feishu
from app.core.config import Settings, settings as default_settings
from app.main import create_app
from app.models.tool_calling import ModelToolCall, ModelTurn, TokenUsage
from app.repositories.sqlite_offerpilot_repository import SQLiteOfferPilotRepository
from app.services.agent_runtime_store import SQLiteAgentRuntimeStore
from app.tools.agent_tool import (
    AgentToolDefinition,
    AgentToolResult,
    FunctionAgentTool,
    ToolApproval,
    ToolEffect,
)
from app.tools.agent_tool_registry import AgentToolRegistry


class ScriptedModel:
    def __init__(self, turns: Iterable[ModelTurn]) -> None:
        self.turns = list(turns)
        self.calls: List[Dict[str, Any]] = []

    async def complete(self, messages, tools, options) -> ModelTurn:
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "tools": list(tools),
                "options": options,
            }
        )
        index = len(self.calls) - 1
        if index >= len(self.turns):
            raise AssertionError(f"Scripted model has no turn at index {index}")
        return self.turns[index]


class TrackingAgentRuntime(AgentRuntime):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.start_calls = 0
        self.resume_message_calls = 0

    async def start(self, *args, **kwargs):
        self.start_calls += 1
        return await super().start(*args, **kwargs)

    async def resume_message(self, *args, **kwargs):
        self.resume_message_calls += 1
        return await super().resume_message(*args, **kwargs)


def _tool_turn(call_id: str, name: str, **arguments: Any) -> ModelTurn:
    call = ModelToolCall(id=call_id, name=name, arguments=arguments)
    return ModelTurn(
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
            ],
        },
        tool_calls=[call],
        content="",
        reasoning_content="",
        finish_reason="tool_calls",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14),
        provider="fake",
        model="scripted",
    )


def _answer_turn(content: str) -> ModelTurn:
    return ModelTurn(
        assistant_message={"role": "assistant", "content": content},
        tool_calls=[],
        content=content,
        reasoning_content="",
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=8, completion_tokens=3, total_tokens=11),
        provider="fake",
        model="scripted",
    )


def _message_event(event_id: str, message_id: str, text: str) -> Dict[str, Any]:
    return {
        "header": {
            "event_id": event_id,
            "event_type": "im.message.receive_v1",
            "tenant_key": "tenant-huawei",
        },
        "event": {
            "sender": {"sender_id": {"open_id": "ou-huawei"}},
            "message": {
                "chat_type": "p2p",
                "chat_id": "chat-huawei",
                "message_id": message_id,
                "message_type": "text",
                "content": {"text": text},
            },
        },
    }


def test_duplicate_feishu_confirmation_replays_completed_write(monkeypatch, tmp_path) -> None:
    database_path = tmp_path / "offerpilot.db"
    repository = SQLiteOfferPilotRepository(str(database_path))
    runtime_store = SQLiteAgentRuntimeStore(str(database_path))
    writes: List[Dict[str, Any]] = []

    async def create_application(arguments: Dict[str, Any]) -> AgentToolResult:
        writes.append(dict(arguments))
        application = repository.create_application(
            company=arguments["company"],
            role=arguments["role"],
            base_location=arguments.get("base_location"),
            owner_id=arguments["owner_id"],
        )
        return AgentToolResult(
            data={"application": application.to_dict()},
            message="投递记录已创建。",
        )

    create_tool = FunctionAgentTool(
        AgentToolDefinition(
            name="create_application",
            description="Create one job application record.",
            parameters={
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "role": {"type": "string"},
                    "base_location": {"type": "string"},
                },
                "required": ["company", "role"],
                "additionalProperties": False,
            },
            effect=ToolEffect.EXTERNAL_WRITE,
            approval=ToolApproval.ALWAYS,
            parallel_safe=False,
            idempotent=True,
        ),
        create_application,
    )
    model = ScriptedModel(
        [
            _tool_turn(
                "create-huawei",
                "create_application",
                company="华为",
                role="软件开发岗位",
                base_location="深圳",
            ),
            _tool_turn(
                "interpret-confirmation",
                AgentRuntime._INTERACTION_INTERPRETER_TOOL_NAME,
                action="approve",
                approval_intent=True,
                has_changes=False,
            ),
            _answer_turn("已记录华为软件开发岗位，工作地点为深圳。"),
        ]
    )
    runtime = TrackingAgentRuntime(
        model=model,
        tool_registry=AgentToolRegistry([create_tool]),
        checkpointer=InMemorySaver(),
        runtime_store=runtime_store,
        settings=replace(default_settings, agent_max_verifier_passes=0),
    )
    runtime.offerpilot_repository = repository

    class UnconfiguredFeishuMessageService:
        def is_configured(self) -> bool:
            return False

    monkeypatch.setattr(feishu, "FeishuMessageService", UnconfiguredFeishuMessageService)
    app = create_app(
        Settings(
            debug_routes_enabled=False,
            feishu_verification_token="",
            feishu_allow_unverified_events=True,
        )
    )
    app.state.agent_runtime = runtime
    client = TestClient(app)

    create_response = client.post(
        "/api/feishu/events",
        json=_message_event(
            "event-create-huawei",
            "om-create-huawei",
            "我投递了华为的软件开发岗位，base 深圳",
        ),
    )
    assert create_response.status_code == 200
    assert create_response.json()["agent_response"]["status"] == "waiting_for_input"

    confirmation_payload = _message_event(
        "event-confirm-huawei",
        "om-confirm-huawei",
        "嗯嗯",
    )
    first_confirmation = client.post("/api/feishu/events", json=confirmation_payload)
    duplicate_confirmation = client.post("/api/feishu/events", json=confirmation_payload)

    assert first_confirmation.status_code == 200
    assert duplicate_confirmation.status_code == 200
    first_agent_response = first_confirmation.json()["agent_response"]
    duplicate_agent_response = duplicate_confirmation.json()["agent_response"]
    assert first_agent_response["status"] == "completed"
    assert duplicate_agent_response == first_agent_response
    assert "当前确认已失效" not in duplicate_agent_response["reply"]

    assert runtime.start_calls == 1
    assert runtime.resume_message_calls == 1
    assert len(model.calls) == 3
    assert len(writes) == 1

    applications = repository.list_applications(owner_id="feishu:ou-huawei")
    assert len(applications) == 1
    assert applications[0].company == "华为"
    assert applications[0].role == "软件开发岗位"
    assert applications[0].base_location == "深圳"

    with sqlite3.connect(database_path) as connection:
        operation_rows = connection.execute(
            "SELECT status, tool_name, owner_id FROM agent_tool_operations"
        ).fetchall()
    assert operation_rows == [("success", "create_application", "feishu:ou-huawei")]
