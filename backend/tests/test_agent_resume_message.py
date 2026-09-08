import asyncio
import json
from dataclasses import replace
from typing import Any, Dict, Iterable, List

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agents.runtime import (
    AgentActorMismatch,
    AgentInteractionError,
    AgentRuntime,
    AgentRuntimeConflict,
)
from app.core.config import settings as default_settings
from app.models.tool_calling import ModelToolCall, ModelTurn, TokenUsage
from app.services.agent_runtime_store import SQLiteAgentRuntimeStore
from app.services.llm_service import LLMRequestError
from app.tools.agent_tool import (
    AgentToolDefinition,
    AgentToolResult,
    FunctionAgentTool,
    ToolApproval,
    ToolEffect,
)
from app.tools.agent_tool_registry import AgentToolRegistry


class ScriptedModel:
    def __init__(self, turns: Iterable[ModelTurn | Exception]) -> None:
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
        turn = self.turns[index]
        if isinstance(turn, Exception):
            raise turn
        return turn


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


def _interpretation_turn(
    action: str = "approve",
    *,
    approval_intent: bool = True,
    has_changes: bool = False,
) -> ModelTurn:
    return _tool_turn(
        "interaction_interpretation",
        AgentRuntime._INTERACTION_INTERPRETER_TOOL_NAME,
        action=action,
        approval_intent=approval_intent,
        has_changes=has_changes,
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


def _write_tool(writes: List[Dict[str, Any]]) -> FunctionAgentTool:
    async def handler(arguments: Dict[str, Any]) -> AgentToolResult:
        writes.append(dict(arguments))
        return AgentToolResult(
            data={"event_id": "event-natural-resume"},
            message="calendar write succeeded",
        )

    return FunctionAgentTool(
        AgentToolDefinition(
            name="sync_calendar",
            description="Synchronize a plan to a calendar.",
            parameters={
                "type": "object",
                "properties": {"plan_id": {"type": "string"}},
                "required": ["plan_id"],
                "additionalProperties": False,
            },
            effect=ToolEffect.EXTERNAL_WRITE,
            approval=ToolApproval.ALWAYS,
            parallel_safe=False,
            idempotent=True,
        ),
        handler,
    )


def _runtime(
    model: ScriptedModel,
    tool: FunctionAgentTool,
    *,
    checkpointer=None,
    runtime_store=None,
) -> AgentRuntime:
    settings = replace(default_settings, agent_max_verifier_passes=0)
    return AgentRuntime(
        model=model,
        tool_registry=AgentToolRegistry([tool]),
        checkpointer=checkpointer or InMemorySaver(),
        runtime_store=runtime_store,
        settings=settings,
    )


def test_resume_message_is_idempotent_and_validates_actor_and_interaction() -> None:
    async def scenario():
        writes: List[Dict[str, Any]] = []
        model = ScriptedModel(
            [
                _tool_turn("calendar-call", "sync_calendar", plan_id="plan-1"),
                _interpretation_turn(),
                _answer_turn("已经同步到日历。"),
            ]
        )
        runtime = _runtime(model, _write_tool(writes))
        paused = await runtime.start(
            "把复习计划同步到日历",
            user_id="user-1",
            source="feishu",
            request_id="event-start",
        )

        with pytest.raises(AgentActorMismatch):
            await runtime.resume_message(
                paused.thread_id,
                paused.interaction.id,
                "确认",
                user_id="other-user",
                source="feishu",
                request_id="event-wrong-actor",
            )
        with pytest.raises(AgentInteractionError, match="stale|match"):
            await runtime.resume_message(
                paused.thread_id,
                "interaction-stale",
                "确认",
                user_id="user-1",
                source="feishu",
                request_id="event-stale",
            )

        resumed = await runtime.resume_message(
            paused.thread_id,
            paused.interaction.id,
            "确认",
            user_id="user-1",
            source="feishu",
            request_id="event-resume",
        )
        duplicate = await runtime.resume_message(
            paused.thread_id,
            paused.interaction.id,
            "确认",
            user_id="user-1",
            source="feishu",
            request_id="event-resume",
        )
        with pytest.raises(AgentRuntimeConflict, match="different payload"):
            await runtime.resume_message(
                paused.thread_id,
                paused.interaction.id,
                "拒绝",
                user_id="user-1",
                source="feishu",
                request_id="event-resume",
            )
        return paused, resumed, duplicate, writes, model

    paused, resumed, duplicate, writes, model = asyncio.run(scenario())

    assert paused.status.value == "waiting_for_input"
    assert resumed.status.value == "completed"
    assert resumed.reply == "已经同步到日历。"
    assert duplicate == resumed
    assert len(writes) == 1
    assert writes[0]["owner_id"] == "feishu:user-1"
    assert len(model.calls) == 3
    interpreter_call = model.calls[1]
    assert len(interpreter_call["tools"]) == 1
    assert (
        interpreter_call["tools"][0]["function"]["name"]
        == AgentRuntime._INTERACTION_INTERPRETER_TOOL_NAME
    )


def test_resume_message_interpreter_failure_reprompts_without_executing() -> None:
    async def scenario():
        writes: List[Dict[str, Any]] = []
        runtime = _runtime(
            ScriptedModel(
                [
                    _tool_turn("calendar-call", "sync_calendar", plan_id="plan-1"),
                    LLMRequestError("interpreter unavailable"),
                ]
            ),
            _write_tool(writes),
        )
        paused = await runtime.start(
            "把复习计划同步到日历",
            user_id="user-1",
            source="feishu",
        )
        response = await runtime.resume_message(
            paused.thread_id,
            paused.interaction.id,
            "确认",
            user_id="user-1",
            source="feishu",
            request_id="event-interpreter-failure",
        )
        return paused, response, writes

    paused, response, writes = asyncio.run(scenario())

    assert response.status.value == "waiting_for_input"
    assert response.interaction.id == paused.interaction.id
    assert "interaction_interpreter_unavailable" in response.warnings
    assert "没有准确理解" in response.reply
    assert "interaction_interpreter" not in response.reply
    assert writes == []


def test_resume_message_survives_restart_and_persists_request_receipt(tmp_path) -> None:
    async def scenario():
        checkpoint_path = tmp_path / "natural_resume_checkpoints.db"
        runtime_path = tmp_path / "natural_resume_runtime.db"
        writes: List[Dict[str, Any]] = []
        tool = _write_tool(writes)

        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
            await saver.setup()
            first_runtime = _runtime(
                ScriptedModel(
                    [_tool_turn("calendar-call", "sync_calendar", plan_id="plan-1")]
                ),
                tool,
                checkpointer=saver,
                runtime_store=SQLiteAgentRuntimeStore(str(runtime_path)),
            )
            paused = await first_runtime.start(
                "把复习计划同步到日历",
                user_id="user-restart",
                source="feishu",
                request_id="restart-start",
            )

        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
            await saver.setup()
            second_model = ScriptedModel(
                [_interpretation_turn(), _answer_turn("重启后已经同步到日历。")]
            )
            second_runtime = _runtime(
                second_model,
                tool,
                checkpointer=saver,
                runtime_store=SQLiteAgentRuntimeStore(str(runtime_path)),
            )
            resumed = await second_runtime.resume_message(
                paused.thread_id,
                paused.interaction.id,
                "确认",
                user_id="user-restart",
                source="feishu",
                request_id="restart-resume",
            )

        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
            await saver.setup()
            third_model = ScriptedModel([])
            third_runtime = _runtime(
                third_model,
                tool,
                checkpointer=saver,
                runtime_store=SQLiteAgentRuntimeStore(str(runtime_path)),
            )
            duplicate = await third_runtime.resume_message(
                paused.thread_id,
                paused.interaction.id,
                "确认",
                user_id="user-restart",
                source="feishu",
                request_id="restart-resume",
            )
        return paused, resumed, duplicate, writes, second_model, third_model

    paused, resumed, duplicate, writes, second_model, third_model = asyncio.run(
        scenario()
    )

    assert paused.status.value == "waiting_for_input"
    assert resumed.status.value == "completed"
    assert resumed.reply == "重启后已经同步到日历。"
    assert duplicate == resumed
    assert len(writes) == 1
    assert len(second_model.calls) == 2
    assert third_model.calls == []
