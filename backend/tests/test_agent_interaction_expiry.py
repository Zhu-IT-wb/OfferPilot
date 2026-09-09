import asyncio
import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone

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
from app.tools.agent_tool import (
    AgentToolDefinition,
    AgentToolResult,
    FunctionAgentTool,
    ToolApproval,
    ToolEffect,
    ToolInteraction,
    ToolOutcomeStatus,
)
from app.tools.agent_tool_registry import AgentToolRegistry


class FakeClock:
    def __init__(self):
        self.now = datetime(2026, 9, 9, 10, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def expire(self):
        self.now += timedelta(hours=4)


class ScriptedModel:
    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []

    async def complete(self, messages, tools, options):
        self.calls.append({"messages": messages, "tools": tools, "options": options})
        assert self.turns, "An expired task must not consume another model turn."
        return self.turns.pop(0)


def _turn(content="", *, name=None, call_id="old-call", **arguments):
    calls = [ModelToolCall(id=call_id, name=name, arguments=arguments)] if name else []
    message = {"role": "assistant", "content": content or None}
    if calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in calls
        ]
    return ModelTurn(
        assistant_message=message,
        tool_calls=calls,
        content=content,
        reasoning_content="",
        finish_reason="tool_calls" if calls else "stop",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
        provider="fake",
        model="scripted",
    )


def _pending_turn(kind):
    if kind == "approval":
        return _turn(name="sync_calendar", plan_id="old-plan")
    return _turn(
        name="request_user_input",
        kind=kind,
        prompt="Please choose how to proceed with the old task.",
        input_schema={"type": "object", "properties": {"choice": {"type": "string"}}},
    )


def _runtime(model, clock, writes, *, saver=None, store=None, max_turns=12, extra_tools=()):
    async def write(arguments):
        writes.append(dict(arguments))
        return AgentToolResult(data={"event_id": "calendar-event"}, message="Saved.")

    async def unexpected_control_execution(arguments):
        raise AssertionError("Input requests must pause before tool execution.")

    tools = [
        FunctionAgentTool(
            AgentToolDefinition(
                name="sync_calendar",
                description="Synchronize a plan to the calendar.",
                parameters={
                    "type": "object",
                    "properties": {"plan_id": {"type": "string"}},
                    "required": ["plan_id"],
                    "additionalProperties": False,
                },
                effect=ToolEffect.EXTERNAL_WRITE,
                approval=ToolApproval.ALWAYS,
                parallel_safe=False,
            ),
            write,
        ),
        FunctionAgentTool(
            AgentToolDefinition(
                name="request_user_input",
                description="Request more information from the user.",
                parameters={
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string"},
                        "prompt": {"type": "string"},
                        "input_schema": {"type": "object"},
                    },
                    "required": ["kind", "prompt"],
                },
                control=True,
                parallel_safe=False,
            ),
            unexpected_control_execution,
        ),
    ]
    return AgentRuntime(
        model=model,
        tool_registry=AgentToolRegistry([*tools, *extra_tools]),
        checkpointer=saver or InMemorySaver(),
        runtime_store=store,
        clock=clock,
        settings=replace(
            default_settings,
            agent_max_verifier_passes=0,
            agent_max_model_turns=max_turns,
            agent_interaction_ttl_seconds=1800,
        ),
    )


async def _pause(runtime):
    result = await runtime.start(
        "Continue the old calendar task.",
        user_id="user-1",
        source="feishu",
        conversation_scope="private-chat",
        request_id="initial-event",
    )
    assert result.status.value == "waiting_for_input"
    return result


def _assert_closed_old_call(messages):
    replies = [
        message
        for message in messages
        if message.get("role") == "tool" and message.get("tool_call_id") == "old-call"
    ]
    assert len(replies) == 1
    assert json.loads(replies[0]["content"])["error_code"] == "interaction_expired"
    call_ids = [
        call["id"]
        for message in messages
        for call in message.get("tool_calls", [])
    ]
    reply_ids = [
        message["tool_call_id"] for message in messages if message.get("role") == "tool"
    ]
    assert sorted(call_ids) == sorted(reply_ids)


@pytest.mark.parametrize("kind", ["approval", "clarification", "preference_form", "selection"])
@pytest.mark.parametrize("entrypoint", ["start", "resume_message"])
def test_expired_interaction_retires_without_work_and_starts_fresh_turn(kind, entrypoint):
    async def scenario():
        clock, writes = FakeClock(), []
        model = ScriptedModel([_pending_turn(kind), _turn("Here is the new answer.")])
        runtime = _runtime(model, clock, writes, max_turns=1)
        paused = await _pause(runtime)
        assert paused.interaction.type.value == kind
        clock.expire()
        message = "LangChain\u4e0eLangGraph\u7684\u4e86\u89e3"
        if entrypoint == "start":
            result = await runtime.start(
                message,
                user_id="user-1",
                source="feishu",
                conversation_scope="private-chat",
                request_id="new-event",
            )
        else:
            result = await runtime.resume_message(
                paused.thread_id,
                paused.interaction.id,
                message,
                user_id="user-1",
                source="feishu",
                request_id="new-event",
            )
        assert result.status.value == "completed"
        assert result.reply == "Here is the new answer."
        assert result.run_id != paused.run_id
        assert result.thread_id == paused.thread_id
        assert result.interaction is None
        assert result.plan is None
        assert result.tool_executions == []
        assert result.usage.model_calls == 1
        assert result.usage.tool_calls == 0
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 3
        assert writes == []
        assert len(model.calls) == 2
        new_call = model.calls[1]
        assert len(new_call["tools"]) == 2
        user_messages = [m["content"] for m in new_call["messages"] if m["role"] == "user"]
        assert user_messages[-1] == message
        _assert_closed_old_call(new_call["messages"])
        snapshot = await runtime.graph.aget_state(runtime._config(paused.thread_id))
        assert snapshot.next == ()
        assert snapshot.values["pending_calls"] == []
        assert snapshot.values["validated_calls"] == []
        assert snapshot.values["approval_granted"] == []

    asyncio.run(scenario())


def test_expired_task_discards_unfinished_plan_and_spent_budget():
    async def scenario():
        clock, writes = FakeClock(), []

        async def save_plan(arguments):
            return AgentToolResult(data=dict(arguments), message="Plan updated.")

        plan_tool = FunctionAgentTool(
            AgentToolDefinition(
                name="update_execution_plan",
                description="Update the optional execution plan.",
                parameters={
                    "type": "object",
                    "properties": {"goal": {"type": "string"}, "steps": {"type": "array"}},
                },
                control=True,
                parallel_safe=False,
            ),
            save_plan,
        )
        model = ScriptedModel([
            _turn(
                name="update_execution_plan", call_id="old-plan-call",
                goal="Old goal", steps=[{"id": "sync", "description": "Synchronize old plan", "status": "pending"}],
            ),
            _pending_turn("approval"),
            _turn("Here is the answer to your new question."),
        ])
        runtime = _runtime(model, clock, writes, max_turns=2, extra_tools=[plan_tool])
        paused = await _pause(runtime)
        assert paused.plan is not None
        assert paused.usage.model_calls == 2
        assert paused.usage.tool_calls == 1
        clock.expire()
        result = await runtime.resume_message(
            paused.thread_id, paused.interaction.id, "A new question.",
            user_id="user-1", source="feishu",
        )
        assert result.status.value == "completed"
        assert result.plan is None
        assert result.usage.model_calls == 1
        assert result.usage.tool_calls == 0
        assert result.usage.verifier_calls == 0
        assert len(model.calls) == 3
        _assert_closed_old_call(model.calls[-1]["messages"])
        assert writes == []

    asyncio.run(scenario())


def test_expired_tool_generated_interaction_closes_existing_call_without_reexecuting():
    async def scenario():
        clock, writes, reads = FakeClock(), [], []

        async def read_choices(arguments):
            reads.append(dict(arguments))
            return AgentToolResult(
                data={},
                status=ToolOutcomeStatus.NEEDS_INPUT,
                interaction=ToolInteraction(
                    kind="selection", prompt="Choose one of the two subscriptions.",
                ),
            )

        read_tool = FunctionAgentTool(
            AgentToolDefinition(
                name="read_subscription_choices",
                description="Look up subscriptions.",
                parameters={"type": "object", "properties": {}},
            ),
            read_choices,
        )
        model = ScriptedModel([
            _turn(name="read_subscription_choices"),
            _turn("Here is the answer to your new question."),
        ])
        runtime = _runtime(model, clock, writes, extra_tools=[read_tool])
        paused = await _pause(runtime)
        assert len(reads) == 1
        assert paused.interaction.type.value == "selection"
        clock.expire()
        result = await runtime.resume_message(
            paused.thread_id, paused.interaction.id, "A new question.",
            user_id="user-1", source="feishu",
        )
        assert result.status.value == "completed"
        assert len(reads) == 1
        assert result.usage.tool_calls == 0
        assert len(model.calls) == 2
        _assert_closed_old_call(model.calls[-1]["messages"])
        assert writes == []

    asyncio.run(scenario())


@pytest.mark.parametrize("decision", ["approve", "revise", "answer", "reject"])
def test_expired_structured_decision_cannot_authorize_or_replan_old_task(decision):
    async def scenario():
        clock, writes = FakeClock(), []
        model = ScriptedModel([_pending_turn("approval")])
        runtime = _runtime(model, clock, writes)
        paused = await _pause(runtime)
        clock.expire()
        with pytest.raises(AgentInteractionError, match="expired|invalid|allowed"):
            await runtime.resume(
                paused.thread_id,
                paused.interaction.id,
                decision,
                user_id="user-1",
                source="feishu",
            )
        current = await runtime.get_state(paused.thread_id, user_id="user-1", source="feishu")
        assert current.interaction.id == paused.interaction.id
        assert len(model.calls) == 1
        assert writes == []

    asyncio.run(scenario())


def test_expired_structured_cancel_releases_pending_without_model_call():
    async def scenario():
        clock, writes = FakeClock(), []
        model = ScriptedModel([_pending_turn("selection")])
        runtime = _runtime(model, clock, writes)
        paused = await _pause(runtime)
        clock.expire()
        result = await runtime.resume(
            paused.thread_id,
            paused.interaction.id,
            "cancel",
            user_id="user-1",
            source="feishu",
        )
        assert result.interaction is None
        assert len(model.calls) == 1
        assert writes == []
        snapshot = await runtime.graph.aget_state(runtime._config(paused.thread_id))
        assert snapshot.next == ()

    asyncio.run(scenario())


def test_expired_natural_cancel_is_forwarded_to_agent_without_keyword_routing():
    async def scenario():
        clock, writes = FakeClock(), []
        model = ScriptedModel([_pending_turn("selection"), _turn("The old task is closed.")])
        runtime = _runtime(model, clock, writes)
        paused = await _pause(runtime)
        clock.expire()
        message = "\u53d6\u6d88\u5f53\u524d\u4efb\u52a1"
        result = await runtime.resume_message(
            paused.thread_id,
            paused.interaction.id,
            message,
            user_id="user-1",
            source="feishu",
        )
        assert result.status.value == "completed"
        assert result.interaction is None
        assert len(model.calls) == 2
        assert model.calls[1]["messages"][-1]["content"] == message
        _assert_closed_old_call(model.calls[1]["messages"])
        assert writes == []

    asyncio.run(scenario())


def test_expired_interaction_still_validates_owner_and_interaction_before_retirement():
    async def scenario():
        clock, writes = FakeClock(), []
        model = ScriptedModel([_pending_turn("approval")])
        runtime = _runtime(model, clock, writes)
        paused = await _pause(runtime)
        clock.expire()
        for user_id, interaction_id, expected in [
            ("other-user", paused.interaction.id, AgentActorMismatch),
            ("user-1", "stale-interaction", AgentInteractionError),
        ]:
            with pytest.raises(expected):
                await runtime.resume_message(
                    paused.thread_id,
                    interaction_id,
                    "New question",
                    user_id=user_id,
                    source="feishu",
                )
        current = await runtime.get_state(paused.thread_id, user_id="user-1", source="feishu")
        assert current.interaction.id == paused.interaction.id
        assert len(model.calls) == 1
        assert writes == []

    asyncio.run(scenario())


def test_expired_approval_new_write_requires_new_confirmation():
    async def scenario():
        clock, writes = FakeClock(), []
        model = ScriptedModel([
            _pending_turn("approval"),
            _turn(name="sync_calendar", call_id="new-call", plan_id="new-plan"),
            _turn("The new plan was synchronized."),
        ])
        runtime = _runtime(model, clock, writes)
        paused = await _pause(runtime)
        clock.expire()
        refreshed = await runtime.resume_message(
            paused.thread_id,
            paused.interaction.id,
            "Please synchronize the new plan instead.",
            user_id="user-1",
            source="feishu",
            request_id="new-write-event",
        )
        assert refreshed.status.value == "waiting_for_input"
        assert refreshed.interaction.id != paused.interaction.id
        assert refreshed.interaction.arguments["plan_id"] == "new-plan"
        assert refreshed.run_id != paused.run_id
        assert writes == []
        with pytest.raises(AgentInteractionError):
            await runtime.resume(
                paused.thread_id, paused.interaction.id, "approve",
                user_id="user-1", source="feishu",
            )
        completed = await runtime.resume(
            refreshed.thread_id, refreshed.interaction.id, "approve",
            user_id="user-1", source="feishu",
        )
        assert completed.status.value == "completed"
        assert len(writes) == 1
        assert writes[0]["plan_id"] == "new-plan"

    asyncio.run(scenario())


def test_expired_resume_survives_sqlite_restart_and_replays_new_request_once(tmp_path):
    async def scenario():
        clock, writes = FakeClock(), []
        checkpoint_path = str(tmp_path / "expiry_checkpoints.db")
        runtime_path = str(tmp_path / "expiry_runtime.db")
        async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
            await saver.setup()
            runtime = _runtime(
                ScriptedModel([_pending_turn("selection")]), clock, writes,
                saver=saver, store=SQLiteAgentRuntimeStore(runtime_path),
            )
            paused = await _pause(runtime)
        clock.expire()
        kwargs = dict(
            thread_id=paused.thread_id,
            interaction_id=paused.interaction.id,
            message="Explain LangGraph.",
            user_id="user-1",
            source="feishu",
            request_id="new-event-after-restart",
        )
        async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
            await saver.setup()
            model = ScriptedModel([_turn("A graph execution runtime.")])
            runtime = _runtime(model, clock, writes, saver=saver, store=SQLiteAgentRuntimeStore(runtime_path))
            completed = await runtime.resume_message(**kwargs)
            assert completed.status.value == "completed"
            assert completed.run_id != paused.run_id
            assert len(model.calls) == 1
        async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
            await saver.setup()
            model = ScriptedModel([])
            runtime = _runtime(model, clock, writes, saver=saver, store=SQLiteAgentRuntimeStore(runtime_path))
            assert await runtime.resume_message(**kwargs) == completed
            with pytest.raises(AgentRuntimeConflict, match="different payload"):
                await runtime.resume_message(**{**kwargs, "message": "Different question"})
            assert model.calls == []
            state = await runtime.get_state(paused.thread_id, user_id="user-1", source="feishu")
            assert state.interaction is None
            assert state.run_id == completed.run_id
        assert writes == []

    asyncio.run(scenario())


def test_expiration_recovery_after_resolver_crash_keeps_new_message(tmp_path, monkeypatch):
    async def scenario():
        clock, writes = FakeClock(), []
        checkpoint_path = str(tmp_path / "crashed_expiry_checkpoints.db")
        runtime_path = str(tmp_path / "crashed_expiry_runtime.db")
        original_resolve = AgentRuntime._resolve_resume

        async def crash_expiration(runtime, state):
            if (state.get("resume_input") or {}).get("mode") == "expiration":
                raise RuntimeError("Simulated crash during expiration cleanup")
            return await original_resolve(runtime, state)

        with monkeypatch.context() as patch:
            patch.setattr(AgentRuntime, "_resolve_resume", crash_expiration)
            async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
                await saver.setup()
                first_model = ScriptedModel([_pending_turn("approval")])
                runtime = _runtime(
                    first_model, clock, writes,
                    saver=saver, store=SQLiteAgentRuntimeStore(runtime_path),
                )
                paused = await _pause(runtime)
                clock.expire()
                kwargs = dict(
                    thread_id=paused.thread_id,
                    interaction_id=paused.interaction.id,
                    message="Explain LangGraph after the restart.",
                    user_id="user-1",
                    source="feishu",
                    request_id="resume-interrupted-by-crash",
                )
                with pytest.raises(RuntimeError, match="Simulated crash"):
                    await runtime.resume_message(**kwargs)
                snapshot = await runtime.graph.aget_state(runtime._config(paused.thread_id))
                assert snapshot.values["resume_input"]["mode"] == "expiration"
                assert "resolve_resume" in snapshot.next
                assert len(first_model.calls) == 1
                assert writes == []

        async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
            await saver.setup()
            model = ScriptedModel([_turn("The new question is answered.")])
            runtime = _runtime(
                model, clock, writes, saver=saver, store=SQLiteAgentRuntimeStore(runtime_path),
            )
            response = await runtime.resume_message(**kwargs)
            assert response.status.value == "completed"
            assert response.run_id != paused.run_id
            assert response.interaction is None
            assert len(model.calls) == 1
            assert model.calls[0]["messages"][-1]["content"] == kwargs["message"]
            _assert_closed_old_call(model.calls[0]["messages"])
            assert await runtime.resume_message(**kwargs) == response
            assert len(model.calls) == 1
            assert writes == []

    asyncio.run(scenario())


@pytest.mark.parametrize("recovery_method", ["resume_message", "replay_request"])
def test_expiration_new_turn_recovers_before_first_checkpoint(tmp_path, monkeypatch, recovery_method):
    async def scenario():
        clock, writes = FakeClock(), []
        checkpoint_path = str(tmp_path / "new_turn_checkpoints.db")
        runtime_path = str(tmp_path / "new_turn_receipts.db")
        message = "LangChain\u4e0eLangGraph\u7684\u4e86\u89e3\uff0c\u8bf7\u7528\u4e2d\u6587\u8bf4\u660e\u3002"
        async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
            await saver.setup()
            first_model = ScriptedModel([_pending_turn("approval")])
            runtime = _runtime(
                first_model, clock, writes,
                saver=saver, store=SQLiteAgentRuntimeStore(runtime_path),
            )
            paused = await _pause(runtime)
            clock.expire()
            kwargs = dict(
                thread_id=paused.thread_id,
                interaction_id=paused.interaction.id,
                message=message,
                user_id="user-1",
                source="feishu",
                request_id="new-turn-checkpoint-crash",
            )
            original_record = runtime._record_event
            failures = []

            async def fail_new_user_event(**event):
                if (
                    not failures
                    and event.get("event_type") == "user_message"
                    and event.get("payload", {}).get("content") == message
                ):
                    failures.append(event["run_id"])
                    raise RuntimeError("Simulated crash before the new turn checkpoint")
                return await original_record(**event)

            with monkeypatch.context() as patch:
                patch.setattr(runtime, "_record_event", fail_new_user_event)
                with pytest.raises(RuntimeError, match="Simulated crash"):
                    await runtime.resume_message(**kwargs)
            assert len(failures) == 1
            new_run_id = failures[0]
            assert new_run_id != paused.run_id
            assert len(first_model.calls) == 1
            assert writes == []
            receipt_scope = runtime._receipt_scope("feishu", "user-1")
            receipt = runtime.runtime_store.get_receipt_record(receipt_scope, kwargs["request_id"])
            assert receipt is not None
            assert receipt.status == "processing"
            assert receipt.run_id == new_run_id
            snapshot = await runtime.graph.aget_state(runtime._config(paused.thread_id))
            assert snapshot.values["pending_interaction"] is None
            _assert_closed_old_call(snapshot.values["messages"])

        async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
            await saver.setup()
            model = ScriptedModel([_turn("The new question is answered.")])
            runtime = _runtime(
                model, clock, writes,
                saver=saver, store=SQLiteAgentRuntimeStore(runtime_path),
            )
            if recovery_method == "resume_message":
                response = await runtime.resume_message(**kwargs)
            else:
                response = await runtime.replay_request(
                    kwargs["request_id"], thread_id=paused.thread_id,
                    user_id="user-1", source="feishu",
                )
            assert response is not None
            assert response.status.value == "completed"
            assert response.reply == "The new question is answered."
            assert response.run_id == new_run_id
            assert response.interaction is None
            assert response.usage.model_calls == 1
            assert len(model.calls) == 1
            assert model.calls[0]["messages"][-1]["content"] == message
            assert sum(
                item.get("role") == "user" and item.get("content") == message
                for item in model.calls[0]["messages"]
            ) == 1
            _assert_closed_old_call(model.calls[0]["messages"])
            assert writes == []

        async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
            await saver.setup()
            replay_model = ScriptedModel([])
            runtime = _runtime(
                replay_model, clock, writes,
                saver=saver, store=SQLiteAgentRuntimeStore(runtime_path),
            )
            assert await runtime.resume_message(**kwargs) == response
            assert await runtime.replay_request(
                kwargs["request_id"], thread_id=paused.thread_id,
                user_id="user-1", source="feishu",
            ) == response
            receipt = runtime.runtime_store.get_receipt_record(receipt_scope, kwargs["request_id"])
            assert receipt.status == "completed"
            assert receipt.run_id == new_run_id
            assert replay_model.calls == []
            assert writes == []
        with sqlite3.connect(runtime_path) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM agent_ingress_receipts WHERE source = ? AND request_id = ?",
                (receipt_scope, kwargs["request_id"]),
            ).fetchone()[0]
        assert count == 1

    asyncio.run(scenario())
