import asyncio
import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agents.runtime import AgentActorMismatch, AgentRuntime, AgentRuntimeConflict
from app.core.config import settings as default_settings
from app.models.tool_calling import ModelToolCall, ModelTurn, TokenUsage
from app.services.agent_runtime_store import (
    InMemoryAgentRuntimeStore,
    SQLiteAgentRuntimeStore,
)
from app.services.llm_service import LLMRequestError
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


ToolHandler = Callable[[Dict[str, Any]], Awaitable[AgentToolResult]]


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


def _tool_turn(*calls: ModelToolCall) -> ModelTurn:
    assistant_calls = [
        {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(call.arguments, ensure_ascii=False),
            },
        }
        for call in calls
    ]
    return ModelTurn(
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_calls,
            "reasoning_content": "hidden reasoning must not enter checkpoints",
        },
        tool_calls=list(calls),
        content="",
        reasoning_content="hidden reasoning must not enter checkpoints",
        finish_reason="tool_calls",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14),
        provider="fake",
        model="scripted",
    )


def _answer_turn(content: str) -> ModelTurn:
    return ModelTurn(
        assistant_message={
            "role": "assistant",
            "content": content,
            "reasoning_content": "hidden reasoning must not enter checkpoints",
        },
        tool_calls=[],
        content=content,
        reasoning_content="hidden reasoning must not enter checkpoints",
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=8, completion_tokens=3, total_tokens=11),
        provider="fake",
        model="scripted",
    )


def _call(call_id: str, name: str, **arguments: Any) -> ModelToolCall:
    return ModelToolCall(id=call_id, name=name, arguments=arguments)


def test_execution_plan_revision_increments_when_the_model_omits_it() -> None:
    initial = AgentRuntime._normalize_plan(
        {
            "goal": "准备面试",
            "steps": [
                {"id": "read", "description": "读取面试", "status": "pending"}
            ],
        },
        {"current_user_message": "准备面试"},
    )
    revised = AgentRuntime._normalize_plan(
        {
            "goal": "准备面试",
            "steps": [
                {
                    "id": "read",
                    "description": "读取面试",
                    "status": "completed",
                }
            ],
        },
        {"plan": initial},
    )

    assert initial["revision"] == 1
    assert revised["revision"] == 2


def _tool(
    name: str,
    handler: ToolHandler,
    *,
    effect: ToolEffect = ToolEffect.READ,
    approval: ToolApproval = ToolApproval.NEVER,
    parallel_safe: bool = True,
    control: bool = False,
    properties: Dict[str, Any] | None = None,
    required: List[str] | None = None,
    timeout_seconds: float = 30.0,
    idempotent: bool | None = None,
) -> FunctionAgentTool:
    return FunctionAgentTool(
        AgentToolDefinition(
            name=name,
            description=f"Test tool {name}",
            parameters={
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
            effect=effect,
            approval=approval,
            parallel_safe=parallel_safe,
            idempotent=effect == ToolEffect.READ if idempotent is None else idempotent,
            timeout_seconds=timeout_seconds,
            control=control,
        ),
        handler,
    )


def _generic_reconciliation_tool() -> FunctionAgentTool:
    async def handler(arguments: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(
            data={
                "operation_key": arguments["operation_key"],
                "reconciled_outcome": arguments["outcome"],
                "evidence": arguments["evidence"],
            },
            message="Reconciliation assertion accepted.",
        )

    return _tool(
        "reconcile_write_operation",
        handler,
        effect=ToolEffect.LOCAL_WRITE,
        approval=ToolApproval.ALWAYS,
        parallel_safe=False,
        control=True,
        idempotent=True,
        properties={
            "operation_key": {"type": "string", "minLength": 1},
            "outcome": {
                "type": "string",
                "enum": ["applied", "not_applied"],
            },
            "evidence": {"type": "string", "minLength": 1},
        },
        required=["operation_key", "outcome", "evidence"],
    )


def _runtime(
    model: ScriptedModel,
    tools: Iterable[FunctionAgentTool] = (),
    *,
    max_model_turns: int = 12,
    max_tool_calls: int = 20,
    no_progress_limit: int = 3,
    max_verifier_passes: int = 0,
) -> AgentRuntime:
    settings = replace(
        default_settings,
        agent_max_model_turns=max_model_turns,
        agent_max_tool_calls=max_tool_calls,
        agent_no_progress_limit=no_progress_limit,
        agent_max_verifier_passes=max_verifier_passes,
    )
    return AgentRuntime(
        model=model,
        tool_registry=AgentToolRegistry(tools),
        checkpointer=InMemorySaver(),
        settings=settings,
    )


def test_runtime_wires_context_compaction_settings() -> None:
    runtime_settings = replace(
        default_settings,
        context_max_input_tokens=5_000,
        context_compaction_enabled=False,
        context_compaction_trigger_tokens=1_200,
        context_keep_recent_turns=4,
        context_summary_max_tokens=250,
    )
    runtime = AgentRuntime(
        model=ScriptedModel([]),
        tool_registry=AgentToolRegistry(),
        checkpointer=InMemorySaver(),
        settings=runtime_settings,
    )

    assert runtime.context_manager.max_input_tokens == 5_000
    assert runtime.context_manager.compaction_enabled is False
    assert runtime.context_manager.compaction_trigger_tokens == 1_200
    assert runtime.context_manager.recent_conversation_turns == 4
    assert runtime.context_manager.conversation_summary_max_tokens == 250


def test_runtime_returns_direct_answer_without_tools() -> None:
    model = ScriptedModel([_answer_turn("这是一个直接回答。")])
    runtime = _runtime(model)

    response = asyncio.run(
        runtime.start("解释什么是 ReAct", user_id="user_1", request_id="req_direct")
    )

    assert response.status.value == "completed"
    assert response.reply == "这是一个直接回答。"
    assert response.tool_executions == []
    assert response.usage.model_calls == 1
    assert response.usage.tool_calls == 0
    assert model.calls[0]["messages"][-1] == {
        "role": "user",
        "content": "解释什么是 ReAct",
    }
    assert all("reasoning" not in message for message in model.calls[0]["messages"])


@pytest.mark.parametrize(
    "failed_attempt",
    [LLMRequestError("temporary model failure"), _answer_turn("")],
    ids=["request_error", "empty_answer"],
)
def test_recovered_model_answer_is_completed_and_persisted(failed_attempt) -> None:
    from app.api.routes.feishu import _build_plain_response_card

    model = ScriptedModel(
        [
            failed_attempt,
            _answer_turn(
                '{"complete":false,"disposition":"recoverable",'
                '"missing":["Respond to the greeting"],"completed_goal_items":[],'
                '"draft_safe_to_show":false}'
            ),
            _answer_turn("你好，有什么需要我帮忙的吗？"),
            _answer_turn(
                '{"complete":true,"disposition":"complete","missing":[], '
                '"completed_goal_items":["Responded to the greeting"],'
                '"draft_safe_to_show":true}'
            ),
        ]
    )
    runtime = _runtime(model, max_verifier_passes=2)

    async def scenario():
        response = await runtime.start("你好", user_id="recovered", request_id="retry")
        persisted = await runtime.get_state(response.thread_id)
        replayed = await runtime.start("你好", user_id="recovered", request_id="retry")
        return response, persisted, replayed

    response, persisted, replayed = asyncio.run(scenario())

    assert response.status.value == "completed"
    assert response.reply == "你好，有什么需要我帮忙的吗？"
    assert persisted.status == replayed.status == response.status
    assert response.usage.model_calls == len(model.calls) == 4
    assert response.usage.verifier_calls == 2
    card = _build_plain_response_card(response.reply, response.status)
    assert card["header"]["template"] == "green"
    assert card["header"]["title"]["content"] == "处理完成"


def test_model_recovery_through_tools_preserves_unresolved_write_status() -> None:
    attempts = 0

    async def write_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        nonlocal attempts
        attempts += 1
        return AgentToolResult(
            status=ToolOutcomeStatus.UNKNOWN,
            message="Write response lost; do not retry.",
        )

    model = ScriptedModel(
        [
            LLMRequestError("temporary failure"),
            _answer_turn('{"complete":false,"missing":["Attempt the requested write"]}'),
            _tool_turn(_call("recovered_write", "write_local")),
            _answer_turn("尚不能确认记录是否保存，请先核对。"),
            _answer_turn('{"complete":true,"missing":[]}'),
        ]
    )
    runtime = _runtime(
        model,
        [_tool("write_local", write_handler, effect=ToolEffect.LOCAL_WRITE)],
        max_verifier_passes=2,
    )

    response = asyncio.run(runtime.start("保存记录"))

    assert response.status.value == "partial"
    assert "write_outcome_unknown" in response.warnings
    assert response.tool_executions[0].status.value == "unknown"
    assert attempts == 1


@pytest.mark.parametrize("prior_status", ["partial", "degraded"])
def test_model_retry_restores_preexisting_outcome(prior_status: str) -> None:
    runtime = _runtime(
        ScriptedModel([LLMRequestError("temporary failure"), _answer_turn("Result")])
    )

    async def scenario():
        state = {
            "status": prior_status,
            "current_user_message": "Continue",
            "messages": [{"role": "user", "content": "Continue"}],
        }
        failed = await runtime._agent(state)
        recovered = await runtime._agent({**state, **failed})
        return failed, recovered

    failed, recovered = asyncio.run(scenario())
    assert failed["status"] == "failed"
    assert recovered["status"] == prior_status
    assert recovered["status_before_model_failure"] is None


def test_unrecovered_model_failure_stays_failed() -> None:
    model = ScriptedModel([LLMRequestError("model unavailable")])
    response = asyncio.run(_runtime(model).start("你好"))

    assert response.status.value == "failed"
    assert "模型服务暂时不可用" in response.reply


def test_runtime_injects_a_stable_trusted_clock_into_every_model_turn() -> None:
    fixed_now = datetime(2026, 9, 2, 1, 30, tzinfo=timezone.utc)
    model = ScriptedModel([_answer_turn("本周范围已确定。")])
    runtime = AgentRuntime(
        model=model,
        tool_registry=AgentToolRegistry(),
        checkpointer=InMemorySaver(),
        runtime_store=InMemoryAgentRuntimeStore(),
        settings=replace(
            default_settings,
            agent_max_verifier_passes=0,
            feishu_calendar_timezone="Asia/Shanghai",
        ),
        clock=lambda: fixed_now,
    )

    asyncio.run(runtime.start("查询本周安排", user_id="clock_user"))

    system_message = model.calls[0]["messages"][0]["content"]
    assert "current_time=2026-09-02T09:30:00+08:00" in system_message
    assert "timezone=Asia/Shanghai" in system_message
    assert "current_week=[2026-08-31, 2026-09-07)" in system_message


def test_private_data_answer_requires_a_successful_read_receipt() -> None:
    async def list_interviews(arguments: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(data={"interviews": [{"company": "Acme"}]})

    model = ScriptedModel(
        [
            _answer_turn("你本周有一场 Acme 面试。"),
            _answer_turn(
                '{"complete":false,"missing":["缺少 list_interviews 成功读取回执"]}'
            ),
            _tool_turn(_call("read_interviews", "list_interviews")),
            _answer_turn("根据查询回执，你本周有一场 Acme 面试。"),
            _answer_turn('{"complete":true,"missing":[]}'),
        ]
    )
    runtime = _runtime(
        model,
        [_tool("list_interviews", list_interviews)],
        max_verifier_passes=2,
    )

    response = asyncio.run(runtime.start("查询我本周的面试", user_id="read_guard"))

    assert response.status.value == "completed"
    assert response.usage.model_calls == 5
    assert response.usage.verifier_calls == 2
    assert [item.name for item in response.tool_executions] == ["list_interviews"]
    assert "缺少 list_interviews 成功读取回执" in str(
        model.calls[2]["messages"]
    )


def test_verifier_feedback_does_not_leak_into_the_next_task() -> None:
    async def list_interviews(arguments: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(data={"interviews": []})

    model = ScriptedModel(
        [
            _answer_turn("你本周没有面试。"),
            _answer_turn(
                '{"complete":false,"missing":["缺少 list_interviews 成功读取回执"]}'
            ),
            _tool_turn(_call("guard_read", "list_interviews")),
            _answer_turn("根据查询回执，你本周没有面试。"),
            _answer_turn('{"complete":true,"missing":[]}'),
            _answer_turn("你好。"),
            _answer_turn('{"complete":true,"missing":[]}'),
        ]
    )
    runtime = _runtime(
        model,
        [_tool("list_interviews", list_interviews)],
        max_verifier_passes=2,
    )

    async def scenario() -> None:
        await runtime.start("查询我本周的面试", user_id="feedback_scope")
        await runtime.start("你好", user_id="feedback_scope")

    asyncio.run(scenario())

    assert "runtime_verifier_untrusted_evidence" in str(model.calls[2]["messages"])
    assert "runtime_verifier_untrusted_evidence" not in str(model.calls[5]["messages"])
    assert "缺少 list_interviews 成功读取回执" not in str(
        model.calls[5]["messages"]
    )


def test_private_data_hallucination_at_budget_fails_without_exposing_verifier() -> None:
    async def list_interviews(arguments: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(data={"interviews": []})

    runtime = _runtime(
        ScriptedModel(
            [
                _answer_turn("你本周没有面试。"),
                _answer_turn(
                    '{"complete":false,"missing":["缺少 list_interviews 成功读取回执"]}'
                ),
            ]
        ),
        [_tool("list_interviews", list_interviews)],
        max_model_turns=2,
        max_verifier_passes=1,
    )

    response = asyncio.run(runtime.start("查看我的面试安排", user_id="read_guard"))

    assert response.status.value == "failed"
    assert "completion_verifier_incomplete" in response.warnings
    assert response.usage.verifier_calls == 1
    assert response.tool_executions == []
    assert "你本周没有面试" not in response.reply
    assert response.reply == "这次没有拿到可靠结果，因此未将操作视为完成。"
    assert "核验" not in response.reply
    assert "回执" not in response.reply


def test_enabled_verifier_fails_closed_when_no_model_budget_remains() -> None:
    response = asyncio.run(
        _runtime(
            ScriptedModel([_answer_turn("投递记录已经创建。")]),
            max_model_turns=1,
            max_verifier_passes=1,
        ).start("帮我新增投递：Example Tech")
    )

    assert response.status.value == "failed"
    assert "completion_unverified_budget" in response.warnings
    assert "投递记录已经创建" not in response.reply
    assert response.reply == "这次没有拿到可靠结果，因此未将操作视为完成。"
    assert "核验" not in response.reply
    assert response.usage.model_calls == 1
    assert response.usage.verifier_calls == 0


def test_bitable_artifact_is_trusted_completion_evidence() -> None:
    table_url = "https://example.feishu.cn/base/app_table?table=applications"

    async def get_link(arguments: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(
            data={"available": True},
            message="已获取投递表链接。",
            artifact_refs=[
                {
                    "type": "feishu_bitable",
                    "id": "applications_table",
                    "title": "投递记录",
                    "url": table_url,
                    "data": {"button_text": "打开多维表格"},
                }
            ],
        )

    model = ScriptedModel(
        [
            _tool_turn(
                _call("get_applications_table", "get_application_bitable_link")
            ),
            _answer_turn("已经找到你的投递多维表格，链接在结果按钮中。"),
            _answer_turn(
                json.dumps(
                    {
                        "complete": True,
                        "disposition": "complete",
                        "missing": [],
                        "completed_goal_items": ["已提供投递记录表链接"],
                        "draft_safe_to_show": True,
                    },
                    ensure_ascii=False,
                )
            ),
        ]
    )
    runtime = _runtime(
        model,
        [_tool("get_application_bitable_link", get_link)],
        max_verifier_passes=1,
    )

    response = asyncio.run(runtime.start("你能把飞书多维表格的链接发给我吗？"))

    assert response.status.value == "completed"
    assert table_url not in response.reply
    assert response.reply == "已经找到你的投递多维表格，链接在结果按钮中。"
    assert response.artifacts[0].id == "applications_table"
    assert response.artifacts[0].url == table_url
    verifier_payload = json.loads(model.calls[2]["messages"][1]["content"])
    assert verifier_payload["tool_executions"][0]["result_refs"] == [
        "applications_table"
    ]
    assert verifier_payload["artifacts"] == [
        {
            "id": "applications_table",
            "type": "feishu_bitable",
            "title": "投递记录",
            "url": table_url,
            "trusted_evidence": True,
            "successful_result_refs": [
                {
                    "tool_call_id": "get_applications_table",
                    "tool_name": "get_application_bitable_link",
                }
            ],
        }
    ]


def test_honest_capability_answer_can_complete_without_a_tool_receipt() -> None:
    answer = "我支持查询投递记录，但目前不支持导出为 Excel。"
    model = ScriptedModel(
        [
            _answer_turn(answer),
            _answer_turn(
                '{"complete":true,"disposition":"complete","missing":[],'
                '"completed_goal_items":[],"draft_safe_to_show":true}'
            ),
        ]
    )

    response = asyncio.run(
        _runtime(model, max_verifier_passes=1).start(
            "你现在支持把投递记录导出成 Excel 吗？"
        )
    )

    assert response.status.value == "completed"
    assert response.reply == answer


def test_unsupported_operation_keeps_safe_limitation_without_retrying() -> None:
    limitation = "我目前还不能把投递记录导出为 PDF。"
    model = ScriptedModel(
        [
            _answer_turn(limitation),
            _answer_turn(
                '{"complete":false,"disposition":"unsupported",'
                '"missing":["无法导出 PDF"],"completed_goal_items":[],'
                '"draft_safe_to_show":true}'
            ),
        ]
    )

    response = asyncio.run(
        _runtime(model, max_verifier_passes=2).start("把投递记录导出为 PDF")
    )

    assert response.status.value == "failed"
    assert response.reply == limitation
    assert response.warnings == ["completion_unsupported"]
    assert response.usage.model_calls == 2
    assert "核验" not in response.reply


def test_partial_reply_lists_only_verified_user_goal_items() -> None:
    async def read_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(data={"ok": True}, message="read complete")

    model = ScriptedModel(
        [
            _tool_turn(_call("incidental_read", "read_a")),
            _answer_turn("所有内容都处理好了。"),
            _answer_turn(
                '{"complete":false,"disposition":"partial",'
                '"missing":["日历同步未完成"],'
                '"completed_goal_items":["已整理本周面试安排"],'
                '"draft_safe_to_show":false}'
            ),
        ]
    )

    response = asyncio.run(
        _runtime(
            model,
            [_tool("read_a", read_handler)],
            max_verifier_passes=2,
        ).start("整理本周面试并同步日历")
    )

    assert response.status.value == "partial"
    assert response.reply == (
        "这次只完成了部分内容，未完成部分没有获得可靠结果。"
        "已完成：已整理本周面试安排。"
    )
    assert "read_a" not in response.reply
    assert "查询" not in response.reply
    assert response.usage.model_calls == 3


def test_unavailable_dependency_with_a_fallback_result_is_degraded() -> None:
    fallback = "已使用本地日程生成计划，但暂时无法读取飞书忙闲。"
    model = ScriptedModel(
        [
            _answer_turn(fallback),
            _answer_turn(
                '{"complete":false,"disposition":"unavailable",'
                '"missing":["飞书忙闲暂不可用"],'
                '"completed_goal_items":["已生成本地复习计划"],'
                '"draft_safe_to_show":true}'
            ),
        ]
    )

    response = asyncio.run(
        _runtime(model, max_verifier_passes=2).start(
            "读取飞书忙闲并生成复习计划"
        )
    )

    assert response.status.value == "degraded"
    assert response.reply == fallback
    assert response.warnings == ["completion_dependency_unavailable"]
    assert response.usage.model_calls == 2


def test_relative_write_time_uses_the_persisted_turn_clock_before_interrupt() -> None:
    async def handler(arguments: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(data={"scheduled": True})

    fixed_now = datetime(2026, 9, 2, 1, 30, tzinfo=timezone.utc)
    model = ScriptedModel(
        [
            _tool_turn(
                _call(
                    "relative_schedule",
                    "schedule_interview",
                    interview_time="明天早上八点",
                )
            )
        ]
    )
    runtime = AgentRuntime(
        model=model,
        tool_registry=AgentToolRegistry(
            [
                _tool(
                    "schedule_interview",
                    handler,
                    effect=ToolEffect.EXTERNAL_WRITE,
                    approval=ToolApproval.ALWAYS,
                    parallel_safe=False,
                    properties={"interview_time": {"type": "string"}},
                    required=["interview_time"],
                )
            ]
        ),
        checkpointer=InMemorySaver(),
        runtime_store=InMemoryAgentRuntimeStore(),
        settings=replace(
            default_settings,
            agent_max_verifier_passes=0,
            feishu_calendar_timezone="Asia/Shanghai",
        ),
        clock=lambda: fixed_now,
    )

    paused = asyncio.run(runtime.start("把面试安排到明天早上八点", user_id="clock_user"))

    assert paused.status.value == "waiting_for_input"
    assert paused.interaction is not None
    assert paused.interaction.arguments["start_at"] == "2026-09-03T08:00:00+08:00"


def test_model_failure_is_counted_and_written_to_raw_audit_events() -> None:
    class FailingModel:
        async def complete(self, messages, tools, options):
            raise LLMRequestError("provider transport failed")

    runtime = AgentRuntime(
        model=FailingModel(),
        tool_registry=AgentToolRegistry(),
        checkpointer=InMemorySaver(),
        settings=replace(default_settings, agent_max_verifier_passes=0),
    )

    response = asyncio.run(runtime.start("查询当前任务", user_id="model_error_user"))

    assert response.status.value == "failed"
    assert response.usage.model_calls == 1
    assert response.reply == "模型服务暂时不可用，本次任务没有继续执行，请稍后重试。"
    assert "provider transport failed" not in response.reply
    event = runtime.runtime_store.events[f"{response.run_id}:model:1"]
    assert event["event_type"] == "model_error"
    assert event["payload"] == {
        "phase": "agent",
        "error_type": "LLMRequestError",
    }


def test_deterministic_command_is_persisted_for_read_only_status_endpoint() -> None:
    async def scenario():
        runtime = _runtime(ScriptedModel([]))
        started = await runtime.start(
            "/help",
            user_id="user_1",
            request_id="req_help",
        )
        current = await runtime.get_state(
            started.thread_id,
            user_id="user_1",
            source="api",
        )
        return started, current

    started, current = asyncio.run(scenario())

    assert started.status.value == "degraded"
    assert current.thread_id == started.thread_id
    assert current.run_id == started.run_id
    assert current.reply == started.reply
    assert current.status.value == "degraded"


def test_deterministic_read_command_uses_standard_execution_audit_and_receipt() -> None:
    executions = 0

    async def list_tasks(arguments: Dict[str, Any]) -> AgentToolResult:
        nonlocal executions
        executions += 1
        assert arguments["owner_id"] == "api:direct_user"
        return AgentToolResult(data={"tasks": []}, message="今天没有待办。")

    runtime = _runtime(ScriptedModel([]), [_tool("list_tasks", list_tasks)])

    async def scenario():
        first = await runtime.start(
            "/today",
            user_id="direct_user",
            request_id="direct_today",
        )
        duplicate = await runtime.start(
            "/today",
            user_id="direct_user",
            request_id="direct_today",
        )
        return first, duplicate

    first, duplicate = asyncio.run(scenario())

    assert duplicate == first
    assert executions == 1
    tool_events = [
        event
        for event in runtime.runtime_store.events.values()
        if event["event_type"] == "tool_result"
    ]
    assert len(tool_events) == 1
    assert tool_events[0]["tool_call_id"] == first.tool_executions[0].call_id


def test_runtime_executes_multiple_tools_serially() -> None:
    execution_order: List[str] = []

    async def first_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        execution_order.append("first")
        assert arguments["owner_id"] == "api:user_1"
        return AgentToolResult(data={"interviews": ["A"]}, message="found interviews")

    async def second_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        execution_order.append("second")
        return AgentToolResult(data={"gaps": ["JVM"]}, message="found gaps")

    model = ScriptedModel(
        [
            _tool_turn(_call("call_interviews", "list_interviews")),
            _tool_turn(_call("call_gaps", "list_learning_gaps")),
            _answer_turn("已结合面试和薄弱点生成建议。"),
        ]
    )
    runtime = _runtime(
        model,
        [
            _tool("list_interviews", first_handler),
            _tool("list_learning_gaps", second_handler),
        ],
    )

    response = asyncio.run(runtime.start("查询面试并分析薄弱点", user_id="user_1"))

    assert execution_order == ["first", "second"]
    assert response.usage.tool_calls == 2
    assert [item.name for item in response.tool_executions] == [
        "list_interviews",
        "list_learning_gaps",
    ]
    assert any(
        message.get("role") == "tool"
        and message.get("tool_call_id") == "call_interviews"
        for message in model.calls[1]["messages"]
    )


def test_runtime_executes_independent_read_tools_in_parallel() -> None:
    async def scenario():
        started: List[str] = []
        both_started = asyncio.Event()

        def handler(name: str) -> ToolHandler:
            async def run(arguments: Dict[str, Any]) -> AgentToolResult:
                started.append(name)
                if len(started) == 2:
                    both_started.set()
                await asyncio.wait_for(both_started.wait(), timeout=1)
                return AgentToolResult(data={"name": name}, message=f"{name} complete")

            return run

        model = ScriptedModel(
            [
                _tool_turn(
                    _call("call_a", "read_a"),
                    _call("call_b", "read_b"),
                ),
                _answer_turn("两个只读查询均已完成。"),
            ]
        )
        runtime = _runtime(
            model,
            [_tool("read_a", handler("a")), _tool("read_b", handler("b"))],
        )
        response = await runtime.start("并行查询两类信息")
        return started, response

    started, response = asyncio.run(scenario())

    assert set(started) == {"a", "b"}
    assert response.status.value == "completed"
    assert response.usage.tool_calls == 2


def test_runtime_rejects_mixed_batch_then_allows_model_to_replan() -> None:
    executed: List[str] = []

    async def read_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        executed.append("read")
        return AgentToolResult(data={"value": 1}, message="read complete")

    async def write_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        executed.append("write")
        return AgentToolResult(data={"saved": True}, message="write complete")

    model = ScriptedModel(
        [
            _tool_turn(
                _call("mixed_read", "read_data"),
                _call("mixed_write", "write_data", value=1),
            ),
            _tool_turn(_call("replanned_read", "read_data")),
            _answer_turn("已根据批次约束重新规划。"),
        ]
    )
    runtime = _runtime(
        model,
        [
            _tool("read_data", read_handler),
            _tool(
                "write_data",
                write_handler,
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.NEVER,
                parallel_safe=False,
                properties={"value": {"type": "integer"}},
                required=["value"],
            ),
        ],
    )

    response = asyncio.run(runtime.start("先查询再写入"))

    assert executed == ["read"]
    assert [item.status.value for item in response.tool_executions] == [
        "rejected",
        "rejected",
        "success",
    ]
    rejected_messages = [
        json.loads(message["content"])
        for message in model.calls[1]["messages"]
        if message.get("role") == "tool"
    ]
    assert len(rejected_messages) == 2
    assert {item["error_code"] for item in rejected_messages} == {"batch_rejected"}
    rejected_events = [
        event
        for event in runtime.runtime_store.events.values()
        if event["tool_call_id"] in {"mixed_read", "mixed_write"}
    ]
    assert len(rejected_events) == 2
    assert {event["payload"]["status"] for event in rejected_events} == {"rejected"}


def test_model_protocol_error_pairs_and_audits_every_unexecuted_tool_call() -> None:
    executed = False

    async def handler(arguments: Dict[str, Any]) -> AgentToolResult:
        nonlocal executed
        executed = True
        return AgentToolResult(data={"unexpected": True})

    invalid_turn = replace(
        _tool_turn(_call("bad_finish", "read_data")),
        finish_reason="stop",
    )
    model = ScriptedModel(
        [invalid_turn, _answer_turn("协议错误已修正，没有执行原工具调用。")]
    )
    runtime = _runtime(model, [_tool("read_data", handler)])

    response = asyncio.run(runtime.start("查询数据"))

    assert executed is False
    assert response.usage.tool_calls == 1
    assert response.tool_executions[0].error_code == "model_protocol_error"
    event = runtime.runtime_store.events[response.run_id + ":tool:bad_finish"]
    assert event["tool_call_id"] == "bad_finish"
    assert event["payload"]["status"] == "error"


def test_runtime_rejects_a_reused_tool_call_id_before_second_execution() -> None:
    executions = 0

    async def handler(arguments: Dict[str, Any]) -> AgentToolResult:
        nonlocal executions
        executions += 1
        return AgentToolResult(data={"attempt": executions}, message="read complete")

    model = ScriptedModel(
        [
            _tool_turn(_call("reused_call", "read_data")),
            _tool_turn(_call("reused_call", "read_data")),
            _answer_turn("检测到重复调用标识，未重复执行。"),
        ]
    )
    runtime = _runtime(model, [_tool("read_data", handler)])

    response = asyncio.run(runtime.start("查询一次即可"))

    assert executions == 1
    assert [item.status.value for item in response.tool_executions] == [
        "success",
        "rejected",
    ]
    assert response.tool_executions[-1].error_code == "reused_tool_call_id"
    events = [
        event
        for event in runtime.runtime_store.events.values()
        if event["tool_call_id"] == "reused_call"
    ]
    assert len(events) == 2


def test_approval_is_bound_to_the_tool_and_normalized_arguments() -> None:
    writes = 0

    async def handler(arguments: Dict[str, Any]) -> AgentToolResult:
        nonlocal writes
        writes += 1
        return AgentToolResult(data={"written": True})

    runtime = _runtime(
        ScriptedModel([]),
        [
            _tool(
                "external_write",
                handler,
                effect=ToolEffect.EXTERNAL_WRITE,
                approval=ToolApproval.ALWAYS,
                parallel_safe=False,
                properties={"value": {"type": "string"}},
                required=["value"],
            )
        ],
    )
    old_call = {
        "id": "write_call",
        "name": "external_write",
        "arguments": {"value": "old"},
        "effect": "external_write",
        "approval": "always",
        "parallel_safe": False,
        "control": False,
    }
    new_call = {**old_call, "arguments": {"value": "new"}}
    old_interaction = runtime._approval_interaction(
        {"run_id": "run_binding"}, old_call
    )
    new_interaction = runtime._approval_interaction(
        {"run_id": "run_binding"}, new_call
    )
    assert old_interaction["id"] != new_interaction["id"]

    update = asyncio.run(
        runtime._execute_tools(
            {
                "thread_id": "thread_binding",
                "run_id": "run_binding",
                "current_user_message": "执行外部写入",
                "validated_calls": [new_call],
                "approval_granted": [old_interaction["call_fingerprint"]],
                "messages": [],
                "tool_executions": [],
                "tool_calls": 0,
            }
        )
    )

    assert writes == 0
    assert update["tool_executions"][-1]["error_code"] == "approval_missing_or_stale"


def test_unresolved_relative_time_is_rejected_before_approval() -> None:
    executed = False

    async def handler(arguments: Dict[str, Any]) -> AgentToolResult:
        nonlocal executed
        executed = True
        return AgentToolResult(data={"scheduled": True})

    model = ScriptedModel(
        [
            _tool_turn(
                _call(
                    "ambiguous_time",
                    "schedule_interview",
                    interview_time="等有空的时候",
                )
            ),
            _answer_turn("这个时间无法确定，请补充明确日期和时间。"),
        ]
    )
    runtime = _runtime(
        model,
        [
            _tool(
                "schedule_interview",
                handler,
                effect=ToolEffect.EXTERNAL_WRITE,
                approval=ToolApproval.ALWAYS,
                parallel_safe=False,
                properties={"interview_time": {"type": "string"}},
                required=["interview_time"],
            )
        ],
    )

    response = asyncio.run(runtime.start("帮我安排面试"))

    assert executed is False
    assert response.interaction is None
    assert response.tool_executions[0].status.value == "rejected"
    assert response.tool_executions[0].error_code == "invalid_arguments"


def test_runtime_replans_with_an_alternative_tool_after_failure() -> None:
    executed: List[str] = []

    async def failing(arguments: Dict[str, Any]) -> AgentToolResult:
        executed.append("primary")
        return AgentToolResult(
            data={},
            is_error=True,
            status=ToolOutcomeStatus.ERROR,
            message="primary source unavailable",
            error_code="source_unavailable",
        )

    async def fallback(arguments: Dict[str, Any]) -> AgentToolResult:
        executed.append("fallback")
        return AgentToolResult(data={"items": [1]}, message="fallback succeeded")

    model = ScriptedModel(
        [
            _tool_turn(_call("primary_call", "primary_search")),
            _tool_turn(_call("fallback_call", "fallback_search")),
            _answer_turn("主数据源失败后，已使用备用来源完成查询。"),
        ]
    )
    runtime = _runtime(
        model,
        [_tool("primary_search", failing), _tool("fallback_search", fallback)],
    )

    response = asyncio.run(runtime.start("查询资料"))

    assert executed == ["primary", "fallback"]
    assert [item.status.value for item in response.tool_executions] == [
        "error",
        "success",
    ]
    assert response.status.value == "completed"


def test_retryable_read_failure_retries_twice_before_returning_observation() -> None:
    attempts = 0

    async def transient_read(arguments: Dict[str, Any]) -> AgentToolResult:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return AgentToolResult(
                data={"attempt": attempts},
                is_error=True,
                status=ToolOutcomeStatus.ERROR,
                message="temporary upstream failure",
                error_code="upstream_unavailable",
                retryable=True,
            )
        return AgentToolResult(data={"items": ["ok"]}, message="read recovered")

    model = ScriptedModel(
        [
            _tool_turn(_call("retry_read", "transient_read")),
            _answer_turn("重试后查询成功。"),
        ]
    )
    response = asyncio.run(
        _runtime(model, [_tool("transient_read", transient_read)]).start("查询数据")
    )

    assert attempts == 3
    assert response.status.value == "completed"
    assert response.tool_executions[-1].status.value == "success"


def test_completion_requires_every_distinct_business_write_to_succeed() -> None:
    async def successful_write(arguments: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(data={"receipt": "write_a"}, message="write A succeeded")

    async def failed_write(arguments: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(
            data={},
            is_error=True,
            status=ToolOutcomeStatus.ERROR,
            message="write B failed",
            error_code="business_rejected",
        )

    model = ScriptedModel(
        [
            _tool_turn(_call("write_a", "write_a")),
            _tool_turn(_call("write_b", "write_b")),
            _answer_turn("两个写操作都完成了。"),
        ]
    )
    runtime = _runtime(
        model,
        [
            _tool(
                "write_a",
                successful_write,
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.NEVER,
                parallel_safe=False,
            ),
            _tool(
                "write_b",
                failed_write,
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.NEVER,
                parallel_safe=False,
            ),
        ],
    )

    response = asyncio.run(runtime.start("依次执行两个写操作"))

    assert response.status.value == "partial"
    assert "write_not_fully_confirmed" in response.warnings


def test_known_write_error_is_superseded_by_a_later_corrected_success() -> None:
    async def write_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        if arguments["value"] == "invalid":
            return AgentToolResult(
                data={},
                is_error=True,
                status=ToolOutcomeStatus.ERROR,
                message="business validation failed before writing",
                error_code="invalid_value",
            )
        return AgentToolResult(data={"receipt": "saved"}, message="write succeeded")

    model = ScriptedModel(
        [
            _tool_turn(
                _call("write_invalid", "save_value", record_id="same", value="invalid")
            ),
            _tool_turn(
                _call(
                    "write_corrected",
                    "save_value",
                    record_id="same",
                    value="corrected",
                )
            ),
            _answer_turn("修正参数后已保存。"),
        ]
    )
    runtime = _runtime(
        model,
        [
            _tool(
                "save_value",
                write_handler,
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.NEVER,
                parallel_safe=False,
                properties={
                    "record_id": {"type": "string"},
                    "value": {"type": "string"},
                },
                required=["record_id", "value"],
            )
        ],
    )

    response = asyncio.run(runtime.start("保存这个值"))

    assert [item.status.value for item in response.tool_executions] == [
        "error",
        "success",
    ]
    assert response.status.value == "completed"
    assert "write_not_fully_confirmed" not in response.warnings


def test_success_for_another_target_does_not_hide_a_known_write_error() -> None:
    async def write_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        if arguments["record_id"] == "record_a":
            return AgentToolResult(
                data={},
                is_error=True,
                status=ToolOutcomeStatus.ERROR,
                message="record A was rejected",
                error_code="record_rejected",
            )
        return AgentToolResult(data={"receipt": "record_b"}, message="record B saved")

    model = ScriptedModel(
        [
            _tool_turn(
                _call("write_a", "save_record", record_id="record_a", value="a")
            ),
            _tool_turn(
                _call("write_b", "save_record", record_id="record_b", value="b")
            ),
            _answer_turn("B 已保存，但 A 仍失败。"),
        ]
    )
    runtime = _runtime(
        model,
        [
            _tool(
                "save_record",
                write_handler,
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.NEVER,
                parallel_safe=False,
                properties={
                    "record_id": {"type": "string"},
                    "value": {"type": "string"},
                },
                required=["record_id", "value"],
            )
        ],
    )

    response = asyncio.run(runtime.start("分别保存 A 和 B"))

    assert response.status.value == "partial"
    assert "write_not_fully_confirmed" in response.warnings


def test_runtime_pauses_for_approval_and_resumes_exact_write() -> None:
    async def scenario():
        received: List[Dict[str, Any]] = []

        async def write_handler(arguments: Dict[str, Any]) -> AgentToolResult:
            received.append(dict(arguments))
            return AgentToolResult(data={"event_id": "evt_1"}, message="calendar synced")

        model = ScriptedModel(
            [
                _tool_turn(_call("sync_call", "sync_calendar", plan_id="plan_1")),
                _answer_turn("复习计划已同步到日历。"),
            ]
        )
        runtime = _runtime(
            model,
            [
                _tool(
                    "sync_calendar",
                    write_handler,
                    effect=ToolEffect.EXTERNAL_WRITE,
                    approval=ToolApproval.ALWAYS,
                    parallel_safe=False,
                    properties={"plan_id": {"type": "string"}},
                    required=["plan_id"],
                )
            ],
        )
        paused = await runtime.start(
            "把复习计划同步到日历",
            user_id="user_1",
            request_id="req_approval_start",
        )
        assert received == []
        resumed = await runtime.resume(
            thread_id=paused.thread_id,
            interaction_id=paused.interaction.id,
            decision="approve",
            user_id="user_1",
            request_id="req_approval_resume",
        )
        assert len(model.calls) == 2
        return paused, resumed, received

    paused, resumed, received = asyncio.run(scenario())

    assert paused.status.value == "waiting_for_input"
    assert paused.interaction.type.value == "approval"
    assert paused.interaction.tool_name == "sync_calendar"
    assert paused.interaction.operation == "待处理操作"
    assert "sync_calendar" not in paused.interaction.prompt
    assert "sync_calendar" not in paused.reply
    assert resumed.status.value == "completed"
    assert resumed.reply == "复习计划已同步到日历。"
    assert len(received) == 1
    assert received[0]["plan_id"] == "plan_1"
    assert received[0]["owner_id"] == "api:user_1"
    assert received[0]["idempotency_key"].startswith("agent_op_")


def test_resume_message_interprets_approval_and_executes_original_call() -> None:
    received: List[Dict[str, Any]] = []

    async def write_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        received.append(dict(arguments))
        return AgentToolResult(data={"saved": True}, message="saved")

    model = ScriptedModel(
        [
            _tool_turn(
                _call(
                    "create_original",
                    "create_application",
                    company="百度",
                    position="AI 应用工程师",
                )
            ),
            _tool_turn(
                _call(
                    "interpret_approval",
                    "submit_interaction_interpretation",
                    action="approve",
                    approval_intent=True,
                    has_changes=False,
                )
            ),
            _answer_turn("已记录这条投递。"),
        ]
    )
    runtime = _runtime(
        model,
        [
            _tool(
                "create_application",
                write_handler,
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.ALWAYS,
                parallel_safe=False,
                properties={
                    "company": {"type": "string"},
                    "position": {"type": "string"},
                },
                required=["company", "position"],
            )
        ],
    )

    async def scenario():
        paused = await runtime.start("记录百度 AI 应用工程师投递", user_id="nl_approve")
        resumed = await runtime.resume_message(
            paused.thread_id,
            paused.interaction.id,
            "没问题，就按这个执行吧",
            user_id="nl_approve",
        )
        return paused, resumed

    paused, resumed = asyncio.run(scenario())

    assert paused.status.value == "waiting_for_input"
    assert resumed.status.value == "completed"
    assert resumed.reply == "已记录这条投递。"
    assert len(received) == 1
    assert received[0]["company"] == "百度"
    assert received[0]["position"] == "AI 应用工程师"
    assert resumed.usage.model_calls == 3
    interpreter_call = model.calls[1]
    assert [item["function"]["name"] for item in interpreter_call["tools"]] == [
        "submit_interaction_interpretation"
    ]
    assert interpreter_call["options"].tool_choice == {
        "type": "function",
        "function": {"name": "submit_interaction_interpretation"},
    }


def test_resume_message_confirmation_with_changes_requires_new_approval() -> None:
    received: List[Dict[str, Any]] = []

    async def write_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        received.append(dict(arguments))
        return AgentToolResult(data={"saved": True}, message="saved")

    model = ScriptedModel(
        [
            _tool_turn(
                _call(
                    "create_before_revision",
                    "create_application",
                    company="百度",
                    position="AI 应用工程师",
                    location="北京",
                )
            ),
            _tool_turn(
                _call(
                    "interpret_revision",
                    "submit_interaction_interpretation",
                    action="approve",
                    approval_intent=True,
                    has_changes=True,
                )
            ),
            _tool_turn(
                _call(
                    "create_after_revision",
                    "create_application",
                    company="百度",
                    position="AI 应用工程师",
                    location="上海",
                )
            ),
            _answer_turn("已按上海工作地点记录投递。"),
        ]
    )
    runtime = _runtime(
        model,
        [
            _tool(
                "create_application",
                write_handler,
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.ALWAYS,
                parallel_safe=False,
                properties={
                    "company": {"type": "string"},
                    "position": {"type": "string"},
                    "location": {"type": "string"},
                },
                required=["company", "position", "location"],
            )
        ],
    )

    async def scenario():
        first = await runtime.start(
            "记录百度 AI 应用工程师投递，工作地点北京",
            user_id="nl_revision",
        )
        revised = await runtime.resume_message(
            first.thread_id,
            first.interaction.id,
            "确认，然后 base 上海",
            user_id="nl_revision",
        )
        assert received == []
        completed = await runtime.resume(
            revised.thread_id,
            revised.interaction.id,
            "approve",
            user_id="nl_revision",
        )
        return first, revised, completed

    first, revised, completed = asyncio.run(scenario())

    assert first.status.value == "waiting_for_input"
    assert revised.status.value == "waiting_for_input"
    assert revised.interaction.arguments["location"] == "上海"
    assert revised.interaction.id != first.interaction.id
    assert any(
        message.get("role") == "user"
        and message.get("content") == "确认，然后 base 上海"
        for message in model.calls[2]["messages"]
    )
    assert completed.status.value == "completed"
    assert len(received) == 1
    assert received[0]["location"] == "上海"


@pytest.mark.parametrize(
    ("interpreter_turn", "warning"),
    [
        (_answer_turn("I cannot call the required function."), "interaction_interpreter_invalid"),
    ],
)
def test_resume_message_invalid_interpretation_keeps_the_interrupt_pending(
    interpreter_turn: ModelTurn,
    warning: str,
) -> None:
    received: List[Dict[str, Any]] = []

    async def write_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        received.append(dict(arguments))
        return AgentToolResult(data={"saved": True})

    model = ScriptedModel(
        [
            _tool_turn(_call("pending_write", "write_local", value="x")),
            interpreter_turn,
        ]
    )
    runtime = _runtime(
        model,
        [
            _tool(
                "write_local",
                write_handler,
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.ALWAYS,
                parallel_safe=False,
                properties={"value": {"type": "string"}},
                required=["value"],
            )
        ],
    )

    async def scenario():
        paused = await runtime.start("写入 x", user_id="nl_invalid")
        resumed = await runtime.resume_message(
            paused.thread_id,
            paused.interaction.id,
            "那就这样吧",
            user_id="nl_invalid",
        )
        return paused, resumed

    paused, resumed = asyncio.run(scenario())

    assert received == []
    assert resumed.status.value == "waiting_for_input"
    assert resumed.interaction.id == paused.interaction.id
    assert warning in resumed.warnings
    assert "我没有准确理解你的意思" in resumed.reply
    assert resumed.usage.model_calls == 2


def test_resume_message_status_preserves_pending_interaction() -> None:
    received: List[Dict[str, Any]] = []

    async def write_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        received.append(dict(arguments))
        return AgentToolResult(data={"saved": True}, message="saved")

    model = ScriptedModel(
        [
            _tool_turn(_call("pending_status", "write_local", value="x")),
            _tool_turn(
                _call(
                    "interpret_status",
                    "submit_interaction_interpretation",
                    action="status",
                    approval_intent=False,
                    has_changes=False,
                )
            ),
            _tool_turn(
                _call(
                    "interpret_after_status",
                    "submit_interaction_interpretation",
                    action="approve",
                    approval_intent=True,
                    has_changes=False,
                )
            ),
            _answer_turn("已完成操作。"),
        ]
    )
    runtime = _runtime(
        model,
        [
            _tool(
                "write_local",
                write_handler,
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.ALWAYS,
                parallel_safe=False,
                properties={"value": {"type": "string"}},
                required=["value"],
            )
        ],
    )

    async def scenario():
        paused = await runtime.start("写入 x", user_id="nl_status")
        status = await runtime.resume_message(
            paused.thread_id,
            paused.interaction.id,
            "现在处理到哪了？",
            user_id="nl_status",
        )
        assert received == []
        snapshot = await runtime.graph.aget_state(runtime._config(status.thread_id))
        assert not any(
            message.get("role") == "tool"
            for message in snapshot.values.get("messages", [])
        )
        completed = await runtime.resume_message(
            status.thread_id,
            status.interaction.id,
            "可以，继续执行",
            user_id="nl_status",
        )
        return paused, status, completed

    paused, status, completed = asyncio.run(scenario())

    assert status.status.value == "waiting_for_input"
    assert status.interaction.id == paused.interaction.id
    assert "当前正在等待你的确认" in status.reply
    assert "write_local" not in status.reply
    assert status.tool_executions == []
    assert status.usage.model_calls == 2
    assert completed.status.value == "completed"
    assert completed.reply == "已完成操作。"
    assert len(received) == 1
    assert len(model.calls) == 4


def test_resume_message_rejects_pending_write_without_executing_it() -> None:
    received: List[Dict[str, Any]] = []

    async def write_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        received.append(dict(arguments))
        return AgentToolResult(data={"saved": True})

    model = ScriptedModel(
        [
            _tool_turn(_call("pending_reject", "write_local", value="x")),
            _tool_turn(
                _call(
                    "interpret_reject",
                    "submit_interaction_interpretation",
                    action="reject",
                    approval_intent=False,
                    has_changes=False,
                )
            ),
            _answer_turn("好的，这项操作没有执行。"),
        ]
    )
    runtime = _runtime(
        model,
        [
            _tool(
                "write_local",
                write_handler,
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.ALWAYS,
                parallel_safe=False,
                properties={"value": {"type": "string"}},
                required=["value"],
            )
        ],
    )

    async def scenario():
        paused = await runtime.start("写入 x", user_id="nl_reject")
        return await runtime.resume_message(
            paused.thread_id,
            paused.interaction.id,
            "不同意，不要执行",
            user_id="nl_reject",
        )

    response = asyncio.run(scenario())

    assert received == []
    assert response.status.value == "partial"
    assert response.reply == "好的，这项操作没有执行。"
    assert len(model.calls) == 3


def test_resume_message_cancels_the_pending_task_without_executing_it() -> None:
    received: List[Dict[str, Any]] = []

    async def write_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        received.append(dict(arguments))
        return AgentToolResult(data={"saved": True})

    model = ScriptedModel(
        [
            _tool_turn(_call("pending_cancel", "write_local", value="x")),
            _tool_turn(
                _call(
                    "interpret_cancel",
                    "submit_interaction_interpretation",
                    action="cancel",
                    approval_intent=False,
                    has_changes=False,
                )
            ),
        ]
    )
    runtime = _runtime(
        model,
        [
            _tool(
                "write_local",
                write_handler,
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.ALWAYS,
                parallel_safe=False,
                properties={"value": {"type": "string"}},
                required=["value"],
            )
        ],
    )

    async def scenario():
        paused = await runtime.start("写入 x", user_id="nl_cancel")
        return await runtime.resume_message(
            paused.thread_id,
            paused.interaction.id,
            "这个任务先取消",
            user_id="nl_cancel",
        )

    response = asyncio.run(scenario())

    assert received == []
    assert response.status.value == "partial"
    assert response.reply == "当前任务已取消，没有继续执行待处理操作。"
    assert len(model.calls) == 2


@pytest.mark.parametrize(
    ("action", "reply", "expected_text", "expected_warning"),
    [
        (
            "unrelated",
            "顺便帮我查一下面试",
            "当前还有待处理事项",
            "interaction_reply_unrelated",
        ),
        (
            "ambiguous",
            "你看着办",
            "我没有准确理解你的意思",
            "interaction_reply_ambiguous",
        ),
    ],
)
def test_resume_message_non_resolving_intents_keep_the_original_interrupt(
    action: str,
    reply: str,
    expected_text: str,
    expected_warning: str,
) -> None:
    received: List[Dict[str, Any]] = []

    async def write_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        received.append(dict(arguments))
        return AgentToolResult(data={"saved": True})

    model = ScriptedModel(
        [
            _tool_turn(_call(f"pending_{action}", "write_local", value="x")),
            _tool_turn(
                _call(
                    f"interpret_{action}",
                    "submit_interaction_interpretation",
                    action=action,
                    approval_intent=False,
                    has_changes=False,
                )
            ),
        ]
    )
    runtime = _runtime(
        model,
        [
            _tool(
                "write_local",
                write_handler,
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.ALWAYS,
                parallel_safe=False,
                properties={"value": {"type": "string"}},
                required=["value"],
            )
        ],
    )

    async def scenario():
        paused = await runtime.start("写入 x", user_id=f"nl_{action}")
        resumed = await runtime.resume_message(
            paused.thread_id,
            paused.interaction.id,
            reply,
            user_id=f"nl_{action}",
        )
        return paused, resumed

    paused, resumed = asyncio.run(scenario())

    assert received == []
    assert resumed.status.value == "waiting_for_input"
    assert resumed.interaction.id == paused.interaction.id
    assert expected_text in resumed.reply
    assert expected_warning in resumed.warnings
    assert len(model.calls) == 2


def test_resume_message_model_failure_keeps_the_interrupt_pending() -> None:
    class FailingInterpreterModel(ScriptedModel):
        async def complete(self, messages, tools, options) -> ModelTurn:
            if len(self.calls) == 1:
                self.calls.append(
                    {
                        "messages": [dict(message) for message in messages],
                        "tools": list(tools),
                        "options": options,
                    }
                )
                raise LLMRequestError("interpreter unavailable")
            return await super().complete(messages, tools, options)

    executed = False

    async def write_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        nonlocal executed
        executed = True
        return AgentToolResult(data={"saved": True})

    model = FailingInterpreterModel(
        [_tool_turn(_call("pending_failure", "write_local", value="x"))]
    )
    runtime = _runtime(
        model,
        [
            _tool(
                "write_local",
                write_handler,
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.ALWAYS,
                parallel_safe=False,
                properties={"value": {"type": "string"}},
                required=["value"],
            )
        ],
    )

    async def scenario():
        paused = await runtime.start("写入 x", user_id="nl_failure")
        resumed = await runtime.resume_message(
            paused.thread_id,
            paused.interaction.id,
            "同意按这个做",
            user_id="nl_failure",
        )
        return paused, resumed

    paused, resumed = asyncio.run(scenario())

    assert executed is False
    assert resumed.status.value == "waiting_for_input"
    assert resumed.interaction.id == paused.interaction.id
    assert "interaction_interpreter_unavailable" in resumed.warnings
    assert resumed.usage.model_calls == 2
    event = runtime.runtime_store.events[f"{resumed.run_id}:model:2"]
    assert event["event_type"] == "model_error"
    assert event["payload"]["phase"] == "interaction_interpreter"


def test_resume_message_supplies_clarification_text_to_the_agent() -> None:
    async def unused_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        raise AssertionError("request_user_input must not execute")

    model = ScriptedModel(
        [
            _tool_turn(
                _call(
                    "clarify_location",
                    "request_user_input",
                    kind="clarification",
                    prompt="请补充工作地点",
                    input_schema={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                )
            ),
            _tool_turn(
                _call(
                    "interpret_answer",
                    "submit_interaction_interpretation",
                    action="answer",
                    approval_intent=False,
                    has_changes=False,
                )
            ),
            _answer_turn("已按上海继续处理。"),
        ]
    )
    runtime = _runtime(
        model,
        [
            _tool(
                "request_user_input",
                unused_handler,
                parallel_safe=False,
                control=True,
                properties={
                    "kind": {"type": "string"},
                    "prompt": {"type": "string"},
                    "input_schema": {"type": "object"},
                },
                required=["kind", "prompt"],
            )
        ],
    )

    async def scenario():
        paused = await runtime.start("记录投递", user_id="nl_answer")
        resumed = await runtime.resume_message(
            paused.thread_id,
            paused.interaction.id,
            "base 上海",
            user_id="nl_answer",
        )
        return paused, resumed

    paused, resumed = asyncio.run(scenario())

    assert paused.status.value == "waiting_for_input"
    assert resumed.status.value == "completed"
    assert resumed.reply == "已按上海继续处理。"
    assert any(
        message.get("role") == "user" and message.get("content") == "base 上海"
        for message in model.calls[2]["messages"]
    )


def test_runtime_redacts_tool_names_from_a_model_final_answer() -> None:
    response = asyncio.run(
        _runtime(
            _model := ScriptedModel(
                [_answer_turn("已通过 create_application 保存投递记录。")]
            )
        ).start("你好")
    )

    assert response.reply == "已通过 新增投递记录 保存投递记录。"
    assert "create_application" not in response.reply
    assert len(_model.calls) == 1


def test_partial_reply_does_not_infer_goal_progress_from_tool_activity() -> None:
    runtime = _runtime(ScriptedModel([]))

    reply = runtime._partial_reply(
        {
            "tool_executions": [
                {"name": "list_interviews", "status": "success"},
                {"name": "create_application", "status": "error"},
            ]
        },
        "任务只完成了一部分",
    )

    assert reply == "任务只完成了一部分。"
    assert "list_interviews" not in reply
    assert "create_application" not in reply


def test_previous_turn_language_cannot_preapprove_a_new_local_write() -> None:
    executed = False

    async def local_write(arguments: Dict[str, Any]) -> AgentToolResult:
        nonlocal executed
        executed = True
        return AgentToolResult(data={"created": True})

    model = ScriptedModel(
        [
            _answer_turn("上一轮只讨论了如何生成复习计划。"),
            _tool_turn(_call("new_plan", "create_study_plan", goal="JVM")),
        ]
    )
    runtime = _runtime(
        model,
        [
            _tool(
                "create_study_plan",
                local_write,
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.IF_IMPLICIT,
                parallel_safe=False,
                properties={"goal": {"type": "string"}},
                required=["goal"],
            )
        ],
    )

    async def scenario():
        await runtime.start("怎么生成复习计划？", user_id="approval_scope_user")
        return await runtime.start("只查询一下现有内容", user_id="approval_scope_user")

    paused = asyncio.run(scenario())

    assert executed is False
    assert paused.status.value == "waiting_for_input"
    assert paused.interaction.tool_name == "create_study_plan"


def test_question_about_a_local_write_does_not_authorize_it() -> None:
    executed = False

    async def local_write(arguments: Dict[str, Any]) -> AgentToolResult:
        nonlocal executed
        executed = True
        return AgentToolResult(data={"created": True})

    runtime = _runtime(
        ScriptedModel(
            [_tool_turn(_call("question_plan", "create_study_plan", goal="JVM"))]
        ),
        [
            _tool(
                "create_study_plan",
                local_write,
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.IF_IMPLICIT,
                parallel_safe=False,
                properties={"goal": {"type": "string"}},
                required=["goal"],
            )
        ],
    )

    paused = asyncio.run(runtime.start("如何生成复习计划？"))

    assert executed is False
    assert paused.status.value == "waiting_for_input"
    assert paused.interaction.tool_name == "create_study_plan"


def test_explicit_local_write_must_match_the_requested_operation() -> None:
    executed = False

    async def update_task(arguments: Dict[str, Any]) -> AgentToolResult:
        nonlocal executed
        executed = True
        return AgentToolResult(data={"updated": True})

    runtime = _runtime(
        ScriptedModel(
            [
                _tool_turn(
                    _call(
                        "mismatched_task_update",
                        "update_task",
                        operation="complete",
                        task_title="JVM 复习",
                    )
                )
            ]
        ),
        [
            _tool(
                "update_task",
                update_task,
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.IF_IMPLICIT,
                parallel_safe=False,
                properties={
                    "operation": {"type": "string", "enum": ["complete", "postpone"]},
                    "task_title": {"type": "string"},
                },
                required=["operation"],
            )
        ],
    )

    paused = asyncio.run(runtime.start("把 JVM 复习任务延期到明天"))

    assert executed is False
    assert paused.status.value == "waiting_for_input"


def test_runtime_resumes_request_user_input_with_answer() -> None:
    async def unused_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        raise AssertionError("request_user_input is a control tool and must not execute")

    input_schema = {
        "type": "object",
        "properties": {"timezone": {"type": "string"}},
        "required": ["timezone"],
    }
    model = ScriptedModel(
        [
            _tool_turn(
                _call(
                    "input_call",
                    "request_user_input",
                    kind="preference_form",
                    prompt="请选择时区",
                    input_schema=input_schema,
                )
            ),
            _answer_turn("已使用 Asia/Shanghai 继续生成计划。"),
        ]
    )
    runtime = _runtime(
        model,
        [
            _tool(
                "request_user_input",
                unused_handler,
                parallel_safe=False,
                control=True,
                properties={
                    "kind": {"type": "string"},
                    "prompt": {"type": "string"},
                    "input_schema": {"type": "object"},
                },
                required=["kind", "prompt"],
            )
        ],
    )

    async def scenario():
        paused = await runtime.start("生成复习计划", user_id="user_1")
        resumed = await runtime.resume(
            paused.thread_id,
            paused.interaction.id,
            "answer",
            value={"timezone": "Asia/Shanghai"},
            text="我的时区是 Asia/Shanghai",
            user_id="user_1",
        )
        return paused, resumed

    paused, resumed = asyncio.run(scenario())

    assert paused.status.value == "waiting_for_input"
    assert paused.interaction.type.value == "preference_form"
    assert paused.interaction.required_fields == ["timezone"]
    assert resumed.status.value == "completed"
    assert any(
        message.get("role") == "user"
        and message.get("content") == "我的时区是 Asia/Shanghai"
        for message in model.calls[1]["messages"]
    )
    input_event = runtime.runtime_store.events[
        resumed.run_id + ":tool:input_call"
    ]
    assert input_event["tool_call_id"] == "input_call"
    assert input_event["payload"]["status"] == "success"


def test_runtime_returns_cached_response_for_duplicate_request_id() -> None:
    model = ScriptedModel([_answer_turn("只执行一次。")])
    runtime = _runtime(model)

    async def scenario():
        first = await runtime.start(
            "第一次请求",
            user_id="user_1",
            request_id="stable_request_1",
        )
        duplicate = await runtime.start(
            "第一次请求",
            user_id="user_1",
            request_id="stable_request_1",
        )
        return first, duplicate

    first, duplicate = asyncio.run(scenario())

    assert duplicate == first
    assert len(model.calls) == 1


def test_replay_request_returns_an_existing_pending_start_response() -> None:
    async def write_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        raise AssertionError("The pending write must not execute during receipt replay.")

    model = ScriptedModel(
        [_tool_turn(_call("pending_replay", "write_local", value="x"))]
    )
    runtime = _runtime(
        model,
        [
            _tool(
                "write_local",
                write_handler,
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.ALWAYS,
                parallel_safe=False,
                properties={"value": {"type": "string"}},
                required=["value"],
            )
        ],
    )

    async def scenario():
        pending = await runtime.start(
            "写入 x",
            user_id="pending_replay_user",
            request_id="pending_replay_request",
        )
        replayed = await runtime.replay_request(
            request_id="pending_replay_request",
            thread_id=pending.thread_id,
            user_id="pending_replay_user",
            source="api",
        )
        return pending, replayed

    pending, replayed = asyncio.run(scenario())

    assert replayed == pending
    assert replayed.status.value == "waiting_for_input"
    assert len(model.calls) == 1


def test_replay_request_returns_completed_resume_without_route_fingerprint() -> None:
    received: List[Dict[str, Any]] = []

    async def write_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        received.append(dict(arguments))
        return AgentToolResult(data={"saved": True}, message="saved")

    model = ScriptedModel(
        [
            _tool_turn(_call("resume_replay_write", "write_local", value="x")),
            _tool_turn(
                _call(
                    "resume_replay_interpretation",
                    "submit_interaction_interpretation",
                    action="approve",
                    approval_intent=True,
                    has_changes=False,
                )
            ),
            _answer_turn("写入完成。"),
        ]
    )
    runtime = _runtime(
        model,
        [
            _tool(
                "write_local",
                write_handler,
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.ALWAYS,
                parallel_safe=False,
                properties={"value": {"type": "string"}},
                required=["value"],
            )
        ],
    )

    async def scenario():
        pending = await runtime.start("写入 x", user_id="resume_replay_user")
        completed = await runtime.resume_message(
            pending.thread_id,
            pending.interaction.id,
            "嗯嗯",
            user_id="resume_replay_user",
            request_id="resume_replay_request",
        )
        replayed = await runtime.replay_request(
            request_id="resume_replay_request",
            thread_id=pending.thread_id,
            user_id="resume_replay_user",
            source="api",
        )
        return completed, replayed

    completed, replayed = asyncio.run(scenario())

    assert replayed == completed
    assert replayed.reply == "写入完成。"
    assert len(received) == 1
    assert len(model.calls) == 3


def test_replay_request_waits_for_an_in_process_request() -> None:
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()

        class BlockingModel(ScriptedModel):
            async def complete(self, messages, tools, options) -> ModelTurn:
                entered.set()
                await release.wait()
                return await super().complete(messages, tools, options)

        model = BlockingModel([_answer_turn("只生成一次。")])
        runtime = _runtime(model)
        thread_id = runtime.thread_id_for("processing_replay_user", "api")
        original_task = asyncio.create_task(
            runtime.start(
                "处理这个请求",
                user_id="processing_replay_user",
                request_id="processing_replay_request",
            )
        )
        await entered.wait()
        replay_task = asyncio.create_task(
            runtime.replay_request(
                request_id="processing_replay_request",
                thread_id=thread_id,
                user_id="processing_replay_user",
                source="api",
            )
        )
        await asyncio.sleep(0)
        assert not replay_task.done()
        release.set()
        return await original_task, await replay_task, model

    original, replayed, model = asyncio.run(scenario())

    assert replayed == original
    assert replayed.reply == "只生成一次。"
    assert len(model.calls) == 1


def test_replay_request_recovers_a_processing_checkpoint_after_restart() -> None:
    class FailFirstStageStore(InMemoryAgentRuntimeStore):
        def __init__(self) -> None:
            super().__init__()
            self.fail_once = True

        def stage_receipt_response(self, source, request_id, response) -> None:
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("simulated crash before response staging")
            super().stage_receipt_response(source, request_id, response)

    async def scenario():
        saver = InMemorySaver()
        store = FailFirstStageStore()
        first_model = ScriptedModel([_answer_turn("checkpoint 中的结果。")])
        first_runtime = AgentRuntime(
            model=first_model,
            tool_registry=AgentToolRegistry(),
            checkpointer=saver,
            runtime_store=store,
            settings=replace(default_settings, agent_max_verifier_passes=0),
        )
        thread_id = first_runtime.thread_id_for("restart_replay_user", "api")
        with pytest.raises(RuntimeError, match="before response staging"):
            await first_runtime.start(
                "完成后模拟崩溃",
                user_id="restart_replay_user",
                request_id="restart_replay_request",
            )

        receipt = store.get_receipt_record(
            "api:api:restart_replay_user", "restart_replay_request"
        )
        assert receipt.status == "processing"
        assert receipt.response is None

        second_model = ScriptedModel([])
        second_runtime = AgentRuntime(
            model=second_model,
            tool_registry=AgentToolRegistry(),
            checkpointer=saver,
            runtime_store=store,
            settings=replace(default_settings, agent_max_verifier_passes=0),
        )
        recovered = await second_runtime.replay_request(
            request_id="restart_replay_request",
            thread_id=thread_id,
            user_id="restart_replay_user",
            source="api",
        )
        return recovered, first_model, second_model, store

    recovered, first_model, second_model, store = asyncio.run(scenario())

    assert recovered.reply == "checkpoint 中的结果。"
    assert len(first_model.calls) == 1
    assert second_model.calls == []
    receipt = store.get_receipt_record(
        "api:api:restart_replay_user", "restart_replay_request"
    )
    assert receipt.status == "completed"


def test_replay_request_commits_a_staged_response_after_restart() -> None:
    async def scenario():
        store = InMemoryAgentRuntimeStore()
        source = "api:api:staged_replay_user"
        request_id = "staged_replay_request"
        thread_id = AgentRuntime.thread_id_for("staged_replay_user", "api")
        assert store.begin_receipt(
            source,
            request_id,
            "original-route-fingerprint",
            thread_id,
            "run_staged_replay",
        )
        store.stage_receipt_response(
            source,
            request_id,
            {
                "thread_id": thread_id,
                "run_id": "run_staged_replay",
                "status": "completed",
                "reply": "已暂存的结果。",
            },
        )
        model = ScriptedModel([])
        runtime = AgentRuntime(
            model=model,
            tool_registry=AgentToolRegistry(),
            checkpointer=InMemorySaver(),
            runtime_store=store,
            settings=replace(default_settings, agent_max_verifier_passes=0),
        )
        replayed = await runtime.replay_request(
            request_id=request_id,
            thread_id=thread_id,
            user_id="staged_replay_user",
            source="api",
        )
        return replayed, model, store

    replayed, model, store = asyncio.run(scenario())

    assert replayed.reply == "已暂存的结果。"
    assert model.calls == []
    receipt = store.get_receipt_record(
        "api:api:staged_replay_user", "staged_replay_request"
    )
    assert receipt.status == "completed"


def test_replay_request_rejects_a_receipt_from_another_thread() -> None:
    model = ScriptedModel([_answer_turn("scope A")])
    runtime = _runtime(model)

    async def scenario():
        completed = await runtime.start(
            "处理 scope A",
            user_id="scoped_replay_user",
            conversation_scope="scope_a",
            request_id="scoped_replay_request",
        )
        other_thread_id = runtime.thread_id_for(
            "scoped_replay_user", "api", "scope_b"
        )
        with pytest.raises(AgentRuntimeConflict, match="different conversation thread"):
            await runtime.replay_request(
                request_id="scoped_replay_request",
                thread_id=other_thread_id,
                user_id="scoped_replay_user",
                source="api",
            )
        return completed

    completed = asyncio.run(scenario())

    assert completed.reply == "scope A"
    assert len(model.calls) == 1


def test_replay_request_does_not_expose_another_users_receipt() -> None:
    model = ScriptedModel([_answer_turn("仅属于用户 A。")])
    runtime = _runtime(model)

    async def scenario():
        completed = await runtime.start(
            "用户 A 的请求",
            user_id="replay_owner_a",
            request_id="shared_transport_id",
        )
        replayed = await runtime.replay_request(
            request_id="shared_transport_id",
            thread_id=completed.thread_id,
            user_id="replay_owner_b",
            source="api",
        )
        return completed, replayed

    completed, replayed = asyncio.run(scenario())

    assert completed.reply == "仅属于用户 A。"
    assert replayed is None
    assert len(model.calls) == 1


def test_completed_start_receipt_rejects_same_request_id_with_different_payload() -> None:
    model = ScriptedModel([_answer_turn("只执行原请求。")])
    runtime = _runtime(model)

    async def scenario():
        await runtime.start(
            "原请求",
            user_id="completed_fingerprint_user",
            request_id="completed_start_request",
        )
        with pytest.raises(AgentRuntimeConflict, match="different payload"):
            await runtime.start(
                "篡改后的请求",
                user_id="completed_fingerprint_user",
                request_id="completed_start_request",
            )

    asyncio.run(scenario())
    assert len(model.calls) == 1


def test_completed_resume_receipt_rejects_same_request_id_with_different_payload() -> None:
    async def unused_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        raise AssertionError("request_user_input is a control tool and must not execute")

    model = ScriptedModel(
        [
            _tool_turn(
                _call(
                    "completed_resume_input",
                    "request_user_input",
                    kind="clarification",
                    prompt="请选择目标城市",
                )
            ),
            _answer_turn("已按上海继续。"),
        ]
    )
    runtime = _runtime(
        model,
        [
            _tool(
                "request_user_input",
                unused_handler,
                parallel_safe=False,
                control=True,
                properties={
                    "kind": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                required=["kind", "prompt"],
            )
        ],
    )

    async def scenario():
        paused = await runtime.start(
            "帮我筛选岗位",
            user_id="completed_resume_user",
        )
        completed = await runtime.resume(
            paused.thread_id,
            paused.interaction.id,
            "answer",
            value="上海",
            text="目标城市是上海",
            user_id="completed_resume_user",
            request_id="completed_resume_request",
        )
        duplicate = await runtime.resume(
            paused.thread_id,
            paused.interaction.id,
            "answer",
            value="上海",
            text="目标城市是上海",
            user_id="completed_resume_user",
            request_id="completed_resume_request",
        )
        with pytest.raises(AgentRuntimeConflict, match="different payload"):
            await runtime.resume(
                paused.thread_id,
                paused.interaction.id,
                "answer",
                value="北京",
                text="目标城市改为北京",
                user_id="completed_resume_user",
                request_id="completed_resume_request",
            )
        return completed, duplicate

    completed, duplicate = asyncio.run(scenario())

    assert duplicate == completed
    assert len(model.calls) == 2


def test_processing_receipt_recovers_final_checkpoint_after_crash() -> None:
    class FailOnceReceiptStore(InMemoryAgentRuntimeStore):
        def __init__(self) -> None:
            super().__init__()
            self.fail_once = True

        def finish_receipt(self, source, request_id, response) -> None:
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("simulated crash before receipt commit")
            super().finish_receipt(source, request_id, response)

    async def scenario():
        saver = InMemorySaver()
        store = FailOnceReceiptStore()
        first_model = ScriptedModel([_answer_turn("checkpoint 已完成。")])
        first_runtime = AgentRuntime(
            model=first_model,
            tool_registry=AgentToolRegistry(),
            checkpointer=saver,
            runtime_store=store,
            settings=replace(default_settings, agent_max_verifier_passes=0),
        )
        with pytest.raises(RuntimeError, match="simulated crash"):
            await first_runtime.start(
                "恢复这次请求",
                user_id="receipt_recovery_user",
                request_id="receipt_crash_1",
            )

        second_model = ScriptedModel([])
        second_runtime = AgentRuntime(
            model=second_model,
            tool_registry=AgentToolRegistry(),
            checkpointer=saver,
            runtime_store=store,
            settings=replace(default_settings, agent_max_verifier_passes=0),
        )
        recovered = await second_runtime.start(
            "恢复这次请求",
            user_id="receipt_recovery_user",
            request_id="receipt_crash_1",
        )
        return recovered, first_model, second_model, store

    recovered, first_model, second_model, store = asyncio.run(scenario())

    assert recovered.reply == "checkpoint 已完成。"
    assert len(first_model.calls) == 1
    assert second_model.calls == []
    assert store.get_receipt("api:api:receipt_recovery_user", "receipt_crash_1")


def test_staged_receipt_recovers_after_a_later_run_replaces_checkpoint() -> None:
    class FailNextFinishStore(InMemoryAgentRuntimeStore):
        def __init__(self) -> None:
            super().__init__()
            self.fail_next = False

        def finish_receipt(self, source, request_id, response) -> None:
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("simulated crash after response staging")
            super().finish_receipt(source, request_id, response)

    async def scenario():
        store = FailNextFinishStore()
        model = ScriptedModel(
            [_answer_turn("A 已完成。"), _answer_turn("B 已完成。")]
        )
        runtime = AgentRuntime(
            model=model,
            tool_registry=AgentToolRegistry(),
            checkpointer=InMemorySaver(),
            runtime_store=store,
            settings=replace(default_settings, agent_max_verifier_passes=0),
        )
        store.fail_next = True
        with pytest.raises(RuntimeError, match="after response staging"):
            await runtime.start(
                "执行 A",
                user_id="receipt_outbox_user",
                request_id="receipt_outbox_a",
            )

        later = await runtime.start(
            "执行 B",
            user_id="receipt_outbox_user",
            request_id="receipt_outbox_b",
        )
        recovered = await runtime.start(
            "执行 A",
            user_id="receipt_outbox_user",
            request_id="receipt_outbox_a",
        )
        return later, recovered, model, store

    later, recovered, model, store = asyncio.run(scenario())

    assert later.reply == "B 已完成。"
    assert recovered.reply == "A 已完成。"
    assert recovered.run_id != later.run_id
    assert len(model.calls) == 2
    receipt = store.get_receipt_record(
        "api:api:receipt_outbox_user", "receipt_outbox_a"
    )
    assert receipt.status == "completed"


def test_staged_processing_receipt_validates_payload_before_returning_response() -> None:
    async def scenario() -> None:
        store = InMemoryAgentRuntimeStore()
        runtime = AgentRuntime(
            model=ScriptedModel([_answer_turn("不应调用模型")]),
            tool_registry=AgentToolRegistry(),
            checkpointer=InMemorySaver(),
            runtime_store=store,
            settings=replace(default_settings, agent_max_verifier_passes=0),
        )
        user_id = "staged_fingerprint_user"
        thread_id = runtime.thread_id_for(user_id, "api")
        original_fingerprint = runtime._receipt_fingerprint(
            "start",
            {
                "thread_id": thread_id,
                "message": "原请求",
                "user_id": user_id,
                "source": "api",
            },
        )
        receipt_scope = f"api:api:{user_id}"
        store.begin_receipt(
            receipt_scope,
            "staged_request",
            original_fingerprint,
            thread_id,
            "run_original",
        )
        store.stage_receipt_response(
            receipt_scope,
            "staged_request",
            {
                "thread_id": thread_id,
                "run_id": "run_original",
                "status": "completed",
                "reply": "原请求结果",
            },
        )

        with pytest.raises(AgentRuntimeConflict, match="different payload"):
            await runtime.start(
                "不同请求",
                user_id=user_id,
                request_id="staged_request",
            )

    asyncio.run(scenario())


def test_processing_receipt_rejects_request_id_reuse_with_different_payload() -> None:
    async def scenario():
        saver = InMemorySaver()
        store = InMemoryAgentRuntimeStore()
        runtime = AgentRuntime(
            model=ScriptedModel([_answer_turn("不会提交回执")]),
            tool_registry=AgentToolRegistry(),
            checkpointer=saver,
            runtime_store=store,
            settings=replace(default_settings, agent_max_verifier_passes=0),
        )
        thread_id = runtime.thread_id_for("fingerprint_user", "api")
        fingerprint = runtime._receipt_fingerprint(
            "start",
            {
                "thread_id": thread_id,
                "message": "原请求",
                "user_id": "fingerprint_user",
                "source": "api",
            },
        )
        store.begin_receipt(
            "api:api:fingerprint_user",
            "same_request_id",
            fingerprint,
            thread_id,
            "run_stale",
        )
        with pytest.raises(AgentRuntimeConflict, match="different payload"):
            await runtime.start(
                "不同请求",
                user_id="fingerprint_user",
                request_id="same_request_id",
            )

    asyncio.run(scenario())


def test_completed_receipt_cannot_leak_across_conversation_threads() -> None:
    async def scenario():
        runtime = _runtime(ScriptedModel([_answer_turn("scope A")]))
        await runtime.start(
            "同一个入站事件",
            user_id="scope_user",
            conversation_scope="scope_a",
            request_id="scope_request",
        )
        with pytest.raises(AgentRuntimeConflict, match="different conversation thread"):
            await runtime.start(
                "同一个入站事件",
                user_id="scope_user",
                conversation_scope="scope_b",
                request_id="scope_request",
            )

    asyncio.run(scenario())


def test_runtime_persists_compacted_history_but_keeps_raw_audit_events() -> None:
    async def scenario():
        saver = InMemorySaver()
        store = InMemoryAgentRuntimeStore()
        model = ScriptedModel(
            [_answer_turn(f"answer-{index}-" + "a" * 500) for index in range(8)]
        )
        runtime = AgentRuntime(
            model=model,
            tool_registry=AgentToolRegistry(),
            checkpointer=saver,
            runtime_store=store,
                settings=replace(
                    default_settings,
                    context_max_input_tokens=1800,
                    context_keep_recent_turns=3,
                    agent_max_verifier_passes=0,
                ),
        )
        response = None
        for index in range(8):
            response = await runtime.start(
                f"question-{index}-" + "q" * 500,
                user_id="context_user",
                request_id=f"context_request_{index}",
            )
        assert response is not None
        snapshot = await runtime.graph.aget_state(runtime._config(response.thread_id))
        return dict(snapshot.values), store

    state, store = asyncio.run(scenario())

    assert state["conversation_summary"] is not None
    assert state["conversation_summary"]["turns"]
    assert any(
        message.get("role") == "system"
        and str(message.get("content", "")).startswith("Historical conversation summary")
        for message in state["messages"]
    )
    assert len(state["messages"]) < 16
    assert len(store.events) == 16


def test_runtime_isolates_owner_context_and_rejects_cross_actor_state_access() -> None:
    observed_owners: List[str] = []

    async def whoami(arguments: Dict[str, Any]) -> AgentToolResult:
        observed_owners.append(arguments["owner_id"])
        return AgentToolResult(data={"owner_id": arguments["owner_id"]}, message="owner")

    model = ScriptedModel(
        [
            _tool_turn(_call("owner_a", "whoami")),
            _answer_turn("user A complete"),
            _tool_turn(_call("owner_b", "whoami")),
            _answer_turn("user B complete"),
        ]
    )
    runtime = _runtime(model, [_tool("whoami", whoami)])

    async def scenario():
        first = await runtime.start(
            "查询我的数据",
            user_id="user_a",
            conversation_scope="shared_scope",
        )
        second = await runtime.start(
            "查询我的数据",
            user_id="user_b",
            conversation_scope="shared_scope",
        )
        with pytest.raises(AgentActorMismatch):
            await runtime.get_state(
                first.thread_id,
                user_id="user_b",
                source="api",
            )
        return first, second

    first, second = asyncio.run(scenario())

    assert first.thread_id != second.thread_id
    assert observed_owners == ["api:user_a", "api:user_b"]
    second_thread_messages = model.calls[2]["messages"]
    assert not any("user A complete" in str(message) for message in second_thread_messages)


def test_runtime_stops_at_model_budget_with_partial_result() -> None:
    calls = [_call("budget_1", "read_value"), _call("budget_2", "read_value")]

    async def read_value(arguments: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(data={"value": 1}, message="same result")

    model = ScriptedModel(
        [
            _tool_turn(calls[0]),
            _tool_turn(calls[1]),
            _answer_turn("模型调用限制已触发，返回部分结果。"),
        ]
    )
    runtime = _runtime(
        model,
        [_tool("read_value", read_value)],
        max_model_turns=2,
    )

    response = asyncio.run(runtime.start("持续查询直到预算结束"))

    assert response.status.value == "partial"
    assert "限制" in response.reply
    assert response.usage.model_calls == 3
    assert response.usage.tool_calls == 2
    assert model.calls[-1]["tools"] == []


def test_runtime_stops_after_repeated_no_progress_observations() -> None:
    executions = 0

    async def unchanged(arguments: Dict[str, Any]) -> AgentToolResult:
        nonlocal executions
        executions += 1
        return AgentToolResult(data={"value": "unchanged"}, message="unchanged")

    model = ScriptedModel(
        [
            _tool_turn(_call("same_observation_1", "unchanged_read")),
            _tool_turn(_call("same_observation_2", "unchanged_read")),
            _tool_turn(_call("same_observation_3", "unchanged_read")),
            _answer_turn("连续无进展，返回已观察到的部分结果。"),
        ]
    )
    runtime = _runtime(
        model,
        [_tool("unchanged_read", unchanged)],
        max_model_turns=10,
        no_progress_limit=2,
    )

    response = asyncio.run(runtime.start("重复查询相同数据"))

    assert executions == 3
    assert response.status.value == "partial"
    assert "无进展" in response.reply
    assert response.usage.model_calls == 4
    assert response.usage.tool_calls == 3
    assert model.calls[-1]["tools"] == []


def test_runtime_detects_an_alternating_no_progress_cycle() -> None:
    executions = 0

    async def alternating(arguments: Dict[str, Any]) -> AgentToolResult:
        nonlocal executions
        executions += 1
        value = "a" if executions % 2 else "b"
        return AgentToolResult(data={"value": value}, message=value)

    model = ScriptedModel(
        [
            _tool_turn(_call(f"cycle_{index}", "alternating_read"))
            for index in range(1, 6)
        ]
        + [_answer_turn("检测到交替回环，停止继续查询。")]
    )
    runtime = _runtime(
        model,
        [_tool("alternating_read", alternating)],
        max_model_turns=10,
        no_progress_limit=2,
    )

    response = asyncio.run(runtime.start("反复查询交替结果"))

    assert executions == 4
    assert response.status.value == "partial"
    assert response.usage.tool_calls == 4
    assert model.calls[-1]["tools"] == []


def test_sqlite_checkpoint_restores_interrupt_without_repeating_write(tmp_path) -> None:
    async def scenario():
        checkpoint_path = tmp_path / "checkpoints.db"
        runtime_path = tmp_path / "runtime.db"
        writes: List[Dict[str, Any]] = []

        async def write_handler(arguments: Dict[str, Any]) -> AgentToolResult:
            writes.append(dict(arguments))
            return AgentToolResult(
                data={"event_id": "event_restart_1"},
                status=ToolOutcomeStatus.SUCCESS,
                message="calendar write succeeded",
            )

        tool = _tool(
            "sync_calendar",
            write_handler,
            effect=ToolEffect.EXTERNAL_WRITE,
            approval=ToolApproval.ALWAYS,
            parallel_safe=False,
            properties={"plan_id": {"type": "string"}},
            required=["plan_id"],
        )
        runtime_settings = replace(
            default_settings,
            agent_max_verifier_passes=0,
        )
        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
            await saver.setup()
            first_runtime = AgentRuntime(
                model=ScriptedModel(
                    [_tool_turn(_call("restart_call", "sync_calendar", plan_id="plan_1"))]
                ),
                tool_registry=AgentToolRegistry([tool]),
                checkpointer=saver,
                runtime_store=SQLiteAgentRuntimeStore(str(runtime_path)),
                settings=runtime_settings,
            )
            paused = await first_runtime.start(
                "同步日历",
                user_id="user_restart",
                request_id="restart_start",
            )
        assert paused.status.value == "waiting_for_input"
        assert writes == []

        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
            await saver.setup()
            second_runtime = AgentRuntime(
                model=ScriptedModel([_answer_turn("重启后同步完成。")]),
                tool_registry=AgentToolRegistry([tool]),
                checkpointer=saver,
                runtime_store=SQLiteAgentRuntimeStore(str(runtime_path)),
                settings=runtime_settings,
            )
            resumed = await second_runtime.resume(
                thread_id=paused.thread_id,
                interaction_id=paused.interaction.id,
                decision="approve",
                user_id="user_restart",
                request_id="restart_resume",
            )
            duplicate = await second_runtime.resume(
                thread_id=paused.thread_id,
                interaction_id=paused.interaction.id,
                decision="approve",
                user_id="user_restart",
                request_id="restart_resume",
            )
        return paused, resumed, duplicate, writes

    paused, resumed, duplicate, writes = asyncio.run(scenario())

    assert resumed.status.value == "completed"
    assert resumed.reply == "重启后同步完成。"
    assert duplicate == resumed
    assert len(writes) == 1
    assert writes[0]["idempotency_key"].startswith("agent_op_")
    assert paused.thread_id == resumed.thread_id


def test_started_write_without_receipt_becomes_unknown_and_is_not_retried() -> None:
    async def scenario():
        writes = 0

        async def write_handler(arguments: Dict[str, Any]) -> AgentToolResult:
            nonlocal writes
            writes += 1
            return AgentToolResult(data={"event_id": "must_not_execute"})

        model = ScriptedModel(
            [
                _tool_turn(_call("unknown_call", "external_write", value="x")),
                _answer_turn("写入结果无法确认，需要先协调。"),
            ]
        )
        store = InMemoryAgentRuntimeStore()
        runtime = AgentRuntime(
            model=model,
            tool_registry=AgentToolRegistry(
                [
                    _tool(
                        "external_write",
                        write_handler,
                        effect=ToolEffect.EXTERNAL_WRITE,
                        approval=ToolApproval.ALWAYS,
                        parallel_safe=False,
                        properties={"value": {"type": "string"}},
                        required=["value"],
                    )
                ]
            ),
            checkpointer=InMemorySaver(),
            runtime_store=store,
            settings=replace(default_settings, agent_max_verifier_passes=0),
        )
        paused = await runtime.start("执行外部写入", user_id="unknown_user")
        snapshot = await runtime.graph.aget_state(runtime._config(paused.thread_id))
        state = dict(snapshot.values)
        call = dict(state["validated_calls"][0])
        arguments = runtime._inject_actor_context(state, call)
        operation_key = runtime._operation_key(state, call)
        store.begin_operation(
            operation_key,
            runtime._call_fingerprint(call["name"], arguments),
        )
        resumed = await runtime.resume(
            paused.thread_id,
            paused.interaction.id,
            "approve",
            user_id="unknown_user",
        )
        return resumed, writes

    resumed, writes = asyncio.run(scenario())

    assert writes == 0
    assert resumed.status.value == "partial"
    assert resumed.tool_executions[-1].status.value == "unknown"
    assert "write_outcome_unknown" in resumed.warnings


def test_semantically_identical_unknown_write_is_blocked_in_a_later_run() -> None:
    async def scenario():
        attempts = 0

        async def uncertain_write(arguments: Dict[str, Any]) -> AgentToolResult:
            nonlocal attempts
            attempts += 1
            raise RuntimeError("connection dropped after dispatch")

        tool = _tool(
            "external_write",
            uncertain_write,
            effect=ToolEffect.EXTERNAL_WRITE,
            approval=ToolApproval.ALWAYS,
            parallel_safe=False,
            properties={"value": {"type": "string"}},
            required=["value"],
        )
        store = InMemoryAgentRuntimeStore()
        runtime = AgentRuntime(
            model=ScriptedModel(
                [
                    _tool_turn(_call("first_write", "external_write", value="same")),
                    _answer_turn("第一次写入结果未知。"),
                ]
            ),
            tool_registry=AgentToolRegistry([tool]),
            checkpointer=InMemorySaver(),
            runtime_store=store,
            settings=replace(default_settings, agent_max_verifier_passes=0),
        )
        first_pause = await runtime.start(
            "perform the external write",
            user_id="cross_run_unknown",
            request_id="cross_run_unknown_1",
        )
        first = await runtime.resume(
            first_pause.thread_id,
            first_pause.interaction.id,
            "approve",
            user_id="cross_run_unknown",
            request_id="cross_run_unknown_2",
        )

        runtime.model = ScriptedModel(
            [
                _tool_turn(_call("second_write", "external_write", value="same")),
                _answer_turn("检测到此前未知写入，未重复执行。"),
            ]
        )
        second_pause = await runtime.start(
            "retry the same external write",
            user_id="cross_run_unknown",
            request_id="cross_run_unknown_3",
        )
        second = await runtime.resume(
            second_pause.thread_id,
            second_pause.interaction.id,
            "approve",
            user_id="cross_run_unknown",
            request_id="cross_run_unknown_4",
        )
        return first, second, attempts

    first, second, attempts = asyncio.run(scenario())

    assert first.status.value == "partial"
    assert second.status.value == "partial"
    assert attempts == 1
    assert second.tool_executions[-1].error_code == "prior_write_outcome_unknown"


def test_unknown_write_can_be_reconciled_as_applied_in_a_later_run() -> None:
    async def scenario():
        attempts = 0

        async def uncertain_write(arguments: Dict[str, Any]) -> AgentToolResult:
            nonlocal attempts
            attempts += 1
            raise RuntimeError("connection dropped after dispatch")

        write_tool = _tool(
            "external_write",
            uncertain_write,
            effect=ToolEffect.EXTERNAL_WRITE,
            approval=ToolApproval.ALWAYS,
            parallel_safe=False,
            properties={"value": {"type": "string"}},
            required=["value"],
        )
        store = InMemoryAgentRuntimeStore()
        runtime = AgentRuntime(
            model=ScriptedModel(
                [
                    _tool_turn(_call("write_unknown", "external_write", value="x")),
                    _answer_turn("写入结果未知，等待核对。"),
                ]
            ),
            tool_registry=AgentToolRegistry(
                [write_tool, _generic_reconciliation_tool()]
            ),
            checkpointer=InMemorySaver(),
            runtime_store=store,
            settings=replace(default_settings, agent_max_verifier_passes=0),
        )
        first_pause = await runtime.start(
            "执行外部写入",
            user_id="reconcile_applied",
            request_id="reconcile_applied_1",
        )
        first = await runtime.resume(
            first_pause.thread_id,
            first_pause.interaction.id,
            "approve",
            user_id="reconcile_applied",
            request_id="reconcile_applied_2",
        )
        operation_key = next(
            key
            for key, record in store.operations.items()
            if record.tool_name == "external_write"
        )

        runtime.model = ScriptedModel(
            [
                _tool_turn(
                    _call(
                        "reconcile_applied",
                        "reconcile_write_operation",
                        operation_key=operation_key,
                        outcome="applied",
                        evidence="The destination system contains the expected record.",
                    )
                ),
                _answer_turn("已核对，原写入确实成功，没有重复执行。"),
            ]
        )
        reconcile_pause = await runtime.start(
            "核对刚才结果未知的写入",
            user_id="reconcile_applied",
            request_id="reconcile_applied_3",
        )
        before_approval = store.get_operation(operation_key)
        reconciled = await runtime.resume(
            reconcile_pause.thread_id,
            reconcile_pause.interaction.id,
            "approve",
            user_id="reconcile_applied",
            request_id="reconcile_applied_4",
        )
        runtime.model = ScriptedModel(
            [
                _tool_turn(
                    _call("model_retried_applied_write", "external_write", value="x")
                ),
                _answer_turn("已复用人工确认的成功回执，没有重复写入。"),
            ]
        )
        retry_pause = await runtime.start(
            "再次检查这项写入",
            user_id="reconcile_applied",
            request_id="reconcile_applied_5",
        )
        deduplicated = await runtime.resume(
            retry_pause.thread_id,
            retry_pause.interaction.id,
            "approve",
            user_id="reconcile_applied",
            request_id="reconcile_applied_6",
        )
        return (
            first,
            reconcile_pause,
            before_approval,
            reconciled,
            deduplicated,
            attempts,
            store,
            operation_key,
        )

    (
        first,
        pause,
        before,
        reconciled,
        deduplicated,
        attempts,
        store,
        operation_key,
    ) = asyncio.run(scenario())

    assert first.status.value == "partial"
    assert pause.status.value == "waiting_for_input"
    assert before is not None and before.status == "unknown"
    assert attempts == 1
    assert reconciled.status.value == "completed"
    assert reconciled.tool_executions[-1].name == "external_write"
    assert reconciled.tool_executions[-1].status.value == "success"
    assert deduplicated.status.value == "completed"
    assert deduplicated.tool_executions[-1].status.value == "success"
    operation = store.get_operation(operation_key)
    assert operation is not None and operation.status == "success"


def test_not_applied_reconciliation_allows_the_original_write_to_retry() -> None:
    async def scenario():
        attempts = 0

        async def uncertain_then_success(arguments: Dict[str, Any]) -> AgentToolResult:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("connection dropped after dispatch")
            return AgentToolResult(data={"record_id": "record_1"}, message="written")

        write_tool = _tool(
            "external_write",
            uncertain_then_success,
            effect=ToolEffect.EXTERNAL_WRITE,
            approval=ToolApproval.ALWAYS,
            parallel_safe=False,
            properties={"value": {"type": "string"}},
            required=["value"],
        )
        store = InMemoryAgentRuntimeStore()
        runtime = AgentRuntime(
            model=ScriptedModel(
                [
                    _tool_turn(_call("write_unknown", "external_write", value="x")),
                    _answer_turn("写入结果未知。"),
                ]
            ),
            tool_registry=AgentToolRegistry(
                [write_tool, _generic_reconciliation_tool()]
            ),
            checkpointer=InMemorySaver(),
            runtime_store=store,
            settings=replace(default_settings, agent_max_verifier_passes=0),
        )
        first_pause = await runtime.start(
            "执行外部写入",
            user_id="reconcile_not_applied",
        )
        first = await runtime.resume(
            first_pause.thread_id,
            first_pause.interaction.id,
            "approve",
            user_id="reconcile_not_applied",
        )
        operation_key = next(
            key
            for key, record in store.operations.items()
            if record.tool_name == "external_write"
        )

        runtime.model = ScriptedModel(
            [
                _tool_turn(
                    _call(
                        "reconcile_not_applied",
                        "reconcile_write_operation",
                        operation_key=operation_key,
                        outcome="not_applied",
                        evidence="The destination has no matching record.",
                    )
                ),
                _tool_turn(_call("retry_write", "external_write", value="x")),
                _answer_turn("已确认第一次未写入，重试后成功。"),
            ]
        )
        reconcile_pause = await runtime.start(
            "确认未写入后重试",
            user_id="reconcile_not_applied",
        )
        retry_pause = await runtime.resume(
            reconcile_pause.thread_id,
            reconcile_pause.interaction.id,
            "approve",
            user_id="reconcile_not_applied",
        )
        resolved_before_retry = store.get_operation(operation_key)
        completed = await runtime.resume(
            retry_pause.thread_id,
            retry_pause.interaction.id,
            "approve",
            user_id="reconcile_not_applied",
        )
        return completed, attempts, resolved_before_retry

    completed, attempts, old_operation = asyncio.run(scenario())

    assert old_operation is not None
    assert old_operation.status == "error"
    assert old_operation.outcome["error_code"] == "write_confirmed_not_applied"
    assert old_operation.outcome["retryable"] is True
    assert attempts == 2
    assert completed.status.value == "completed"
    assert completed.tool_executions[-1].name == "external_write"
    assert completed.tool_executions[-1].status.value == "success"


@pytest.mark.parametrize("case", ["different_owner", "unknown_key"])
def test_write_reconciliation_rejects_unowned_or_unknown_operation_keys(case: str) -> None:
    async def scenario():
        store = InMemoryAgentRuntimeStore()
        runtime = AgentRuntime(
            model=ScriptedModel(
                [
                    _tool_turn(
                        _call(
                            "invalid_reconcile",
                            "reconcile_write_operation",
                            operation_key="foreign_operation",
                            outcome="applied",
                            evidence="unsupported assertion",
                        )
                    ),
                    _answer_turn("无法协调该写入。"),
                ]
            ),
            tool_registry=AgentToolRegistry([_generic_reconciliation_tool()]),
            checkpointer=InMemorySaver(),
            runtime_store=store,
            settings=replace(default_settings, agent_max_verifier_passes=0),
        )
        if case == "different_owner":
            thread_id = runtime.thread_id_for("attacker", "api")
            store.begin_operation(
                "foreign_operation",
                "fingerprint",
                thread_id=thread_id,
                owner_id="api:victim",
                run_id="victim_run",
                tool_name="external_write",
            )
            store.finish_operation(
                "foreign_operation",
                ToolOutcomeStatus.UNKNOWN.value,
                AgentToolResult(
                    data={},
                    is_error=True,
                    status=ToolOutcomeStatus.UNKNOWN,
                    message="unknown",
                ).to_outcome_dict(),
            )
        response = await runtime.start(
            "协调一个写入",
            user_id="attacker",
        )
        return response, store.get_operation("foreign_operation")

    response, operation = asyncio.run(scenario())

    assert response.status.value != "waiting_for_input"
    assert response.tool_executions[0].status.value == "rejected"
    assert response.tool_executions[0].error_code == "invalid_write_reconciliation"
    if operation is not None:
        assert operation.status == "unknown"


def test_semantically_identical_non_idempotent_partial_write_is_blocked_later() -> None:
    async def scenario():
        attempts = 0

        async def partial_write(arguments: Dict[str, Any]) -> AgentToolResult:
            nonlocal attempts
            attempts += 1
            return AgentToolResult(
                data={"application_id": "application_1"},
                is_error=True,
                status=ToolOutcomeStatus.PARTIAL,
                message="Application saved, but downstream sync failed.",
                error_code="downstream_sync_failed",
            )

        tool = _tool(
            "create_application",
            partial_write,
            effect=ToolEffect.LOCAL_WRITE,
            approval=ToolApproval.NEVER,
            parallel_safe=False,
            properties={"company": {"type": "string"}},
            required=["company"],
            idempotent=False,
        )
        store = InMemoryAgentRuntimeStore()
        runtime = AgentRuntime(
            model=ScriptedModel(
                [
                    _tool_turn(
                        _call(
                            "first_partial_write",
                            "create_application",
                            company="Acme",
                        )
                    ),
                    _answer_turn("投递已保存，但同步失败。"),
                ]
            ),
            tool_registry=AgentToolRegistry([tool]),
            checkpointer=InMemorySaver(),
            runtime_store=store,
            settings=replace(default_settings, agent_max_verifier_passes=0),
        )
        first = await runtime.start(
            "create application for Acme",
            user_id="cross_run_partial",
            request_id="cross_run_partial_1",
        )

        runtime.model = ScriptedModel(
            [
                _tool_turn(
                    _call(
                        "second_partial_write",
                        "create_application",
                        company="Acme",
                    )
                ),
                _answer_turn("检测到此前部分写入，未重复创建。"),
            ]
        )
        second = await runtime.start(
            "retry creating application for Acme",
            user_id="cross_run_partial",
            request_id="cross_run_partial_2",
        )
        return first, second, attempts

    first, second, attempts = asyncio.run(scenario())

    assert first.status.value == "partial"
    assert second.status.value == "partial"
    assert attempts == 1
    assert second.tool_executions[-1].error_code == "downstream_sync_failed"


def test_semantically_identical_idempotent_partial_write_can_retry_later() -> None:
    async def scenario():
        attempts = 0

        async def partial_then_success(arguments: Dict[str, Any]) -> AgentToolResult:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return AgentToolResult(
                    data={"synced": 1, "failed": 1},
                    is_error=True,
                    status=ToolOutcomeStatus.PARTIAL,
                    message="One item failed.",
                    error_code="partial_sync",
                )
            return AgentToolResult(data={"synced": 2})

        tool = _tool(
            "idempotent_write",
            partial_then_success,
            effect=ToolEffect.EXTERNAL_WRITE,
            approval=ToolApproval.ALWAYS,
            parallel_safe=False,
            properties={"batch": {"type": "string"}},
            required=["batch"],
            idempotent=True,
        )
        store = InMemoryAgentRuntimeStore()
        runtime = AgentRuntime(
            model=ScriptedModel(
                [
                    _tool_turn(_call("partial_once", "idempotent_write", batch="b1")),
                    _answer_turn("首次同步部分成功。"),
                ]
            ),
            tool_registry=AgentToolRegistry([tool]),
            checkpointer=InMemorySaver(),
            runtime_store=store,
            settings=replace(default_settings, agent_max_verifier_passes=0),
        )
        first_pause = await runtime.start(
            "sync batch b1",
            user_id="cross_run_idempotent",
        )
        first = await runtime.resume(
            first_pause.thread_id,
            first_pause.interaction.id,
            "approve",
            user_id="cross_run_idempotent",
        )

        runtime.model = ScriptedModel(
            [
                _tool_turn(_call("retry_idempotent", "idempotent_write", batch="b1")),
                _answer_turn("重试同步完成。"),
            ]
        )
        second_pause = await runtime.start(
            "retry syncing batch b1",
            user_id="cross_run_idempotent",
        )
        second = await runtime.resume(
            second_pause.thread_id,
            second_pause.interaction.id,
            "approve",
            user_id="cross_run_idempotent",
        )
        return first, second, attempts

    first, second, attempts = asyncio.run(scenario())

    assert first.status.value == "partial"
    assert second.status.value == "completed"
    assert attempts == 2


def test_unknown_write_cannot_be_retried_by_changing_only_tool_call_id() -> None:
    async def scenario():
        attempts = 0

        async def uncertain_write(arguments: Dict[str, Any]) -> AgentToolResult:
            nonlocal attempts
            attempts += 1
            raise RuntimeError("connection dropped after dispatch")

        model = ScriptedModel(
            [
                _tool_turn(_call("write_1", "external_write", value="same")),
                _tool_turn(_call("write_2", "external_write", value="same")),
                _answer_turn("写入结果未知，已停止重复写入。"),
            ]
        )
        runtime = _runtime(
            model,
            [
                _tool(
                    "external_write",
                    uncertain_write,
                    effect=ToolEffect.EXTERNAL_WRITE,
                    approval=ToolApproval.ALWAYS,
                    parallel_safe=False,
                    properties={"value": {"type": "string"}},
                    required=["value"],
                )
            ],
        )

        first_pause = await runtime.start("执行外部写入", user_id="unknown_retry")
        second_pause = await runtime.resume(
            first_pause.thread_id,
            first_pause.interaction.id,
            "approve",
            user_id="unknown_retry",
        )
        completed = await runtime.resume(
            second_pause.thread_id,
            second_pause.interaction.id,
            "approve",
            user_id="unknown_retry",
        )
        return attempts, completed

    attempts, completed = asyncio.run(scenario())

    assert attempts == 1
    assert completed.status.value == "partial"
    assert [item.status.value for item in completed.tool_executions] == [
        "unknown",
        "unknown",
    ]


def test_unhandled_local_write_exception_is_unknown_and_not_retried() -> None:
    attempts = 0

    async def uncertain_local_write(arguments: Dict[str, Any]) -> AgentToolResult:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("database connection dropped after commit")

    model = ScriptedModel(
        [
            _tool_turn(_call("local_write_1", "write_local", value="same")),
            _tool_turn(_call("local_write_2", "write_local", value="same")),
            _answer_turn("本地写入结果未知，已停止重试。"),
        ]
    )
    runtime = _runtime(
        model,
        [
            _tool(
                "write_local",
                uncertain_local_write,
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.NEVER,
                parallel_safe=False,
                properties={"value": {"type": "string"}},
                required=["value"],
            )
        ],
    )

    response = asyncio.run(runtime.start("写入本地记录"))

    assert attempts == 1
    assert response.status.value == "partial"
    assert [item.status.value for item in response.tool_executions] == [
        "unknown",
        "unknown",
    ]
    assert all(
        item.error_code == "write_outcome_unknown"
        for item in response.tool_executions
    )


def test_revising_approval_does_not_create_a_false_write_receipt() -> None:
    async def scenario():
        writes = 0

        async def write_handler(arguments: Dict[str, Any]) -> AgentToolResult:
            nonlocal writes
            writes += 1
            return AgentToolResult(data={"written": True})

        model = ScriptedModel(
            [
                _tool_turn(_call("revise_write", "external_write", value="old")),
                _answer_turn("已按你的修改要求停止原操作。"),
            ]
        )
        runtime = _runtime(
            model,
            [
                _tool(
                    "external_write",
                    write_handler,
                    effect=ToolEffect.EXTERNAL_WRITE,
                    approval=ToolApproval.ALWAYS,
                    parallel_safe=False,
                    properties={"value": {"type": "string"}},
                    required=["value"],
                )
            ],
        )
        paused = await runtime.start("写入 old", user_id="revision_user")
        response = await runtime.resume(
            paused.thread_id,
            paused.interaction.id,
            "revise",
            text="改成 new，先不要执行",
            user_id="revision_user",
        )
        return writes, response

    writes, response = asyncio.run(scenario())

    assert writes == 0
    assert response.status.value == "partial"
    assert response.tool_executions[-1].status.value == "rejected"
    assert response.tool_executions[-1].error_code == "revision_requested"


def test_tool_requested_input_is_counted_once_after_resume() -> None:
    async def needs_input(arguments: Dict[str, Any]) -> AgentToolResult:
        interaction = ToolInteraction(
            kind="preference_form",
            prompt="请补充偏好",
            input_schema={
                "type": "object",
                "properties": {"minutes": {"type": "integer"}},
                "required": ["minutes"],
                "additionalProperties": False,
            },
            allowed_actions=["answer", "cancel"],
        )
        return AgentToolResult(
            data={},
            status=ToolOutcomeStatus.NEEDS_INPUT,
            message=interaction.prompt,
            interaction=interaction,
        )

    async def scenario():
        model = ScriptedModel(
            [
                _tool_turn(_call("preferences", "create_plan")),
                _answer_turn("已收到偏好，后续可以继续排期。"),
            ]
        )
        runtime = _runtime(
            model,
            [
                _tool(
                    "create_plan",
                    needs_input,
                    effect=ToolEffect.LOCAL_WRITE,
                    approval=ToolApproval.NEVER,
                    parallel_safe=False,
                )
            ],
        )
        paused = await runtime.start("生成计划", user_id="preference_user")
        response = await runtime.resume(
            paused.thread_id,
            paused.interaction.id,
            "answer",
            value={"minutes": 45},
            user_id="preference_user",
        )
        return paused, response

    paused, response = asyncio.run(scenario())

    assert paused.usage.tool_calls == 1
    assert response.usage.tool_calls == 1
    assert response.status.value == "completed"


def test_duplicate_request_id_is_serialized_across_conversation_scopes() -> None:
    async def scenario():
        model = ScriptedModel([_answer_turn("只执行一次。")])
        runtime = _runtime(model)
        results = await asyncio.gather(
            runtime.start(
                "同一个请求",
                user_id="receipt_user",
                conversation_scope="scope_a",
                request_id="same_request",
            ),
            runtime.start(
                "同一个请求",
                user_id="receipt_user",
                conversation_scope="scope_b",
                request_id="same_request",
            ),
            return_exceptions=True,
        )
        return model, results

    model, results = asyncio.run(scenario())

    assert len(model.calls) == 1
    assert sum(isinstance(item, AgentRuntimeConflict) for item in results) == 1
    assert sum(not isinstance(item, Exception) for item in results) == 1


def test_timed_out_write_is_unknown_and_not_retryable() -> None:
    async def slow_write(arguments: Dict[str, Any]) -> AgentToolResult:
        await asyncio.sleep(1)
        return AgentToolResult(data={"written": True})

    async def scenario():
        model = ScriptedModel(
            [
                _tool_turn(_call("slow_write", "external_write", value="x")),
                _answer_turn("写入超时，结果未知。"),
            ]
        )
        runtime = _runtime(
            model,
            [
                _tool(
                    "external_write",
                    slow_write,
                    effect=ToolEffect.EXTERNAL_WRITE,
                    approval=ToolApproval.ALWAYS,
                    parallel_safe=False,
                    timeout_seconds=0.1,
                    properties={"value": {"type": "string"}},
                    required=["value"],
                )
            ],
        )
        paused = await runtime.start("执行慢写入", user_id="timeout_user")
        return await runtime.resume(
            paused.thread_id,
            paused.interaction.id,
            "approve",
            user_id="timeout_user",
        )

    response = asyncio.run(scenario())

    assert response.status.value == "partial"
    assert response.tool_executions[-1].status.value == "unknown"
    assert response.tool_executions[-1].retryable is False
    assert response.tool_executions[-1].error_code == "write_outcome_unknown"


def test_cancel_pairs_pending_tool_call_and_allows_a_new_turn() -> None:
    async def write_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        raise AssertionError("cancelled write must not execute")

    async def scenario():
        model = ScriptedModel(
            [
                _tool_turn(_call("cancel_me", "external_write", value="x")),
                _answer_turn("新的任务已处理。"),
            ]
        )
        runtime = _runtime(
            model,
            [
                _tool(
                    "external_write",
                    write_handler,
                    effect=ToolEffect.EXTERNAL_WRITE,
                    approval=ToolApproval.ALWAYS,
                    parallel_safe=False,
                    properties={"value": {"type": "string"}},
                    required=["value"],
                )
            ],
        )
        paused = await runtime.start("执行写入", user_id="cancel_user")
        cancelled = await runtime.resume(
            paused.thread_id,
            paused.interaction.id,
            "cancel",
            user_id="cancel_user",
        )
        snapshot = await runtime.graph.aget_state(runtime._config(paused.thread_id))
        next_turn = await runtime.start("处理新的任务", user_id="cancel_user")
        return runtime, cancelled, dict(snapshot.values), next_turn

    runtime, cancelled, state, next_turn = asyncio.run(scenario())

    assert cancelled.status.value == "partial"
    assert cancelled.tool_executions[-1].error_code == "cancelled_by_user"
    assert runtime._unpaired_tool_call_ids(state["messages"]) == []
    cancel_event = runtime.runtime_store.events[
        cancelled.run_id + ":tool:cancel_me"
    ]
    assert cancel_event["payload"]["status"] == "rejected"
    assert next_turn.status.value == "completed"
    assert next_turn.reply == "新的任务已处理。"


def test_parallel_needs_input_keeps_one_interrupt_and_pairs_every_call() -> None:
    async def needs_input(arguments: Dict[str, Any]) -> AgentToolResult:
        interaction = ToolInteraction(
            kind="clarification",
            prompt=f"请补充 {arguments['field']}",
            allowed_actions=["answer", "cancel"],
        )
        return AgentToolResult(
            data={},
            status=ToolOutcomeStatus.NEEDS_INPUT,
            message=interaction.prompt,
            interaction=interaction,
        )

    async def scenario():
        model = ScriptedModel(
            [
                _tool_turn(
                    _call("need_a", "read_a", field="A"),
                    _call("need_b", "read_b", field="B"),
                ),
                _answer_turn("已处理当前补充，另一个输入稍后再询问。"),
            ]
        )
        tools = [
            _tool(
                name,
                needs_input,
                properties={"field": {"type": "string"}},
                required=["field"],
            )
            for name in ("read_a", "read_b")
        ]
        runtime = _runtime(model, tools)
        paused = await runtime.start("并行读取", user_id="multi_input_user")
        response = await runtime.resume(
            paused.thread_id,
            paused.interaction.id,
            "answer",
            text="A 的补充信息",
            user_id="multi_input_user",
        )
        snapshot = await runtime.graph.aget_state(runtime._config(paused.thread_id))
        return runtime, paused, response, dict(snapshot.values)

    runtime, paused, response, state = asyncio.run(scenario())

    assert paused.interaction.tool_name == "read_a"
    assert paused.usage.tool_calls == 2
    assert response.usage.tool_calls == 2
    assert {item.call_id for item in response.tool_executions} == {"need_a", "need_b"}
    assert runtime._unpaired_tool_call_ids(state["messages"]) == []


def test_complex_run_uses_tool_free_verifier_and_reenters_agent_on_gap() -> None:
    async def update_plan(arguments: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(data={"plan": arguments}, message="plan updated")

    plan_tool = _tool(
        "update_execution_plan",
        update_plan,
        effect=ToolEffect.LOCAL_WRITE,
        approval=ToolApproval.NEVER,
        parallel_safe=False,
        control=True,
        properties={
            "goal": {"type": "string"},
            "steps": {"type": "array"},
        },
        required=["goal", "steps"],
    )
    model = ScriptedModel(
        [
            _tool_turn(
                _call(
                    "plan_call",
                    "update_execution_plan",
                    goal="完成任务",
                    steps=[
                        {
                            "id": "step_1",
                            "description": "完成",
                            "status": "completed",
                        }
                    ],
                )
            ),
            _answer_turn("第一版答复。"),
            _answer_turn('{"complete":false,"missing":["补充结果"]}'),
            _answer_turn("补充后的答复。"),
            _answer_turn('{"complete":true,"missing":[]}'),
        ]
    )
    runtime = _runtime(
        model,
        [plan_tool],
        max_verifier_passes=2,
    )

    response = asyncio.run(runtime.start("执行复杂任务"))

    assert response.status.value == "completed"
    assert response.reply == "补充后的答复。"
    assert response.usage.model_calls == 5
    assert response.usage.verifier_calls == 2
    assert model.calls[2]["tools"] == []
    assert model.calls[4]["tools"] == []
    assert any(
        message.get("role") == "user"
        and '"runtime_feedback": "completion_incomplete"'
        in str(message.get("content"))
        for message in model.calls[3]["messages"]
    )
    verifier_events = [
        event
        for event in runtime.runtime_store.events.values()
        if event["event_type"] == "verifier_result"
    ]
    assert len(verifier_events) == 2


def test_invalid_verifier_json_degrades_an_otherwise_complete_read_run() -> None:
    async def read_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(data={"ok": True}, message="read complete")

    model = ScriptedModel(
        [
            _tool_turn(
                _call("verify_read_a", "read_a"),
                _call("verify_read_b", "read_b"),
            ),
            _answer_turn("两个查询已完成。"),
            _answer_turn("not-json"),
        ]
    )
    runtime = _runtime(
        model,
        [_tool("read_a", read_handler), _tool("read_b", read_handler)],
        max_verifier_passes=1,
    )

    response = asyncio.run(runtime.start("查询两项数据并汇总"))

    assert response.status.value == "degraded"
    assert "verifier_unavailable" in response.warnings
    assert response.usage.verifier_calls == 1
    assert "两个查询已完成" not in response.reply
    assert response.reply == "这次没有拿到可靠结果，因此未将操作视为完成。"
    assert "核验" not in response.reply


def test_verifier_cannot_claim_complete_while_reporting_missing_work() -> None:
    async def read_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(data={"ok": True}, message="read complete")

    model = ScriptedModel(
        [
            _tool_turn(
                _call("contradiction_a", "read_a"),
                _call("contradiction_b", "read_b"),
            ),
            _answer_turn("查询完成。"),
            _answer_turn('{"complete":true,"missing":["仍缺少最终证据"]}'),
        ]
    )
    response = asyncio.run(
        _runtime(
            model,
            [_tool("read_a", read_handler), _tool("read_b", read_handler)],
            max_verifier_passes=1,
        ).start("查询后核验")
    )

    assert response.status.value == "failed"
    assert "completion_verifier_incomplete" in response.warnings
    assert "查询完成" not in response.reply
    assert response.reply == "这次没有拿到可靠结果，因此未将操作视为完成。"
    assert "核验" not in response.reply


def test_unavailable_verifier_degrades_a_write_with_a_success_receipt() -> None:
    class FailingVerifierModel(ScriptedModel):
        async def complete(self, messages, tools, options) -> ModelTurn:
            if len(self.calls) == 2:
                self.calls.append(
                    {
                        "messages": [dict(message) for message in messages],
                        "tools": list(tools),
                        "options": options,
                    }
                )
                raise LLMRequestError("verifier transport failed")
            return await super().complete(messages, tools, options)

    async def write_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(data={"receipt": "saved"}, message="saved")

    model = FailingVerifierModel(
        [
            _tool_turn(_call("verified_write", "write_local", value="x")),
            _answer_turn("写入完成。"),
        ]
    )
    runtime = _runtime(
        model,
        [
            _tool(
                "write_local",
                write_handler,
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.NEVER,
                parallel_safe=False,
                properties={"value": {"type": "string"}},
                required=["value"],
            )
        ],
        max_verifier_passes=1,
    )

    response = asyncio.run(runtime.start("写入 x"))

    assert response.status.value == "degraded"
    assert "verifier_unavailable" in response.warnings
    assert "写入完成" not in response.reply
    assert response.reply == "这次没有拿到可靠结果，因此未将操作视为完成。"
    assert "核验" not in response.reply
    assert response.usage.model_calls == 3
    event = runtime.runtime_store.events[f"{response.run_id}:model:3"]
    assert event["event_type"] == "model_error"
    assert event["payload"]["phase"] == "verifier"


def test_required_skipped_plan_step_is_not_considered_complete() -> None:
    async def update_plan(arguments: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(data={"plan": arguments}, message="plan updated")

    plan_tool = _tool(
        "update_execution_plan",
        update_plan,
        effect=ToolEffect.LOCAL_WRITE,
        approval=ToolApproval.NEVER,
        parallel_safe=False,
        control=True,
        properties={
            "goal": {"type": "string"},
            "revision": {"type": "integer"},
            "steps": {"type": "array"},
        },
        required=["goal", "steps"],
    )
    model = ScriptedModel(
        [
            _tool_turn(
                _call(
                    "plan_skipped",
                    "update_execution_plan",
                    goal="完成必选步骤",
                    steps=[
                        {
                            "id": "required_step",
                            "description": "必须完成",
                            "status": "skipped",
                            "required": True,
                        }
                    ],
                )
            ),
            _answer_turn("已完成。"),
            _tool_turn(
                _call(
                    "plan_completed",
                    "update_execution_plan",
                    goal="完成必选步骤",
                    revision=2,
                    steps=[
                        {
                            "id": "required_step",
                            "description": "必须完成",
                            "status": "completed",
                            "required": True,
                        }
                    ],
                )
            ),
            _answer_turn("现在确实完成了。"),
        ]
    )

    response = asyncio.run(_runtime(model, [plan_tool]).start("执行带计划的任务"))

    assert response.status.value == "completed"
    assert response.reply == "现在确实完成了。"
    assert len(model.calls) == 4
    assert any(
        "plan steps remain" in str(message.get("content"))
        for message in model.calls[2]["messages"]
    )


def test_completion_gaps_at_model_budget_are_reported_partial() -> None:
    async def update_plan(arguments: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(data={"plan": arguments}, message="plan updated")

    plan_tool = _tool(
        "update_execution_plan",
        update_plan,
        effect=ToolEffect.LOCAL_WRITE,
        approval=ToolApproval.NEVER,
        parallel_safe=False,
        control=True,
        properties={"goal": {"type": "string"}, "steps": {"type": "array"}},
        required=["goal", "steps"],
    )
    model = ScriptedModel(
        [
            _tool_turn(
                _call(
                    "budget_plan",
                    "update_execution_plan",
                    goal="必须完成",
                    steps=[
                        {
                            "id": "unfinished",
                            "description": "尚未完成的步骤",
                            "status": "pending",
                            "required": True,
                        }
                    ],
                )
            ),
            _answer_turn("我先声称完成。"),
        ]
    )

    response = asyncio.run(
        _runtime(model, [plan_tool], max_model_turns=2).start("执行预算受限任务")
    )

    assert response.status.value == "partial"
    assert "completion_guard_incomplete" in response.warnings
    assert response.usage.model_calls == 2


def test_unknown_checkpoint_schema_cannot_resume_old_pending_state() -> None:
    async def scenario() -> None:
        runtime = _runtime(ScriptedModel([_answer_turn("完成。")]))
        response = await runtime.start("建立会话")
        config = {"configurable": {"thread_id": response.thread_id}}
        await runtime.graph.aupdate_state(config, {"schema_version": 0})
        with pytest.raises(AgentRuntimeConflict, match="old pending action is invalid"):
            await runtime.get_state(response.thread_id)

    asyncio.run(scenario())


def test_explicit_write_request_cannot_finish_without_a_success_receipt() -> None:
    async def create_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(
            data={"application_id": "app_1"},
            status=ToolOutcomeStatus.SUCCESS,
            message="投递记录已创建。",
        )

    model = ScriptedModel(
        [
            _answer_turn("投递记录已经创建。"),
            _answer_turn(
                '{"complete":false,"missing":["缺少 create_application 成功写入回执"]}'
            ),
            _tool_turn(
                _call(
                    "create_after_guard",
                    "create_application",
                    company="Example Tech",
                )
            ),
            _answer_turn("已根据写入回执创建投递记录。"),
            _answer_turn('{"complete":true,"missing":[]}'),
        ]
    )
    tool = _tool(
        "create_application",
        create_handler,
        effect=ToolEffect.LOCAL_WRITE,
        approval=ToolApproval.NEVER,
        parallel_safe=False,
        properties={"company": {"type": "string"}},
        required=["company"],
    )

    response = asyncio.run(
        _runtime(model, [tool], max_verifier_passes=2).start(
            "帮我新增投递：Example Tech"
        )
    )

    assert response.status.value == "completed"
    assert response.usage.model_calls == 5
    assert response.usage.verifier_calls == 2
    assert response.tool_executions[0].name == "create_application"
    assert response.tool_executions[0].status.value == "success"
    assert "缺少 create_application 成功写入回执" in str(
        model.calls[2]["messages"]
    )


@pytest.mark.parametrize(
    ("message", "expected_explicit"),
    [
        ("我现在投递了几家公司的什么岗位？", False),
        ("帮我看看现在投递了几家公司", False),
        ("我今天投递了携程 AI Agent 岗位", True),
        ("帮我新增投递：Example Tech", True),
    ],
)
def test_write_approval_classifier_distinguishes_queries_from_explicit_writes(
    message: str,
    expected_explicit: bool,
) -> None:
    call = {"name": "create_application", "arguments": {}}
    assert AgentRuntime._explicitly_requests_write(call, message) is expected_explicit


def test_application_information_question_completes_after_read_receipt() -> None:
    async def list_handler(arguments: Dict[str, Any]) -> AgentToolResult:
        return AgentToolResult(
            data={
                "applications": [
                    {
                        "company": "携程",
                        "position": "AI Agent 岗位",
                        "status": "applied",
                    }
                ]
            },
            status=ToolOutcomeStatus.SUCCESS,
            message="当前投递记录：1. 携程 - AI Agent 岗位（已投递）",
        )

    model = ScriptedModel(
        [
            _tool_turn(_call("list_applications_1", "list_applications")),
            _answer_turn("你目前投递了 1 家公司：携程，AI Agent 岗位。"),
            _answer_turn('{"complete":true,"missing":[]}'),
        ]
    )
    tool = _tool("list_applications", list_handler)

    response = asyncio.run(
        _runtime(model, [tool], max_verifier_passes=1).start(
            "我现在投递了几家公司的什么岗位？"
        )
    )

    assert response.status.value == "completed"
    assert response.reply == "你目前投递了 1 家公司：携程，AI Agent 岗位。"
    assert response.usage.model_calls == 3
    assert response.usage.verifier_calls == 1
    assert [item.name for item in response.tool_executions] == ["list_applications"]
