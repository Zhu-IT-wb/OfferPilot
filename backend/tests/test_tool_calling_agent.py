import asyncio
import json

import pytest

from app.agents.tool_calling_agent import (
    AgentLimitExceededError,
    AgentProtocolError,
    ToolCallingAgent,
    ToolCallingAgentLimits,
)
from app.models.tool_calling import (
    ModelOptions,
    ModelToolCall,
    ModelTurn,
    TokenUsage,
)
from app.tools.agent_tool import (
    AgentToolDefinition,
    AgentToolResult,
    FunctionAgentTool,
)
from app.tools.agent_tool_registry import (
    AgentToolRegistry,
)


class ScriptedModel:
    def __init__(self, turns):
        self.turns = turns
        self.calls = []

    async def complete(
        self,
        messages,
        tools,
        options,
    ):
        self.calls.append(
            {
                "messages": [
                    dict(message)
                    for message in messages
                ],
                "tools": tools,
                "options": options,
            }
        )

        turn_index = len(self.calls) - 1
        return self.turns[turn_index]


def test_agent_executes_tool_then_returns_final_answer() -> None:
    first_turn = ModelTurn(
        assistant_message={
            "role": "assistant",
            "content": None,
            "reasoning_content": (
                "I should inspect README.md."
            ),
            "tool_calls": [
                {
                    "id": "call_001",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": (
                            '{"path": "README.md"}'
                        ),
                    },
                }
            ],
        },
        tool_calls=[
            ModelToolCall(
                id="call_001",
                name="read_file",
                arguments={
                    "path": "README.md",
                },
            )
        ],
        content="",
        reasoning_content=(
            "I should inspect README.md."
        ),
        finish_reason="tool_calls",
        usage=TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
        provider="fake",
        model="fake-model",
    )

    second_turn = ModelTurn(
        assistant_message={
            "role": "assistant",
            "content": "这是一个秋招助手项目。",
            "reasoning_content": (
                "The README provides enough evidence."
            ),
        },
        tool_calls=[],
        content="这是一个秋招助手项目。",
        reasoning_content=(
            "The README provides enough evidence."
        ),
        finish_reason="stop",
        usage=TokenUsage(
            prompt_tokens=20,
            completion_tokens=8,
            total_tokens=28,
        ),
        provider="fake",
        model="fake-model",
    )

    model = ScriptedModel(
        turns=[
            first_turn,
            second_turn,
        ]
    )

    async def read_file_handler(arguments):
        assert arguments == {
            "path": "README.md",
        }

        return AgentToolResult(
            data={
                "path": "README.md",
                "content": "OfferPilot README",
            }
        )

    read_file_tool = FunctionAgentTool(
        definition=AgentToolDefinition(
            name="read_file",
            description="Read a repository file.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        handler=read_file_handler,
    )

    registry = AgentToolRegistry(
        tools=[
            read_file_tool,
        ]
    )

    agent = ToolCallingAgent(
        model=model,
        tool_registry=registry,
    )
    progress = []

    result = asyncio.run(
        agent.run(
            messages=[
                {
                    "role": "user",
                    "content": "分析这个项目。",
                }
            ],
            options=ModelOptions(),
            progress_callback=progress.append,
        )
    )

    assert result.content == "这是一个秋招助手项目。"
    assert result.tool_call_count == 1
    assert len(result.turns) == 2

    assert result.usage == TokenUsage(
        prompt_tokens=30,
        completion_tokens=13,
        total_tokens=43,
    )

    assert len(model.calls) == 2
    assert [
        item.tool_call_count
        for item in progress
    ] == [0, 1, 1, 1]
    assert progress[0].last_event["event"] == "model_turn"
    assert progress[2].last_event["tools"][0]["name"] == "read_file"

    second_request_messages = (
        model.calls[1]["messages"]
    )

    assistant_message = (
        second_request_messages[-2]
    )
    tool_message = second_request_messages[-1]

    assert assistant_message["reasoning_content"] == (
        "I should inspect README.md."
    )
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_001"

    tool_content = json.loads(
        tool_message["content"]
    )

    assert tool_content == {
        "success": True,
        "data": {
            "path": "README.md",
            "content": "OfferPilot README",
        },
    }


def test_agent_reports_usage_before_cumulative_token_failure() -> None:
    """验证累计预算失败前会报告可持久化的最后运行指标。"""

    model = ScriptedModel(
        turns=[
            ModelTurn(
                assistant_message={
                    "role": "assistant",
                    "content": "最终答案",
                },
                tool_calls=[],
                content="最终答案",
                reasoning_content=None,
                finish_reason="stop",
                usage=TokenUsage(
                    prompt_tokens=90,
                    completion_tokens=30,
                    total_tokens=120,
                ),
                provider="fake",
                model="fake-model",
            )
        ]
    )
    progress = []
    agent = ToolCallingAgent(
        model=model,
        tool_registry=AgentToolRegistry(),
        limits=ToolCallingAgentLimits(
            max_total_tokens=100,
        ),
    )

    with pytest.raises(
        AgentLimitExceededError,
        match=(
            r"120/100 cumulative tokens.*"
            r"1 model turns.*0 executed tool calls"
        ),
    ):
        asyncio.run(
            agent.run(
                messages=[
                    {
                        "role": "user",
                        "content": "分析项目。",
                    }
                ],
                progress_callback=progress.append,
            )
        )

    assert len(progress) == 1
    assert progress[0].model_turn_count == 1
    assert progress[0].tool_call_count == 0
    assert progress[0].usage.total_tokens == 120
    assert progress[0].finish_reason == "stop"


def test_agent_rejects_oversized_request_before_calling_model() -> None:
    """验证当前历史超过字符预算时不会继续请求模型供应商。"""

    model = ScriptedModel(turns=[])
    agent = ToolCallingAgent(
        model=model,
        tool_registry=AgentToolRegistry(),
        limits=ToolCallingAgentLimits(
            max_request_chars=10,
        ),
    )

    with pytest.raises(
        AgentLimitExceededError,
        match=r"current request size budget",
    ):
        asyncio.run(
            agent.run(
                messages=[
                    {
                        "role": "user",
                        "content": "这段输入明显超过十个字符。",
                    }
                ]
            )
        )

    assert model.calls == []


def _tool_turn(call_id: str, name: str, arguments: dict) -> ModelTurn:
    """创建包含一次函数调用的确定性模型轮次。"""

    return ModelTurn(
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        },
        tool_calls=[
            ModelToolCall(
                id=call_id,
                name=name,
                arguments=arguments,
            )
        ],
        content="",
        reasoning_content=None,
        finish_reason="tool_calls",
        usage=TokenUsage(10, 2, 12),
        provider="fake",
        model="fake-model",
    )


def test_terminal_tool_completes_without_another_model_call() -> None:
    """模型主动提交有效结果后，运行时立即成功结束。"""

    model = ScriptedModel(
        [_tool_turn("submit-1", "submit_analysis", {"findings": ["F1"]})]
    )

    async def submit(arguments):
        return AgentToolResult(
            data={"accepted": True},
            terminal_content=json.dumps(arguments),
        )

    registry = AgentToolRegistry(
        [
            FunctionAgentTool(
                definition=AgentToolDefinition(
                    name="submit_analysis",
                    description="Submit the completed analysis.",
                    parameters={"type": "object"},
                ),
                handler=submit,
            )
        ]
    )
    result = asyncio.run(
        ToolCallingAgent(
            model=model,
            tool_registry=registry,
            terminal_tool_name="submit_analysis",
        ).run([{"role": "user", "content": "分析"}])
    )

    assert len(model.calls) == 1
    assert json.loads(result.content) == {"findings": ["F1"]}
    assert result.completion_reason == "terminal_tool"


def test_terminal_submission_is_accepted_when_turn_crosses_token_budget() -> None:
    """最后一轮的有效提交优先于累计 token 阈值生效。"""

    model = ScriptedModel(
        [_tool_turn("submit-over", "submit_analysis", {"findings": ["F1"]})]
    )
    model.turns[0] = ModelTurn(
        **{
            **model.turns[0].__dict__,
            "usage": TokenUsage(90, 20, 110),
        }
    )

    async def submit(arguments):
        return AgentToolResult(
            data={"accepted": True},
            terminal_content=json.dumps(arguments),
        )

    registry = AgentToolRegistry([
        FunctionAgentTool(
            AgentToolDefinition(
                "submit_analysis",
                "Submit.",
                {"type": "object"},
            ),
            submit,
        )
    ])
    result = asyncio.run(
        ToolCallingAgent(
            model=model,
            tool_registry=registry,
            limits=ToolCallingAgentLimits(max_total_tokens=100),
            terminal_tool_name="submit_analysis",
        ).run([{"role": "user", "content": "分析"}])
    )

    assert len(model.calls) == 1
    assert result.completion_reason == "terminal_tool"
    assert result.usage.total_tokens == 110


def test_duplicate_read_only_tool_call_reuses_cached_result() -> None:
    """同一提交内完全相同的只读调用只执行一次。"""

    model = ScriptedModel(
        [
            _tool_turn("read-1", "read_file", {"path": "README.md"}),
            _tool_turn("read-2", "read_file", {"path": "README.md"}),
            ModelTurn(
                assistant_message={"role": "assistant", "content": "完成"},
                tool_calls=[],
                content="完成",
                reasoning_content=None,
                finish_reason="stop",
                usage=TokenUsage(10, 2, 12),
                provider="fake",
                model="fake-model",
            ),
        ]
    )
    executions = []

    async def read_file(arguments):
        executions.append(arguments)
        return AgentToolResult(data={"content": "README"})

    registry = AgentToolRegistry(
        [
            FunctionAgentTool(
                definition=AgentToolDefinition(
                    name="read_file",
                    description="Read a file.",
                    parameters={"type": "object"},
                    cacheable=True,
                ),
                handler=read_file,
            )
        ]
    )
    result = asyncio.run(
        ToolCallingAgent(model=model, tool_registry=registry).run(
            [{"role": "user", "content": "分析"}]
        )
    )

    assert executions == [{"path": "README.md"}]
    cached_event = next(
        event
        for event in result.trace
        if event.get("event") == "tool_call" and event.get("cache_hit")
    )
    assert cached_event["made_progress"] is False


def test_required_terminal_tool_reprompts_after_plain_stop() -> None:
    """专用 Agent 不能用普通文本绕过结构化提交协议。"""

    plain_stop = ModelTurn(
        assistant_message={"role": "assistant", "content": "我分析完了"},
        tool_calls=[],
        content="我分析完了",
        reasoning_content=None,
        finish_reason="stop",
        usage=TokenUsage(10, 2, 12),
        provider="fake",
        model="fake-model",
    )
    submit_turn = _tool_turn(
        "submit-2",
        "submit_analysis",
        {"findings": ["F1"]},
    )
    model = ScriptedModel([plain_stop, submit_turn])

    async def submit(arguments):
        return AgentToolResult(
            data={"accepted": True},
            terminal_content=json.dumps(arguments),
        )

    registry = AgentToolRegistry(
        [
            FunctionAgentTool(
                AgentToolDefinition(
                    "submit_analysis",
                    "Submit.",
                    {"type": "object"},
                ),
                submit,
            )
        ]
    )
    result = asyncio.run(
        ToolCallingAgent(
            model=model,
            tool_registry=registry,
            terminal_tool_name="submit_analysis",
        ).run([{"role": "user", "content": "分析"}])
    )

    assert len(model.calls) == 2
    assert "submit_analysis" in model.calls[1]["messages"][-1]["content"]
    assert result.completion_reason == "terminal_tool"


def test_old_tool_results_are_compacted_before_next_model_call() -> None:
    """超过上下文软阈值后保留最近结果并压缩较老的工具正文。"""

    model = ScriptedModel(
        [
            _tool_turn("read-1", "read_file", {"path": "one.py"}),
            _tool_turn("read-2", "read_file", {"path": "two.py"}),
            ModelTurn(
                assistant_message={"role": "assistant", "content": "完成"},
                tool_calls=[],
                content="完成",
                reasoning_content=None,
                finish_reason="stop",
                usage=TokenUsage(1, 1, 2),
                provider="fake",
                model="fake-model",
            ),
        ]
    )

    async def read_file(arguments):
        return AgentToolResult(
            data={
                "path": arguments["path"],
                "start_line": 1,
                "end_line": 200,
                "has_more": False,
                "content": "x" * 2000,
            }
        )

    registry = AgentToolRegistry([
        FunctionAgentTool(
            AgentToolDefinition(
                "read_file",
                "Read.",
                {"type": "object"},
            ),
            read_file,
        )
    ])
    result = asyncio.run(
        ToolCallingAgent(
            model=model,
            tool_registry=registry,
            limits=ToolCallingAgentLimits(
                context_compact_chars=1000,
                max_request_chars=10_000,
                recent_tool_results=1,
            ),
        ).run([{"role": "user", "content": "分析"}])
    )

    third_request_tools = [
        message
        for message in model.calls[2]["messages"]
        if message.get("role") == "tool"
    ]
    assert json.loads(third_request_tools[0]["content"])["data"]["compacted"] is True
    assert "x" * 100 not in third_request_tools[0]["content"]
    assert "x" * 100 in third_request_tools[1]["content"]
    assert any(event["event"] == "context_compacted" for event in result.trace)


def test_three_duplicate_calls_inject_no_progress_nudge() -> None:
    """连续三次缓存命中后，模型收到重新评估完成状态的提醒。"""

    turns = [
        _tool_turn(f"read-{index}", "read_file", {"path": "README.md"})
        for index in range(4)
    ]
    turns.append(ModelTurn(
        assistant_message={"role": "assistant", "content": "完成"},
        tool_calls=[],
        content="完成",
        reasoning_content=None,
        finish_reason="stop",
        usage=TokenUsage(1, 1, 2),
        provider="fake",
        model="fake-model",
    ))

    async def read_file(arguments):
        return AgentToolResult(data={"content": "README"})

    registry = AgentToolRegistry([
        FunctionAgentTool(
            AgentToolDefinition(
                "read_file",
                "Read.",
                {"type": "object"},
                cacheable=True,
            ),
            read_file,
        )
    ])
    model = ScriptedModel(turns)
    result = asyncio.run(
        ToolCallingAgent(model=model, tool_registry=registry).run(
            [{"role": "user", "content": "分析"}]
        )
    )

    assert "最近三次操作没有产生新证据" in model.calls[4]["messages"][-1]["content"]
    assert any(event["event"] == "no_progress_nudge" for event in result.trace)


def test_non_cacheable_tool_executes_every_identical_call() -> None:
    """未显式标记为只读幂等的通用工具不得复用结果。"""

    model = ScriptedModel([
        _tool_turn("write-1", "side_effect", {"value": 1}),
        _tool_turn("write-2", "side_effect", {"value": 1}),
        ModelTurn(
            assistant_message={"role": "assistant", "content": "完成"},
            tool_calls=[],
            content="完成",
            reasoning_content=None,
            finish_reason="stop",
            usage=TokenUsage(1, 1, 2),
            provider="fake",
            model="fake-model",
        ),
    ])
    executions = []

    async def side_effect(arguments):
        executions.append(arguments)
        return AgentToolResult(data={"execution": len(executions)})

    registry = AgentToolRegistry([
        FunctionAgentTool(
            AgentToolDefinition(
                "side_effect",
                "A tool with unspecified side effects.",
                {"type": "object"},
            ),
            side_effect,
        )
    ])

    asyncio.run(
        ToolCallingAgent(model=model, tool_registry=registry).run(
            [{"role": "user", "content": "执行"}]
        )
    )

    assert executions == [{"value": 1}, {"value": 1}]


def test_two_invalid_terminal_submissions_fail_without_third_attempt() -> None:
    """terminal 修正机会最多两次，不能隐式追加第三次提交。"""

    model = ScriptedModel([
        _tool_turn("bad-1", "submit_analysis", {"bad": 1}),
        _tool_turn("bad-2", "submit_analysis", {"bad": 2}),
    ])

    async def reject(arguments):
        return AgentToolResult(
            data={"error": {"message": "invalid submission"}},
            is_error=True,
        )

    registry = AgentToolRegistry([
        FunctionAgentTool(
            AgentToolDefinition(
                "submit_analysis",
                "Submit.",
                {"type": "object"},
            ),
            reject,
        )
    ])

    with pytest.raises(AgentProtocolError, match="after 2 attempts"):
        asyncio.run(
            ToolCallingAgent(
                model=model,
                tool_registry=registry,
                terminal_tool_name="submit_analysis",
            ).run([{"role": "user", "content": "分析"}])
        )

    assert len(model.calls) == 2


def test_terminal_failure_reports_each_validation_error() -> None:
    """两次 terminal 校验都失败时，最终异常必须保留具体修正原因。"""

    model = ScriptedModel([
        _tool_turn("bad-detail-1", "submit_analysis", {"attempt": 1}),
        _tool_turn("bad-detail-2", "submit_analysis", {"attempt": 2}),
    ])
    errors = iter([
        "evidence_by_field contains unknown field: project_overview",
        "finding F7 quote does not match the declared source range",
    ])

    async def reject(arguments):
        return AgentToolResult(
            data={"error": {"message": next(errors)}},
            is_error=True,
        )

    registry = AgentToolRegistry([
        FunctionAgentTool(
            AgentToolDefinition(
                "submit_analysis",
                "Submit.",
                {"type": "object"},
            ),
            reject,
        )
    ])
    progress = []

    with pytest.raises(AgentProtocolError) as raised:
        asyncio.run(
            ToolCallingAgent(
                model=model,
                tool_registry=registry,
                terminal_tool_name="submit_analysis",
            ).run(
                [{"role": "user", "content": "分析"}],
                progress_callback=progress.append,
            )
        )

    message = str(raised.value)
    assert "after 2 attempts" in message
    assert "1. evidence_by_field contains unknown field" in message
    assert "2. finding F7 quote does not match" in message
    last_model_event = next(
        snapshot.last_event
        for snapshot in reversed(progress)
        if snapshot.last_event.get("event") == "model_turn"
    )
    assert last_model_event["tools"][0]["error"].startswith(
        "finding F7 quote does not match"
    )


def test_second_invalid_submission_over_budget_still_has_no_third_attempt() -> None:
    """第二次坏提交同时越过预算时也必须优先执行提交次数上限。"""

    first = _tool_turn("bad-budget-1", "submit_analysis", {"bad": 1})
    second_base = _tool_turn("bad-budget-2", "submit_analysis", {"bad": 2})
    second = ModelTurn(
        **{
            **second_base.__dict__,
            "usage": TokenUsage(90, 20, 110),
        }
    )
    model = ScriptedModel([first, second])

    async def reject(arguments):
        return AgentToolResult(
            data={"error": {"message": "invalid submission"}},
            is_error=True,
        )

    registry = AgentToolRegistry([
        FunctionAgentTool(
            AgentToolDefinition(
                "submit_analysis",
                "Submit.",
                {"type": "object"},
            ),
            reject,
        )
    ])

    with pytest.raises(AgentProtocolError, match="after 2 attempts"):
        asyncio.run(
            ToolCallingAgent(
                model=model,
                tool_registry=registry,
                limits=ToolCallingAgentLimits(max_total_tokens=100),
                terminal_tool_name="submit_analysis",
            ).run([{"role": "user", "content": "分析"}])
        )

    assert len(model.calls) == 2


def test_compacting_an_already_compacted_result_is_idempotent() -> None:
    """多次跨过压缩阈值时必须保留首次生成的证据摘要。"""

    history = [{
        "role": "tool",
        "tool_call_id": "read-1",
        "content": json.dumps({
            "success": True,
            "data": {
                "path": "src/app.py",
                "start_line": 10,
                "end_line": 20,
                "content": "x" * 2000,
            },
        }),
    }]

    first = ToolCallingAgent._compact_tool_results(history, keep_recent=0)
    second = ToolCallingAgent._compact_tool_results(first, keep_recent=0)

    assert second == first
    summary = json.loads(second[0]["content"])["data"]["summary"]
    assert summary["path"] == "src/app.py"
    assert summary["start_line"] == 10
    assert summary["end_line"] == 20


def test_hard_turn_limit_uses_one_forced_terminal_submission() -> None:
    """探索轮次耗尽后只暴露完成工具，并保留一次应急提交调用。"""

    model = ScriptedModel([
        _tool_turn("read-1", "read_file", {"path": "README.md"}),
        _tool_turn("submit-3", "submit_analysis", {"findings": ["F1"]}),
    ])

    async def read_file(arguments):
        return AgentToolResult(data={"content": "README"})

    async def submit(arguments):
        return AgentToolResult(
            data={"accepted": True},
            terminal_content=json.dumps(arguments),
        )

    registry = AgentToolRegistry([
        FunctionAgentTool(
            AgentToolDefinition("read_file", "Read.", {"type": "object"}),
            read_file,
        ),
        FunctionAgentTool(
            AgentToolDefinition("submit_analysis", "Submit.", {"type": "object"}),
            submit,
        ),
    ])
    result = asyncio.run(
        ToolCallingAgent(
            model=model,
            tool_registry=registry,
            limits=ToolCallingAgentLimits(max_model_turns=1),
            terminal_tool_name="submit_analysis",
        ).run([{"role": "user", "content": "分析"}])
    )

    assert len(model.calls) == 2
    assert [tool["function"]["name"] for tool in model.calls[1]["tools"]] == [
        "submit_analysis"
    ]
    assert model.calls[1]["options"].tool_choice == {
        "type": "function",
        "function": {"name": "submit_analysis"},
    }
    assert result.completion_reason == "emergency_finalize"
