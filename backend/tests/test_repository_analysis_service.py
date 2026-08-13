import asyncio
import json

import pytest

from app.agents.tool_calling_agent import (
    ToolCallingAgentResult,
)
from app.models.tool_calling import (
    ModelToolCall,
    ModelTurn,
    TokenUsage,
)
from app.project_analysis.repository_analysis_result import (
    RepositoryAnalysisResult,
    RepositoryAnalysisValidationError,
)
from app.project_analysis.repository_analysis_agent import (
    PROJECT_ANALYSIS_SYSTEM_PROMPT,
    RepositoryAnalysisAgent,
    build_project_analysis_system_prompt,
)
from app.project_analysis.repository_analysis_service import (
    RepositoryAnalysisService,
    RepositoryAnalysisServiceError,
)


class FakeWorkspace:
    """记录异步上下文进入和退出状态的临时仓库替身。"""

    def __init__(
        self,
        repository_url,
        paths=("README.md", "app/main.py"),
    ) -> None:
        """初始化仓库元数据、安全路径和生命周期状态。"""

        self.repository_url = repository_url
        self.commit_sha = "abc123"
        self.default_branch = "main"
        self.all_paths = tuple(paths)
        self.entered = False
        self.exited = False
        self.exit_exception_type = None

    async def __aenter__(self):
        """模拟打开临时仓库并返回当前 Workspace。"""

        self.entered = True
        return self

    async def __aexit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        """记录退出状态，用于验证异常时仍会清理仓库。"""

        self.exited = True
        self.exit_exception_type = exc_type

    async def read_file(
        self,
        path,
        start_line=1,
        end_line=None,
    ):
        """返回可供 submit_analysis 后端校验的固定代码证据。"""

        return {
            "path": path,
            "content": "app = FastAPI()",
            "start_line": start_line,
            "end_line": end_line or start_line,
            "total_lines": 1,
            "truncated": False,
            "has_more": False,
            "next_start_line": None,
        }


def test_analysis_prompt_uses_compact_grounded_output_contract() -> None:
    """验证最终协议不重复输出后端不会采信的完整项目档案。"""

    assert '"project_profile"' not in (
        PROJECT_ANALYSIS_SYSTEM_PROMPT
    )
    assert "findings 最多返回 24 条" in (
        PROJECT_ANALYSIS_SYSTEM_PROMPT
    )
    assert "quote" in PROJECT_ANALYSIS_SYSTEM_PROMPT
    assert "run_repository_command" in (
        PROJECT_ANALYSIS_SYSTEM_PROMPT
    )
    assert "不得执行仓库代码、测试或构建" in (
        PROJECT_ANALYSIS_SYSTEM_PROMPT
    )

    custom_prompt = build_project_analysis_system_prompt(
        max_findings=7,
        max_quote_chars=123,
    )
    assert "findings 最多返回 7 条" in custom_prompt
    assert "最多 123 个字符" in custom_prompt


class CapturingModel:
    """记录 Agent 暴露给模型的 Tool Schema。"""

    def __init__(self) -> None:
        """初始化调用记录。"""

        self.calls = []

    async def complete(self, messages, tools, options):
        """记录请求并立即返回最终 JSON。"""

        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "options": options,
            }
        )
        payload = {
            "findings": [
                {
                    "id": "F1",
                    "claim": "后端使用 FastAPI",
                    "topic": "architecture",
                    "target_field": "architecture",
                    "path": "app/main.py",
                    "start_line": 1,
                    "end_line": 1,
                    "quote": "app = FastAPI()",
                    "confidence": 0.95,
                }
            ],
            "evidence_by_field": {"architecture": ["F1"]},
            "coverage": {
                name: {
                    "status": (
                        "covered" if name == "architecture" else "not_found"
                    ),
                    "evidence_ids": ["F1"] if name == "architecture" else [],
                }
                for name in (
                    "project_overview",
                    "tech_stack",
                    "architecture",
                    "business_flows",
                    "data_and_integrations",
                    "testing_and_reliability",
                    "deployment",
                )
            },
            "warnings": [
                f"[{name}] 未找到可验证证据。"
                for name in (
                    "project_overview",
                    "tech_stack",
                    "business_flows",
                    "data_and_integrations",
                    "testing_and_reliability",
                    "deployment",
                )
            ],
        }
        arguments = json.dumps(payload, ensure_ascii=False)
        return ModelTurn(
            assistant_message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "submit-1",
                        "type": "function",
                        "function": {
                            "name": "submit_analysis",
                            "arguments": arguments,
                        },
                    }
                ],
            },
            tool_calls=[
                ModelToolCall(
                    id="submit-1",
                    name="submit_analysis",
                    arguments=payload,
                )
            ],
            content="",
            reasoning_content=None,
            finish_reason="tool_calls",
            usage=TokenUsage(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
            provider="fake",
            model="fake-model",
        )


def test_analysis_agent_exposes_constrained_repository_command(
    tmp_path,
) -> None:
    """分析 Agent 会把受控命令与原有只读工具一并暴露给模型。"""

    workspace = FakeWorkspace(
        "https://github.com/example/project"
    )
    workspace.repository_dir = str(tmp_path)
    model = CapturingModel()
    agent = RepositoryAnalysisAgent(model=model)

    asyncio.run(agent.analyze(workspace))

    assert [
        schema["function"]["name"]
        for schema in model.calls[0]["tools"]
    ] == [
        "list_files",
        "read_file",
        "search_code",
        "run_repository_command",
        "submit_analysis",
    ]
    assert model.calls[0]["options"].response_format is None
    assert len(model.calls) == 1


def test_submit_analysis_schema_restricts_evidence_mapping_fields(
    tmp_path,
) -> None:
    """模型 Schema 必须在生成阶段阻止 coverage 字段混入证据映射。"""

    workspace = FakeWorkspace("https://github.com/example/project")
    workspace.repository_dir = str(tmp_path)
    model = CapturingModel()

    asyncio.run(RepositoryAnalysisAgent(model=model).analyze(workspace))

    submit_schema = next(
        tool["function"]["parameters"]
        for tool in model.calls[0]["tools"]
        if tool["function"]["name"] == "submit_analysis"
    )
    mapping_schema = submit_schema["properties"]["evidence_by_field"]

    assert set(mapping_schema["properties"]) == {
        "name",
        "background",
        "tech_stack",
        "architecture",
        "key_decisions",
        "technical_challenges",
        "resume_description",
        "supplemental_text",
    }
    assert mapping_schema["additionalProperties"] is False
    assert "project_overview" not in mapping_schema["properties"]


def test_submit_analysis_rejects_incomplete_coverage_and_allows_retry(
    tmp_path,
) -> None:
    """无效提交作为工具错误返回模型，修正后可正常完成。"""

    workspace = FakeWorkspace("https://github.com/example/project")
    workspace.repository_dir = str(tmp_path)
    model = CapturingModel()
    original_complete = model.complete

    async def complete(messages, tools, options):
        if not model.calls:
            bad_payload = {
                "findings": [],
                "evidence_by_field": {},
                "coverage": {},
                "warnings": [],
            }
            model.calls.append(
                {
                    "messages": [dict(message) for message in messages],
                    "tools": tools,
                    "options": options,
                }
            )
            return ModelTurn(
                assistant_message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "bad-submit",
                        "type": "function",
                        "function": {
                            "name": "submit_analysis",
                            "arguments": json.dumps(bad_payload),
                        },
                    }],
                },
                tool_calls=[ModelToolCall(
                    id="bad-submit",
                    name="submit_analysis",
                    arguments=bad_payload,
                )],
                content="",
                reasoning_content=None,
                finish_reason="tool_calls",
                usage=TokenUsage(1, 1, 2),
                provider="fake",
                model="fake-model",
            )
        return await original_complete(messages, tools, options)

    model.complete = complete
    result = asyncio.run(RepositoryAnalysisAgent(model=model).analyze(workspace))

    assert len(model.calls) == 2
    bad_result_message = next(
        message
        for message in model.calls[1]["messages"]
        if message.get("tool_call_id") == "bad-submit"
    )
    retry_tool_result = json.loads(bad_result_message["content"])
    assert retry_tool_result["success"] is False
    assert "coverage" in retry_tool_result["data"]["error"]["message"]
    assert result.completion_reason == "terminal_tool"


def test_submit_analysis_rejects_any_invalid_finding_and_allows_retry(
    tmp_path,
) -> None:
    """terminal 提交包含一条伪造证据时必须整体拒绝而非静默丢弃。"""

    workspace = FakeWorkspace("https://github.com/example/project")
    workspace.repository_dir = str(tmp_path)
    model = CapturingModel()
    original_complete = model.complete

    async def complete(messages, tools, options):
        if not model.calls:
            valid_turn = await original_complete(messages, tools, options)
            bad_payload = dict(valid_turn.tool_calls[0].arguments)
            bad_payload["findings"] = [
                *bad_payload["findings"],
                {
                    **bad_payload["findings"][0],
                    "id": "F_BAD",
                    "path": "missing.py",
                },
            ]
            bad_payload["evidence_by_field"] = {
                **bad_payload["evidence_by_field"],
                "tech_stack": ["F_BAD"],
            }
            model.calls.clear()
            model.calls.append({
                "messages": [dict(message) for message in messages],
                "tools": tools,
                "options": options,
            })
            return ModelTurn(
                assistant_message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "bad-finding-submit",
                        "type": "function",
                        "function": {
                            "name": "submit_analysis",
                            "arguments": json.dumps(bad_payload),
                        },
                    }],
                },
                tool_calls=[ModelToolCall(
                    id="bad-finding-submit",
                    name="submit_analysis",
                    arguments=bad_payload,
                )],
                content="",
                reasoning_content=None,
                finish_reason="tool_calls",
                usage=TokenUsage(1, 1, 2),
                provider="fake",
                model="fake-model",
            )
        return await original_complete(messages, tools, options)

    model.complete = complete
    result = asyncio.run(RepositoryAnalysisAgent(model=model).analyze(workspace))

    assert len(model.calls) == 2
    rejected = next(
        message
        for message in model.calls[1]["messages"]
        if message.get("tool_call_id") == "bad-finding-submit"
    )
    payload = json.loads(rejected["content"])
    assert payload["success"] is False
    assert "F_BAD" in payload["data"]["error"]["message"]
    assert result.completion_reason == "terminal_tool"


def test_submit_analysis_returns_relocated_evidence_range(tmp_path) -> None:
    """terminal 工具必须返回后端校正后的证据范围，而非原始坏行号。"""

    workspace = FakeWorkspace("https://github.com/example/project")
    workspace.repository_dir = str(tmp_path)

    async def locate_exact_quote(path, quote, near_line=1):
        assert path == "app/main.py"
        assert quote == "app = FastAPI()"
        return 7, 7

    async def read_file(path, start_line=1, end_line=None):
        return {
            "path": path,
            "content": (
                "different source line"
                if start_line == 3
                else "app = FastAPI()"
            ),
            "start_line": start_line,
            "end_line": end_line or start_line,
            "total_lines": 7,
            "truncated": False,
            "has_more": False,
            "next_start_line": None,
        }

    workspace.locate_exact_quote = locate_exact_quote
    workspace.read_file = read_file
    model = CapturingModel()
    original_complete = model.complete

    async def complete(messages, tools, options):
        turn = await original_complete(messages, tools, options)
        payload = dict(turn.tool_calls[0].arguments)
        payload["findings"] = [
            {
                **payload["findings"][0],
                "start_line": 3,
                "end_line": 3,
            }
        ]
        return ModelTurn(
            assistant_message={
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "relocate-submit",
                    "type": "function",
                    "function": {
                        "name": "submit_analysis",
                        "arguments": json.dumps(payload),
                    },
                }],
            },
            tool_calls=[ModelToolCall(
                id="relocate-submit",
                name="submit_analysis",
                arguments=payload,
            )],
            content="",
            reasoning_content=None,
            finish_reason="tool_calls",
            usage=TokenUsage(1, 1, 2),
            provider="fake",
            model="fake-model",
        )

    model.complete = complete
    result = asyncio.run(RepositoryAnalysisAgent(model=model).analyze(workspace))
    normalized = json.loads(result.content)

    assert normalized["findings"][0]["start_line"] == 7
    assert normalized["findings"][0]["end_line"] == 7


class FakeAgent:
    """返回固定 Agent 结果或抛出预设异常的分析 Agent 替身。"""

    def __init__(
        self,
        error=None,
    ) -> None:
        """初始化调用记录和可选失败行为。"""

        self.error = error
        self.calls = []

    async def analyze(
        self,
        workspace,
        progress_callback=None,
    ):
        """记录 Workspace，并返回带运行指标的固定结果。"""

        self.calls.append(workspace)

        if self.error is not None:
            raise self.error

        return ToolCallingAgentResult(
            content='{"findings": []}',
            messages=[],
            turns=[
                _model_turn(),
                _model_turn(),
            ],
            usage=TokenUsage(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
            ),
            tool_call_count=7,
        )


def _model_turn() -> ModelTurn:
    """创建满足 ToolCallingAgentResult 类型契约的最小模型轮次。"""

    return ModelTurn(
        assistant_message={
            "role": "assistant",
            "content": "complete",
        },
        tool_calls=[],
        content="complete",
        reasoning_content=None,
        finish_reason="stop",
        usage=TokenUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        ),
        provider="fake",
        model="fake-model",
    )


class FakeParser:
    """记录解析参数并返回固定可信分析结果的解析器替身。"""

    def __init__(
        self,
        error=None,
    ) -> None:
        """初始化调用记录和可选失败行为。"""

        self.error = error
        self.calls = []

    async def parse(
        self,
        raw_content,
        workspace,
    ):
        """记录原始内容和 Workspace，并返回固定结构化结果。"""

        self.calls.append(
            (raw_content, workspace)
        )

        if self.error is not None:
            raise self.error

        return RepositoryAnalysisResult(
            draft={
                "name": "offerpilot",
            },
            findings=[
                {
                    "id": "F1",
                }
            ],
            evidence_by_field={
                "architecture": ["F1"],
            },
            warnings=[],
        )


class FakeWorkspaceFactory:
    """创建并保留最后一个 FakeWorkspace 供测试断言。"""

    def __init__(
        self,
        paths=("README.md", "app/main.py"),
    ) -> None:
        """保存每次创建 Workspace 时使用的安全路径。"""

        self.paths = paths
        self.calls = []
        self.workspace = None

    def __call__(self, repository_url):
        """根据仓库地址创建一个新的 FakeWorkspace。"""

        self.calls.append(repository_url)
        self.workspace = FakeWorkspace(
            repository_url,
            paths=self.paths,
        )
        return self.workspace


def test_service_runs_agent_parser_and_cleans_workspace() -> None:
    """验证成功流程会返回元数据、运行指标并清理临时仓库。"""

    factory = FakeWorkspaceFactory()
    agent = FakeAgent()
    parser = FakeParser()
    progress = []
    service = RepositoryAnalysisService(
        agent=agent,
        parser=parser,
        workspace_factory=factory,
    )

    result = asyncio.run(
        service.analyze(
            "https://github.com/example/offerpilot",
            progress_callback=(
                lambda stage, value: progress.append(
                    (stage, value)
                )
            ),
        )
    )

    workspace = factory.workspace

    assert workspace is not None
    assert workspace.entered is True
    assert workspace.exited is True
    assert workspace.exit_exception_type is None
    assert agent.calls == [workspace]
    assert parser.calls == [
        ('{"findings": []}', workspace)
    ]
    assert result.repository_url == (
        "https://github.com/example/offerpilot"
    )
    assert result.commit_sha == "abc123"
    assert result.default_branch == "main"
    assert result.safe_file_count == 2
    assert result.analysis.draft == {
        "name": "offerpilot",
    }
    assert result.usage == TokenUsage(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
    )
    assert result.model_turn_count == 2
    assert result.tool_call_count == 7
    assert result.completion_reason == "model_stop"
    assert result.result_status == "complete"
    assert result.trace == ()
    assert progress == [
        ("cloning", 5),
        ("inventory", 20),
        ("analyzing", 30),
        ("validating", 90),
        ("validating", 100),
    ]


def test_service_rejects_empty_workspace_and_cleans_it() -> None:
    """验证没有安全文件时不会调用 Agent，并仍然退出 Workspace。"""

    factory = FakeWorkspaceFactory(paths=())
    agent = FakeAgent()
    service = RepositoryAnalysisService(
        agent=agent,
        parser=FakeParser(),
        workspace_factory=factory,
    )

    with pytest.raises(
        RepositoryAnalysisServiceError,
        match="no safe files",
    ):
        asyncio.run(
            service.analyze(
                "https://github.com/example/empty"
            )
        )

    assert agent.calls == []
    assert factory.workspace is not None
    assert factory.workspace.exited is True
    assert factory.workspace.exit_exception_type is (
        RepositoryAnalysisServiceError
    )


def test_service_cleans_workspace_when_agent_fails() -> None:
    """验证 Agent 抛出异常时异步上下文仍然执行清理。"""

    factory = FakeWorkspaceFactory()
    agent_error = RuntimeError(
        "model unavailable"
    )
    service = RepositoryAnalysisService(
        agent=FakeAgent(error=agent_error),
        parser=FakeParser(),
        workspace_factory=factory,
    )

    with pytest.raises(
        RuntimeError,
        match="model unavailable",
    ):
        asyncio.run(
            service.analyze(
                "https://github.com/example/offerpilot"
            )
        )

    assert factory.workspace is not None
    assert factory.workspace.exited is True
    assert factory.workspace.exit_exception_type is (
        RuntimeError
    )


def test_service_cleans_workspace_when_parser_fails() -> None:
    """验证最终 JSON 校验失败时临时仓库仍然会被清理。"""

    factory = FakeWorkspaceFactory()
    parser_error = (
        RepositoryAnalysisValidationError(
            "invalid analysis"
        )
    )
    service = RepositoryAnalysisService(
        agent=FakeAgent(),
        parser=FakeParser(
            error=parser_error
        ),
        workspace_factory=factory,
    )

    with pytest.raises(
        RepositoryAnalysisValidationError,
        match="invalid analysis",
    ):
        asyncio.run(
            service.analyze(
                "https://github.com/example/offerpilot"
            )
        )

    assert factory.workspace is not None
    assert factory.workspace.exited is True
    assert factory.workspace.exit_exception_type is (
        RepositoryAnalysisValidationError
    )


def test_service_cleans_workspace_when_progress_callback_fails() -> None:
    """验证 Workspace 打开后的进度回调异常同样触发清理。"""

    factory = FakeWorkspaceFactory()
    service = RepositoryAnalysisService(
        agent=FakeAgent(),
        parser=FakeParser(),
        workspace_factory=factory,
    )

    def failing_progress(stage, value):
        """在进入分析阶段时模拟任务取消或租约丢失。"""

        if stage == "analyzing":
            raise RuntimeError(
                "analysis cancelled"
            )

    with pytest.raises(
        RuntimeError,
        match="analysis cancelled",
    ):
        asyncio.run(
            service.analyze(
                "https://github.com/example/offerpilot",
                progress_callback=failing_progress,
            )
        )

    assert factory.workspace is not None
    assert factory.workspace.exited is True
    assert factory.workspace.exit_exception_type is (
        RuntimeError
    )
