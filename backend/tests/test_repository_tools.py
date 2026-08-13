import asyncio
import json

from app.models.tool_calling import ModelToolCall
from app.project_analysis.repository_tools import (
    build_repository_tools,
)
from app.project_analysis.repository_command_executor import (
    RepositoryCommandRejectedError,
    RepositoryCommandResult,
)
from app.project_analysis.repository_workspace import (
    RepositoryAccessError,
)
from app.tools.agent_tool_registry import AgentToolRegistry


class FakeWorkspace:
    """记录工具调用参数并返回确定结果的 Workspace 测试替身。"""

    all_paths = (
        "README.md",
        "src/app.py",
    )

    def __init__(self) -> None:
        """初始化调用记录。"""

        self.calls = []

    def list_files(
        self,
        prefix="",
        limit=200,
    ):
        """记录文件列表参数并返回固定路径。"""

        self.calls.append(
            ("list_files", prefix, limit)
        )
        return list(self.all_paths)[:limit]

    async def read_file(
        self,
        path,
        start_line=1,
        start_column=1,
        end_line=None,
    ):
        """记录文件读取参数，并模拟成功或安全拒绝。"""

        self.calls.append(
            (
                "read_file",
                path,
                start_line,
                start_column,
                end_line,
            )
        )

        if path == "blocked.env":
            raise RepositoryAccessError(
                "Repository file is blocked."
            )

        return {
            "path": path,
            "content": "hello",
            "start_line": start_line,
            "end_line": end_line or 1,
            "total_lines": 1,
            "truncated": False,
        }

    async def search_code(
        self,
        query,
        path_prefix="",
        limit=50,
    ):
        """记录代码搜索参数并返回固定空结果。"""

        self.calls.append(
            (
                "search_code",
                query,
                path_prefix,
                limit,
            )
        )
        return {
            "query": query,
            "path_prefix": path_prefix,
            "matches": [],
            "truncated": False,
        }


class FakeCommandExecutor:
    """记录受控命令参数并返回确定结果的执行器测试替身。"""

    def __init__(self) -> None:
        """初始化调用记录。"""

        self.calls = []

    async def execute(
        self,
        argv,
        timeout_seconds=None,
        max_output_bytes=None,
    ):
        """记录参数，并模拟安全拒绝或成功。"""

        self.calls.append(
            (
                argv,
                timeout_seconds,
                max_output_bytes,
            )
        )
        if argv == ["git", "checkout"]:
            raise RepositoryCommandRejectedError(
                "Git subcommand is not allowed: checkout."
            )
        return RepositoryCommandResult(
            argv=tuple(argv),
            exit_code=0,
            stdout="README.md\n",
            stderr="",
            timed_out=False,
            truncated=False,
            duration_ms=3,
        )


def _tools(workspace=None, executor=None):
    """用默认测试替身创建完整仓库工具集合。"""

    return build_repository_tools(
        workspace or FakeWorkspace(),
        executor or FakeCommandExecutor(),
    )


def _payload(message):
    """把 Registry 返回的 Tool 消息内容解析成字典。"""

    return json.loads(message["content"])


def test_repository_tools_publish_expected_model_schemas() -> None:
    """验证四个仓库工具以预期名称和参数 Schema 暴露给模型。"""

    registry = AgentToolRegistry(
        _tools()
    )

    schemas = registry.model_schemas()

    assert [
        item["function"]["name"]
        for item in schemas
    ] == [
        "list_files",
        "read_file",
        "search_code",
        "run_repository_command",
    ]
    assert schemas[1]["function"]["parameters"][
        "required"
    ] == ["path"]
    assert schemas[2]["function"]["parameters"][
        "required"
    ] == ["query"]
    assert schemas[3]["function"]["parameters"][
        "required"
    ] == ["argv"]


def test_repository_tools_forward_defaults_and_arguments() -> None:
    """验证工具 handler 把默认值和模型参数正确转发给 Workspace。"""

    workspace = FakeWorkspace()
    executor = FakeCommandExecutor()
    registry = AgentToolRegistry(
        _tools(workspace, executor)
    )

    list_message = asyncio.run(
        registry.dispatch(
            ModelToolCall(
                id="call_list",
                name="list_files",
                arguments={},
            )
        )
    )
    assert workspace.calls[-1] == (
        "list_files",
        "",
        120,
    )
    assert _payload(list_message)["success"] is True

    read_message = asyncio.run(
        registry.dispatch(
            ModelToolCall(
                id="call_read",
                name="read_file",
                arguments={
                    "path": "README.md",
                    "start_line": 2,
                    "start_column": 3,
                    "end_line": 4,
                },
            )
        )
    )
    assert workspace.calls[-1] == (
        "read_file",
        "README.md",
        2,
        3,
        4,
    )
    assert _payload(read_message)["data"]["path"] == (
        "README.md"
    )

    search_message = asyncio.run(
        registry.dispatch(
            ModelToolCall(
                id="call_search",
                name="search_code",
                arguments={
                    "query": "FastAPI",
                },
            )
        )
    )
    assert workspace.calls[-1] == (
        "search_code",
        "FastAPI",
        "",
        20,
    )
    assert _payload(search_message)["success"] is True

    command_message = asyncio.run(
        registry.dispatch(
            ModelToolCall(
                id="call_command",
                name="run_repository_command",
                arguments={
                    "argv": [
                        "git",
                        "status",
                        "--short",
                    ],
                    "timeout_seconds": 8,
                    "max_output_bytes": 5000,
                },
            )
        )
    )
    assert executor.calls[-1] == (
        ["git", "status", "--short"],
        8,
        5000,
    )
    assert _payload(command_message)["data"][
        "stdout"
    ] == "README.md\n"


def test_repository_tools_reject_unknown_and_invalid_arguments() -> None:
    """验证未知字段和错误参数类型统一返回 invalid_arguments。"""

    registry = AgentToolRegistry(
        _tools()
    )

    calls = [
        ModelToolCall(
            id="invalid_list",
            name="list_files",
            arguments={"unknown": True},
        ),
        ModelToolCall(
            id="invalid_command",
            name="run_repository_command",
            arguments={"argv": "git status"},
        ),
        ModelToolCall(
            id="invalid_read",
            name="read_file",
            arguments={"path": 123},
        ),
        ModelToolCall(
            id="invalid_search",
            name="search_code",
            arguments={
                "query": "FastAPI",
                "unknown": True,
            },
        ),
    ]

    for call in calls:
        message = asyncio.run(
            registry.dispatch(call)
        )
        payload = _payload(message)

        assert payload["success"] is False
        assert payload["data"]["error"]["code"] == (
            "invalid_arguments"
        )


def test_repository_access_error_becomes_invalid_arguments() -> None:
    """验证 Workspace 的安全拒绝转换成模型可修正的参数错误。"""

    registry = AgentToolRegistry(
        _tools()
    )

    message = asyncio.run(
        registry.dispatch(
            ModelToolCall(
                id="blocked_read",
                name="read_file",
                arguments={
                    "path": "blocked.env",
                },
            )
        )
    )
    payload = _payload(message)

    assert payload["success"] is False
    assert payload["data"]["error"]["code"] == (
        "invalid_arguments"
    )


def test_repository_command_policy_error_becomes_invalid_arguments() -> None:
    """执行策略拒绝会返回给模型修正，而不是中止 Agent。"""

    registry = AgentToolRegistry(_tools())

    message = asyncio.run(
        registry.dispatch(
            ModelToolCall(
                id="blocked_command",
                name="run_repository_command",
                arguments={
                    "argv": ["git", "checkout"],
                },
            )
        )
    )
    payload = _payload(message)

    assert payload["success"] is False
    assert payload["data"]["error"]["code"] == (
        "invalid_arguments"
    )
