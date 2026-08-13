from typing import Any, Dict, List, Optional, Set

from app.project_analysis.repository_workspace import (
    GitHubRepositoryWorkspace,
    RepositoryAccessError,
)
from app.project_analysis.repository_command_executor import (
    RepositoryCommandExecutor,
    RepositoryCommandRejectedError,
)
from app.tools.agent_tool import (
    AgentToolDefinition,
    AgentToolResult,
    FunctionAgentTool,
)
from app.tools.agent_tool_registry import (
    AgentToolInputError,
)


def build_repository_tools(
    workspace: GitHubRepositoryWorkspace,
    command_executor: RepositoryCommandExecutor,
) -> List[FunctionAgentTool]:
    """创建绑定当前 Workspace 的只读代码分析工具。"""

    async def list_files_handler(
        arguments: Dict[str,Any],
    ) -> AgentToolResult:
        """校验模型参数，并返回仓库中的安全文件路径。"""
        _reject_unknown_arguments(
                arguments,
            {
                "prefix",
                "limit",
            },
        )
        prefix = _optional_string(
            arguments,
            "prefix",
            default="",
        )
        limit = _optional_integer(
            arguments,
            "limit",
            default=120,
        )

        try:
            paths = workspace.list_files(
                prefix=prefix,
                limit=limit,
            )
        except RepositoryAccessError as exc:
            raise AgentToolInputError(
                str(exc)
            ) from exc

        return AgentToolResult(
            data={
                "prefix": prefix,
                "paths": paths,
                "returned_count": len(paths),
                "total_safe_files": len(
                    workspace.all_paths
                ),
            }
        )
    async def read_file_handler(
        arguments: Dict[str, Any],
    ) -> AgentToolResult:
        """校验模型参数，并读取仓库文本文件的指定行范围。"""
        _reject_unknown_arguments(
            arguments,
            {
                "path",
                "start_line",
                "start_column",
                "end_line",
            },
        )
        path = _required_string(
            arguments,
            "path",
        )
        start_line = _optional_integer(
            arguments,
            "start_line",
            default=1,
        )
        start_column = _optional_integer(
            arguments,
            "start_column",
            default=1,
        )
        end_line = _nullable_integer(
            arguments,
            "end_line",
        )

        try:
            result = await workspace.read_file(
                path=path,
                start_line=start_line,
                start_column=start_column,
                end_line=end_line,
            )
        except RepositoryAccessError as exc:
            raise AgentToolInputError(
                str(exc)
            ) from exc

        return AgentToolResult(
            data=result
        )

    async def search_code_handler(
        arguments: Dict[str, Any],
    ) -> AgentToolResult:
        """校验模型参数，并在仓库中执行固定字符串代码搜索。"""
        _reject_unknown_arguments(
            arguments,
            {
                "query",
                "path_prefix",
                "limit",
            },
        )
        query = _required_string(
            arguments,
            "query",
        )
        path_prefix = _optional_string(
            arguments,
            "path_prefix",
            default="",
        )
        limit = _optional_integer(
            arguments,
            "limit",
            default=20,
        )

        try:
            result = await workspace.search_code(
                query=query,
                path_prefix=path_prefix,
                limit=limit,
            )
        except RepositoryAccessError as exc:
            raise AgentToolInputError(
                str(exc)
            ) from exc

        return AgentToolResult(
            data=result
        )

    async def run_repository_command_handler(
        arguments: Dict[str, Any],
    ) -> AgentToolResult:
        """校验模型参数，并执行严格受限的仓库只读命令。"""

        _reject_unknown_arguments(
            arguments,
            {
                "argv",
                "timeout_seconds",
                "max_output_bytes",
            },
        )
        argv = _required_string_list(
            arguments,
            "argv",
        )
        timeout_seconds = _optional_integer(
            arguments,
            "timeout_seconds",
            default=(
                RepositoryCommandExecutor
                .DEFAULT_TIMEOUT_SECONDS
            ),
        )
        max_output_bytes = _optional_integer(
            arguments,
            "max_output_bytes",
            default=(
                RepositoryCommandExecutor
                .DEFAULT_MAX_OUTPUT_BYTES
            ),
        )

        try:
            result = await command_executor.execute(
                argv,
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
            )
        except RepositoryCommandRejectedError as exc:
            raise AgentToolInputError(str(exc)) from exc

        return AgentToolResult(data=result.to_dict())

    return [
        FunctionAgentTool(
            definition=AgentToolDefinition(
                name="list_files",
                description=(
                    "List safe files in the repository. "
                    "Use prefix to inspect a specific "
                    "directory before reading files."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "prefix": {
                            "type": "string",
                            "description": (
                                "Optional repository-relative "
                                "directory prefix."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 300,
                            "default": 120,
                        },
                    },
                    "additionalProperties": False,
                },
                cacheable=True,
            ),
            handler=list_files_handler,
        ),
        FunctionAgentTool(
            definition=AgentToolDefinition(
                name="read_file",
                description=(
                    "Read a safe UTF-8 text file from the "
                    "repository. Continue only when has_more is true, "
                    "using both next_start_line and next_start_column. A "
                    "caller-selected partial range does not imply that the "
                    "rest of the file must be read."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Repository-relative file path."
                            ),
                        },
                        "start_line": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 1,
                        },
                        "start_column": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 1,
                            "description": (
                                "One-based character column used only to "
                                "resume a character-truncated line."
                            ),
                        },
                        "end_line": {
                            "type": "integer",
                            "minimum": 1,
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                cacheable=True,
            ),
            handler=read_file_handler,
        ),
        FunctionAgentTool(
            definition=AgentToolDefinition(
                name="search_code",
                description=(
                    "Search for an exact fixed string in safe "
                    "repository text files. Use the returned "
                    "path and line number with read_file to "
                    "inspect surrounding code."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Exact text to search for."
                            ),
                        },
                        "path_prefix": {
                            "type": "string",
                            "description": (
                                "Optional repository-relative "
                                "directory prefix."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 200,
                            "default": 20,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                cacheable=True,
            ),
            handler=search_code_handler,
        ),
        FunctionAgentTool(
            definition=AgentToolDefinition(
                name="run_repository_command",
                description=(
                    "Run one constrained read-only command in the "
                    "repository root without a shell. Allowed programs: "
                    "rg, git, head, tail, wc. Git is limited to read-only "
                    "subcommands. Network, writes, builds, tests, package "
                    "installation, interpreters, and project execution "
                    "are unavailable."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "argv": {
                            "type": "array",
                            "description": (
                                "Program and arguments as separate items; "
                                "do not pass a shell command string."
                            ),
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": (
                                    RepositoryCommandExecutor
                                    .MAX_ARGUMENT_CHARS
                                ),
                            },
                            "minItems": (
                                RepositoryCommandExecutor
                                .MIN_ARGV_ITEMS
                            ),
                            "maxItems": (
                                RepositoryCommandExecutor
                                .MAX_ARGV_ITEMS
                            ),
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "minimum": (
                                RepositoryCommandExecutor
                                .MIN_TIMEOUT_SECONDS
                            ),
                            "maximum": (
                                RepositoryCommandExecutor
                                .MAX_TIMEOUT_SECONDS
                            ),
                            "default": (
                                RepositoryCommandExecutor
                                .DEFAULT_TIMEOUT_SECONDS
                            ),
                        },
                        "max_output_bytes": {
                            "type": "integer",
                            "minimum": (
                                RepositoryCommandExecutor
                                .MIN_OUTPUT_BYTES
                            ),
                            "maximum": (
                                RepositoryCommandExecutor
                                .MAX_OUTPUT_BYTES
                            ),
                            "default": (
                                RepositoryCommandExecutor
                                .DEFAULT_MAX_OUTPUT_BYTES
                            ),
                        },
                    },
                    "required": ["argv"],
                    "additionalProperties": False,
                },
                cacheable=True,
            ),
            handler=run_repository_command_handler,
        ),
    ]


def _required_string_list(
    arguments: Dict[str, Any],
    name: str,
) -> List[str]:
    """读取必填字符串数组，并拒绝字符串冒充 argv 数组。"""

    value = arguments.get(name)
    if not isinstance(value, list):
        raise AgentToolInputError(
            f"{name} must be an array of strings."
        )
    if not value:
        raise AgentToolInputError(
            f"{name} cannot be empty."
        )
    if not all(isinstance(item, str) for item in value):
        raise AgentToolInputError(
            f"{name} must contain only strings."
        )
    return list(value)

def _required_string(
    arguments: Dict[str, Any],
    name: str,
) -> str:
    """读取必填字符串参数，并拒绝缺失、空值和错误类型。"""

    value = arguments.get(name)

    if not isinstance(value, str):
        raise AgentToolInputError(
            f"{name} must be a string."
        )

    normalized = value.strip()

    if not normalized:
        raise AgentToolInputError(
            f"{name} cannot be empty."
        )

    return normalized

def _nullable_integer(
    arguments: Dict[str, Any],
    name: str,
) -> Optional[int]:
    """读取允许为空的整数参数，并校验实际参数类型。"""

    value = arguments.get(name)

    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise AgentToolInputError(
            f"{name} must be an integer."
        )

    return value

def _optional_string(
    arguments: Dict[str, Any],
    name: str,
    default: str,
) -> str:
    """读取可选字符串参数，未提供时返回指定默认值。"""

    value = arguments.get(name)

    if value is None:
        return default

    if not isinstance(value, str):
        raise AgentToolInputError(
            f"{name} must be a string."
        )

    return value.strip()

def _optional_integer(
    arguments: Dict[str, Any],
    name: str,
    default: int,
) -> int:
    """读取可选整数参数，并避免把布尔值误当作整数。"""

    value = arguments.get(name)

    if value is None:
        return default

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise AgentToolInputError(
            f"{name} must be an integer."
        )

    return value

def _reject_unknown_arguments(
    arguments: Dict[str, Any],
    allowed_names: Set[str],
) -> None:
    """拒绝 Tool Schema 中没有声明的模型参数。"""

    unknown_names = sorted(
        set(arguments) - allowed_names
    )

    if unknown_names:
        raise AgentToolInputError(
            "Unknown tool arguments: "
            + ", ".join(unknown_names)
        )
