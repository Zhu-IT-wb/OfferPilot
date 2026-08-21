import inspect
from typing import Awaitable, Callable, Dict, List, Optional, Union

from app.schemas.tool import ToolResult, ToolSpec


ToolHandler = Callable[[dict], Union[ToolResult, Awaitable[ToolResult]]]


# 表示当前模块抛出的业务异常。
class ToolNotFoundError(RuntimeError):
    """Raised when an agent action has no registered tool."""


# 保存工具名称到处理函数的映射，并统一执行工具。
class ToolRegistry:
    # 初始化当前组件所需的依赖和配置。
    def __init__(self) -> None:
        self._handlers: Dict[str, ToolHandler] = {}
        self._specs: Dict[str, ToolSpec] = {}

    # 把工具处理函数注册到工具表中。
    def register(
        self,
        tool_name: str,
        handler: ToolHandler,
        description: str = "",
        mutating: bool = False,
        required_slots: Optional[List[str]] = None,
        optional_slots: Optional[List[str]] = None,
        examples: Optional[List[str]] = None,
    ) -> None:
        self._handlers[tool_name] = handler
        self._specs[tool_name] = ToolSpec(
            name=tool_name,
            description=description,
            mutating=mutating,
            required_slots=required_slots or [],
            optional_slots=optional_slots or [],
            examples=examples or [],
        )

    # 判断是否存在 tool。
    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._handlers

    # 返回当前可用工具的结构化说明，供 LLM Planner 选择。
    def describe_tools(self) -> List[ToolSpec]:
        return list(self._specs.values())

    # 查询单个工具说明。
    def get_spec(self, tool_name: str) -> Optional[ToolSpec]:
        return self._specs.get(tool_name)

    # 按工具名称查找并执行对应工具。
    def run(self, tool_name: str, arguments: dict) -> ToolResult:
        handler = self._handlers.get(tool_name)
        if handler is None:
            raise ToolNotFoundError(f"Tool is not registered: {tool_name}")

        result = handler(arguments)
        if inspect.isawaitable(result):
            close = getattr(result, "close", None)
            if callable(close):
                close()
            raise RuntimeError(
                f"Tool {tool_name} is asynchronous; use run_async instead."
            )
        return result

    async def run_async(self, tool_name: str, arguments: dict) -> ToolResult:
        handler = self._handlers.get(tool_name)
        if handler is None:
            raise ToolNotFoundError(f"Tool is not registered: {tool_name}")
        result = handler(arguments)
        if inspect.isawaitable(result):
            return await result
        return result
