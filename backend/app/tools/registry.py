from typing import Callable, Dict

from app.schemas.tool import ToolResult


ToolHandler = Callable[[dict], ToolResult]


class ToolNotFoundError(RuntimeError):
    """Raised when an agent action has no registered tool."""


class ToolRegistry:
    def __init__(self) -> None:
        self._handlers: Dict[str, ToolHandler] = {}

    def register(self, tool_name: str, handler: ToolHandler) -> None:
        self._handlers[tool_name] = handler

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._handlers

    def run(self, tool_name: str, arguments: dict) -> ToolResult:
        handler = self._handlers.get(tool_name)
        if handler is None:
            raise ToolNotFoundError(f"Tool is not registered: {tool_name}")

        return handler(arguments)
