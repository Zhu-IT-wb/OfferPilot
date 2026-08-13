import logging
from typing import Dict, Iterable, List

from app.models.tool_calling import ModelToolCall
from app.tools.agent_tool import (
    AgentTool,
    AgentToolResult,
)

logger = logging.getLogger(__name__)


class AgentToolInputError(ValueError):
    """Tool arguments are missing or invalid."""


class DuplicateAgentToolError(ValueError):
    """Two tools were registered with the same name."""

class AgentToolRegistry:
    def __init__(
        self,
        tools: Iterable[AgentTool] = (),
    ) -> None:
        self._tools: Dict[str, AgentTool] = {}

        for tool in tools:
            self.register(tool)
    def register(
        self,
        tool: AgentTool,
    ) -> None:
        tool_name = tool.definition.name.strip()

        if not tool_name:
            raise ValueError(
                "Agent tool name cannot be empty"
            )
        if tool_name in self._tools:
            raise DuplicateAgentToolError(
                f"Agent tool is already registered:{tool_name}"
            )
        self._tools[tool_name] = tool

    def model_schemas(self) -> List[dict]:
        return [
            tool.definition.to_model_schema()
            for tool in self._tools.values()
        ]
    def model_schema(self, tool_name: str) -> dict:
        """返回单个已注册工具的模型 Schema。"""

        tool = self._tools.get(tool_name)
        if tool is None:
            raise KeyError(tool_name)
        return tool.definition.to_model_schema()

    def is_cacheable(self, tool_name: str) -> bool:
        """仅允许显式声明为只读幂等的工具参与结果缓存。"""

        tool = self._tools.get(tool_name)
        return bool(tool and tool.definition.cacheable)

    async def execute(
        self,
        tool_call: ModelToolCall,
    ) -> AgentToolResult:
        """执行工具并保留 terminal 等运行时元数据。"""

        tool = self._tools.get(tool_call.name)

        if tool is None:
            return AgentToolResult(
                data={
                    "error": {
                        "code": "tool_not_found",
                        "message": f"Unknown tool: {tool_call.name}",
                        "available_tools": sorted(self._tools.keys()),
                    }
                },
                is_error=True,
            )
        try:
            result = await tool.execute(tool_call.arguments)
            if not isinstance(result, AgentToolResult):
                raise TypeError("Agent tool must return AgentToolResult")
            return result
        except AgentToolInputError as exc:
            return AgentToolResult(
                data={
                    "error": {
                        "code": "invalid_arguments",
                        "message": str(exc),
                    }
                },
                is_error=True,
            )
        except Exception:
            logger.exception("Agent tool excution failed: %s", tool_call.name)
            return AgentToolResult(
                data={
                    "error": {
                        "code": "tool_execution_failed",
                        "message": "The tool failed while executing",
                    }
                },
                is_error=True,
            )
    async def dispatch(
        self,
        tool_call: ModelToolCall,
    ) -> dict:
        result = await self.execute(tool_call)
        return result.to_model_message(tool_call.id)
