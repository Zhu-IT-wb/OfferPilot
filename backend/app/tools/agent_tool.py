import json
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Protocol,
    Optional,
)

@dataclass(frozen=True)
class AgentToolDefinition:
    name: str
    description: str
    parameters: Dict[str,Any]
    cacheable: bool = False

    def to_model_schema(self) -> Dict[str,Any]:
        return {
            "type": "function",
            "function":{
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            },
        }
@dataclass(frozen=True)
class AgentToolResult:
    data: Dict[str,Any]
    is_error: bool = False
    terminal_content: Optional[str] = None

    def to_model_message(
        self,
        tool_call_id: str,
    ) -> Dict[str,Any]:
        content = json.dumps(
            {
            "success": not self.is_error,
            "data": self.data,
            },
            ensure_ascii= False,
        )
        return {
            "role":"tool",
            "tool_call_id":tool_call_id,
            "content": content,
        }

class AgentTool(Protocol):
    @property
    def definition(self) -> AgentToolDefinition:
        ...
    async def execute(
        self,
        arguments: Dict[str,Any],
    ) -> AgentToolResult:
        ...

AgentToolHandler = Callable[
    [Dict[str,Any]],
    Awaitable[AgentToolResult],
]

class FunctionAgentTool:
    def __init__(
        self,
        definition: AgentToolDefinition,
        handler: AgentToolHandler,
    ) -> None:
        self._definition = definition
        self._handler = handler

    @property
    def definition(self) -> AgentToolDefinition:
        return self._definition
    async def execute(
        self,
        arguments: Dict[str,Any],
    ) -> AgentToolResult:
        return await self._handler(arguments)
