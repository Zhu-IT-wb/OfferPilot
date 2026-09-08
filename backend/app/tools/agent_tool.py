import json
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Protocol,
    Optional,
    Type,
)

from pydantic import BaseModel


class ToolEffect(str, Enum):
    READ = "read"
    LOCAL_WRITE = "local_write"
    EXTERNAL_WRITE = "external_write"
    DESTRUCTIVE = "destructive"


class ToolApproval(str, Enum):
    NEVER = "never"
    IF_IMPLICIT = "if_implicit"
    ALWAYS = "always"


class ToolOutcomeStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    NEEDS_INPUT = "needs_input"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ToolInteraction:
    kind: str
    prompt: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    allowed_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "prompt": self.prompt,
            "input_schema": dict(self.input_schema),
            "allowed_actions": list(self.allowed_actions),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ToolPresentation:
    """User-facing metadata kept separate from the model-visible tool contract."""

    display_name: str = "待处理操作"
    detail_fields: tuple[str, ...] = ()
    impact: str = ""


@dataclass(frozen=True)
class AgentToolDefinition:
    name: str
    description: str
    parameters: Dict[str,Any]
    cacheable: bool = False
    input_model: Optional[Type[BaseModel]] = None
    effect: ToolEffect = ToolEffect.READ
    approval: ToolApproval = ToolApproval.NEVER
    parallel_safe: bool = True
    idempotent: bool = True
    timeout_seconds: float = 30.0
    control: bool = False
    presentation: ToolPresentation = field(default_factory=ToolPresentation)

    def to_model_schema(self) -> Dict[str,Any]:
        parameters = self.parameters
        if self.input_model is not None:
            parameters = self.input_model.model_json_schema()
        return {
            "type": "function",
            "function":{
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }
@dataclass(frozen=True)
class AgentToolResult:
    data: Dict[str,Any]
    is_error: bool = False
    terminal_content: Optional[str] = None
    verified_partial: bool = False
    status: Optional[ToolOutcomeStatus] = None
    message: str = ""
    error_code: Optional[str] = None
    retryable: bool = False
    interaction: Optional[ToolInteraction] = None
    artifact_refs: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """只允许错误工具显式标记已核验的部分 terminal 结果。"""

        if self.status is None:
            object.__setattr__(
                self,
                "status",
                ToolOutcomeStatus.ERROR if self.is_error else ToolOutcomeStatus.SUCCESS,
            )

        if self.verified_partial and (
            not self.is_error
            or self.terminal_content is None
        ):
            raise ValueError(
                "verified_partial requires an error result with "
                "terminal_content."
            )
        if self.status == ToolOutcomeStatus.SUCCESS and self.is_error:
            raise ValueError("A successful outcome cannot be marked as an error.")
        if self.status == ToolOutcomeStatus.NEEDS_INPUT and self.interaction is None:
            raise ValueError("needs_input outcomes require an interaction payload.")

    @property
    def effective_status(self) -> ToolOutcomeStatus:
        if self.status is not None:
            return self.status
        return ToolOutcomeStatus.ERROR if self.is_error else ToolOutcomeStatus.SUCCESS

    def to_outcome_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "status": self.effective_status.value,
            "success": self.effective_status == ToolOutcomeStatus.SUCCESS,
            "message": self.message,
            "data": self.data,
            "retryable": self.retryable,
            "artifact_refs": list(self.artifact_refs),
        }
        if self.error_code:
            value["error_code"] = self.error_code
        if self.interaction is not None:
            value["interaction"] = self.interaction.to_dict()
        return value

    def to_model_message(
        self,
        tool_call_id: str,
    ) -> Dict[str,Any]:
        payload = self.to_outcome_dict()
        content = json.dumps(
            payload,
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
