import logging
import re
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Tuple

from pydantic import ValidationError

from app.models.tool_calling import ModelToolCall
from app.tools.agent_tool import (
    AgentTool,
    AgentToolResult,
    ToolEffect,
    ToolOutcomeStatus,
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

    def definitions(self) -> List[object]:
        return [tool.definition for tool in self._tools.values()]

    def get_definition(self, tool_name: str):
        tool = self._tools.get(tool_name)
        return tool.definition if tool is not None else None

    def validate_arguments(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Tuple[object, Dict[str, Any]]:
        tool = self._tools.get(tool_name)
        if tool is None:
            raise KeyError(tool_name)

        definition = tool.definition
        input_model = definition.input_model
        if input_model is not None:
            try:
                if hasattr(input_model, "model_validate"):
                    value = input_model.model_validate(arguments)
                    normalized = value.model_dump(mode="json", exclude_none=True)
                else:
                    value = input_model.parse_obj(arguments)
                    normalized = value.dict(exclude_none=True)
            except ValidationError as exc:
                raise AgentToolInputError(str(exc)) from exc
            return definition, normalized

        parameters = definition.parameters or {}
        required = parameters.get("required") or []
        missing = [name for name in required if arguments.get(name) is None]
        if missing:
            raise AgentToolInputError(
                "Missing required arguments: " + ", ".join(sorted(missing))
            )
        properties = parameters.get("properties")
        if parameters.get("additionalProperties") is False and isinstance(properties, dict):
            unknown = sorted(set(arguments) - set(properties))
            if unknown:
                raise AgentToolInputError(
                    "Unknown arguments: " + ", ".join(unknown)
                )
        self._validate_json_schema(arguments, parameters, "arguments")
        return definition, dict(arguments)

    @classmethod
    def _validate_json_schema(
        cls,
        value: Any,
        schema: Dict[str, Any],
        path: str,
    ) -> None:
        if not schema:
            return
        expected = schema.get("type")
        valid_type = {
            "object": isinstance(value, dict),
            "array": isinstance(value, list),
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
            "null": value is None,
        }.get(expected, True)
        if not valid_type:
            raise AgentToolInputError(f"{path} must be {expected}.")
        if "enum" in schema and value not in schema["enum"]:
            raise AgentToolInputError(
                f"{path} must be one of: {', '.join(map(str, schema['enum']))}."
            )

        if isinstance(value, dict):
            properties = schema.get("properties") or {}
            required = schema.get("required") or []
            missing = [name for name in required if value.get(name) is None]
            if missing:
                raise AgentToolInputError(
                    f"{path} is missing: {', '.join(sorted(missing))}."
                )
            if schema.get("additionalProperties") is False:
                unknown = sorted(set(value) - set(properties))
                if unknown:
                    raise AgentToolInputError(
                        f"{path} has unknown fields: {', '.join(unknown)}."
                    )
            for name, item in value.items():
                child_schema = properties.get(name)
                if isinstance(child_schema, dict):
                    cls._validate_json_schema(item, child_schema, f"{path}.{name}")
        elif isinstance(value, list):
            minimum = schema.get("minItems")
            maximum = schema.get("maxItems")
            if minimum is not None and len(value) < int(minimum):
                raise AgentToolInputError(f"{path} requires at least {minimum} items.")
            if maximum is not None and len(value) > int(maximum):
                raise AgentToolInputError(f"{path} allows at most {maximum} items.")
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for index, item in enumerate(value):
                    cls._validate_json_schema(item, item_schema, f"{path}[{index}]")
        elif isinstance(value, str):
            if schema.get("minLength") is not None and len(value) < int(schema["minLength"]):
                raise AgentToolInputError(f"{path} is too short.")
            if schema.get("maxLength") is not None and len(value) > int(schema["maxLength"]):
                raise AgentToolInputError(f"{path} is too long.")
            pattern = schema.get("pattern")
            if pattern and re.fullmatch(str(pattern), value) is None:
                raise AgentToolInputError(f"{path} has an invalid format.")
            try:
                if schema.get("format") == "date":
                    date.fromisoformat(value)
                elif schema.get("format") == "date-time":
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    if parsed.tzinfo is None:
                        raise ValueError("timezone offset is required")
            except ValueError as exc:
                raise AgentToolInputError(
                    f"{path} must be a valid {schema.get('format')}."
                ) from exc
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if schema.get("minimum") is not None and value < schema["minimum"]:
                raise AgentToolInputError(f"{path} must be >= {schema['minimum']}.")
            if schema.get("maximum") is not None and value > schema["maximum"]:
                raise AgentToolInputError(f"{path} must be <= {schema['maximum']}.")

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
                status=ToolOutcomeStatus.REJECTED,
                message=f"Unknown tool: {tool_call.name}",
                error_code="tool_not_found",
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
                status=ToolOutcomeStatus.REJECTED,
                message=str(exc),
                error_code="invalid_arguments",
            )
        except Exception:
            logger.exception("Agent tool excution failed: %s", tool_call.name)
            write_effect = tool.definition.effect != ToolEffect.READ
            return AgentToolResult(
                data={
                    "error": {
                        "code": "tool_execution_failed",
                        "message": "The tool failed while executing",
                    }
                },
                is_error=True,
                status=(
                    ToolOutcomeStatus.UNKNOWN
                    if write_effect
                    else ToolOutcomeStatus.ERROR
                ),
                message=(
                    "The write outcome is unknown; reconcile before retrying."
                    if write_effect
                    else "The tool failed while executing."
                ),
                error_code=(
                    "write_outcome_unknown"
                    if write_effect
                    else "tool_execution_failed"
                ),
                retryable=tool.definition.effect == ToolEffect.READ,
            )
    async def dispatch(
        self,
        tool_call: ModelToolCall,
    ) -> dict:
        result = await self.execute(tool_call)
        return result.to_model_message(tool_call.id)
