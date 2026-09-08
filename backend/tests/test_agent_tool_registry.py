import asyncio
import json

from app.models.tool_calling import ModelToolCall
from app.tools.agent_tool import (
    AgentToolDefinition,
    AgentToolResult,
    FunctionAgentTool,
)
from app.tools.agent_tool_registry import AgentToolRegistry


def test_registry_dispatches_tool_call_to_handler() -> None:
    captured_arguments = {}

    async def read_file_handler(arguments):
        captured_arguments.update(arguments)

        return AgentToolResult(
            data={
                "path": arguments["path"],
                "content": "README content",
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

    tool_call = ModelToolCall(
        id="call_001",
        name="read_file",
        arguments={
            "path": "README.md",
        },
    )

    message = asyncio.run(
        registry.dispatch(tool_call)
    )

    assert captured_arguments == {
        "path": "README.md",
    }

    assert message["role"] == "tool"
    assert message["tool_call_id"] == "call_001"

    content = json.loads(message["content"])

    assert content == {
        "status": "success",
        "success": True,
        "message": "",
        "data": {
            "path": "README.md",
            "content": "README content",
        },
        "retryable": False,
        "artifact_refs": [],
    }


def test_registry_returns_error_for_unknown_tool() -> None:
    registry = AgentToolRegistry()

    tool_call = ModelToolCall(
        id="call_missing",
        name="missing_tool",
        arguments={},
    )

    message = asyncio.run(
        registry.dispatch(tool_call)
    )

    content = json.loads(message["content"])

    assert message["role"] == "tool"
    assert message["tool_call_id"] == "call_missing"
    assert content["success"] is False
    assert content["status"] == "rejected"
    assert content["error_code"] == "tool_not_found"
    assert content["data"]["error"]["code"] == (
        "tool_not_found"
    )
