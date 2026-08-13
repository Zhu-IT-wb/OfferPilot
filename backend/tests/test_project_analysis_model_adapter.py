import asyncio

import httpx
import pytest

from app.models.tool_calling import ModelOptions
from app.services.llm_service import LLMRequestError
from app.services.tool_calling_model import (
    DeepSeekToolCallingModel,
)


def test_complete_sends_tools_and_parses_tool_call() -> None:
    model = DeepSeekToolCallingModel(
        api_key="test-key",
        default_model="deepseek-chat",
    )
    captured_payload = {}

    async def fake_post(payload):
        captured_payload.update(payload)

        return {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": (
                            "I should inspect the README."
                        ),
                        "tool_calls": [
                            {
                                "id": "call_001",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": (
                                        '{"path": "README.md"}'
                                    ),
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
        }

    model._post_chat_completions = fake_post

    messages = [
        {
            "role": "user",
            "content": "分析这个项目。",
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a repository file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                        }
                    },
                    "required": ["path"],
                },
            },
        }
    ]

    turn = asyncio.run(
        model.complete(
            messages=messages,
            tools=tools,
            options=ModelOptions(
                max_tokens=2048,
                temperature=0.0,
                thinking={"type": "enabled"},
                response_format={
                    "type": "json_object",
                },
            ),
        )
    )

    assert captured_payload["model"] == "deepseek-chat"
    assert captured_payload["messages"] == messages
    assert captured_payload["tools"] == tools
    assert captured_payload["max_tokens"] == 2048
    assert captured_payload["thinking"] == {
        "type": "enabled"
    }
    assert captured_payload["response_format"] == {
        "type": "json_object"
    }
    assert captured_payload["stream"] is False

    assert turn.finish_reason == "tool_calls"
    assert turn.content == ""
    assert turn.reasoning_content == (
        "I should inspect the README."
    )
    assert turn.tool_calls[0].id == "call_001"
    assert turn.tool_calls[0].name == "read_file"
    assert turn.tool_calls[0].arguments == {
        "path": "README.md"
    }
    assert turn.usage.prompt_tokens == 100
    assert turn.usage.completion_tokens == 20
    assert turn.usage.total_tokens == 120


def test_complete_passes_forced_tool_choice_to_provider() -> None:
    """应急收尾指定的 terminal 工具会传入供应商请求。"""

    model = DeepSeekToolCallingModel(
        api_key="test-key",
        default_model="deepseek-chat",
    )
    captured_payload = {}

    async def fake_post(payload):
        captured_payload.update(payload)
        return {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "done",
                    },
                }
            ],
            "usage": {},
        }

    model._post_chat_completions = fake_post
    forced_choice = {
        "type": "function",
        "function": {"name": "submit_analysis"},
    }

    asyncio.run(
        model.complete(
            messages=[{"role": "user", "content": "提交结果。"}],
            tools=[],
            options=ModelOptions(tool_choice=forced_choice),
        )
    )

    assert captured_payload["tool_choice"] == forced_choice


def test_complete_reports_timeout_type_and_configured_limit() -> None:
    """验证模型超时时保留异常类型和已配置的等待时长。"""

    model = DeepSeekToolCallingModel(
        api_key="test-key",
        default_model="deepseek-chat",
        timeout_seconds=180,
    )

    async def fake_post(payload):
        """模拟供应商在等待响应正文时超时。"""

        raise httpx.ReadTimeout("")

    model._post_chat_completions = fake_post

    with pytest.raises(
        LLMRequestError,
        match=(
            r"timed out after 180 seconds "
            r"\(ReadTimeout\)"
        ),
    ):
        asyncio.run(
            model.complete(
                messages=[
                    {
                        "role": "user",
                        "content": "分析项目。",
                    }
                ],
                tools=[],
                options=ModelOptions(),
            )
        )


def test_complete_reports_http_error_type_when_detail_is_empty() -> None:
    """验证空字符串 HTTP 异常不会退化成只有冒号的错误。"""

    model = DeepSeekToolCallingModel(
        api_key="test-key",
        default_model="deepseek-chat",
    )

    async def fake_post(payload):
        """模拟没有附加错误文本的远端协议异常。"""

        raise httpx.RemoteProtocolError("")

    model._post_chat_completions = fake_post

    with pytest.raises(
        LLMRequestError,
        match=r"RemoteProtocolError.*no additional detail",
    ):
        asyncio.run(
            model.complete(
                messages=[
                    {
                        "role": "user",
                        "content": "分析项目。",
                    }
                ],
                tools=[],
                options=ModelOptions(),
            )
        )


def test_complete_reports_status_type_when_response_body_is_empty() -> None:
    """验证空响应体的 HTTP 状态错误仍包含类型、状态码和兜底说明。"""

    model = DeepSeekToolCallingModel(
        api_key="test-key",
        default_model="deepseek-chat",
    )

    async def fake_post(payload):
        """模拟供应商返回没有正文的网关错误。"""

        request = httpx.Request(
            "POST",
            "https://api.deepseek.com/chat/completions",
        )
        response = httpx.Response(
            502,
            content=b"",
            request=request,
        )
        raise httpx.HTTPStatusError(
            "",
            request=request,
            response=response,
        )

    model._post_chat_completions = fake_post

    with pytest.raises(
        LLMRequestError,
        match=(
            r"HTTPStatusError, HTTP 502.*"
            r"empty response body"
        ),
    ):
        asyncio.run(
            model.complete(
                messages=[
                    {
                        "role": "user",
                        "content": "分析项目。",
                    }
                ],
                tools=[],
                options=ModelOptions(),
            )
        )
