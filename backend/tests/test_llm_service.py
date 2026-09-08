import asyncio

from app.services.llm_service import LLMService


def test_generate_text_passes_json_response_format_to_provider() -> None:
    service = LLMService(api_key="test-key", provider="deepseek")
    captured = {}

    async def fake_post(payload):
        captured.update(payload)
        return {
            "choices": [
                {
                    "message": {"content": "{}"},
                    "finish_reason": "stop",
                }
            ]
        }

    service._post_chat_completions = fake_post

    asyncio.run(
        service.generate_text(
            prompt="Return JSON.",
            response_format={"type": "json_object"},
            thinking={"type": "disabled"},
        )
    )

    assert captured["response_format"] == {"type": "json_object"}
    assert captured["thinking"] == {"type": "disabled"}


def test_generate_text_uses_openai_request_parameters() -> None:
    service = LLMService(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        provider="openai",
        default_model="gpt-4.1-mini",
    )
    captured = {}

    async def fake_post(payload):
        captured.update(payload)
        return {
            "choices": [
                {
                    "message": {"content": "ok"},
                    "finish_reason": "stop",
                }
            ]
        }

    service._post_chat_completions = fake_post

    result = asyncio.run(
        service.generate_text(
            prompt="Return JSON.",
            max_tokens=321,
            response_format={"type": "json_object"},
            thinking={"type": "disabled"},
        )
    )

    assert result.provider == "openai"
    assert result.model == "gpt-4.1-mini"
    assert result.content == "ok"
    assert captured["max_completion_tokens"] == 321
    assert captured["response_format"] == {"type": "json_object"}
    assert "max_tokens" not in captured
    assert "thinking" not in captured
