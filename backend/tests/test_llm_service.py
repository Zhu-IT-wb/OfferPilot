import asyncio

from app.services.llm_service import LLMService


def test_generate_text_passes_json_response_format_to_provider() -> None:
    service = LLMService(api_key="test-key")
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
