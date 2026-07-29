import asyncio
import json

import httpx
import pytest

from app.core.config import Settings
from app.services.speech_transcription import (
    BailianSpeechTranscriptionProvider,
    NoSpeechDetectedError,
    SpeechTranscriptionConfigurationError,
    SpeechTranscriptionError,
    create_speech_transcription_service,
)


def test_bailian_provider_sends_bearer_base64_format_and_safe_context() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["sse"] = request.headers["x-dashscope-sse"]
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "output": {"text": "JVM 使用 CAS 实现并发控制。"},
                "usage": {"duration": 4},
            },
        )

    provider = BailianSpeechTranscriptionProvider(
        api_key="sk-test",
        workspace_id="ws-test",
        model="fun-asr-test",
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(
        provider.transcribe(
            audio=b"test-audio",
            audio_format="webm",
            mime_type="audio/webm",
            context="模块：Java；技术术语：JVM、CAS",
        )
    )

    assert result.transcript == "JVM 使用 CAS 实现并发控制。"
    assert result.provider == "bailian"
    assert result.model == "fun-asr-test"
    assert result.duration_seconds == 4
    assert captured["url"].startswith("https://ws-test.cn-beijing.maas.aliyuncs.com/")
    assert captured["authorization"] == "Bearer sk-test"
    assert captured["sse"] == "disable"
    payload = captured["payload"]
    assert payload["model"] == "fun-asr-test"
    assert payload["parameters"] == {"format": "webm"}
    assert payload["input"]["messages"][0]["content"][0] == {
        "type": "input_text",
        "text": "模块：Java；技术术语：JVM、CAS",
    }
    audio_input = payload["input"]["messages"][-1]["content"][0]["input_audio"]
    assert audio_input["data"] == "data:audio/webm;base64,dGVzdC1hdWRpbw=="


def test_transcription_service_limits_context_to_300_characters() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"output": {"text": "识别结果"}})

    service = create_speech_transcription_service(
        Settings(
            asr_api_key="sk-test",
            asr_workspace_id="ws-test",
            asr_model="fun-asr-test",
        ),
        transport=httpx.MockTransport(handler),
    )

    asyncio.run(service.transcribe(b"audio", "aac", "audio/aac", "术" * 400))

    context = captured["payload"]["input"]["messages"][0]["content"][0]["text"]
    assert len(context) == 300


def test_bailian_provider_rejects_missing_configuration() -> None:
    provider = BailianSpeechTranscriptionProvider(
        api_key="",
        workspace_id="",
        model="fun-asr-test",
    )

    with pytest.raises(SpeechTranscriptionConfigurationError, match="ASR_API_KEY"):
        asyncio.run(provider.transcribe(b"audio", "aac", "audio/aac"))


def test_bailian_provider_rejects_empty_transcript() -> None:
    provider = BailianSpeechTranscriptionProvider(
        api_key="sk-test",
        workspace_id="ws-test",
        model="fun-asr-test",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"output": {"text": "  "}})
        ),
    )

    with pytest.raises(NoSpeechDetectedError, match="没有识别到"):
        asyncio.run(provider.transcribe(b"audio", "aac", "audio/aac"))


def test_bailian_provider_wraps_timeout_and_provider_errors() -> None:
    async def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    timeout_provider = BailianSpeechTranscriptionProvider(
        api_key="sk-test",
        workspace_id="ws-test",
        model="fun-asr-test",
        transport=httpx.MockTransport(timeout_handler),
    )
    with pytest.raises(SpeechTranscriptionError, match="超时"):
        asyncio.run(timeout_provider.transcribe(b"audio", "aac", "audio/aac"))

    error_provider = BailianSpeechTranscriptionProvider(
        api_key="sk-test",
        workspace_id="ws-test",
        model="fun-asr-test",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                429,
                json={"code": "Throttling", "message": "rate limited"},
            )
        ),
    )
    with pytest.raises(SpeechTranscriptionError, match="rate limited"):
        asyncio.run(error_provider.transcribe(b"audio", "aac", "audio/aac"))


def test_unknown_asr_provider_is_rejected() -> None:
    with pytest.raises(SpeechTranscriptionConfigurationError, match="暂不支持"):
        create_speech_transcription_service(Settings(asr_provider="unknown"))
