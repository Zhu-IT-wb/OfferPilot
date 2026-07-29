import base64
from dataclasses import dataclass
from typing import Optional, Protocol

import httpx

from app.core.config import Settings


class SpeechTranscriptionError(RuntimeError):
    """A speech transcription request could not be completed."""


class SpeechTranscriptionConfigurationError(SpeechTranscriptionError):
    """The configured ASR provider is missing required settings."""


class NoSpeechDetectedError(SpeechTranscriptionError):
    """The provider returned no usable speech transcript."""


@dataclass(frozen=True)
class SpeechTranscriptionResult:
    transcript: str
    provider: str
    model: str
    duration_seconds: Optional[float] = None


class SpeechTranscriptionProvider(Protocol):
    async def transcribe(
        self,
        audio: bytes,
        audio_format: str,
        mime_type: str,
        context: str = "",
    ) -> SpeechTranscriptionResult:
        ...


class SpeechTranscriptionService:
    """Provider-neutral entry point for short answer audio transcription."""

    def __init__(self, provider: SpeechTranscriptionProvider) -> None:
        self._provider = provider

    async def transcribe(
        self,
        audio: bytes,
        audio_format: str,
        mime_type: str,
        context: str = "",
    ) -> SpeechTranscriptionResult:
        return await self._provider.transcribe(
            audio=audio,
            audio_format=audio_format,
            mime_type=mime_type,
            context=context[:300],
        )


class BailianSpeechTranscriptionProvider:
    provider_name = "bailian"
    endpoint_path = "/api/v1/services/aigc/multimodal-generation/generation"

    def __init__(
        self,
        *,
        api_key: str,
        workspace_id: str,
        model: str,
        timeout_seconds: float = 60,
        base_url: str = "",
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.workspace_id = workspace_id.strip()
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds
        self.base_url = base_url.strip().rstrip("/")
        self.transport = transport

    async def transcribe(
        self,
        audio: bytes,
        audio_format: str,
        mime_type: str,
        context: str = "",
    ) -> SpeechTranscriptionResult:
        self._ensure_configured()
        encoded_audio = base64.b64encode(audio).decode("ascii")
        messages = []
        normalized_context = " ".join(context.split())[:300]
        if normalized_context:
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": normalized_context}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": f"data:{mime_type};base64,{encoded_audio}"
                        },
                    }
                ],
            }
        )
        payload = {
            "model": self.model,
            "input": {"messages": messages},
            "parameters": {"format": audio_format},
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    self._endpoint_url(),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "X-DashScope-SSE": "disable",
                    },
                    json=payload,
                )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise SpeechTranscriptionError(
                "语音转写超时，请检查网络后重试。"
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = _provider_error_detail(exc.response)
            raise SpeechTranscriptionError(
                f"语音服务暂时不可用（{detail}），请稍后重试。"
            ) from exc
        except httpx.HTTPError as exc:
            raise SpeechTranscriptionError(
                "无法连接语音服务，请检查网络后重试。"
            ) from exc

        try:
            response_payload = response.json()
            output = (
                response_payload.get("output")
                if isinstance(response_payload, dict)
                else None
            )
            text = output.get("text") if isinstance(output, dict) else None
            transcript = text.strip() if isinstance(text, str) else ""
        except ValueError as exc:
            raise SpeechTranscriptionError("语音服务返回了无法解析的结果。") from exc
        if not transcript:
            raise NoSpeechDetectedError(
                "没有识别到清晰语音，请靠近麦克风后重新录制。"
            )
        usage = response_payload.get("usage")
        raw_duration = usage.get("duration") if isinstance(usage, dict) else None
        duration_seconds = (
            float(raw_duration)
            if isinstance(raw_duration, (int, float)) and raw_duration >= 0
            else None
        )
        return SpeechTranscriptionResult(
            transcript=transcript,
            provider=self.provider_name,
            model=self.model,
            duration_seconds=duration_seconds,
        )

    def _ensure_configured(self) -> None:
        missing = []
        if not self.api_key:
            missing.append("OFFERPILOT_ASR_API_KEY")
        if not self.workspace_id and not self.base_url:
            missing.append("OFFERPILOT_ASR_WORKSPACE_ID")
        if not self.model:
            missing.append("OFFERPILOT_ASR_MODEL")
        if missing:
            raise SpeechTranscriptionConfigurationError(
                "语音转写尚未配置，请联系管理员补充 " + "、".join(missing) + "。"
            )

    def _endpoint_url(self) -> str:
        if self.base_url:
            if self.base_url.endswith(self.endpoint_path):
                return self.base_url
            return self.base_url + self.endpoint_path
        return (
            f"https://{self.workspace_id}.cn-beijing.maas.aliyuncs.com"
            f"{self.endpoint_path}"
        )


def create_speech_transcription_service(
    settings: Settings,
    *,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> SpeechTranscriptionService:
    provider_name = settings.asr_provider.strip().lower()
    if provider_name != "bailian":
        raise SpeechTranscriptionConfigurationError(
            f"暂不支持语音供应商 {settings.asr_provider!r}。"
        )
    return SpeechTranscriptionService(
        BailianSpeechTranscriptionProvider(
            api_key=settings.asr_api_key,
            workspace_id=settings.asr_workspace_id,
            model=settings.asr_model,
            timeout_seconds=settings.asr_timeout_seconds,
            base_url=settings.asr_base_url,
            transport=transport,
        )
    )


def _provider_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    if not isinstance(payload, dict):
        return f"HTTP {response.status_code}"
    message = payload.get("message") or payload.get("code")
    return str(message)[:160] if message else f"HTTP {response.status_code}"
