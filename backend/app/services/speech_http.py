from dataclasses import dataclass

from fastapi import HTTPException, Request, status

from app.services.speech_transcription import (
    NoSpeechDetectedError,
    SpeechTranscriptionConfigurationError,
    SpeechTranscriptionError,
)


MAX_BAILIAN_DATA_URI_BYTES = 10 * 1024 * 1024
MAX_AUDIO_BYTES = ((MAX_BAILIAN_DATA_URI_BYTES - 64) // 4) * 3
MAX_AUDIO_DURATION_MS = 180_000
AUDIO_FORMATS = {
    "audio/webm": "webm",
    "audio/mp4": "mp4",
    "audio/x-m4a": "m4a",
    "audio/m4a": "m4a",
    "audio/ogg": "ogg",
    "audio/opus": "opus",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/aac": "aac",
}


async def transcribe_audio_request(request: Request, service, context: str):
    mime_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    audio_format = AUDIO_FORMATS.get(mime_type)
    if audio_format is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="不支持这种录音格式，请改用 WebM、MP4、M4A、Ogg、WAV、MP3 或 AAC。",
        )
    audio_duration_ms(request)
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_AUDIO_BYTES:
                raise _audio_too_large()
        except ValueError:
            pass
    audio = await read_limited_audio(request)
    if not audio:
        raise HTTPException(status_code=400, detail="录音内容为空，请重新录制。")
    try:
        result = await service.transcribe(
            audio=audio,
            audio_format=audio_format,
            mime_type=mime_type,
            context=context[:300],
        )
    except SpeechTranscriptionConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except NoSpeechDetectedError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SpeechTranscriptionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if (
        result.duration_seconds is not None
        and result.duration_seconds * 1000 > MAX_AUDIO_DURATION_MS
    ):
        raise HTTPException(
            status_code=400,
            detail="语音服务检测到录音超过 3 分钟，请缩短后重新录制。",
        )
    return result


def audio_duration_ms(request: Request) -> int:
    raw_duration = request.headers.get("x-audio-duration-ms", "")
    try:
        duration_ms = int(raw_duration)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="缺少有效的录音时长，请重新录制。") from exc
    if duration_ms <= 0:
        raise HTTPException(status_code=400, detail="录音时长必须大于 0。")
    if duration_ms > MAX_AUDIO_DURATION_MS:
        raise HTTPException(status_code=400, detail="单次语音回答不能超过 3 分钟。")
    return duration_ms


async def read_limited_audio(request: Request) -> bytes:
    chunks = []
    total_bytes = 0
    async for chunk in request.stream():
        total_bytes += len(chunk)
        if total_bytes > MAX_AUDIO_BYTES:
            raise _audio_too_large()
        chunks.append(chunk)
    return b"".join(chunks)


def _audio_too_large() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        detail="录音文件不能超过约 7.5 MB（Base64 输入上限为 10 MB）。",
    )
