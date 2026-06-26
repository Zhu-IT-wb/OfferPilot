import json
from hmac import compare_digest
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status

from app.agents.orchestrator import AgentOrchestrator
from app.core.config import settings
from app.schemas.feishu import FeishuEventProcessResponse
from app.services.feishu_service import (
    FeishuConfigurationError,
    FeishuMessageService,
    FeishuRequestError,
)
from app.services.llm_service import LLMRequestError

router = APIRouter(prefix="/feishu")


@router.post("/events")
async def handle_feishu_event(payload: Dict[str, Any]):
    _verify_event_token(payload)

    challenge = _extract_challenge(payload)
    if challenge:
        return {"challenge": challenge}

    event_type = _extract_event_type(payload)
    message = _extract_text_message(payload)
    user_id = _extract_user_id(payload)

    if not message:
        return _response(
            FeishuEventProcessResponse(
                handled=False,
                event_type=event_type,
                user_id=user_id,
                message="当前只处理文本消息事件。",
            )
        )

    orchestrator = AgentOrchestrator()
    try:
        agent_response = await orchestrator.handle_message(
            message=message,
            confirmed=False,
        )
    except LLMRequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    reply_status = await _send_agent_reply(user_id=user_id, text=agent_response.reply)
    return _response(
        FeishuEventProcessResponse(
            handled=True,
            event_type=event_type,
            message=message,
            user_id=user_id,
            agent_response=agent_response,
            reply_sent=reply_status["sent"],
            reply_error=reply_status.get("error"),
            reply_message_id=reply_status.get("message_id"),
        )
    )


def _response(response: FeishuEventProcessResponse) -> Dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(exclude_none=True)
    return response.dict(exclude_none=True)


async def _send_agent_reply(user_id: Optional[str], text: str) -> Dict[str, Any]:
    if not user_id:
        return {
            "sent": False,
            "error": "Missing Feishu user open_id; reply was not sent.",
        }

    service = FeishuMessageService()
    if not service.is_configured():
        return {
            "sent": False,
            "error": "Feishu app credentials are not configured; reply was not sent.",
        }

    try:
        result = await service.send_text_message(receive_id=user_id, text=text)
    except (FeishuConfigurationError, FeishuRequestError) as exc:
        return {
            "sent": False,
            "error": str(exc),
        }

    return {
        "sent": True,
        "message_id": result.message_id,
    }


def _verify_event_token(payload: Dict[str, Any]) -> None:
    expected_token = settings.feishu_verification_token.strip()
    if not expected_token:
        return

    actual_token = _extract_verification_token(payload)
    if not actual_token or not compare_digest(actual_token, expected_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid Feishu verification token.",
        )


def _extract_verification_token(payload: Dict[str, Any]) -> Optional[str]:
    token = payload.get("token")
    if isinstance(token, str) and token:
        return token

    header = payload.get("header")
    if isinstance(header, dict) and isinstance(header.get("token"), str):
        return header["token"]

    event = payload.get("event")
    if isinstance(event, dict) and isinstance(event.get("token"), str):
        return event["token"]

    return None


def _extract_challenge(payload: Dict[str, Any]) -> Optional[str]:
    challenge = payload.get("challenge")
    if isinstance(challenge, str) and challenge:
        return challenge

    event = payload.get("event")
    if isinstance(event, dict) and isinstance(event.get("challenge"), str):
        return event["challenge"]

    return None


def _extract_event_type(payload: Dict[str, Any]) -> Optional[str]:
    header = payload.get("header")
    if isinstance(header, dict) and isinstance(header.get("event_type"), str):
        return header["event_type"]

    event_type = payload.get("type")
    if isinstance(event_type, str):
        return event_type

    return None


def _extract_text_message(payload: Dict[str, Any]) -> Optional[str]:
    event = payload.get("event")
    if not isinstance(event, dict):
        return None

    message = event.get("message")
    if not isinstance(message, dict):
        return None

    message_type = message.get("message_type")
    if message_type and message_type != "text":
        return None

    content = message.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return content.strip() or None

    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text.strip() or None

    return None


def _extract_user_id(payload: Dict[str, Any]) -> Optional[str]:
    event = payload.get("event")
    if not isinstance(event, dict):
        return None

    sender = event.get("sender")
    if not isinstance(sender, dict):
        return None

    sender_id = sender.get("sender_id")
    if not isinstance(sender_id, dict):
        return None

    for key in ("open_id", "union_id", "user_id"):
        value = sender_id.get(key)
        if isinstance(value, str) and value:
            return value

    return None
