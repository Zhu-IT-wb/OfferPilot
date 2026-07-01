import json
import logging
from collections import OrderedDict
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
logger = logging.getLogger(__name__)
_processed_event_ids: "OrderedDict[str, None]" = OrderedDict()
_MAX_PROCESSED_EVENT_IDS = 500


@router.post("/events")
async def handle_feishu_event(payload: Dict[str, Any]):
    _verify_event_token(payload)

    challenge = _extract_challenge(payload)
    if challenge:
        return {"challenge": challenge}

    event_type = _extract_event_type(payload)
    event_id = _extract_event_id(payload)
    if event_id and _is_duplicate_event(event_id):
        logger.info("Feishu duplicate event ignored: event_type=%s", event_type)
        return _response(
            FeishuEventProcessResponse(
                handled=False,
                event_type=event_type,
                message="重复事件已忽略。",
            )
        )

    message = _extract_text_message(payload)
    user_id = _extract_user_id(payload)

    if not message:
        logger.info(
            "Feishu event ignored: event_type=%s user_id_present=%s reason=non_text_message",
            event_type,
            bool(user_id),
        )
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
            user_id=user_id or "unknown_feishu_user",
            source="feishu",
        )
    except LLMRequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    reply_status = await _send_agent_reply(user_id=user_id, text=agent_response.reply)
    if reply_status["sent"]:
        logger.info(
            "Feishu event handled: event_type=%s user_id_present=%s reply_sent=True reply_message_id_present=%s",
            event_type,
            bool(user_id),
            bool(reply_status.get("message_id")),
        )
    else:
        logger.warning(
            "Feishu reply failed: event_type=%s user_id_present=%s error=%s",
            event_type,
            bool(user_id),
            _summarize_reply_error(reply_status.get("error")),
        )
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


def _summarize_reply_error(error: Optional[str]) -> str:
    if not error:
        return "unknown"

    summary_parts = []
    if "HTTP " in error:
        http_status = error.split("HTTP ", 1)[1].split(":", 1)[0].strip()
        if http_status:
            summary_parts.append(f"http_status={http_status}")

    json_start = error.find("{")
    if json_start >= 0:
        try:
            payload = json.loads(error[json_start:])
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and payload.get("code") is not None:
            summary_parts.append(f"feishu_code={payload['code']}")

    if summary_parts:
        return " ".join(summary_parts)

    first_line = error.splitlines()[0].strip()
    return first_line[:160]


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


def _extract_event_id(payload: Dict[str, Any]) -> Optional[str]:
    header = payload.get("header")
    if isinstance(header, dict):
        event_id = header.get("event_id")
        if isinstance(event_id, str) and event_id:
            return event_id

    event = payload.get("event")
    if isinstance(event, dict):
        message = event.get("message")
        if isinstance(message, dict):
            message_id = message.get("message_id")
            if isinstance(message_id, str) and message_id:
                return message_id

    return None


def _is_duplicate_event(event_id: str) -> bool:
    if event_id in _processed_event_ids:
        _processed_event_ids.move_to_end(event_id)
        return True

    _processed_event_ids[event_id] = None
    while len(_processed_event_ids) > _MAX_PROCESSED_EVENT_IDS:
        _processed_event_ids.popitem(last=False)
    return False


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
