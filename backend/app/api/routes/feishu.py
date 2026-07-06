import json
import logging
from collections import OrderedDict
from datetime import datetime
from hmac import compare_digest
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status

from app.agents.orchestrator import AgentOrchestrator
from app.core.config import settings
from app.schemas.feishu import FeishuEventProcessResponse
from app.services.bitable_sync_service import sync_bitable_record_to_repository
from app.services.feishu_service import (
    FeishuBitableService,
    FeishuConfigurationError,
    FeishuMessageService,
    FeishuRequestError,
)
from app.services.llm_service import LLMRequestError
from app.tools.offerpilot_tools import get_default_offerpilot_repository

router = APIRouter(prefix="/feishu")
logger = logging.getLogger(__name__)
_processed_event_ids: "OrderedDict[str, None]" = OrderedDict()
_MAX_PROCESSED_EVENT_IDS = 500


# 处理飞书事件回调入口。
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

    _audit_feishu_event(payload=payload, event_type=event_type, event_id=event_id)

    bitable_event_context = _extract_bitable_record_event_context(payload)
    if bitable_event_context is not None:
        return _response(
            _handle_bitable_record_event(
                event_type=event_type,
                context=bitable_event_context,
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


# 处理 response 相关逻辑。
def _response(response: FeishuEventProcessResponse) -> Dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(exclude_none=True)
    return response.dict(exclude_none=True)


# 处理 send_agent_reply 相关逻辑。
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


# 压缩并说明 reply error。
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


# 处理 audit_feishu_event 相关逻辑。
def _audit_feishu_event(
    payload: Dict[str, Any],
    event_type: Optional[str],
    event_id: Optional[str],
) -> None:
    try:
        app_token = _extract_first_string_value(
            payload,
            keys={"app_token", "base_token", "base_id", "file_token", "fileToken"},
        )
        table_id = _extract_first_string_value(
            payload,
            keys={"table_id", "tableId", "tableID"},
        )
        record_ids = _extract_unique_string_values(
            payload,
            keys={
                "record_id",
                "record_ids",
                "recordId",
                "recordIds",
                "recordID",
                "recordIDs",
                "record_id_list",
                "recordIdList",
                "recordIDList",
            },
        )
        if not record_ids:
            record_ids = _extract_candidate_bitable_record_ids(payload)

        audit_payload = {
            "received_at": datetime.now().isoformat(timespec="seconds"),
            "event_type": event_type,
            "event_id": event_id,
            "event_keys": _extract_event_keys(payload),
            "app_token": app_token,
            "table_id": table_id,
            "record_ids": record_ids,
            "event": payload.get("event"),
        }
        _feishu_event_audit_path().parent.mkdir(parents=True, exist_ok=True)
        with _feishu_event_audit_path().open("a", encoding="utf-8") as file:
            file.write(json.dumps(audit_payload, ensure_ascii=False, default=str))
            file.write("\n")
    except Exception as exc:
        logger.warning("Failed to write Feishu event audit log: %s", exc)


# 处理 audit_bitable_sync_result 相关逻辑。
def _audit_bitable_sync_result(
    event_type: Optional[str],
    context: Dict[str, Any],
    results: list[Any],
) -> None:
    try:
        audit_payload = {
            "received_at": datetime.now().isoformat(timespec="seconds"),
            "phase": "bitable_sync_result",
            "event_type": event_type,
            "app_token": context.get("app_token"),
            "table_id": context.get("table_id"),
            "record_ids": context.get("record_ids"),
            "results": [
                {
                    "synced": result.synced,
                    "status": result.status,
                    "application_id": result.application_id,
                    "record_id": result.record_id,
                    "updated_fields": list(result.updated_fields),
                    "error": result.error,
                }
                for result in results
            ],
        }
        _feishu_event_audit_path().parent.mkdir(parents=True, exist_ok=True)
        with _feishu_event_audit_path().open("a", encoding="utf-8") as file:
            file.write(json.dumps(audit_payload, ensure_ascii=False, default=str))
            file.write("\n")
    except Exception as exc:
        logger.warning("Failed to write Feishu bitable sync audit log: %s", exc)


# 处理 feishu_event_audit_path 相关逻辑。
def _feishu_event_audit_path() -> Path:
    sqlite_path = Path(settings.sqlite_path).expanduser()
    if not sqlite_path.is_absolute():
        sqlite_path = Path.cwd() / sqlite_path
    return sqlite_path.parent / "feishu_events.log"


# 处理 verify_event_token 相关逻辑。
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


# 从输入数据中提取 verification token。
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


# 从输入数据中提取 challenge。
def _extract_challenge(payload: Dict[str, Any]) -> Optional[str]:
    challenge = payload.get("challenge")
    if isinstance(challenge, str) and challenge:
        return challenge

    event = payload.get("event")
    if isinstance(event, dict) and isinstance(event.get("challenge"), str):
        return event["challenge"]

    return None


# 从输入数据中提取 event type。
def _extract_event_type(payload: Dict[str, Any]) -> Optional[str]:
    header = payload.get("header")
    if isinstance(header, dict) and isinstance(header.get("event_type"), str):
        return header["event_type"]

    event_type = payload.get("type")
    if isinstance(event_type, str):
        return event_type

    return None


# 从输入数据中提取 event id。
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


# 判断 duplicate event 是否成立。
def _is_duplicate_event(event_id: str) -> bool:
    if event_id in _processed_event_ids:
        _processed_event_ids.move_to_end(event_id)
        return True

    _processed_event_ids[event_id] = None
    while len(_processed_event_ids) > _MAX_PROCESSED_EVENT_IDS:
        _processed_event_ids.popitem(last=False)
    return False


# 从输入数据中提取 text message。
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


# 从输入数据中提取 user id。
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


# 处理 handle_bitable_record_event 相关逻辑。
def _handle_bitable_record_event(
    event_type: Optional[str],
    context: Dict[str, Any],
) -> FeishuEventProcessResponse:
    repository = get_default_offerpilot_repository()
    bitable_service = FeishuBitableService()
    results = [
        sync_bitable_record_to_repository(
            repository=repository,
            bitable_service=bitable_service,
            app_token=context.get("app_token"),
            table_id=context.get("table_id"),
            record_id=record_id,
        )
        for record_id in context["record_ids"]
    ]
    _audit_bitable_sync_result(
        event_type=event_type,
        context=context,
        results=results,
    )
    synced_results = [result for result in results if result.synced]
    failed_results = [result for result in results if result.status == "failed"]

    if synced_results:
        application_ids = ", ".join(
            result.application_id or result.record_id or "unknown"
            for result in synced_results
        )
        logger.info(
            "Feishu bitable event synced: event_type=%s records=%s applications=%s",
            event_type,
            len(synced_results),
            application_ids,
        )
        return FeishuEventProcessResponse(
            handled=True,
            event_type=event_type,
            message=f"多维表格记录已回写数据库：{application_ids}",
        )

    if failed_results:
        error = failed_results[0].error or "unknown"
        logger.warning(
            "Feishu bitable event sync failed: event_type=%s records=%s error=%s",
            event_type,
            len(failed_results),
            error,
        )
        return FeishuEventProcessResponse(
            handled=True,
            event_type=event_type,
            message=f"多维表格事件已收到，但同步失败：{error}",
        )

    statuses = ", ".join(sorted({result.status for result in results})) or "no_record"
    logger.info(
        "Feishu bitable event ignored: event_type=%s statuses=%s",
        event_type,
        statuses,
    )
    return FeishuEventProcessResponse(
        handled=False,
        event_type=event_type,
        message=f"多维表格事件未写入数据库：{statuses}",
    )


# 从输入数据中提取 bitable record event context。
def _extract_bitable_record_event_context(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    event_type = _extract_event_type(payload) or ""
    app_token = _extract_first_string_value(
        payload,
        keys={"app_token", "base_token", "base_id", "file_token", "fileToken"},
    )
    table_id = _extract_first_string_value(
        payload,
        keys={"table_id", "tableId", "tableID"},
    )
    record_ids = _extract_unique_string_values(
        payload,
        keys={
            "record_id",
            "record_ids",
            "recordId",
            "recordIds",
            "recordID",
            "recordIDs",
            "record_id_list",
            "recordIdList",
            "recordIDList",
        },
    )
    if not record_ids and _looks_like_bitable_event(event_type, app_token, table_id):
        record_ids = _extract_candidate_bitable_record_ids(payload)

    if not record_ids:
        if _looks_like_bitable_event(event_type, app_token, table_id):
            logger.warning(
                "Feishu bitable event has no record id: event_type=%s app_token_present=%s table_id_present=%s event_keys=%s",
                event_type,
                bool(app_token),
                bool(table_id),
                _extract_event_keys(payload),
            )
        return None
    if not _looks_like_bitable_event(event_type, app_token, table_id):
        return None

    return {
        "app_token": app_token,
        "table_id": table_id,
        "record_ids": record_ids,
    }


# 判断输入是否像 bitable event。
def _looks_like_bitable_event(
    event_type: str,
    app_token: Optional[str],
    table_id: Optional[str],
) -> bool:
    normalized_event_type = event_type.replace("_", ".").lower()
    if any(keyword in normalized_event_type for keyword in ("bitable", "base", "table.record")):
        return True
    return bool(app_token and table_id)


# 从输入数据中提取 first string value。
def _extract_first_string_value(payload: Any, keys: set[str]) -> Optional[str]:
    values = _extract_unique_string_values(payload, keys)
    return values[0] if values else None


# 从输入数据中提取 unique string values。
def _extract_unique_string_values(payload: Any, keys: set[str]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()

    # 递归遍历嵌套数据并收集目标值。
    def collect(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in keys:
                    if isinstance(value, str) and value:
                        if value not in seen:
                            seen.add(value)
                            values.append(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and item and item not in seen:
                                seen.add(item)
                                values.append(item)
                collect(value)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(payload)
    return values


# 从输入数据中提取 candidate bitable record ids。
def _extract_candidate_bitable_record_ids(payload: Any) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()

    # 递归遍历嵌套数据并收集目标值。
    def collect(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for item in node:
                collect(item)
        elif isinstance(node, str) and _looks_like_bitable_record_id(node):
            if node not in seen:
                seen.add(node)
                values.append(node)

    collect(payload)
    return values


# 判断输入是否像 bitable record id。
def _looks_like_bitable_record_id(value: str) -> bool:
    if not value.startswith("rec"):
        return False
    if len(value) < 8:
        return False
    return all(char.isalnum() or char in {"_", "-"} for char in value)


# 从输入数据中提取 event keys。
def _extract_event_keys(payload: Dict[str, Any]) -> list[str]:
    event = payload.get("event")
    if not isinstance(event, dict):
        return []
    return sorted(event.keys())
