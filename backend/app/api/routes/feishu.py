import asyncio
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from hmac import compare_digest
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote
from uuid import NAMESPACE_URL, uuid5
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Request, status

from app.agents.runtime import (
    AgentActorMismatch,
    AgentInteractionError,
    AgentRuntimeConflict,
    AgentThreadNotFound,
)
from app.core.config import Settings, settings
from app.schemas.agent import (
    AgentRunPublicResponse,
    AgentRunResponse,
    AgentRunStatus,
)
from app.schemas.feishu import FeishuEventProcessResponse
from app.services.bitable_sync_service import sync_bitable_record_to_repository
from app.services.feishu_service import (
    FeishuBitableService,
    FeishuConfigurationError,
    FeishuMessageService,
    FeishuRequestError,
)
from app.services.llm_service import LLMConfigurationError, LLMRequestError
from app.services.leetcode_card_service import LeetCodeCardService
from app.tools.offerpilot_tools import (
    get_default_leetcode_repository,
    get_default_offerpilot_repository,
)

router = APIRouter(prefix="/feishu")
logger = logging.getLogger(__name__)


# 处理飞书事件回调入口。
@router.post("/events")
async def handle_feishu_event(request: Request, payload: Dict[str, Any]):
    _verify_event_token(payload, app_settings=request.app.state.settings)

    challenge = _extract_challenge(payload)
    if challenge:
        return {"challenge": challenge}

    event_type = _extract_event_type(payload)
    event_id = _extract_event_id(payload)
    _audit_feishu_event(payload=payload, event_type=event_type, event_id=event_id)
    runtime = getattr(request.app.state, "agent_runtime", None)

    card_action = _extract_card_action(payload)
    if card_action is not None:
        async def execute_card_action() -> Dict[str, Any]:
            return await asyncio.to_thread(
                _handle_leetcode_card_action,
                payload=payload,
                action=card_action,
                repository=getattr(runtime, "leetcode_repository", None),
            )

        return await _run_idempotent_feishu_event(
            runtime=runtime,
            event_id=event_id,
            event_kind="card_action",
            payload=payload,
            execute=execute_card_action,
        )

    bitable_event_context = _extract_bitable_record_event_context(payload)
    if bitable_event_context is not None:
        async def execute_bitable_event() -> Dict[str, Any]:
            result = await asyncio.to_thread(
                _handle_bitable_record_event,
                event_type=event_type,
                context=bitable_event_context,
                repository=getattr(runtime, "offerpilot_repository", None),
                bitable_service=getattr(runtime, "feishu_bitable_service", None),
            )
            return _response(result)

        return await _run_idempotent_feishu_event(
            runtime=runtime,
            event_id=event_id,
            event_kind="bitable_record",
            payload=payload,
            execute=execute_bitable_event,
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

    if runtime is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent runtime is not initialized.",
        )

    if not user_id:
        logger.warning(
            "Feishu text event ignored because sender identity is missing: event_type=%s",
            event_type,
        )
        return _response(
            FeishuEventProcessResponse(
                handled=False,
                event_type=event_type,
                message="无法识别消息发送者，本次请求未执行任何用户数据操作。",
            )
        )

    _require_feishu_event_id(event_id, event_kind="text_message")

    runtime_user_id = user_id
    conversation_scope = _extract_conversation_scope(
        payload,
        user_id=runtime_user_id,
    )
    thread_id = _resolve_runtime_thread_id(
        runtime=runtime,
        user_id=runtime_user_id,
        conversation_scope=conversation_scope,
    )
    progress_status = await _send_progress_reply(
        payload=payload,
        user_id=user_id,
        request_id=event_id,
    )
    if not progress_status.get("sent") and progress_status.get("error"):
        logger.info(
            "Feishu progress reply unavailable: event_type=%s error=%s",
            event_type,
            _summarize_reply_error(progress_status.get("error")),
        )
    try:
        agent_response = await _dispatch_agent_message(
            runtime=runtime,
            message=message,
            user_id=runtime_user_id,
            conversation_scope=conversation_scope,
            thread_id=thread_id,
            request_id=event_id,
        )
    except (LLMConfigurationError, LLMRequestError) as exc:
        logger.warning(
            "Agent runtime unavailable for Feishu event: event_type=%s error_type=%s",
            event_type,
            type(exc).__name__,
        )
        agent_response = _runtime_failure_response(
            thread_id=thread_id,
            warning="llm_unavailable",
            reply="当前智能服务暂时不可用，这次请求没有执行完成，请稍后再试。",
        )
    except AgentActorMismatch:
        agent_response = AgentRunResponse(
            thread_id=thread_id,
            run_id="run_actor_conflict",
            status=AgentRunStatus.DEGRADED,
            reply="这个群聊任务由另一位成员发起，请由发起人继续处理或另开一条会话。",
            warnings=["actor_mismatch"],
        )
    except (AgentRuntimeConflict, AgentInteractionError):
        agent_response = AgentRunResponse(
            thread_id=thread_id,
            run_id="run_interaction_conflict",
            status=AgentRunStatus.DEGRADED,
            reply="当前确认已失效或任务状态已经变化，请查看当前状态后再操作。",
            warnings=["interaction_conflict"],
        )
    except Exception as exc:
        logger.exception(
            "Agent runtime failed for Feishu event: event_type=%s error_type=%s",
            event_type,
            type(exc).__name__,
        )
        agent_response = _runtime_failure_response(
            thread_id=thread_id,
            warning="runtime_failure",
            reply="处理请求时遇到问题，这次任务没有执行完成，请稍后重试。",
        )

    reply_status = await _send_agent_reply(
        payload=payload,
        user_id=user_id,
        agent_response=agent_response,
        request_id=event_id,
        progress_message_id=progress_status.get("message_id"),
        message_service=progress_status.get("service"),
    )
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
        if reply_status.get("retryable"):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Feishu reply delivery failed; the event can be retried.",
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
        value = response.model_dump(exclude_none=True, exclude={"agent_response"})
    else:
        value = response.dict(exclude_none=True, exclude={"agent_response"})
    if response.agent_response is not None:
        public_agent_response = AgentRunPublicResponse.model_validate(
            response.agent_response.model_dump(mode="json")
        )
        value["agent_response"] = public_agent_response.model_dump(
            mode="json",
            exclude_none=True,
        )
    return value


def _resolve_runtime_thread_id(
    runtime: Any,
    user_id: str,
    conversation_scope: Optional[str],
) -> str:
    return runtime.thread_id_for(
        user_id=user_id,
        source="feishu",
        conversation_scope=conversation_scope,
    )


async def _dispatch_agent_message(
    runtime: Any,
    message: str,
    user_id: str,
    conversation_scope: Optional[str],
    thread_id: str,
    request_id: Optional[str],
) -> AgentRunResponse:
    if request_id:
        replay_request = getattr(runtime, "replay_request", None)
        if callable(replay_request):
            replayed = _coerce_agent_response(
                await replay_request(
                    thread_id=thread_id,
                    user_id=user_id,
                    source="feishu",
                    request_id=request_id,
                )
            )
            if replayed is not None:
                return replayed

    try:
        stored_response = await runtime.get_state(
            thread_id=thread_id,
            user_id=user_id,
            source="feishu",
        )
    except AgentThreadNotFound:
        stored_response = None
    current = _coerce_agent_response(stored_response)
    interaction = current.interaction if current is not None else None
    if interaction is not None:
        return _require_agent_response(
            await runtime.resume_message(
                thread_id=current.thread_id,
                interaction_id=interaction.id,
                message=message,
                user_id=user_id,
                source="feishu",
                request_id=request_id,
            )
        )

    if _is_status_question(message):
        if current is not None:
            return _build_status_response(current)
        return AgentRunResponse(
            thread_id=thread_id,
            run_id="run_state_not_found",
            status=AgentRunStatus.DEGRADED,
            reply="当前会话还没有可查询的任务。",
            warnings=["thread_state_not_found"],
        )

    return _require_agent_response(
        await runtime.start(
            message=message,
            user_id=user_id,
            source="feishu",
            conversation_scope=conversation_scope,
            request_id=request_id,
        )
    )


def _coerce_agent_response(value: Any) -> Optional[AgentRunResponse]:
    if value is None:
        return None
    if isinstance(value, AgentRunResponse):
        return value
    if hasattr(value, "values") and isinstance(value.values, dict):
        value = value.values.get("response") or value.values
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, dict) and isinstance(value.get("response"), (dict, AgentRunResponse)):
        value = value["response"]
        if isinstance(value, AgentRunResponse):
            return value
    if isinstance(value, dict):
        return AgentRunResponse.model_validate(value)
    raise TypeError(f"Unsupported agent runtime response: {type(value).__name__}")


def _require_agent_response(value: Any) -> AgentRunResponse:
    response = _coerce_agent_response(value)
    if response is None:
        raise RuntimeError("Agent runtime returned no response.")
    return response


def _build_status_response(response: AgentRunResponse) -> AgentRunResponse:
    status_labels = {
        AgentRunStatus.COMPLETED: "已完成",
        AgentRunStatus.WAITING_FOR_INPUT: "等待你的输入",
        AgentRunStatus.PARTIAL: "部分完成",
        AgentRunStatus.FAILED: "执行失败",
        AgentRunStatus.DEGRADED: "降级完成",
    }
    reply = f"当前任务状态：{status_labels.get(response.status, response.status.value)}。"
    if response.interaction is not None:
        reply = f"{reply}\n\n{response.interaction.prompt}"
    elif response.reply:
        reply = f"{reply}\n\n{response.reply}"
    return response.model_copy(update={"reply": reply})


def _normalize_control_message(message: str) -> str:
    return "".join(message.strip().lower().split()).strip("。！!?？,.，")


def _is_status_question(message: str) -> bool:
    normalized = _normalize_control_message(message)
    return normalized in {
        "当前任务状态",
        "任务状态",
        "现在什么状态",
        "执行到哪了",
        "执行进度",
        "进度怎么样了",
        "现在进度如何",
        "待确认的是什么",
        "需要我确认什么",
    }


def _runtime_failure_response(
    thread_id: str,
    warning: str,
    reply: str,
) -> AgentRunResponse:
    return AgentRunResponse(
        thread_id=thread_id,
        run_id="run_feishu_failure",
        status=AgentRunStatus.FAILED,
        reply=reply,
        warnings=[warning],
    )


async def _send_progress_reply(
    payload: Dict[str, Any],
    user_id: Optional[str],
    request_id: Optional[str],
) -> Dict[str, Any]:
    message_context = _extract_message_context(payload)
    if not message_context.get("message_id") and not message_context.get("chat_id") and not user_id:
        return {"sent": False, "error": "Missing Feishu reply target."}

    service: Optional[FeishuMessageService] = None
    try:
        service = FeishuMessageService()
        if not service.is_configured():
            return {
                "sent": False,
                "error": "Feishu app credentials are not configured.",
                "service": service,
            }

        # Do not create a placeholder when this client cannot replace it later.
        if not callable(getattr(service, "update_interactive_message", None)):
            return {
                "sent": False,
                "error": "Feishu message updates are unavailable.",
                "service": service,
            }

        card = _build_progress_card()
        idempotency_key = _stable_reply_uuid(
            request_id=request_id,
            payload=payload,
            message_kind="progress",
        )
        message_id = message_context.get("message_id")
        reply_in_thread = (
            message_context.get("chat_type") == "group"
            or bool(message_context.get("root_id"))
        )
        if message_id:
            result = await service.reply_interactive_message(
                message_id=message_id,
                card=card,
                idempotency_key=idempotency_key,
                reply_in_thread=reply_in_thread,
            )
        else:
            receive_id = message_context.get("chat_id") or user_id
            receive_id_type = "chat_id" if message_context.get("chat_id") else "open_id"
            result = await service.send_interactive_message(
                receive_id=receive_id,
                card=card,
                receive_id_type=receive_id_type,
                idempotency_key=idempotency_key,
            )
        return {
            "sent": True,
            "message_id": result.message_id,
            "idempotency_key": idempotency_key,
            "service": service,
        }
    except Exception as exc:
        return {
            "sent": False,
            "error": f"{type(exc).__name__}: {exc}",
            "retryable": isinstance(exc, FeishuRequestError),
            "service": service,
        }


async def _send_agent_reply(
    payload: Dict[str, Any],
    user_id: Optional[str],
    agent_response: AgentRunResponse,
    request_id: Optional[str],
    progress_message_id: Optional[str] = None,
    message_service: Optional[FeishuMessageService] = None,
) -> Dict[str, Any]:
    message_context = _extract_message_context(payload)
    if not message_context.get("message_id") and not message_context.get("chat_id") and not user_id:
        return {
            "sent": False,
            "error": "Missing Feishu message, chat, and user identifiers; reply was not sent.",
        }

    service = message_service or FeishuMessageService()
    if not service.is_configured():
        return {
            "sent": False,
            "error": "Feishu app credentials are not configured; reply was not sent.",
        }

    public_response = AgentRunPublicResponse.model_validate(
        agent_response.model_dump(mode="json")
    )
    public_reply = public_response.reply
    artifact_card = _build_artifact_card(
        artifacts=public_response.artifacts,
        fallback_text=public_reply,
        user_id=user_id,
    )
    final_card = artifact_card or _build_plain_response_card(
        text=public_reply,
        run_status=public_response.status,
    )

    if progress_message_id:
        try:
            update_message = getattr(service, "update_interactive_message")
            result = await update_message(
                message_id=progress_message_id,
                card=final_card,
            )
            return {
                "sent": True,
                "message_id": result.message_id or progress_message_id,
                "updated": True,
            }
        except Exception as exc:
            logger.warning(
                "Feishu progress card update failed; sending final reply separately: error=%s",
                _summarize_reply_error(f"{type(exc).__name__}: {exc}"),
            )

    card = final_card if progress_message_id else artifact_card
    message_kind = "interactive" if card is not None else "text"
    idempotency_key = _stable_reply_uuid(
        request_id=request_id,
        payload=payload,
        message_kind=message_kind,
    )
    reply_in_thread = (
        message_context.get("chat_type") == "group"
        or bool(message_context.get("root_id"))
    )
    try:
        message_id = message_context.get("message_id")
        if message_id:
            if card is not None:
                result = await service.reply_interactive_message(
                    message_id=message_id,
                    card=card,
                    idempotency_key=idempotency_key,
                    reply_in_thread=reply_in_thread,
                )
            else:
                result = await service.reply_text_message(
                    message_id=message_id,
                    text=public_reply,
                    idempotency_key=idempotency_key,
                    reply_in_thread=reply_in_thread,
                )
        else:
            receive_id = message_context.get("chat_id") or user_id
            receive_id_type = "chat_id" if message_context.get("chat_id") else "open_id"
            if card is not None:
                result = await service.send_interactive_message(
                    receive_id=receive_id,
                    card=card,
                    receive_id_type=receive_id_type,
                    idempotency_key=idempotency_key,
                )
            else:
                result = await service.send_text_message(
                    receive_id=receive_id,
                    text=public_reply,
                    receive_id_type=receive_id_type,
                    idempotency_key=idempotency_key,
                )
    except FeishuConfigurationError as exc:
        return {
            "sent": False,
            "error": str(exc),
            "retryable": False,
        }
    except FeishuRequestError as exc:
        return {
            "sent": False,
            "error": str(exc),
            "retryable": True,
        }

    return {
        "sent": True,
        "message_id": result.message_id,
    }


def _stable_reply_uuid(
    request_id: Optional[str],
    payload: Dict[str, Any],
    message_kind: str,
) -> str:
    stable_event_key = request_id or json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return str(
        uuid5(
            NAMESPACE_URL,
            f"offerpilot:feishu-reply:{stable_event_key}:{message_kind}",
        )
    )


def _build_progress_card() -> Dict[str, Any]:
    return {
        "config": {"wide_screen_mode": True, "update_multi": True},
        "header": {
            "template": "blue",
            "title": {"tag": "plain_text", "content": "OfferPilot 正在处理"},
        },
        "elements": [
            {
                "tag": "markdown",
                "content": "正在处理你的请求，请稍候。",
            }
        ],
    }


def _build_plain_response_card(
    text: str,
    run_status: AgentRunStatus,
) -> Dict[str, Any]:
    presentation = {
        AgentRunStatus.COMPLETED: ("green", "处理完成"),
        AgentRunStatus.WAITING_FOR_INPUT: ("orange", "需要你的回复"),
        AgentRunStatus.PARTIAL: ("orange", "部分完成"),
        AgentRunStatus.FAILED: ("red", "本次未完成"),
        AgentRunStatus.DEGRADED: ("orange", "处理结果"),
    }
    template, title = presentation.get(run_status, ("blue", "OfferPilot"))
    return {
        "config": {"wide_screen_mode": True, "update_multi": True},
        "header": {
            "template": template,
            "title": {"tag": "plain_text", "content": title},
        },
        "elements": [
            {
                "tag": "markdown",
                "content": text.strip() or "请求已处理。",
            }
        ],
    }


def _build_artifact_card(
    artifacts: list[Any],
    fallback_text: str,
    user_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not artifacts:
        return None

    for artifact in artifacts:
        artifact_data = getattr(artifact, "data", None) or {}
        explicit_card = artifact_data.get("feishu_card") or artifact_data.get("card")
        if isinstance(explicit_card, dict):
            return _enable_card_updates(explicit_card)

    artifact_types = {str(getattr(item, "type", "")) for item in artifacts}
    if artifact_types & {"leetcode_today", "leetcode_recommendations"} and user_id:
        return _enable_card_updates(
            LeetCodeCardService(get_default_leetcode_repository()).build_today_card(
                owner_id=f"feishu:{user_id}",
                today=datetime.now(ZoneInfo("Asia/Shanghai")).date(),
            )
        )

    first = artifacts[0]
    first_data = getattr(first, "data", None) or {}
    title = (
        getattr(first, "title", None)
        or first_data.get("project_name")
        or ("复习计划" if getattr(first, "type", "") == "study_plan" else "OfferPilot")
    )
    content_parts = [fallback_text.strip()] if fallback_text.strip() else []
    for artifact in artifacts[:5]:
        data = getattr(artifact, "data", None) or {}
        artifact_title = getattr(artifact, "title", None)
        summary = data.get("summary") or data.get("message") or data.get("description")
        if artifact_title and artifact_title != title:
            content_parts.append(f"**{artifact_title}**")
        if summary and str(summary) not in content_parts:
            content_parts.append(str(summary))
        if getattr(artifact, "type", "") == "study_plan":
            sessions = data.get("sessions")
            if isinstance(sessions, list) and sessions:
                session_lines = [
                    _format_study_session(item)
                    for item in sessions[:8]
                    if isinstance(item, dict)
                ]
                content_parts.extend(line for line in session_lines if line)
            unscheduled = data.get("unscheduled")
            if isinstance(unscheduled, list) and unscheduled:
                content_parts.append(f"另有 {len(unscheduled)} 项暂未排入日历。")
    if not content_parts:
        content_parts.append(f"已生成 {len(artifacts)} 项结果。")

    elements: list[Dict[str, Any]] = [
        {"tag": "markdown", "content": "\n\n".join(content_parts)}
    ]
    actions = _build_artifact_actions(artifacts)
    if actions:
        elements.append({"tag": "action", "actions": actions})
    return {
        "config": {"wide_screen_mode": True, "update_multi": True},
        "header": {
            "template": "blue",
            "title": {"tag": "plain_text", "content": str(title)[:80]},
        },
        "elements": elements,
    }


def _enable_card_updates(card: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(card)
    raw_config = card.get("config")
    config = dict(raw_config) if isinstance(raw_config, dict) else {}
    config["wide_screen_mode"] = True
    config["update_multi"] = True
    normalized["config"] = config
    return normalized


def _format_study_session(session: Dict[str, Any]) -> str:
    topic = str(session.get("topic") or "复习任务")
    start_at = str(session.get("start_at") or "待安排")
    duration = session.get("duration_minutes")
    duration_text = f"，{duration} 分钟" if duration else ""
    return f"- {start_at}：{topic}{duration_text}"


def _build_artifact_actions(artifacts: list[Any]) -> list[Dict[str, Any]]:
    actions: list[Dict[str, Any]] = []
    for artifact in artifacts:
        data = getattr(artifact, "data", None) or {}
        url = getattr(artifact, "url", None) or data.get("launch_url") or data.get("url")
        candidates = data.get("candidates")
        if isinstance(url, str) and url and isinstance(candidates, list):
            separator = "&" if "?" in url else "?"
            for candidate in candidates:
                if not isinstance(candidate, dict) or not candidate.get("id"):
                    continue
                actions.append(
                    {
                        "tag": "button",
                        "type": "primary" if not actions else "default",
                        "text": {
                            "tag": "plain_text",
                            "content": str(candidate.get("name") or "选择项目")[:40],
                        },
                        "url": (
                            f"{url}{separator}start_project_id="
                            f"{quote(str(candidate['id']), safe='')}"
                        ),
                    }
                )
                if len(actions) >= 5:
                    return actions
            continue
        if isinstance(url, str) and url:
            label = data.get("button_text") or getattr(artifact, "title", None) or "查看详情"
            actions.append(
                {
                    "tag": "button",
                    "type": "primary" if not actions else "default",
                    "text": {"tag": "plain_text", "content": str(label)[:40]},
                    "url": url,
                }
            )
        if len(actions) >= 5:
            break
    return actions


def _handle_leetcode_card_action(
    payload: Dict[str, Any],
    action: Dict[str, Any],
    repository: Any = None,
) -> Dict[str, Any]:
    open_id = _extract_card_operator_open_id(payload)
    if not open_id:
        return _card_toast("error", "无法识别点击用户，请重新打开机器人后再试。")
    value = action.get("value")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = None
    if not isinstance(value, dict) or value.get("action") != "leetcode_result":
        return _card_toast("error", "暂不支持这个卡片操作。")
    assignment_id = str(value.get("assignment_id") or "").strip()
    result = str(value.get("result") or "").strip()
    if not assignment_id:
        return _card_toast("error", "题目标识缺失，请刷新题目卡片后重试。")

    owner_id = f"feishu:{open_id}"
    selected_repository = repository or get_default_leetcode_repository()
    tool_result = LeetCodeCardService(selected_repository).record_result(
        owner_id=owner_id,
        assignment_id=assignment_id,
        result=result,
    )
    if not tool_result.success:
        logger.info(
            "LeetCode card action rejected: owner_id=%s assignment_id_present=%s message=%s",
            owner_id,
            bool(assignment_id),
            tool_result.message,
        )
        return _card_toast("error", tool_result.message)
    return _card_toast("success", tool_result.message)


def _card_toast(toast_type: str, content: str) -> Dict[str, Any]:
    return {"toast": {"type": toast_type, "content": content}}


def _extract_card_action(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    event = payload.get("event")
    if isinstance(event, dict) and isinstance(event.get("action"), dict):
        return event["action"]
    action = payload.get("action")
    return action if isinstance(action, dict) else None


def _extract_card_operator_open_id(payload: Dict[str, Any]) -> Optional[str]:
    open_id = payload.get("open_id")
    if isinstance(open_id, str) and open_id:
        return open_id
    event = payload.get("event")
    if not isinstance(event, dict):
        return None
    operator = event.get("operator")
    if not isinstance(operator, dict):
        return None
    operator_id = operator.get("operator_id")
    if isinstance(operator_id, dict):
        open_id = operator_id.get("open_id")
        if isinstance(open_id, str) and open_id:
            return open_id
    open_id = operator.get("open_id")
    return open_id if isinstance(open_id, str) and open_id else None


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
def _verify_event_token(payload: Dict[str, Any], *, app_settings: Settings) -> None:
    expected_token = app_settings.feishu_verification_token.strip()
    if not expected_token:
        environment = app_settings.environment.strip().lower()
        if (
            app_settings.feishu_allow_unverified_events
            and environment in {"local", "dev", "development", "test"}
        ):
            return
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feishu verification token is not configured.",
        )

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

    open_id = sender_id.get("open_id")
    if not isinstance(open_id, str):
        return None
    return open_id.strip() or None


def _extract_message_context(payload: Dict[str, Any]) -> Dict[str, Optional[str]]:
    event = payload.get("event")
    if not isinstance(event, dict):
        return {}
    message = event.get("message")
    if not isinstance(message, dict):
        return {}

    def optional_string(key: str) -> Optional[str]:
        value = message.get(key)
        return value if isinstance(value, str) and value else None

    return {
        "chat_id": optional_string("chat_id"),
        "message_id": optional_string("message_id"),
        "root_id": optional_string("root_id") or optional_string("thread_id"),
        "chat_type": optional_string("chat_type"),
    }


def _extract_conversation_scope(
    payload: Dict[str, Any],
    user_id: Optional[str] = None,
) -> Optional[str]:
    message_context = _extract_message_context(payload)
    chat_id = message_context.get("chat_id")
    if not chat_id:
        return None

    tenant_key = "unknown_tenant"
    header = payload.get("header")
    if isinstance(header, dict):
        raw_tenant_key = header.get("tenant_key")
        if isinstance(raw_tenant_key, str) and raw_tenant_key:
            tenant_key = raw_tenant_key

    chat_type = message_context.get("chat_type")
    root_id = message_context.get("root_id")
    is_group = chat_type == "group" or bool(root_id)
    if is_group:
        conversation_id = root_id or message_context.get("message_id")
    else:
        conversation_id = user_id or _extract_user_id(payload)
    if not conversation_id:
        return None
    return ":".join((tenant_key, chat_id, conversation_id))


# 处理 handle_bitable_record_event 相关逻辑。
def _handle_bitable_record_event(
    event_type: Optional[str],
    context: Dict[str, Any],
    repository: Any = None,
    bitable_service: Any = None,
) -> FeishuEventProcessResponse:
    selected_repository = repository or get_default_offerpilot_repository()
    selected_bitable_service = bitable_service or FeishuBitableService()
    results = [
        sync_bitable_record_to_repository(
            repository=selected_repository,
            bitable_service=selected_bitable_service,
            app_token=context.get("app_token"),
            table_id=context.get("table_id"),
            record_id=record_id,
            fallback_owner_id=context.get("owner_id"),
            allow_default_local_owner=False,
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


async def _run_idempotent_feishu_event(
    *,
    runtime: Any,
    event_id: Optional[str],
    event_kind: str,
    payload: Dict[str, Any],
    execute: Callable[[], Awaitable[Dict[str, Any]]],
) -> Dict[str, Any]:
    _require_feishu_event_id(event_id, event_kind=event_kind)
    store = getattr(runtime, "runtime_store", None)
    if store is None:
        return await execute()

    source = "feishu:webhook"
    fingerprint = hashlib.sha256(
        json.dumps(
            {"event_kind": event_kind, "payload": payload},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    thread_id = f"feishu_event_{event_id}"
    run_id = "webhook_" + hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:24]

    record = await asyncio.to_thread(store.get_receipt_record, source, event_id)
    if record is None:
        created = await asyncio.to_thread(
            store.begin_receipt,
            source,
            event_id,
            fingerprint,
            thread_id,
            run_id,
        )
        if created:
            result = await execute()
            await asyncio.to_thread(
                store.stage_receipt_response,
                source,
                event_id,
                result,
            )
            await asyncio.to_thread(store.finish_receipt, source, event_id, result)
            return result
        record = await asyncio.to_thread(store.get_receipt_record, source, event_id)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This Feishu event is already being processed.",
        )
    if record.fingerprint != fingerprint:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This Feishu event id was reused with a different payload.",
        )
    if record.response is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This Feishu event is already being processed.",
        )
    result = dict(record.response)
    if record.status != "completed":
        await asyncio.to_thread(store.finish_receipt, source, event_id, result)
    return result


def _require_feishu_event_id(
    event_id: Optional[str],
    *,
    event_kind: str,
) -> str:
    if event_id:
        return event_id
    logger.warning(
        "Feishu event rejected because a stable event id is missing: event_kind=%s",
        event_kind,
    )
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Feishu event id is required for idempotent processing.",
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

    operator_open_id = _extract_bitable_operator_open_id(payload)

    return {
        "app_token": app_token,
        "table_id": table_id,
        "record_ids": record_ids,
        "owner_id": f"feishu:{operator_open_id}" if operator_open_id else None,
    }


def _extract_bitable_operator_open_id(payload: Dict[str, Any]) -> Optional[str]:
    event = payload.get("event")
    if not isinstance(event, dict):
        return None

    candidates = [
        event.get("operator_id"),
        event.get("operatorId"),
        event.get("operator"),
    ]
    for candidate in candidates:
        open_id = _extract_open_id_from_operator(candidate)
        if open_id:
            return open_id
    return None


def _extract_open_id_from_operator(candidate: Any) -> Optional[str]:
    if not isinstance(candidate, dict):
        return None
    open_id = candidate.get("open_id") or candidate.get("openId")
    if isinstance(open_id, str) and open_id:
        return open_id
    nested_operator_id = candidate.get("operator_id") or candidate.get("operatorId")
    if isinstance(nested_operator_id, dict):
        open_id = nested_operator_id.get("open_id") or nested_operator_id.get("openId")
        if isinstance(open_id, str) and open_id:
            return open_id
    return None


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
