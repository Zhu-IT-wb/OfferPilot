import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from app.agents.conversation import (
    ConversationEvent,
    ConversationStore,
    ConversationSummary,
)
from app.core.config import settings
from app.services.llm_service import LLMConfigurationError, LLMRequestError, LLMService


_TEXT_FIELDS = (
    "user_goals",
    "target_roles",
    "confirmed_facts",
    "decisions",
    "completed_actions",
    "open_loops",
)
_OBJECT_FIELDS = (
    "active_applications",
    "upcoming_interviews",
    "training_progress",
    "tool_evidence",
)
_SUMMARY_LIMITS = {
    "user_goals": 12,
    "target_roles": 12,
    "active_applications": 20,
    "upcoming_interviews": 20,
    "training_progress": 12,
    "confirmed_facts": 30,
    "decisions": 20,
    "completed_actions": 30,
    "open_loops": 20,
    "tool_evidence": 30,
}


@dataclass(frozen=True)
class PreparedConversationContext:
    summary: Optional[ConversationSummary]
    events: List[ConversationEvent]
    estimated_tokens: int
    compacted: bool = False


class ContextCompactionService:
    SYSTEM_PROMPT = """
You maintain compact working memory for OfferPilot, a campus recruiting Agent.
Return exactly one JSON object and no Markdown. Use only facts supported by the supplied
summary, conversation events, and deterministic seed. Do not invent companies, roles,
interviews, task results, tool outcomes, dates, user preferences, or decisions.

Keep these fields:
user_goals, target_roles, active_applications, upcoming_interviews, training_progress,
confirmed_facts, decisions, completed_actions, open_loops, tool_evidence.

Each text field is an array of concise strings. The application, interview, training, and
tool_evidence fields are arrays of JSON objects. Preserve unresolved work and user constraints.
Keep tool evidence compact and retain source_sequences so the raw event can be audited. Do not
include raw tool output dumps.
""".strip()

    def __init__(
        self,
        conversation_store: ConversationStore,
        llm_service: Optional[Any] = None,
        enabled: Optional[bool] = None,
        trigger_tokens: Optional[int] = None,
        max_context_tokens: Optional[int] = None,
        keep_recent_turns: Optional[int] = None,
        summary_max_tokens: Optional[int] = None,
    ) -> None:
        self.conversation_store = conversation_store
        self.llm_service = llm_service or LLMService()
        self.enabled = (
            settings.context_compaction_enabled if enabled is None else enabled
        )
        self.trigger_tokens = max(
            1,
            trigger_tokens
            if trigger_tokens is not None
            else settings.context_compaction_trigger_tokens,
        )
        self.max_context_tokens = max(
            self.trigger_tokens,
            max_context_tokens
            if max_context_tokens is not None
            else settings.context_max_input_tokens,
        )
        self.keep_recent_turns = max(
            1,
            keep_recent_turns
            if keep_recent_turns is not None
            else settings.context_keep_recent_turns,
        )
        self.summary_max_tokens = max(
            256,
            summary_max_tokens
            if summary_max_tokens is not None
            else settings.context_summary_max_tokens,
        )

    async def prepare_context(self, conversation_id: str) -> PreparedConversationContext:
        summary = self.conversation_store.get_conversation_summary(conversation_id)
        summary_sequence = summary.summary_upto_sequence if summary is not None else 0
        events = self.conversation_store.get_events_after(conversation_id, summary_sequence)
        estimated_tokens = estimate_context_tokens(summary, events)
        compacted = False

        if (
            self.enabled
            and estimated_tokens >= self.trigger_tokens
            and len(_group_complete_turns(events)) > self.keep_recent_turns
        ):
            compacted = await self._compact(conversation_id, summary, events)
            latest_summary = self.conversation_store.get_conversation_summary(
                conversation_id
            )
            latest_sequence = (
                latest_summary.summary_upto_sequence
                if latest_summary is not None
                else 0
            )
            if compacted or latest_sequence > summary_sequence:
                summary = latest_summary
                summary_sequence = summary.summary_upto_sequence if summary is not None else 0
                events = self.conversation_store.get_events_after(
                    conversation_id,
                    summary_sequence,
                )

        fitted_events = _fit_recent_turns(
            summary=summary,
            events=events,
            max_context_tokens=self.max_context_tokens,
        )
        return PreparedConversationContext(
            summary=summary,
            events=fitted_events,
            estimated_tokens=estimate_context_tokens(summary, fitted_events),
            compacted=compacted,
        )

    async def _compact(
        self,
        conversation_id: str,
        existing_summary: Optional[ConversationSummary],
        events: List[ConversationEvent],
    ) -> bool:
        turns = _group_complete_turns(events)
        compacted_turns = turns[: -self.keep_recent_turns]
        compacted_events = [event for turn in compacted_turns for event in turn]
        if not compacted_events:
            return False

        previous_sequence = (
            existing_summary.summary_upto_sequence if existing_summary is not None else 0
        )
        compacted_through = compacted_events[-1].sequence
        deterministic = _build_deterministic_summary(
            existing_summary=existing_summary,
            events=compacted_events,
            compacted_through=compacted_through,
        )
        model_summary = await self._summarize_with_model(
            existing_summary=existing_summary,
            events=compacted_events,
            deterministic=deterministic,
        )
        merged = _merge_model_summary(deterministic, model_summary)
        merged = _shrink_summary(merged, self.summary_max_tokens)
        return self.conversation_store.save_conversation_summary(
            conversation_id=conversation_id,
            summary=merged,
            expected_upto_sequence=previous_sequence,
        )

    async def _summarize_with_model(
        self,
        existing_summary: Optional[ConversationSummary],
        events: List[ConversationEvent],
        deterministic: ConversationSummary,
    ) -> Optional[ConversationSummary]:
        if isinstance(self.llm_service, LLMService) and not self.llm_service.api_key:
            return None

        prompt = json.dumps(
            {
                "existing_summary": (
                    existing_summary.to_prompt_context()
                    if existing_summary is not None
                    else None
                ),
                "deterministic_seed": deterministic.to_prompt_context(),
                "events_to_compact": [
                    _limit_prompt_value(event.to_prompt_context())
                    for event in events
                ],
            },
            ensure_ascii=False,
        )
        try:
            result = await self.llm_service.generate_text(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=self.summary_max_tokens,
                response_format={"type": "json_object"},
            )
            value = _parse_json_object(result.content)
            value["summary_upto_sequence"] = deterministic.summary_upto_sequence
            value["compaction_count"] = deterministic.compaction_count
            value["schema_version"] = deterministic.schema_version
            return ConversationSummary.from_dict(value)
        except (LLMConfigurationError, LLMRequestError, ValueError, TypeError):
            return None


def estimate_context_tokens(
    summary: Optional[ConversationSummary],
    events: Sequence[ConversationEvent],
) -> int:
    payload = {
        "summary": summary.to_prompt_context() if summary is not None else None,
        "history": [event.to_prompt_context() for event in events],
    }
    return estimate_value_tokens(payload)


def estimate_value_tokens(value: Any) -> int:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    cjk_chars = 0
    ascii_chars = 0
    whitespace_chars = 0
    for char in text:
        if char.isspace():
            whitespace_chars += 1
        elif _is_cjk(char):
            cjk_chars += 1
        else:
            ascii_chars += 1
    return max(
        1,
        cjk_chars
        + math.ceil(ascii_chars / 4)
        + math.ceil(whitespace_chars / 8),
    )


def _is_cjk(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _group_complete_turns(
    events: Sequence[ConversationEvent],
) -> List[List[ConversationEvent]]:
    turns: List[List[ConversationEvent]] = []
    for event in events:
        if not turns or turns[-1][0].turn_id != event.turn_id:
            turns.append([event])
        else:
            turns[-1].append(event)
    return turns


def _fit_recent_turns(
    summary: Optional[ConversationSummary],
    events: List[ConversationEvent],
    max_context_tokens: int,
) -> List[ConversationEvent]:
    if estimate_context_tokens(summary, events) <= max_context_tokens:
        return events

    selected: List[List[ConversationEvent]] = []
    for turn in reversed(_group_complete_turns(events)):
        candidate = [event for item in reversed(selected) for event in item]
        candidate = turn + candidate
        if selected and estimate_context_tokens(summary, candidate) > max_context_tokens:
            break
        selected.append(turn)
    return [event for turn in reversed(selected) for event in turn]


def _build_deterministic_summary(
    existing_summary: Optional[ConversationSummary],
    events: Sequence[ConversationEvent],
    compacted_through: int,
) -> ConversationSummary:
    value = (
        existing_summary.to_dict()
        if existing_summary is not None
        else ConversationSummary.empty().to_dict()
    )
    value["summary_upto_sequence"] = compacted_through
    value["compaction_count"] = int(value.get("compaction_count", 0)) + 1

    for event in events:
        payload = event.payload
        if event.event_type == "user_message":
            text = str(payload.get("text", "")).strip()
            if text and any(
                marker in text.lower()
                for marker in ("目标", "希望", "想找", "优先", "准备", "target", "goal")
            ):
                _append_unique(value["user_goals"], _truncate_text(text, 300))
            continue

        if event.event_type == "assistant_message":
            _extract_assistant_memory(value, payload)
            continue

        if event.event_type == "tool_result":
            _extract_tool_memory(value, payload, event.sequence)

    return _normalize_summary(ConversationSummary.from_dict(value))


def _extract_assistant_memory(value: Dict[str, Any], payload: Dict[str, Any]) -> None:
    action = str(payload.get("action", "")).strip()
    reply = str(payload.get("reply", "")).strip()
    slots = payload.get("slots")
    slots = slots if isinstance(slots, dict) else {}
    role = str(slots.get("role", "")).strip()
    if role:
        _append_unique(value["target_roles"], _truncate_text(role, 120))

    missing_slots = payload.get("missing_slots")
    has_open_loop = bool(payload.get("need_confirmation")) or bool(missing_slots)
    if has_open_loop and reply:
        _append_unique(value["open_loops"], _truncate_text(reply, 300))
    elif action and action not in {"answer_help", "ask_clarification", "no_op"}:
        detail = f"{action}: {reply}" if reply else action
        _append_unique(value["completed_actions"], _truncate_text(detail, 300))

    if any(slots.get(key) for key in ("company", "role", "application_id")):
        _append_unique_dict(
            value["active_applications"],
            _select_fields(
                slots,
                ("application_id", "company", "role", "status", "round"),
            ),
        )


def _extract_tool_memory(
    value: Dict[str, Any],
    payload: Dict[str, Any],
    sequence: int,
) -> None:
    tool_name = str(payload.get("tool_name", "unknown_tool"))
    success = bool(payload.get("success", False))
    message = _truncate_text(str(payload.get("message", "")).strip(), 240)
    data = payload.get("data")
    data = data if isinstance(data, dict) else {}
    evidence = {
        "tool_name": tool_name,
        "success": success,
        "result_digest": _tool_result_digest(message, data),
        "source_sequences": [sequence],
    }
    _append_unique_dict(value["tool_evidence"], evidence)
    if success:
        _append_unique(
            value["confirmed_facts"],
            _truncate_text(f"{tool_name}: {evidence['result_digest']}", 360),
        )

    applications = data.get("applications")
    if isinstance(applications, list):
        for application in applications[:20]:
            if isinstance(application, dict):
                _append_unique_dict(
                    value["active_applications"],
                    _select_fields(
                        application,
                        ("id", "application_id", "company", "role", "status", "round"),
                    ),
                )
    application = data.get("application")
    if isinstance(application, dict):
        _append_unique_dict(
            value["active_applications"],
            _select_fields(
                application,
                ("id", "application_id", "company", "role", "status", "round"),
            ),
        )

    schedules = data.get("interview_schedules")
    if isinstance(schedules, list):
        for schedule in schedules[:20]:
            if isinstance(schedule, dict):
                _append_unique_dict(
                    value["upcoming_interviews"],
                    _select_fields(
                        schedule,
                        (
                            "id",
                            "schedule_id",
                            "company",
                            "role",
                            "round",
                            "interview_time",
                            "start_at",
                            "status",
                        ),
                    ),
                )

    if "project_training" in tool_name or "mock_interview" in tool_name:
        progress = _select_fields(
            data,
            ("session_id", "project_name", "status", "stage", "score", "completed"),
        )
        if progress:
            _append_unique_dict(value["training_progress"], progress)


def _tool_result_digest(message: str, data: Dict[str, Any]) -> str:
    parts = [message] if message else []
    for key, item in list(data.items())[:10]:
        if isinstance(item, list):
            parts.append(f"{key}={len(item)} item(s)")
        elif isinstance(item, dict):
            parts.append(f"{key}={_truncate_text(json.dumps(item, ensure_ascii=False), 160)}")
        elif item is not None:
            parts.append(f"{key}={_truncate_text(str(item), 100)}")
    return _truncate_text("; ".join(parts) or "tool completed", 480)


def _merge_model_summary(
    deterministic: ConversationSummary,
    model_summary: Optional[ConversationSummary],
) -> ConversationSummary:
    if model_summary is None:
        return deterministic

    value = deterministic.to_dict()
    model_value = model_summary.to_dict()
    for field in _TEXT_FIELDS:
        for item in model_value[field]:
            _append_unique(value[field], item)
    for field in _OBJECT_FIELDS:
        for item in model_value[field]:
            _append_unique_dict(value[field], item)
    value["summary_upto_sequence"] = deterministic.summary_upto_sequence
    value["compaction_count"] = deterministic.compaction_count
    value["schema_version"] = deterministic.schema_version
    return _normalize_summary(ConversationSummary.from_dict(value))


def _normalize_summary(summary: ConversationSummary) -> ConversationSummary:
    value = summary.to_dict()
    for field in _TEXT_FIELDS:
        normalized: List[str] = []
        for item in value[field]:
            _append_unique(normalized, _truncate_text(str(item).strip(), 500))
        value[field] = normalized[-_SUMMARY_LIMITS[field] :]
    for field in _OBJECT_FIELDS:
        normalized_objects: List[Dict[str, Any]] = []
        for item in value[field]:
            _append_unique_dict(
                normalized_objects,
                _limit_prompt_value(item, max_string_chars=500, max_list_items=12),
            )
        value[field] = normalized_objects[-_SUMMARY_LIMITS[field] :]
    return ConversationSummary.from_dict(value)


def _shrink_summary(
    summary: ConversationSummary,
    max_tokens: int,
) -> ConversationSummary:
    value = summary.to_dict()
    removal_order = (
        "completed_actions",
        "confirmed_facts",
        "tool_evidence",
        "active_applications",
        "training_progress",
        "decisions",
        "upcoming_interviews",
        "target_roles",
        "user_goals",
        "open_loops",
    )
    while estimate_value_tokens(value) > max_tokens:
        removed = False
        for field in removal_order:
            items = value[field]
            if len(items) > 1:
                del items[0]
                removed = True
                break
        if not removed:
            break
    return ConversationSummary.from_dict(value)


def _append_unique(items: List[str], value: str) -> None:
    normalized = value.strip()
    if normalized and normalized not in items:
        items.append(normalized)


def _append_unique_dict(items: List[Dict[str, Any]], value: Dict[str, Any]) -> None:
    if not value:
        return
    key = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    existing_keys = {
        json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        for item in items
    }
    if key not in existing_keys:
        items.append(value)


def _select_fields(value: Dict[str, Any], fields: Sequence[str]) -> Dict[str, Any]:
    return {
        field: _limit_prompt_value(value[field])
        for field in fields
        if field in value and value[field] not in (None, "", [], {})
    }


def _limit_prompt_value(
    value: Any,
    depth: int = 0,
    max_string_chars: int = 2000,
    max_list_items: int = 20,
) -> Any:
    if depth >= 8:
        return "[nested content omitted]"
    if isinstance(value, str):
        return _truncate_text(value, max_string_chars)
    if isinstance(value, list):
        return [
            _limit_prompt_value(
                item,
                depth + 1,
                max_string_chars=max_string_chars,
                max_list_items=max_list_items,
            )
            for item in value[:max_list_items]
        ]
    if isinstance(value, dict):
        return {
            str(key): _limit_prompt_value(
                item,
                depth + 1,
                max_string_chars=max_string_chars,
                max_list_items=max_list_items,
            )
            for key, item in value.items()
        }
    return value


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}...[truncated]"


def _parse_json_object(content: str) -> Dict[str, Any]:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Compaction model did not return a JSON object.")
        text = text[start : end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Compaction model response must be a JSON object.")
    return value
