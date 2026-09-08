import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol, Sequence
from zoneinfo import ZoneInfo

from app.models.interview_knowledge import KnowledgeMasteryStatus
from app.models.interview_schedule import InterviewScheduleStatus
from app.models.study import (
    StudyPlan,
    StudyPlanStatus,
    StudyPriority,
    StudySession,
    StudySessionStatus,
    StudySessionSyncStatus,
    StudyWindow,
)
from app.repositories.interview_knowledge_repository import InterviewKnowledgeRepository
from app.repositories.offerpilot_repository import OfferPilotRepository
from app.services.study_plan_service import (
    MissingStudyPreferencesError,
    StudyPlanNotFoundError,
    StudyPlanService,
    StudySessionNotFoundError,
)
from app.services.study_calendar_provider import (
    StudyCalendarAuthorizationRequired,
    StudyCalendarCredentialRefreshError,
)
from app.services.study_scheduler import BusyInterval, StudySchedulingError, StudyTopic
from app.tools.agent_tool import (
    AgentToolDefinition,
    AgentToolResult,
    FunctionAgentTool,
    ToolApproval,
    ToolEffect,
    ToolInteraction,
    ToolOutcomeStatus,
)
from app.tools.runtime_tool_inputs import RUNTIME_TOOL_INPUT_MODELS


@dataclass(frozen=True)
class CalendarSyncReceipt:
    event_id: str
    calendar_id: Optional[str] = None


@dataclass(frozen=True)
class StudyAvailabilitySnapshot:
    busy: List[BusyInterval]
    intervals: List[Dict[str, Any]]
    provider: str
    degraded: bool
    authorization_required: bool
    artifacts: List[Dict[str, Any]]

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "busy_intervals": list(self.intervals),
            "provider": self.provider,
            "degraded": self.degraded,
            "authorization_required": self.authorization_required,
        }


class StudyCalendarProvider(Protocol):
    async def list_busy(
        self,
        owner_id: str,
        start_at: datetime,
        end_at: datetime,
    ) -> Sequence[BusyInterval]: ...

    async def create_study_event(
        self,
        owner_id: str,
        session: StudySession,
        operation_key: str,
    ) -> CalendarSyncReceipt: ...

    async def update_study_event(
        self,
        owner_id: str,
        session: StudySession,
        event_id: str,
        operation_key: str,
    ) -> CalendarSyncReceipt: ...

    async def delete_study_event(
        self,
        owner_id: str,
        event_id: str,
        operation_key: str,
    ) -> CalendarSyncReceipt: ...

    def authorization_url(self, owner_id: str) -> Optional[str]: ...


def build_study_agent_tools(
    repository: OfferPilotRepository,
    knowledge_repository: Optional[InterviewKnowledgeRepository] = None,
    calendar_provider: Optional[StudyCalendarProvider] = None,
) -> List[FunctionAgentTool]:
    service = StudyPlanService(repository)
    return [
        FunctionAgentTool(
            AgentToolDefinition(
                name="get_study_preferences",
                description="读取当前用户的复习时间偏好。若尚未设置，会请求用户填写。",
                parameters=_object_schema({}),
                input_model=RUNTIME_TOOL_INPUT_MODELS["get_study_preferences"],
                cacheable=True,
                effect=ToolEffect.READ,
                approval=ToolApproval.NEVER,
                parallel_safe=True,
            ),
            _get_preferences_handler(service),
        ),
        FunctionAgentTool(
            AgentToolDefinition(
                name="save_study_preferences",
                description="保存复习时区、可用时间窗、每日上限和单次时长。",
                parameters=_preferences_schema(),
                input_model=RUNTIME_TOOL_INPUT_MODELS["save_study_preferences"],
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.IF_IMPLICIT,
                parallel_safe=False,
                idempotent=True,
            ),
            _save_preferences_handler(service),
        ),
        FunctionAgentTool(
            AgentToolDefinition(
                name="get_calendar_availability",
                description=(
                    "读取给定绝对时间范围内的本地面试、既有复习安排和用户主日历忙闲。"
                    "range_start/range_end 必须是带时区 ISO 8601 时间。"
                ),
                parameters=_object_schema(
                    {
                        "range_start": {"type": "string", "format": "date-time"},
                        "range_end": {"type": "string", "format": "date-time"},
                    },
                    ["range_start", "range_end"],
                ),
                input_model=RUNTIME_TOOL_INPUT_MODELS["get_calendar_availability"],
                cacheable=False,
                effect=ToolEffect.READ,
                approval=ToolApproval.NEVER,
                parallel_safe=True,
            ),
            _availability_handler(repository, service, calendar_provider),
        ),
        FunctionAgentTool(
            AgentToolDefinition(
                name="create_study_plan",
                description=(
                    "根据主题、时长、优先级和截止时间进行确定性排期并保存本地复习计划。"
                    "工具会在执行时重新读取本地面试、已有复习 session 和可用的用户日历忙闲；"
                    "日期和时间必须是绝对值。"
                ),
                parameters=_create_plan_schema(),
                input_model=RUNTIME_TOOL_INPUT_MODELS["create_study_plan"],
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.IF_IMPLICIT,
                parallel_safe=False,
                idempotent=False,
            ),
            _create_plan_handler(repository, service, calendar_provider),
        ),
        FunctionAgentTool(
            AgentToolDefinition(
                name="get_study_plan",
                description="按计划 ID 读取当前用户的复习计划、session 和未排期项。",
                parameters=_object_schema(
                    {"plan_id": {"type": "string", "minLength": 1}},
                    ["plan_id"],
                ),
                input_model=RUNTIME_TOOL_INPUT_MODELS["get_study_plan"],
                cacheable=False,
                effect=ToolEffect.READ,
                approval=ToolApproval.NEVER,
                parallel_safe=True,
            ),
            _get_plan_handler(service),
        ),
        FunctionAgentTool(
            AgentToolDefinition(
                name="list_study_plans",
                description="列出当前用户的复习计划，可按状态筛选。",
                parameters=_object_schema(
                    {
                        "status": {
                            "type": "string",
                            "enum": [item.value for item in StudyPlanStatus],
                        },
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    }
                ),
                input_model=RUNTIME_TOOL_INPUT_MODELS["list_study_plans"],
                cacheable=False,
                effect=ToolEffect.READ,
                approval=ToolApproval.NEVER,
                parallel_safe=True,
            ),
            _list_plans_handler(service),
        ),
        FunctionAgentTool(
            AgentToolDefinition(
                name="sync_study_plan_to_calendar",
                description="把已保存的复习计划逐 session 幂等同步到用户日历。",
                parameters=_object_schema(
                    {"plan_id": {"type": "string", "minLength": 1}},
                    ["plan_id"],
                ),
                input_model=RUNTIME_TOOL_INPUT_MODELS["sync_study_plan_to_calendar"],
                effect=ToolEffect.EXTERNAL_WRITE,
                approval=ToolApproval.ALWAYS,
                parallel_safe=False,
                idempotent=True,
                timeout_seconds=60.0,
            ),
            _sync_plan_handler(service, calendar_provider),
        ),
        FunctionAgentTool(
            AgentToolDefinition(
                name="update_study_session",
                description="更新一个复习 session 的完成状态或绝对起止时间。",
                parameters=_update_session_schema(),
                input_model=RUNTIME_TOOL_INPUT_MODELS["update_study_session"],
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.IF_IMPLICIT,
                parallel_safe=False,
                idempotent=True,
            ),
            _update_session_handler(service),
        ),
        FunctionAgentTool(
            AgentToolDefinition(
                name="reconcile_study_calendar_session",
                description=(
                    "在人工核对飞书日历后，协调结果未知的复习 session。"
                    "按原操作类型确认 created/updated/deleted 或对应的 not_* 结果。"
                ),
                parameters=_object_schema(
                    {
                        "session_id": {"type": "string", "minLength": 1},
                        "operation_key": {"type": "string", "minLength": 1},
                        "outcome": {
                            "type": "string",
                            "enum": [
                                "created",
                                "not_created",
                                "updated",
                                "not_updated",
                                "deleted",
                                "not_deleted",
                            ],
                        },
                        "event_id": {"type": "string", "minLength": 1},
                    },
                    ["session_id", "operation_key", "outcome"],
                ),
                input_model=RUNTIME_TOOL_INPUT_MODELS[
                    "reconcile_study_calendar_session"
                ],
                effect=ToolEffect.LOCAL_WRITE,
                approval=ToolApproval.ALWAYS,
                parallel_safe=False,
                idempotent=True,
            ),
            _reconcile_session_handler(service),
        ),
        FunctionAgentTool(
            AgentToolDefinition(
                name="list_learning_gaps",
                description="按掌握度列出当前用户的薄弱面试知识点及最近暴露的问题。",
                parameters=_object_schema(
                    {"limit": {"type": "integer", "minimum": 1, "maximum": 30}}
                ),
                input_model=RUNTIME_TOOL_INPUT_MODELS["list_learning_gaps"],
                cacheable=True,
                effect=ToolEffect.READ,
                approval=ToolApproval.NEVER,
                parallel_safe=True,
            ),
            _learning_gaps_handler(knowledge_repository),
        ),
    ]


def _object_schema(
    properties: Dict[str, Any], required: Optional[List[str]] = None
) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required or []),
        "additionalProperties": False,
    }


def _window_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "start_time": {
                "type": "string",
                "pattern": "^([01]\\d|2[0-3]):[0-5]\\d$",
            },
            "end_time": {
                "type": "string",
                "pattern": "^([01]\\d|2[0-3]):[0-5]\\d$",
            },
        },
        ["start_time", "end_time"],
    )


def _preferences_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "timezone": {"type": "string", "minLength": 1},
            "weekday_windows": {"type": "array", "items": _window_schema()},
            "weekend_windows": {"type": "array", "items": _window_schema()},
            "daily_max_minutes": {"type": "integer", "minimum": 15, "maximum": 720},
            "session_minutes": {"type": "integer", "minimum": 10, "maximum": 240},
        },
        [
            "timezone",
            "weekday_windows",
            "weekend_windows",
            "daily_max_minutes",
            "session_minutes",
        ],
    )


def _create_plan_schema() -> Dict[str, Any]:
    topic_schema = _object_schema(
        {
            "topic": {"type": "string", "minLength": 1},
            "duration_minutes": {"type": "integer", "minimum": 10, "maximum": 1440},
            "priority": {"type": "integer", "enum": [1, 2, 3]},
            "deadline": {"type": "string", "format": "date-time"},
            "source_refs": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        ["topic", "duration_minutes", "priority"],
    )
    return _object_schema(
        {
            "goal": {"type": "string", "minLength": 1},
            "range_start": {"type": "string", "format": "date"},
            "range_end": {"type": "string", "format": "date"},
            "topics": {"type": "array", "minItems": 1, "items": topic_schema},
            "source_interview_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        ["goal", "range_start", "range_end", "topics"],
    )


def _update_session_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "session_id": {"type": "string", "minLength": 1},
            "status": {
                "type": "string",
                "enum": [item.value for item in StudySessionStatus],
            },
            "start_at": {"type": "string", "format": "date-time"},
            "end_at": {"type": "string", "format": "date-time"},
        },
        ["session_id"],
    )


def _preference_interaction() -> ToolInteraction:
    return ToolInteraction(
        kind="preference_form",
        prompt="首次生成复习计划前，请设置你的可用时间和单次复习时长。",
        input_schema=_preferences_schema(),
        allowed_actions=["answer", "cancel"],
    )


def _get_preferences_handler(service: StudyPlanService):
    async def handler(arguments: Dict[str, Any]) -> AgentToolResult:
        preferences = await asyncio.to_thread(
            service.get_preferences, str(arguments.get("owner_id") or "local_user")
        )
        if preferences is None:
            interaction = _preference_interaction()
            return AgentToolResult(
                data={},
                status=ToolOutcomeStatus.NEEDS_INPUT,
                message=interaction.prompt,
                interaction=interaction,
            )
        return AgentToolResult(
            data={"preferences": preferences.to_dict()},
            status=ToolOutcomeStatus.SUCCESS,
            message="已读取复习偏好。",
        )

    return handler


def _save_preferences_handler(service: StudyPlanService):
    async def handler(arguments: Dict[str, Any]) -> AgentToolResult:
        try:
            preferences = await asyncio.to_thread(
                service.save_preferences,
                str(arguments.get("owner_id") or "local_user"),
                str(arguments["timezone"]),
                [_study_window(item) for item in arguments.get("weekday_windows") or []],
                [_study_window(item) for item in arguments.get("weekend_windows") or []],
                int(arguments["daily_max_minutes"]),
                int(arguments["session_minutes"]),
            )
        except (KeyError, TypeError, ValueError, StudySchedulingError) as exc:
            return _business_error("invalid_study_preferences", str(exc))
        return AgentToolResult(
            data={"preferences": preferences.to_dict()},
            status=ToolOutcomeStatus.SUCCESS,
            message="复习偏好已保存。",
        )

    return handler


def _get_plan_handler(service: StudyPlanService):
    async def handler(arguments: Dict[str, Any]) -> AgentToolResult:
        owner_id = str(arguments.get("owner_id") or "local_user")
        plan_id = str(arguments.get("plan_id") or "")
        plan = await asyncio.to_thread(service.get_plan, owner_id, plan_id)
        if plan is None:
            return _business_error(
                "study_plan_not_found",
                "Study plan was not found.",
            )
        sessions = await asyncio.to_thread(
            service.list_sessions,
            owner_id,
            plan_id,
        )
        return AgentToolResult(
            data=_study_plan_details(plan, sessions),
            status=ToolOutcomeStatus.SUCCESS,
            message="已读取复习计划。",
            artifact_refs=[_study_plan_artifact(plan, sessions)],
        )

    return handler


def _list_plans_handler(service: StudyPlanService):
    async def handler(arguments: Dict[str, Any]) -> AgentToolResult:
        owner_id = str(arguments.get("owner_id") or "local_user")
        raw_status = arguments.get("status")
        status = StudyPlanStatus(str(raw_status)) if raw_status else None
        limit = int(arguments.get("limit") or 20)
        plans = await asyncio.to_thread(service.list_plans, owner_id, status)
        plans = sorted(
            plans,
            key=lambda item: (item.created_at, item.id),
            reverse=True,
        )[:limit]
        summaries: List[Dict[str, Any]] = []
        for plan in plans:
            sessions = await asyncio.to_thread(
                service.list_sessions,
                owner_id,
                plan.id,
            )
            summaries.append(
                {
                    **plan.to_dict(),
                    "session_count": len(sessions),
                    "scheduled_minutes": sum(
                        session.duration_minutes for session in sessions
                    ),
                }
            )
        return AgentToolResult(
            data={"plans": summaries, "count": len(summaries)},
            status=ToolOutcomeStatus.SUCCESS,
            message=("已读取复习计划。" if summaries else "当前没有复习计划。"),
        )

    return handler


def _availability_handler(
    repository: OfferPilotRepository,
    service: StudyPlanService,
    calendar_provider: Optional[StudyCalendarProvider],
):
    async def handler(arguments: Dict[str, Any]) -> AgentToolResult:
        try:
            start_at = _aware_datetime(arguments["range_start"])
            end_at = _aware_datetime(arguments["range_end"])
            if end_at <= start_at:
                raise ValueError("range_end must be after range_start")
        except (KeyError, TypeError, ValueError) as exc:
            return _business_error("invalid_time_range", str(exc))

        owner_id = str(arguments.get("owner_id") or "local_user")
        availability = await _collect_availability(
            repository=repository,
            service=service,
            calendar_provider=calendar_provider,
            owner_id=owner_id,
            start_at=start_at,
            end_at=end_at,
            include_study_sessions=True,
        )
        return AgentToolResult(
            data={
                "range_start": start_at.isoformat(),
                "range_end": end_at.isoformat(),
                **availability.to_public_dict(),
            },
            status=ToolOutcomeStatus.SUCCESS,
            message=(
                "已合并本地安排；用户主日历不可用，后续将按本地忙闲排期。"
                if availability.degraded
                else "已合并本地安排和用户主日历忙闲。"
            ),
            artifact_refs=availability.artifacts,
        )

    return handler


async def _collect_availability(
    *,
    repository: OfferPilotRepository,
    service: StudyPlanService,
    calendar_provider: Optional[StudyCalendarProvider],
    owner_id: str,
    start_at: datetime,
    end_at: datetime,
    include_study_sessions: bool,
) -> StudyAvailabilitySnapshot:
    schedules_task = asyncio.to_thread(
        repository.list_interview_schedules,
        None,
        owner_id,
    )
    if include_study_sessions:
        schedules, sessions = await asyncio.gather(
            schedules_task,
            asyncio.to_thread(
                service.list_sessions,
                owner_id,
                None,
                start_at,
                end_at,
                None,
            ),
        )
    else:
        schedules = await schedules_task
        sessions = []

    busy: List[BusyInterval] = []
    intervals: List[Dict[str, Any]] = []
    for schedule in schedules:
        if schedule.status != InterviewScheduleStatus.SCHEDULED or not schedule.start_at:
            continue
        try:
            interview_start = _aware_datetime(schedule.start_at)
        except ValueError:
            continue
        interview_end = interview_start + timedelta(minutes=60)
        if interview_end <= start_at or interview_start >= end_at:
            continue
        busy.append(BusyInterval(interview_start, interview_end))
        intervals.append(
            {
                "start_at": interview_start.isoformat(),
                "end_at": interview_end.isoformat(),
                "source": "local_interview",
                "source_id": schedule.id,
            }
        )
    for session in sessions:
        if session.status in {StudySessionStatus.SKIPPED, StudySessionStatus.CANCELLED}:
            continue
        busy.append(
            BusyInterval(
                session.start_at,
                session.end_at,
                counts_toward_daily_cap=True,
            )
        )
        intervals.append(
            {
                "start_at": session.start_at.isoformat(),
                "end_at": session.end_at.isoformat(),
                "source": "local_study_session",
                "source_id": session.id,
            }
        )

    degraded = False
    authorization_required = False
    artifacts: List[Dict[str, Any]] = []
    if calendar_provider is None:
        degraded = True
    else:
        try:
            remote = await calendar_provider.list_busy(owner_id, start_at, end_at)
            for item in remote:
                busy.append(BusyInterval(item.start_at, item.end_at))
                intervals.append(
                    {
                        "start_at": item.start_at.isoformat(),
                        "end_at": item.end_at.isoformat(),
                        "source": "feishu_primary_calendar",
                    }
                )
        except StudyCalendarAuthorizationRequired as exc:
            degraded = True
            authorization_required = True
            if exc.authorization_url:
                artifacts.append(_authorization_artifact(exc.authorization_url))
        except Exception:
            degraded = True

    intervals.sort(key=lambda item: (item["start_at"], item["end_at"]))
    return StudyAvailabilitySnapshot(
        busy=busy,
        intervals=intervals,
        provider="local_only" if degraded else "local_and_feishu",
        degraded=degraded,
        authorization_required=authorization_required,
        artifacts=artifacts,
    )


def _create_plan_handler(
    repository: OfferPilotRepository,
    service: StudyPlanService,
    calendar_provider: Optional[StudyCalendarProvider],
):
    async def handler(arguments: Dict[str, Any]) -> AgentToolResult:
        try:
            owner_id = str(arguments.get("owner_id") or "local_user")
            range_start = date.fromisoformat(str(arguments["range_start"]))
            range_end = date.fromisoformat(str(arguments["range_end"]))
            preferences = await asyncio.to_thread(service.get_preferences, owner_id)
            if preferences is None:
                interaction = _preference_interaction()
                return AgentToolResult(
                    data={},
                    status=ToolOutcomeStatus.NEEDS_INPUT,
                    message=interaction.prompt,
                    interaction=interaction,
                )
            topics = [
                StudyTopic(
                    topic=str(item["topic"]),
                    duration_minutes=int(item["duration_minutes"]),
                    priority=StudyPriority(int(item["priority"])),
                    deadline=_aware_datetime(item["deadline"]) if item.get("deadline") else None,
                    source_refs=list(item.get("source_refs") or []),
                    rationale=str(item.get("rationale") or ""),
                )
                for item in arguments.get("topics") or []
            ]
            zone = ZoneInfo(preferences.timezone)
            availability = await _collect_availability(
                repository=repository,
                service=service,
                calendar_provider=calendar_provider,
                owner_id=owner_id,
                start_at=datetime.combine(range_start, datetime.min.time(), zone),
                end_at=datetime.combine(
                    range_end + timedelta(days=1), datetime.min.time(), zone
                ),
                include_study_sessions=False,
            )
            result = await asyncio.to_thread(
                service.create_plan,
                owner_id,
                str(arguments["goal"]),
                range_start,
                range_end,
                topics,
                list(arguments.get("source_interview_ids") or []),
                availability.busy,
            )
        except MissingStudyPreferencesError:
            interaction = _preference_interaction()
            return AgentToolResult(
                data={},
                status=ToolOutcomeStatus.NEEDS_INPUT,
                message=interaction.prompt,
                interaction=interaction,
            )
        except (KeyError, TypeError, ValueError, StudySchedulingError) as exc:
            return _business_error("study_plan_invalid", str(exc))

        outcome_status = (
            ToolOutcomeStatus.PARTIAL
            if result.unscheduled_items
            else ToolOutcomeStatus.SUCCESS
        )
        return AgentToolResult(
            data={
                **result.to_dict(),
                "availability": availability.to_public_dict(),
            },
            status=outcome_status,
            is_error=outcome_status == ToolOutcomeStatus.PARTIAL,
            message=(
                "复习计划已创建，但部分时长因容量不足未排入。"
                if result.unscheduled_items
                else (
                    "复习计划已创建；用户主日历不可用，已按本地面试和复习记录排期。"
                    if availability.degraded
                    else "复习计划已创建。"
                )
            ),
            artifact_refs=[
                _study_plan_artifact(result.plan, result.sessions),
                *availability.artifacts,
            ],
        )

    return handler


def _sync_plan_handler(
    service: StudyPlanService,
    calendar_provider: Optional[StudyCalendarProvider],
):
    async def handler(arguments: Dict[str, Any]) -> AgentToolResult:
        owner_id = str(arguments.get("owner_id") or "local_user")
        plan_id = str(arguments.get("plan_id") or "")
        operation_key = str(arguments.get("idempotency_key") or plan_id)
        plan = await asyncio.to_thread(service.get_plan, owner_id, plan_id)
        if plan is None:
            return AgentToolResult(
                data={
                    "plan_id": plan_id,
                    "operation_key": operation_key,
                    "error": {
                        "code": "study_plan_not_found",
                        "message": "Study plan was not found.",
                    },
                },
                is_error=True,
                status=ToolOutcomeStatus.ERROR,
                message="Study plan was not found.",
                error_code="study_plan_not_found",
            )
        if calendar_provider is None:
            return AgentToolResult(
                data={"plan_id": plan_id, "operation_key": operation_key},
                is_error=True,
                status=ToolOutcomeStatus.ERROR,
                message="用户日历尚未授权，计划保留在本地。",
                error_code="calendar_authorization_required",
            )

        sessions = await asyncio.to_thread(service.list_sessions, owner_id, plan_id)
        synced: List[Dict[str, str]] = []
        updated_events: List[Dict[str, str]] = []
        deleted_events: List[Dict[str, str]] = []
        failed: List[Dict[str, str]] = []
        uncertain: List[Dict[str, str]] = []
        authorization_required = False
        authorization_url: Optional[str] = None
        for session in sessions:
            operation_kind = (
                "delete"
                if session.status
                in {StudySessionStatus.SKIPPED, StudySessionStatus.CANCELLED}
                and session.calendar_event_id
                else ("update" if session.calendar_event_id else "create")
            )
            if session.sync_status == StudySessionSyncStatus.UNKNOWN:
                failed.append(
                    {
                        "session_id": session.id,
                        "operation": operation_kind,
                        "event_id": session.calendar_event_id,
                        "error": "previous_write_outcome_unknown_requires_reconciliation",
                    }
                )
                continue
            try:
                session_operation_key = f"{operation_key}:{session.id}"
                if session.status in {
                    StudySessionStatus.SKIPPED,
                    StudySessionStatus.CANCELLED,
                }:
                    if not session.calendar_event_id:
                        deleted_events.append(
                            {"session_id": session.id, "event_id": ""}
                        )
                        continue
                    receipt = await calendar_provider.delete_study_event(
                        owner_id,
                        session.calendar_event_id,
                        session_operation_key,
                    )
                    await asyncio.to_thread(
                        service.update_session,
                        owner_id,
                        session.id,
                        None,
                        StudySessionSyncStatus.PENDING,
                        None,
                        True,
                    )
                    deleted_events.append(
                        {"session_id": session.id, "event_id": receipt.event_id}
                    )
                    continue
                if (
                    session.sync_status == StudySessionSyncStatus.SYNCED
                    and session.calendar_event_id
                ):
                    synced.append(
                        {"session_id": session.id, "event_id": session.calendar_event_id}
                    )
                    continue
                if session.calendar_event_id:
                    receipt = await calendar_provider.update_study_event(
                        owner_id,
                        session,
                        session.calendar_event_id,
                        session_operation_key,
                    )
                    updated_events.append(
                        {"session_id": session.id, "event_id": receipt.event_id}
                    )
                else:
                    receipt = await calendar_provider.create_study_event(
                        owner_id,
                        session,
                        session_operation_key,
                    )
                await asyncio.to_thread(
                    service.update_session,
                    owner_id,
                    session.id,
                    None,
                    StudySessionSyncStatus.SYNCED,
                    receipt.event_id,
                )
                synced.append({"session_id": session.id, "event_id": receipt.event_id})
            except StudyCalendarAuthorizationRequired as exc:
                authorization_required = True
                authorization_url = exc.authorization_url
                failed.append(
                    {
                        "session_id": session.id,
                        "operation": operation_kind,
                        "event_id": session.calendar_event_id,
                        "error": "calendar_authorization_required",
                    }
                )
                break
            except (StudyCalendarCredentialRefreshError, ValueError) as exc:
                await asyncio.to_thread(
                    service.update_session,
                    owner_id,
                    session.id,
                    None,
                    StudySessionSyncStatus.FAILED,
                )
                failed.append(
                    {
                        "session_id": session.id,
                        "operation": operation_kind,
                        "event_id": session.calendar_event_id,
                        "error": str(exc)[:500] or "calendar_preflight_failed",
                    }
                )
            except Exception as exc:
                await asyncio.to_thread(
                    service.update_session,
                    owner_id,
                    session.id,
                    None,
                    StudySessionSyncStatus.UNKNOWN,
                )
                item = {
                    "session_id": session.id,
                    "operation": operation_kind,
                    "event_id": session.calendar_event_id,
                    "error": str(exc)[:500],
                }
                failed.append(item)
                uncertain.append(item)

        any_success = bool(synced or updated_events or deleted_events)
        if authorization_required and not any_success:
            status = ToolOutcomeStatus.ERROR
        elif uncertain and not any_success:
            status = ToolOutcomeStatus.UNKNOWN
        elif failed and not any_success:
            status = ToolOutcomeStatus.ERROR
        elif uncertain or failed:
            status = ToolOutcomeStatus.PARTIAL
        else:
            status = ToolOutcomeStatus.SUCCESS
        return AgentToolResult(
            data={
                "plan_id": plan_id,
                "operation_key": operation_key,
                "synced": synced,
                "updated": updated_events,
                "deleted": deleted_events,
                "failed": failed,
            },
            is_error=status != ToolOutcomeStatus.SUCCESS,
            status=status,
            message=(
                "复习计划已同步到日历。"
                if status == ToolOutcomeStatus.SUCCESS
                else (
                    "用户日历尚未授权，计划保留在本地。"
                    if authorization_required
                    else "日历同步未完全确认，请先协调失败或未知的 session，禁止盲目重试。"
                )
            ),
            error_code=(
                "calendar_authorization_required"
                if authorization_required
                else ("calendar_sync_incomplete" if failed else None)
            ),
            artifact_refs=[
                {
                    "type": "study_plan",
                    "id": plan_id,
                    "title": plan.goal,
                    "data": {
                        "synced_count": len(synced),
                        "updated_count": len(updated_events),
                        "deleted_count": len(deleted_events),
                        "failed_count": len(failed),
                    },
                },
                *(
                    [_authorization_artifact(authorization_url)]
                    if authorization_url
                    else []
                ),
            ],
        )

    return handler


def _reconcile_session_handler(service: StudyPlanService):
    async def handler(arguments: Dict[str, Any]) -> AgentToolResult:
        owner_id = str(arguments.get("owner_id") or "local_user")
        session_id = str(arguments.get("session_id") or "")
        operation_key = str(arguments.get("operation_key") or "").strip()
        outcome = str(arguments.get("outcome") or "")
        event_id = str(arguments.get("event_id") or "").strip()
        if not operation_key:
            return _business_error(
                "calendar_operation_key_required",
                "operation_key is required to reconcile a calendar write.",
            )
        allowed_outcomes = {
            "created",
            "not_created",
            "updated",
            "not_updated",
            "deleted",
            "not_deleted",
        }
        if outcome not in allowed_outcomes:
            return _business_error(
                "invalid_reconciliation_outcome",
                "outcome does not describe a supported calendar reconciliation.",
            )
        sessions = await asyncio.to_thread(service.list_sessions, owner_id)
        session = next((item for item in sessions if item.id == session_id), None)
        if session is None:
            return _business_error(
                "study_session_not_found",
                f"Study session not found: {session_id}",
            )
        if session.sync_status not in {
            StudySessionSyncStatus.PENDING,
            StudySessionSyncStatus.UNKNOWN,
            StudySessionSyncStatus.FAILED,
        }:
            return _business_error(
                "study_session_not_reconcilable",
                "Only pending, failed, or unknown calendar writes can be reconciled.",
            )
        operation_by_outcome = {
            "created": "create",
            "not_created": "create",
            "updated": "update",
            "not_updated": "update",
            "deleted": "delete",
            "not_deleted": "delete",
        }
        expected_operation = (
            "delete"
            if session.status
            in {StudySessionStatus.SKIPPED, StudySessionStatus.CANCELLED}
            and session.calendar_event_id
            else ("update" if session.calendar_event_id else "create")
        )
        if operation_by_outcome[outcome] != expected_operation:
            return _business_error(
                "calendar_reconciliation_operation_mismatch",
                f"This session requires a {expected_operation} reconciliation.",
            )
        if outcome == "created" and not event_id:
            return _business_error(
                "calendar_event_id_required",
                "event_id is required when the calendar event was confirmed created.",
            )
        successful_event_outcomes = {"created", "updated"}
        absent_event_outcomes = {"not_created", "deleted"}
        next_event_id = event_id or session.calendar_event_id
        if outcome in successful_event_outcomes and not next_event_id:
            return _business_error(
                "calendar_event_id_required",
                "A confirmed calendar event requires an event_id.",
            )
        try:
            updated = await asyncio.to_thread(
                service.update_session,
                owner_id,
                session_id,
                None,
                (
                    StudySessionSyncStatus.SYNCED
                    if outcome in successful_event_outcomes
                    else StudySessionSyncStatus.PENDING
                ),
                next_event_id,
                outcome in absent_event_outcomes,
            )
        except (StudySessionNotFoundError, StudySchedulingError, ValueError) as exc:
            return _business_error("study_session_reconciliation_failed", str(exc))
        plan = await asyncio.to_thread(service.get_plan, owner_id, updated.plan_id)
        return AgentToolResult(
            data={
                "session": updated.to_dict(),
                "plan_id": updated.plan_id,
                "plan_status": plan.status.value if plan is not None else None,
                "plan_fully_synced": bool(
                    plan is not None and plan.status.value == "synced"
                ),
                "operation_key": operation_key,
                "reconciled_outcome": outcome,
            },
            status=ToolOutcomeStatus.SUCCESS,
            message=(
                "已确认飞书日历操作成功。"
                if outcome in {"created", "updated", "deleted"}
                else "已确认飞书日历操作未生效，可以重新发起同步。"
            ),
        )

    return handler


def _update_session_handler(service: StudyPlanService):
    async def handler(arguments: Dict[str, Any]) -> AgentToolResult:
        try:
            session = await asyncio.to_thread(
                service.update_session,
                str(arguments.get("owner_id") or "local_user"),
                str(arguments["session_id"]),
                (
                    StudySessionStatus(str(arguments["status"]))
                    if arguments.get("status")
                    else None
                ),
                None,
                None,
                False,
                _aware_datetime(arguments["start_at"]) if arguments.get("start_at") else None,
                _aware_datetime(arguments["end_at"]) if arguments.get("end_at") else None,
            )
        except StudySessionNotFoundError as exc:
            return _business_error("study_session_not_found", str(exc))
        except (KeyError, TypeError, ValueError, StudySchedulingError) as exc:
            return _business_error("study_session_invalid", str(exc))
        return AgentToolResult(
            data={"session": session.to_dict()},
            status=ToolOutcomeStatus.SUCCESS,
            message="复习 session 已更新。",
        )

    return handler


def _learning_gaps_handler(
    repository: Optional[InterviewKnowledgeRepository],
):
    async def handler(arguments: Dict[str, Any]) -> AgentToolResult:
        if repository is None:
            return _business_error(
                "knowledge_repository_unavailable",
                "Knowledge progress is not available.",
            )
        owner_id = str(arguments.get("owner_id") or "local_user")
        limit = max(1, min(int(arguments.get("limit") or 10), 30))
        progress_items = await asyncio.to_thread(repository.list_progress, owner_id)
        questions = {
            item.id: item for item in await asyncio.to_thread(repository.list_questions)
        }
        candidates = sorted(
            (
                item
                for item in progress_items
                if item.mastery_status != KnowledgeMasteryStatus.MASTERED
            ),
            key=lambda item: (
                item.mastery_score,
                item.last_score if item.last_score is not None else -1,
                item.question_id,
            ),
        )[:limit]
        gaps = []
        for item in candidates:
            question = questions.get(item.question_id)
            gaps.append(
                {
                    "question_id": item.question_id,
                    "topic": question.prompt if question is not None else item.question_id,
                    "module": question.module_title if question is not None else None,
                    "mastery_status": item.mastery_status.value,
                    "mastery_score": item.mastery_score,
                    "last_score": item.last_score,
                    "detected_gaps": list(item.last_detected_gaps),
                    "next_review_on": (
                        item.next_review_on.isoformat() if item.next_review_on else None
                    ),
                }
            )
        return AgentToolResult(
            data={"learning_gaps": gaps, "count": len(gaps)},
            status=ToolOutcomeStatus.SUCCESS,
            message=f"已找到 {len(gaps)} 个薄弱知识点。",
        )

    return handler


def _study_window(value: Dict[str, Any]) -> StudyWindow:
    return StudyWindow(
        start_time=str(value["start_time"]),
        end_time=str(value["end_time"]),
    )


def _aware_datetime(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("datetime must include a timezone offset")
    return parsed


def _business_error(code: str, message: str) -> AgentToolResult:
    return AgentToolResult(
        data={"error": {"code": code, "message": message}},
        is_error=True,
        status=ToolOutcomeStatus.ERROR,
        message=message,
        error_code=code,
    )


def _study_plan_details(
    plan: StudyPlan,
    sessions: Sequence[StudySession],
) -> Dict[str, Any]:
    return {
        "plan": plan.to_dict(),
        "sessions": [session.to_dict() for session in sessions],
        "unscheduled": [item.to_dict() for item in plan.unscheduled_items],
    }


def _study_plan_artifact(
    plan: StudyPlan,
    sessions: Sequence[StudySession],
) -> Dict[str, Any]:
    compact_sessions = [
        {
            "id": session.id,
            "topic": session.topic,
            "start_at": session.start_at.isoformat(),
            "end_at": session.end_at.isoformat(),
            "duration_minutes": session.duration_minutes,
            "priority": int(session.priority),
            "status": session.status.value,
            "sync_status": session.sync_status.value,
        }
        for session in sessions
    ]
    return {
        "type": "study_plan",
        "id": plan.id,
        "title": plan.goal,
        "data": {
            "plan_id": plan.id,
            "status": plan.status.value,
            "range_start": plan.range_start.isoformat(),
            "range_end": plan.range_end.isoformat(),
            "session_count": len(compact_sessions),
            "sessions": compact_sessions,
            "unscheduled": [
                item.to_dict() for item in plan.unscheduled_items
            ],
        },
    }


def _authorization_artifact(url: str) -> Dict[str, Any]:
    return {
        "type": "authorization",
        "id": "feishu_calendar_authorization",
        "title": "授权飞书日历",
        "url": url,
        "data": {"url": url},
    }
