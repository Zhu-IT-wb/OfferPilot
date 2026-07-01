from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.models.application import (
    Application,
    ApplicationStatus,
    INTERVIEW_APPLICATION_STATUSES,
    application_passed_status_from_round,
    application_status_label,
    application_status_from_round,
)
from app.models.interview_schedule import InterviewSchedule
from app.repositories.offerpilot_repository import (
    InMemoryOfferPilotRepository,
    OfferPilotRepository,
)
from app.repositories.sqlite_offerpilot_repository import SQLiteOfferPilotRepository
from app.schemas.agent import AgentActionName
from app.schemas.tool import ToolResult
from app.services.feishu_service import (
    FeishuCalendarService,
    FeishuConfigurationError,
    FeishuRequestError,
    parse_chinese_datetime,
)
from app.tools.registry import ToolRegistry


_OFFERPILOT_CALENDAR_ID_SETTING = "feishu.offerpilot_calendar_id"


def build_default_offerpilot_repository() -> OfferPilotRepository:
    if settings.storage_backend.strip().lower() == "sqlite":
        return SQLiteOfferPilotRepository(settings.sqlite_path)
    return InMemoryOfferPilotRepository()


def build_default_feishu_calendar_service() -> Optional[FeishuCalendarService]:
    if not settings.feishu_calendar_sync_enabled:
        return None
    return FeishuCalendarService()


def build_offerpilot_tool_registry(
    repository: Optional[OfferPilotRepository] = None,
    calendar_service: Optional[FeishuCalendarService] = None,
) -> ToolRegistry:
    selected_repository = repository or build_default_offerpilot_repository()
    selected_calendar_service = calendar_service or build_default_feishu_calendar_service()
    registry = ToolRegistry()

    registry.register(
        AgentActionName.LIST_TODAY_TASKS.value,
        lambda arguments: list_today_tasks(selected_repository),
    )
    registry.register(
        AgentActionName.CREATE_APPLICATION.value,
        lambda arguments: create_application(
            selected_repository,
            arguments,
            calendar_service=selected_calendar_service,
        ),
    )
    registry.register(
        AgentActionName.QUERY_APPLICATION.value,
        lambda arguments: query_application(selected_repository, arguments),
    )
    registry.register(
        AgentActionName.UPDATE_APPLICATION.value,
        lambda arguments: update_application(
            selected_repository,
            arguments,
            calendar_service=selected_calendar_service,
        ),
    )
    registry.register(
        AgentActionName.COMPLETE_TASK.value,
        lambda arguments: complete_task(selected_repository, arguments),
    )
    registry.register(
        AgentActionName.POSTPONE_TASK.value,
        lambda arguments: postpone_task(selected_repository, arguments),
    )
    registry.register(
        AgentActionName.CREATE_INTERVIEW_REVIEW.value,
        lambda arguments: create_interview_review(selected_repository, arguments),
    )
    registry.register(
        AgentActionName.START_MOCK_INTERVIEW.value,
        start_mock_interview,
    )
    return registry


def list_today_tasks(repository: OfferPilotRepository) -> ToolResult:
    tasks = repository.list_today_tasks()
    task_data = [task.to_dict() for task in tasks]
    if not task_data:
        return ToolResult(
            tool_name=AgentActionName.LIST_TODAY_TASKS.value,
            success=True,
            message="今天暂时没有待办任务。",
            data={"tasks": []},
        )

    lines = [
        f"{index}. {task.title}（{task.task_type.value}，{task.priority.value}）"
        for index, task in enumerate(tasks, start=1)
    ]
    return ToolResult(
        tool_name=AgentActionName.LIST_TODAY_TASKS.value,
        success=True,
        message="今天的任务：\n" + "\n".join(lines),
        data={"tasks": task_data},
    )


def create_application(
    repository: OfferPilotRepository,
    arguments: Dict[str, Any],
    calendar_service: Optional[FeishuCalendarService] = None,
) -> ToolResult:
    missing = [name for name in ("company", "role") if not arguments.get(name)]
    if missing:
        return ToolResult(
            tool_name=AgentActionName.CREATE_APPLICATION.value,
            success=False,
            message=f"创建投递记录失败，缺少字段：{', '.join(missing)}。",
            data={"missing_slots": missing},
        )

    round_name = _resolve_application_round(arguments)
    application = repository.create_application(
        company=arguments["company"],
        role=arguments["role"],
        interview_time=arguments.get("interview_time"),
        round_name=round_name,
        jd_keywords=arguments.get("jd_keywords", []),
    )
    schedule = None
    if _should_create_schedule_for_created_application(arguments, round_name):
        start_at = arguments.get("start_at") or _normalize_interview_start_at(arguments.get("interview_time"))
        schedule = repository.create_interview_schedule(
            company=application.company,
            round_name=round_name or "面试",
            application_id=application.id,
            role=application.role,
            start_time=arguments.get("interview_time"),
            start_at=start_at,
            reminder_minutes=30,
            raw_message=arguments.get("raw_message", ""),
        )

    calendar_sync = _sync_interview_schedule_to_calendar(
        repository=repository,
        schedule=schedule,
        calendar_service=calendar_service,
        attendee_user_id=arguments.get("attendee_user_id"),
        attendee_user_id_type=arguments.get("attendee_user_id_type", "open_id"),
    )
    application_data = application.to_dict()
    return ToolResult(
        tool_name=AgentActionName.CREATE_APPLICATION.value,
        success=True,
        message=_format_application_created_message(application, schedule, calendar_sync),
        data={
            "application": application_data,
            "interview_schedule": schedule.to_dict() if schedule else None,
            "calendar_sync": calendar_sync,
        },
    )


def query_application(repository: OfferPilotRepository, arguments: Dict[str, Any]) -> ToolResult:
    query_type = arguments.get("query_type") or "list"
    company = arguments.get("company")
    applications = repository.list_applications(company=company)
    schedules = []

    if query_type == "upcoming_interviews":
        schedules = repository.list_interview_schedules(company=company)
        applications = [
            application
            for application in applications
            if application.interview_time or application.status in INTERVIEW_APPLICATION_STATUSES
        ]
        message = _format_upcoming_interviews(applications, schedules)
    elif company:
        message = _format_company_status(company, applications)
    else:
        message = _format_application_list(applications)

    return ToolResult(
        tool_name=AgentActionName.QUERY_APPLICATION.value,
        success=True,
        message=message,
        data={
            "query_type": query_type,
            "company": company,
            "applications": [application.to_dict() for application in applications],
            "interview_schedules": [schedule.to_dict() for schedule in schedules],
        },
    )


def update_application(
    repository: OfferPilotRepository,
    arguments: Dict[str, Any],
    calendar_service: Optional[FeishuCalendarService] = None,
) -> ToolResult:
    company = arguments.get("company")
    if not company:
        return ToolResult(
            tool_name=AgentActionName.UPDATE_APPLICATION.value,
            success=False,
            message="更新投递进度失败，缺少公司。请补充公司后再试。",
            data={"missing_slots": ["company"]},
        )

    status = _resolve_application_status(arguments)
    interview_time = arguments.get("interview_time")
    round_name = arguments.get("round")
    role = arguments.get("role")

    if status is None and interview_time is None and round_name is None and role is None:
        return ToolResult(
            tool_name=AgentActionName.UPDATE_APPLICATION.value,
            success=False,
            message="更新投递进度失败，我还没有识别到要更新的状态、轮次或时间。",
            data={},
        )

    application = repository.update_application(
        company=company,
        status=status,
        interview_time=interview_time,
        round_name=round_name,
        role=role,
    )
    if application is None:
        return ToolResult(
            tool_name=AgentActionName.UPDATE_APPLICATION.value,
            success=False,
            message=f"没有找到 {company} 的投递记录。你可以先说“我投递了{company}的某岗位”。",
            data={"company": company},
        )

    schedule = None
    if _should_create_interview_schedule(arguments, round_name):
        start_at = arguments.get("start_at") or _normalize_interview_start_at(interview_time)
        schedule = repository.create_interview_schedule(
            company=application.company,
            round_name=round_name or application.round or "面试",
            application_id=application.id,
            role=application.role,
            start_time=interview_time,
            start_at=start_at,
            reminder_minutes=30,
            raw_message=arguments.get("raw_message", ""),
        )

    calendar_sync = _sync_interview_schedule_to_calendar(
        repository=repository,
        schedule=schedule,
        calendar_service=calendar_service,
        attendee_user_id=arguments.get("attendee_user_id"),
        attendee_user_id_type=arguments.get("attendee_user_id_type", "open_id"),
    )

    return ToolResult(
        tool_name=AgentActionName.UPDATE_APPLICATION.value,
        success=True,
        message=_format_application_updated_message(application, schedule, calendar_sync),
        data={
            "application": application.to_dict(),
            "interview_schedule": schedule.to_dict() if schedule else None,
            "calendar_sync": calendar_sync,
        },
    )


def complete_task(repository: OfferPilotRepository, arguments: Dict[str, Any]) -> ToolResult:
    task = repository.complete_task(
        task_title=arguments.get("task_title"),
        task_type=arguments.get("task_type"),
    )
    if task is None:
        return ToolResult(
            tool_name=AgentActionName.COMPLETE_TASK.value,
            success=False,
            message="没有找到可以完成的匹配任务。",
            data={},
        )

    return ToolResult(
        tool_name=AgentActionName.COMPLETE_TASK.value,
        success=True,
        message=f"已记录完成：{task.title}。",
        data={"task": task.to_dict()},
    )


def postpone_task(repository: OfferPilotRepository, arguments: Dict[str, Any]) -> ToolResult:
    task = repository.postpone_task(
        task_title=arguments.get("task_title"),
        task_type=arguments.get("task_type"),
    )
    if task is None:
        return ToolResult(
            tool_name=AgentActionName.POSTPONE_TASK.value,
            success=False,
            message="没有找到可以延期的匹配任务。",
            data={},
        )

    return ToolResult(
        tool_name=AgentActionName.POSTPONE_TASK.value,
        success=True,
        message=f"已延期任务：{task.title}。",
        data={"task": task.to_dict()},
    )


def create_interview_review(repository: OfferPilotRepository, arguments: Dict[str, Any]) -> ToolResult:
    review = repository.create_interview_review(
        company=arguments.get("company"),
        round_name=arguments.get("round"),
        topics=arguments.get("topics", []),
        raw_message=arguments.get("raw_message", ""),
    )
    return ToolResult(
        tool_name=AgentActionName.CREATE_INTERVIEW_REVIEW.value,
        success=True,
        message="已创建面试复盘记录。后续会在这里接入薄弱点提取和补强任务生成。",
        data={"review": review.to_dict()},
    )


def _format_application_created_message(
    application: Application,
    schedule: Optional[InterviewSchedule] = None,
    calendar_sync: Optional[Dict[str, Any]] = None,
) -> str:
    lines = [
        "已记录投递：",
        f"公司：{application.company}",
        f"岗位：{application.role}",
        f"状态：{application_status_label(application.status)}",
    ]
    if application.round:
        lines.append(f"轮次：{application.round}")
    if application.interview_time:
        lines.append(f"时间：{application.interview_time}")
    if schedule:
        lines.append(f"已记录面试安排，默认提前 {schedule.reminder_minutes} 分钟提醒。")
        lines.extend(_format_calendar_sync_lines(calendar_sync))
    return "\n".join(lines)


def _format_application_list(applications: list[Application]) -> str:
    if not applications:
        return "目前还没有投递记录。你可以说“我投递了某公司某岗位”来创建第一条记录。"

    lines = ["当前投递记录："]
    for index, application in enumerate(applications, start=1):
        detail = f"{index}. {application.company} - {application.role}（{application_status_label(application.status)}）"
        if application.round:
            detail += f"，{application.round}"
        if application.interview_time:
            detail += f"，时间：{application.interview_time}"
        lines.append(detail)
    return "\n".join(lines)


def _format_company_status(company: str, applications: list[Application]) -> str:
    if not applications:
        return f"没有找到 {company} 的投递记录。"

    if len(applications) == 1:
        application = applications[0]
        lines = [
            f"{application.company} 当前进度：",
            f"岗位：{application.role}",
            f"状态：{application_status_label(application.status)}",
        ]
        if application.round:
            lines.append(f"当前轮次：{application.round}")
        if application.interview_time:
            lines.append(f"下一步时间：{application.interview_time}")
        return "\n".join(lines)

    return _format_application_list(applications)


def _format_upcoming_interviews(
    applications: List[Application],
    schedules: Optional[List[InterviewSchedule]] = None,
) -> str:
    if schedules:
        lines = ["近期笔试/面试安排："]
        for index, schedule in enumerate(schedules, start=1):
            role = schedule.role or "岗位待补充"
            time_text = schedule.start_time or "时间待补充"
            lines.append(
                f"{index}. {schedule.company} - {role}，{schedule.round}，{time_text}，提前 {schedule.reminder_minutes} 分钟提醒"
            )
        return "\n".join(lines)

    if not applications:
        return "目前没有记录到即将进行的笔试或面试。"

    lines = ["近期笔试/面试安排："]
    for index, application in enumerate(applications, start=1):
        round_name = application.round or application_status_label(application.status)
        time_text = application.interview_time or "时间待补充"
        lines.append(f"{index}. {application.company} - {application.role}，{round_name}，{time_text}")
    return "\n".join(lines)


def _resolve_application_status(arguments: Dict[str, Any]) -> Optional[ApplicationStatus]:
    raw_status = arguments.get("status")
    if raw_status:
        try:
            return ApplicationStatus(raw_status)
        except ValueError:
            pass

    update_type = arguments.get("update_type")
    round_name = arguments.get("round")
    if update_type == "schedule_interview":
        return application_status_from_round(round_name)
    if update_type == "pass_round":
        return application_passed_status_from_round(round_name) or application_status_from_round(round_name)
    if update_type == "reject":
        return ApplicationStatus.REJECTED
    if update_type == "offer":
        return ApplicationStatus.OFFER
    if update_type == "submitted":
        return ApplicationStatus.SUBMITTED
    if round_name:
        return application_status_from_round(round_name)
    return None


def _should_create_interview_schedule(arguments: Dict[str, Any], round_name: Optional[str]) -> bool:
    if not round_name:
        return False
    if arguments.get("update_type") == "schedule_interview":
        return True
    return bool(arguments.get("interview_time") and arguments.get("status") in INTERVIEW_APPLICATION_STATUSES)


def _resolve_application_round(arguments: Dict[str, Any]) -> Optional[str]:
    if arguments.get("round"):
        return arguments["round"]
    if _looks_like_created_application_has_interview(arguments):
        return "面试"
    return None


def _should_create_schedule_for_created_application(
    arguments: Dict[str, Any],
    round_name: Optional[str],
) -> bool:
    return bool(arguments.get("interview_time") and round_name)


def _looks_like_created_application_has_interview(arguments: Dict[str, Any]) -> bool:
    raw_message = str(arguments.get("raw_message", ""))
    compact = raw_message.replace(" ", "").lower()
    return any(
        word in compact
        for word in ("面试", "笔试", "约我", "约了", "一面", "二面", "三面", "hr面")
    )


def _sync_interview_schedule_to_calendar(
    repository: OfferPilotRepository,
    schedule: Optional[InterviewSchedule],
    calendar_service: Optional[FeishuCalendarService],
    attendee_user_id: Optional[str] = None,
    attendee_user_id_type: str = "open_id",
) -> Dict[str, Any]:
    if schedule is None:
        return {"synced": False, "status": "not_applicable"}
    if calendar_service is None:
        return {"synced": False, "status": "disabled"}
    if not calendar_service.is_calendar_sync_enabled():
        return {"synced": False, "status": "disabled"}
    if not schedule.start_at and not schedule.start_time:
        return {"synced": False, "status": "missing_start_time"}

    try:
        calendar_context = _ensure_calendar_for_sync(repository, calendar_service)
        result = calendar_service.create_interview_event(
            company=schedule.company,
            role=schedule.role,
            round_name=schedule.round,
            start_time_text=schedule.start_time,
            start_at=schedule.start_at,
            reminder_minutes=schedule.reminder_minutes,
            description=schedule.raw_message,
            calendar_id=calendar_context.get("calendar_id"),
        )
    except (FeishuConfigurationError, FeishuRequestError) as exc:
        return {
            "synced": False,
            "status": "failed",
            "error": _summarize_calendar_sync_error(str(exc)),
        }

    if result.event_id:
        repository.update_interview_schedule_calendar_event(
            schedule_id=schedule.id,
            calendar_event_id=result.event_id,
        )
        schedule.calendar_event_id = result.event_id

    attendee_sync = _sync_calendar_attendee(
        calendar_service=calendar_service,
        calendar_id=calendar_context.get("calendar_id"),
        event_id=result.event_id,
        attendee_user_id=attendee_user_id,
        attendee_user_id_type=attendee_user_id_type,
    )

    return {
        "synced": True,
        "status": "synced",
        "calendar_event_id": result.event_id,
        "attendee_sync": attendee_sync,
        **calendar_context,
    }


def _normalize_interview_start_at(interview_time: Optional[str]) -> Optional[str]:
    parsed = parse_chinese_datetime(interview_time)
    return parsed.isoformat() if parsed else None


def _format_application_updated_message(
    application: Application,
    schedule: Optional[InterviewSchedule],
    calendar_sync: Optional[Dict[str, Any]] = None,
) -> str:
    lines = [
        "已更新投递进度：",
        f"公司：{application.company}",
        f"岗位：{application.role}",
        f"状态：{application_status_label(application.status)}",
    ]
    if application.round:
        lines.append(f"轮次：{application.round}")
    if application.interview_time:
        lines.append(f"时间：{application.interview_time}")
    if schedule:
        lines.append(f"已记录面试安排，默认提前 {schedule.reminder_minutes} 分钟提醒。")
        lines.extend(_format_calendar_sync_lines(calendar_sync))
    return "\n".join(lines)


def _format_calendar_sync_lines(calendar_sync: Optional[Dict[str, Any]]) -> List[str]:
    if not calendar_sync:
        return []
    if calendar_sync.get("synced"):
        lines = []
        if calendar_sync.get("calendar_auto_created"):
            lines.append("已自动创建 OfferPilot 秋招日历。")
        lines.append("已同步到飞书日历。")
        attendee_sync = calendar_sync.get("attendee_sync")
        if isinstance(attendee_sync, dict):
            if attendee_sync.get("synced"):
                lines.append("已将你加入日程参与人。")
            elif attendee_sync.get("status") == "failed":
                lines.append(f"暂未将你加入日程参与人：{attendee_sync.get('error', 'unknown')}。")
        return lines
    if calendar_sync.get("status") == "disabled":
        return ["飞书日历同步未开启，当前只保存到 OfferPilot。"]
    if calendar_sync.get("status") == "missing_start_time":
        return ["暂未同步飞书日历：缺少可识别的面试时间。"]
    if calendar_sync.get("status") == "failed":
        return [f"飞书日历同步失败：{calendar_sync.get('error', 'unknown')}。"]
    return []


def _summarize_calendar_sync_error(error: str) -> str:
    if not error:
        return "unknown"
    first_line = error.splitlines()[0].strip()
    return first_line[:160]


def _ensure_calendar_for_sync(
    repository: OfferPilotRepository,
    calendar_service: FeishuCalendarService,
) -> Dict[str, Any]:
    should_manage = (
        calendar_service.should_manage_offerpilot_calendar()
        if hasattr(calendar_service, "should_manage_offerpilot_calendar")
        else False
    )
    if not should_manage:
        return {
            "calendar_id": getattr(calendar_service, "calendar_id", None),
            "calendar_managed": False,
            "calendar_auto_created": False,
        }

    stored_calendar_id = repository.get_runtime_setting(_OFFERPILOT_CALENDAR_ID_SETTING)
    if stored_calendar_id:
        return {
            "calendar_id": stored_calendar_id,
            "calendar_managed": True,
            "calendar_auto_created": False,
        }

    calendar_result = calendar_service.create_shared_calendar()
    if not calendar_result.calendar_id:
        raise FeishuRequestError("Feishu calendar response does not contain calendar_id.")

    repository.set_runtime_setting(
        _OFFERPILOT_CALENDAR_ID_SETTING,
        calendar_result.calendar_id,
    )
    return {
        "calendar_id": calendar_result.calendar_id,
        "calendar_managed": True,
        "calendar_auto_created": True,
    }


def _sync_calendar_attendee(
    calendar_service: FeishuCalendarService,
    calendar_id: Optional[str],
    event_id: Optional[str],
    attendee_user_id: Optional[str],
    attendee_user_id_type: str = "open_id",
) -> Dict[str, Any]:
    if not attendee_user_id:
        return {"synced": False, "status": "missing_user"}
    if not calendar_id or not event_id:
        return {"synced": False, "status": "missing_event"}
    if not hasattr(calendar_service, "add_event_attendee"):
        return {"synced": False, "status": "not_supported"}

    try:
        attendee_result = calendar_service.add_event_attendee(
            calendar_id=calendar_id,
            event_id=event_id,
            user_id=attendee_user_id,
            user_id_type=attendee_user_id_type or "open_id",
            need_notification=True,
        )
    except (FeishuConfigurationError, FeishuRequestError) as exc:
        return {
            "synced": False,
            "status": "failed",
            "error": _summarize_calendar_sync_error(str(exc)),
        }

    return {
        "synced": True,
        "status": "synced",
        "attendee_ids": attendee_result.attendee_ids,
    }


def start_mock_interview(arguments: Dict[str, Any]) -> ToolResult:
    project = arguments.get("project")
    role = arguments.get("role")
    if project:
        question = f"请先用 2 分钟介绍一下 {project}，重点说清楚你的职责、核心设计和最能体现后端能力的技术点。"
    elif role:
        question = f"我们开始 {role} 模拟面试。请先做一个 1 分钟自我介绍，并说明你最近最想展示的项目。"
    else:
        question = "我们开始模拟面试。请先选择面试方向或项目，例如 Java 后端、Agent 开发或云聚图库。"

    return ToolResult(
        tool_name=AgentActionName.START_MOCK_INTERVIEW.value,
        success=True,
        message=question,
        data={"first_question": question},
    )


_default_repository = build_default_offerpilot_repository()
_default_registry = build_offerpilot_tool_registry(_default_repository)


def get_default_tool_registry() -> ToolRegistry:
    return _default_registry
