import json
from typing import Any, Dict, List, Optional
from datetime import datetime

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
from app.services.bitable_event_subscription_service import ensure_bitable_event_subscription
from app.services.feishu_service import (
    FeishuBitableService,
    FeishuCalendarService,
    FeishuConfigurationError,
    FeishuRequestError,
    parse_chinese_datetime,
)
from app.tools.registry import ToolRegistry


_OFFERPILOT_CALENDAR_ID_SETTING = "feishu.offerpilot_calendar_id"
_OFFERPILOT_BITABLE_APP_TOKEN_SETTING = "feishu.offerpilot_bitable_app_token"
_OFFERPILOT_BITABLE_TABLE_ID_SETTING = "feishu.offerpilot_bitable_table_id"
_OFFERPILOT_BITABLE_SCHEMA_VERSION_SETTING = "feishu.offerpilot_bitable_schema_version"
_OFFERPILOT_BITABLE_RECORD_ID_PREFIX = "feishu.offerpilot_bitable_record_id."
_OFFERPILOT_BITABLE_COLLABORATOR_PREFIX = "feishu.offerpilot_bitable_collaborator."
_CURRENT_BITABLE_SCHEMA_VERSION = "v2"
_DEFAULT_EXTERNAL_SERVICE = object()


# 根据配置创建默认的 OfferPilot 数据仓库。
def build_default_offerpilot_repository() -> OfferPilotRepository:
    if settings.storage_backend.strip().lower() == "sqlite":
        return SQLiteOfferPilotRepository(settings.sqlite_path)
    return InMemoryOfferPilotRepository()


# 根据配置创建飞书日历服务实例。
def build_default_feishu_calendar_service() -> Optional[FeishuCalendarService]:
    if not settings.feishu_calendar_sync_enabled:
        return None
    return FeishuCalendarService()


# 根据配置创建飞书多维表格服务实例。
def build_default_feishu_bitable_service() -> Optional[FeishuBitableService]:
    if not settings.feishu_bitable_sync_enabled:
        return None
    return FeishuBitableService()


# 注册 OfferPilot 当前可执行的业务工具。
def build_offerpilot_tool_registry(
    repository: Optional[OfferPilotRepository] = None,
    calendar_service: Any = _DEFAULT_EXTERNAL_SERVICE,
    bitable_service: Any = _DEFAULT_EXTERNAL_SERVICE,
) -> ToolRegistry:
    selected_repository = repository or build_default_offerpilot_repository()
    selected_calendar_service = (
        build_default_feishu_calendar_service()
        if calendar_service is _DEFAULT_EXTERNAL_SERVICE
        else calendar_service
    )
    selected_bitable_service = (
        build_default_feishu_bitable_service()
        if bitable_service is _DEFAULT_EXTERNAL_SERVICE
        else bitable_service
    )
    registry = ToolRegistry()

    registry.register(
        AgentActionName.LIST_TODAY_TASKS.value,
        lambda arguments: list_today_tasks(selected_repository),
        description="查看今天的 LeetCode、八股、项目深挖和投递相关任务。",
        mutating=False,
        examples=["今天任务是什么", "查看今日任务", "/today"],
    )
    registry.register(
        AgentActionName.CREATE_APPLICATION.value,
        lambda arguments: create_application(
            selected_repository,
            arguments,
            calendar_service=selected_calendar_service,
            bitable_service=selected_bitable_service,
        ),
        description="新增一条投递记录；如果包含面试时间和轮次，也会记录面试安排并按需同步日历和多维表格。",
        mutating=True,
        required_slots=["company", "role"],
        optional_slots=[
            "round",
            "interview_time",
            "calendar_reminder",
            "attendee_user_id",
            "attendee_user_id_type",
            "bitable_collaborator_user_id",
            "bitable_collaborator_user_id_type",
        ],
        examples=["我投递了美团 Java 后端实习", "新增投递深信服 AI 应用开发"],
    )
    registry.register(
        AgentActionName.QUERY_APPLICATION.value,
        lambda arguments: query_application(selected_repository, arguments),
        description="查询投递记录、公司进度或近期笔试/面试安排。",
        mutating=False,
        optional_slots=["query_type", "company"],
        examples=["我有哪些面试", "查询投递进度", "深信服现在什么状态"],
    )
    registry.register(
        AgentActionName.UPDATE_APPLICATION.value,
        lambda arguments: update_application(
            selected_repository,
            arguments,
            calendar_service=selected_calendar_service,
            bitable_service=selected_bitable_service,
        ),
        description="更新投递进度，例如安排一面/二面、通过某轮、被拒、拿到 offer；面试安排会按需同步日历和多维表格。",
        mutating=True,
        required_slots=["company"],
        optional_slots=[
            "role",
            "round",
            "interview_time",
            "update_type",
            "status",
            "calendar_reminder",
            "attendee_user_id",
            "attendee_user_id_type",
            "bitable_collaborator_user_id",
            "bitable_collaborator_user_id_type",
        ],
        examples=["明天下午三点美团一面", "字节二面过了", "阿里给 offer 了"],
    )
    registry.register(
        AgentActionName.COMPLETE_TASK.value,
        lambda arguments: complete_task(selected_repository, arguments),
        description="把某个秋招任务标记为已完成。",
        mutating=True,
        optional_slots=["task_title", "task_type"],
        examples=["我做完反转链表了", "完成今天的八股任务"],
    )
    registry.register(
        AgentActionName.POSTPONE_TASK.value,
        lambda arguments: postpone_task(selected_repository, arguments),
        description="把某个秋招任务延期。",
        mutating=True,
        optional_slots=["task_title", "task_type"],
        examples=["这道算法题明天再做", "延期项目深挖任务"],
    )
    registry.register(
        AgentActionName.CREATE_INTERVIEW_REVIEW.value,
        lambda arguments: create_interview_review(selected_repository, arguments),
        description="记录一场面试复盘，后续用于提取薄弱点和补强任务。",
        mutating=True,
        optional_slots=["company", "round", "topics"],
        examples=["复盘今天的面试", "记录一下美团一面复盘"],
    )
    registry.register(
        AgentActionName.START_MOCK_INTERVIEW.value,
        start_mock_interview,
        description="启动一次文本模拟面试。",
        mutating=False,
        optional_slots=["project", "role"],
        examples=["开始模拟面试，项目问云聚图库", "模拟 Java 后端一面"],
    )
    return registry


# 查询并格式化今天的秋招任务。
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


# 创建投递记录，并按需同步面试日程和飞书多维表格。
def create_application(
    repository: OfferPilotRepository,
    arguments: Dict[str, Any],
    calendar_service: Optional[FeishuCalendarService] = None,
    bitable_service: Optional[FeishuBitableService] = None,
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
    bitable_sync = _sync_application_to_bitable(
        repository=repository,
        application=application,
        schedule=schedule,
        calendar_sync=calendar_sync,
        bitable_service=bitable_service,
        collaborator_user_id=arguments.get("bitable_collaborator_user_id") or arguments.get("attendee_user_id"),
        collaborator_user_id_type=arguments.get("bitable_collaborator_user_id_type")
        or arguments.get("attendee_user_id_type", "open_id"),
    )
    application_data = application.to_dict()
    return ToolResult(
        tool_name=AgentActionName.CREATE_APPLICATION.value,
        success=True,
        message=_format_application_created_message(
            application,
            schedule,
            calendar_sync,
            bitable_sync,
        ),
        data={
            "application": application_data,
            "interview_schedule": schedule.to_dict() if schedule else None,
            "calendar_sync": calendar_sync,
            "bitable_sync": bitable_sync,
        },
    )


# 查询投递记录或近期面试安排。
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


# 更新投递进度，并按需同步日历和多维表格。
def update_application(
    repository: OfferPilotRepository,
    arguments: Dict[str, Any],
    calendar_service: Optional[FeishuCalendarService] = None,
    bitable_service: Optional[FeishuBitableService] = None,
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

    if schedule and arguments.get("calendar_reminder") is False:
        calendar_sync = {"synced": False, "status": "skipped_by_user"}
    else:
        calendar_sync = _sync_interview_schedule_to_calendar(
            repository=repository,
            schedule=schedule,
            calendar_service=calendar_service,
            attendee_user_id=arguments.get("attendee_user_id"),
            attendee_user_id_type=arguments.get("attendee_user_id_type", "open_id"),
        )
    bitable_sync = _sync_application_to_bitable(
        repository=repository,
        application=application,
        schedule=schedule,
        calendar_sync=calendar_sync,
        bitable_service=bitable_service,
        collaborator_user_id=arguments.get("bitable_collaborator_user_id") or arguments.get("attendee_user_id"),
        collaborator_user_id_type=arguments.get("bitable_collaborator_user_id_type")
        or arguments.get("attendee_user_id_type", "open_id"),
    )

    return ToolResult(
        tool_name=AgentActionName.UPDATE_APPLICATION.value,
        success=True,
        message=_format_application_updated_message(
            application,
            schedule,
            calendar_sync,
            bitable_sync,
        ),
        data={
            "application": application.to_dict(),
            "interview_schedule": schedule.to_dict() if schedule else None,
            "calendar_sync": calendar_sync,
            "bitable_sync": bitable_sync,
        },
    )


# 将指定任务标记为已完成。
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


# 将指定任务延期处理。
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


# 记录一次面试复盘。
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


# 格式化 application created message。
def _format_application_created_message(
    application: Application,
    schedule: Optional[InterviewSchedule] = None,
    calendar_sync: Optional[Dict[str, Any]] = None,
    bitable_sync: Optional[Dict[str, Any]] = None,
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
        if calendar_sync and calendar_sync.get("status") == "skipped_by_user":
            lines.append("已记录面试安排，未添加飞书日历提醒。")
        elif calendar_sync and calendar_sync.get("synced"):
            lines.append(f"已记录面试安排，默认提前 {schedule.reminder_minutes} 分钟提醒。")
        else:
            lines.append("已记录面试安排。")
        lines.extend(_format_calendar_sync_lines(calendar_sync))
    lines.extend(_format_bitable_sync_lines(bitable_sync))
    return "\n".join(lines)


# 格式化 application list。
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


# 格式化 company status。
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


# 格式化 upcoming interviews。
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


# 解析并确定 application status。
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


# 判断是否需要 create interview schedule。
def _should_create_interview_schedule(arguments: Dict[str, Any], round_name: Optional[str]) -> bool:
    if not round_name:
        return False
    if arguments.get("update_type") == "schedule_interview":
        return True
    return bool(arguments.get("interview_time") and arguments.get("status") in INTERVIEW_APPLICATION_STATUSES)


# 解析并确定 application round。
def _resolve_application_round(arguments: Dict[str, Any]) -> Optional[str]:
    if arguments.get("round"):
        return arguments["round"]
    if _looks_like_created_application_has_interview(arguments):
        return "面试"
    return None


# 判断是否需要 create schedule for created application。
def _should_create_schedule_for_created_application(
    arguments: Dict[str, Any],
    round_name: Optional[str],
) -> bool:
    return bool(arguments.get("interview_time") and round_name)


# 判断输入是否像 created application has interview。
def _looks_like_created_application_has_interview(arguments: Dict[str, Any]) -> bool:
    raw_message = str(arguments.get("raw_message", ""))
    compact = raw_message.replace(" ", "").lower()
    return any(
        word in compact
        for word in ("面试", "笔试", "约我", "约了", "一面", "二面", "三面", "hr面")
    )


# 同步 interview schedule to calendar。
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


# 标准化 interview start at。
def _normalize_interview_start_at(interview_time: Optional[str]) -> Optional[str]:
    parsed = parse_chinese_datetime(interview_time)
    return parsed.isoformat() if parsed else None


# 格式化 application updated message。
def _format_application_updated_message(
    application: Application,
    schedule: Optional[InterviewSchedule],
    calendar_sync: Optional[Dict[str, Any]] = None,
    bitable_sync: Optional[Dict[str, Any]] = None,
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
        if calendar_sync and calendar_sync.get("status") == "skipped_by_user":
            lines.append("已记录面试安排，未添加飞书日历提醒。")
        else:
            lines.append(f"已记录面试安排，默认提前 {schedule.reminder_minutes} 分钟提醒。")
        lines.extend(_format_calendar_sync_lines(calendar_sync))
    lines.extend(_format_bitable_sync_lines(bitable_sync))
    return "\n".join(lines)


# 格式化 calendar sync lines。
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
    if calendar_sync.get("status") == "skipped_by_user":
        return []
    if calendar_sync.get("status") == "missing_start_time":
        return ["暂未同步飞书日历：缺少可识别的面试时间。"]
    if calendar_sync.get("status") == "failed":
        return [f"飞书日历同步失败：{calendar_sync.get('error', 'unknown')}。"]
    return []


# 压缩并说明 calendar sync error。
def _summarize_calendar_sync_error(error: str) -> str:
    if not error:
        return "unknown"
    first_line = error.splitlines()[0].strip()
    return first_line[:160]


# 同步 application to bitable。
def _sync_application_to_bitable(
    repository: OfferPilotRepository,
    application: Application,
    schedule: Optional[InterviewSchedule],
    calendar_sync: Optional[Dict[str, Any]],
    bitable_service: Optional[FeishuBitableService],
    collaborator_user_id: Optional[str] = None,
    collaborator_user_id_type: str = "open_id",
) -> Dict[str, Any]:
    if bitable_service is None:
        return {"synced": False, "status": "disabled"}
    if not bitable_service.is_bitable_sync_enabled():
        return {"synced": False, "status": "disabled"}

    try:
        bitable_context = _ensure_bitable_for_sync(repository, bitable_service)
        fields = _build_application_bitable_fields(
            application=application,
            schedule=schedule,
            calendar_sync=calendar_sync,
        )
        record_setting_key = _bitable_record_setting_key(
            table_id=bitable_context["table_id"],
            application_id=application.id,
        )
        existing_record_id = repository.get_runtime_setting(record_setting_key)

        event_subscription = None
        if existing_record_id:
            result = bitable_service.update_record(
                app_token=bitable_context["app_token"],
                table_id=bitable_context["table_id"],
                record_id=existing_record_id,
                fields=fields,
            )
            record_id = result.record_id or existing_record_id
            operation = "updated"
        else:
            result = bitable_service.create_record(
                app_token=bitable_context["app_token"],
                table_id=bitable_context["table_id"],
                fields=fields,
            )
            if not result.record_id:
                raise FeishuRequestError("Feishu bitable record response does not contain record_id.")
            record_id = result.record_id
            repository.set_runtime_setting(record_setting_key, record_id)
            operation = "created"
        event_subscription = ensure_bitable_event_subscription(
            repository=repository,
            bitable_service=bitable_service,
        )
        collaborator_sync = _sync_bitable_collaborator(
            repository=repository,
            bitable_service=bitable_service,
            app_token=bitable_context["app_token"],
            user_id=collaborator_user_id,
            user_id_type=collaborator_user_id_type,
        )
    except (FeishuConfigurationError, FeishuRequestError) as exc:
        return {
            "synced": False,
            "status": "failed",
            "error": _summarize_bitable_collaborator_error(str(exc)),
        }

    return {
        "synced": True,
        "status": "synced",
        "operation": operation,
        "record_id": record_id,
        "event_subscription": event_subscription.__dict__ if event_subscription else None,
        "collaborator_sync": collaborator_sync,
        "web_url": _build_bitable_web_url(
            app_token=bitable_context["app_token"],
            table_id=bitable_context["table_id"],
        ),
        **bitable_context,
    }


# 确保 bitable for sync 存在或已配置。
def _ensure_bitable_for_sync(
    repository: OfferPilotRepository,
    bitable_service: FeishuBitableService,
) -> Dict[str, Any]:
    should_manage = (
        bitable_service.should_manage_offerpilot_bitable()
        if hasattr(bitable_service, "should_manage_offerpilot_bitable")
        else False
    )
    if not should_manage:
        return {
            "app_token": getattr(bitable_service, "app_token", None),
            "table_id": getattr(bitable_service, "table_id", None),
            "bitable_managed": False,
            "bitable_auto_created": False,
        }

    stored_app_token = repository.get_runtime_setting(_OFFERPILOT_BITABLE_APP_TOKEN_SETTING)
    stored_table_id = repository.get_runtime_setting(_OFFERPILOT_BITABLE_TABLE_ID_SETTING)
    stored_schema_version = repository.get_runtime_setting(_OFFERPILOT_BITABLE_SCHEMA_VERSION_SETTING)

    app_token = stored_app_token
    auto_created = False
    if not app_token:
        app_result = bitable_service.create_app()
        if not app_result.app_token:
            raise FeishuRequestError("Feishu bitable app response does not contain app_token.")
        app_token = app_result.app_token
        repository.set_runtime_setting(_OFFERPILOT_BITABLE_APP_TOKEN_SETTING, app_token)
        auto_created = True

    table_id = stored_table_id
    if not table_id or stored_schema_version != _CURRENT_BITABLE_SCHEMA_VERSION:
        table_name = getattr(bitable_service, "table_name", None) or settings.feishu_offerpilot_bitable_table_name
        if table_id and stored_schema_version != _CURRENT_BITABLE_SCHEMA_VERSION:
            table_name = f"{table_name} Pro"
        table_result = bitable_service.create_application_table(
            app_token=app_token,
            table_name=table_name,
        )
        if not table_result.table_id:
            raise FeishuRequestError("Feishu bitable table response does not contain table_id.")
        table_id = table_result.table_id
        repository.set_runtime_setting(_OFFERPILOT_BITABLE_TABLE_ID_SETTING, table_id)
        repository.set_runtime_setting(
            _OFFERPILOT_BITABLE_SCHEMA_VERSION_SETTING,
            _CURRENT_BITABLE_SCHEMA_VERSION,
        )
        auto_created = True

    return {
        "app_token": app_token,
        "table_id": table_id,
        "bitable_managed": True,
        "bitable_auto_created": auto_created,
    }


# 构造 application bitable fields。
def _build_application_bitable_fields(
    application: Application,
    schedule: Optional[InterviewSchedule],
    calendar_sync: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    calendar_event_id = application.to_dict().get("calendar_event_id")
    if schedule and schedule.calendar_event_id:
        calendar_event_id = schedule.calendar_event_id
    elif calendar_sync and calendar_sync.get("calendar_event_id"):
        calendar_event_id = calendar_sync["calendar_event_id"]

    fields: Dict[str, Any] = {
        "OfferPilot记录ID": application.id,
        "公司": application.company,
        "岗位": application.role,
        "投递状态": application_status_label(application.status),
        "状态值": application.status.value,
        "优先级": _infer_application_priority(application),
        "来源": "飞书助手",
        "下一步": _build_application_next_action(application, schedule),
        "最后同步时间": _datetime_to_bitable_timestamp_ms(datetime.now()),
    }
    round_name = _normalize_bitable_round(application.round)
    if round_name:
        fields["面试轮次"] = round_name
    if application.interview_time:
        fields["面试时间文本"] = application.interview_time
    if schedule and schedule.start_at:
        start_timestamp = _iso_datetime_to_bitable_timestamp_ms(schedule.start_at)
        if start_timestamp is not None:
            fields["面试开始时间"] = start_timestamp
    if schedule:
        fields["提醒分钟"] = schedule.reminder_minutes
    if calendar_event_id:
        fields["日历事件ID"] = calendar_event_id
    if application.jd_keywords:
        fields["JD关键词"] = list(application.jd_keywords)
    return fields


# 格式化 bitable sync lines。
def _format_bitable_sync_lines(bitable_sync: Optional[Dict[str, Any]]) -> List[str]:
    if not bitable_sync:
        return []
    if bitable_sync.get("synced"):
        lines = []
        if bitable_sync.get("bitable_auto_created"):
            lines.append("已自动创建 OfferPilot 秋招投递多维表格。")
        lines.append("已同步到飞书多维表格。")
        collaborator_sync = bitable_sync.get("collaborator_sync")
        if isinstance(collaborator_sync, dict):
            if collaborator_sync.get("synced"):
                lines.append("已授予你飞书多维表格编辑权限。")
            elif collaborator_sync.get("status") == "failed":
                lines.append(f"暂未授予多维表格编辑权限：{collaborator_sync.get('error', 'unknown')}。")
        if bitable_sync.get("web_url"):
            lines.append(f"打开多维表格：{bitable_sync['web_url']}")
        return lines
    if bitable_sync.get("status") == "failed":
        return [f"飞书多维表格同步失败：{bitable_sync.get('error', 'unknown')}。"]
    return []


# 压缩并说明 sync error。
def _summarize_sync_error(error: str) -> str:
    if not error:
        return "unknown"
    first_line = error.splitlines()[0].strip()
    return first_line[:160]


# 处理 bitable_record_setting_key 相关逻辑。
def _bitable_record_setting_key(table_id: str, application_id: str) -> str:
    return f"{_OFFERPILOT_BITABLE_RECORD_ID_PREFIX}{table_id}.{application_id}"


# 构造 bitable web url。
def _build_bitable_web_url(app_token: str, table_id: str) -> str:
    base_url = settings.feishu_bitable_web_base_url.rstrip("/")
    if not app_token:
        return ""
    if table_id:
        return f"{base_url}/{app_token}?table={table_id}"
    return f"{base_url}/{app_token}"


# 同步 bitable collaborator。
def _sync_bitable_collaborator(
    repository: OfferPilotRepository,
    bitable_service: FeishuBitableService,
    app_token: str,
    user_id: Optional[str],
    user_id_type: str = "open_id",
) -> Dict[str, Any]:
    if not user_id:
        return {"synced": False, "status": "missing_user"}
    if not app_token:
        return {"synced": False, "status": "missing_app"}
    if not hasattr(bitable_service, "add_bitable_collaborator"):
        return {"synced": False, "status": "not_supported"}

    setting_key = _bitable_collaborator_setting_key(app_token=app_token, user_id=user_id)
    if repository.get_runtime_setting(setting_key):
        return {"synced": True, "status": "cached"}

    try:
        result = bitable_service.add_bitable_collaborator(
            app_token=app_token,
            member_id=user_id,
            member_id_type=user_id_type or "open_id",
            perm="edit",
            need_notification=False,
        )
    except (FeishuConfigurationError, FeishuRequestError) as exc:
        return {
            "synced": False,
            "status": "failed",
            "error": _summarize_sync_error(str(exc)),
        }

    repository.set_runtime_setting(setting_key, result.member_id or user_id)
    return {
        "synced": True,
        "status": "synced",
        "member_id": result.member_id or user_id,
    }


# 处理 bitable_collaborator_setting_key 相关逻辑。
def _bitable_collaborator_setting_key(app_token: str, user_id: str) -> str:
    return f"{_OFFERPILOT_BITABLE_COLLABORATOR_PREFIX}{app_token}.{user_id}"


# 压缩并说明 bitable collaborator error。
def _summarize_bitable_collaborator_error(error: str) -> str:
    parsed = _extract_feishu_error_payload(error)
    if parsed and parsed.get("code") == 99991672:
        return (
            "缺少云文档协作者授权权限。请在飞书开放平台的“应用身份权限”里开通 "
            "drive:drive 或 drive:file，或云文档/文档/表格相关编辑权限，并发布版本后重试。"
        )
    return _summarize_sync_error(error)


# 从输入数据中提取 feishu error payload。
def _extract_feishu_error_payload(error: str) -> Optional[Dict[str, Any]]:
    if not error:
        return None
    json_start = error.find("{")
    if json_start < 0:
        return None
    try:
        payload = json.loads(error[json_start:])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


# 处理 iso_datetime_to_bitable_timestamp_ms 相关逻辑。
def _iso_datetime_to_bitable_timestamp_ms(value: str) -> Optional[int]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _datetime_to_bitable_timestamp_ms(parsed)


# 处理 datetime_to_bitable_timestamp_ms 相关逻辑。
def _datetime_to_bitable_timestamp_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


# 标准化 bitable round。
def _normalize_bitable_round(round_name: Optional[str]) -> Optional[str]:
    if not round_name:
        return None
    normalized = round_name.replace(" ", "").lower()
    mapping = {
        "笔试": "笔试",
        "一面": "一面",
        "二面": "二面",
        "三面": "三面",
        "hr": "HR 面",
        "hr面": "HR 面",
        "面试": "面试",
    }
    return mapping.get(normalized, round_name)


# 处理 infer_application_priority 相关逻辑。
def _infer_application_priority(application: Application) -> str:
    text = f"{application.company}{application.role}".lower()
    if any(word in text for word in ("腾讯", "阿里", "字节", "美团", "快手", "百度", "deepseek")):
        return "高"
    if any(word in text for word in ("ai", "agent", "java", "后端")):
        return "中"
    return "中"


# 构造 application next action。
def _build_application_next_action(
    application: Application,
    schedule: Optional[InterviewSchedule],
) -> str:
    if application.status == ApplicationStatus.PLANNED:
        return "确认投递信息，补充投递渠道或 JD 关键词"
    if application.status == ApplicationStatus.SUBMITTED:
        return "等待反馈，超过 7 天可跟进"
    if application.status in INTERVIEW_APPLICATION_STATUSES:
        if schedule and schedule.start_time:
            return f"准备 {application.round or schedule.round}，时间：{schedule.start_time}"
        return f"准备 {application.round or '面试/笔试'}"
    if application.status == ApplicationStatus.OFFER:
        return "整理 Offer 信息，准备横向比较"
    if application.status in {ApplicationStatus.REJECTED, ApplicationStatus.WITHDRAWN}:
        return "归档并复盘原因"
    return "持续跟进"


# 确保 calendar for sync 存在或已配置。
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


# 同步 calendar attendee。
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


# 生成模拟面试的启动回复。
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


# 返回全局默认工具注册表。
def get_default_tool_registry() -> ToolRegistry:
    return _default_registry


# 返回全局默认数据仓库实例。
def get_default_offerpilot_repository() -> OfferPilotRepository:
    return _default_repository
