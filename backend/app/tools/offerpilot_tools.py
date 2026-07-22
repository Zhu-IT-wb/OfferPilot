import hashlib
import json
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

from app.core.config import settings
from app.models.application import (
    Application,
    ApplicationStatus,
    INTERVIEW_APPLICATION_STATUSES,
    application_passed_status_from_round,
    application_status_label,
    application_status_from_round,
)
from app.models.interview_schedule import InterviewSchedule, InterviewScheduleStatus
from app.models.leetcode import (
    LeetCodePracticeResult,
    LeetCodeRecommendation,
    LeetCodeSubscription,
)
from app.repositories.leetcode_repository import InMemoryLeetCodeRepository, LeetCodeRepository
from app.repositories.offerpilot_repository import (
    InMemoryOfferPilotRepository,
    OfferPilotRepository,
)
from app.repositories.sqlite_offerpilot_repository import SQLiteOfferPilotRepository
from app.repositories.sqlite_leetcode_repository import SQLiteLeetCodeRepository
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
from app.services.leetcode_catalog import load_hot100_snapshot
from app.services.leetcode_messages import format_leetcode_recommendations
from app.services.leetcode_recommendation import LeetCodeRecommendationWorkflow
from app.tools.registry import ToolRegistry


_OFFERPILOT_CALENDAR_ID_SETTING = "feishu.offerpilot_calendar_id"
_OFFERPILOT_BITABLE_APP_TOKEN_SETTING = "feishu.offerpilot_bitable_app_token"
_OFFERPILOT_BITABLE_TABLE_ID_SETTING = "feishu.offerpilot_bitable_table_id"
_OFFERPILOT_BITABLE_SCHEMA_VERSION_SETTING = "feishu.offerpilot_bitable_schema_version"
_OFFERPILOT_BITABLE_RECORD_ID_PREFIX = "feishu.offerpilot_bitable_record_id."
_OFFERPILOT_BITABLE_COLLABORATOR_PREFIX = "feishu.offerpilot_bitable_collaborator."
_OFFERPILOT_SYNC_IDEMPOTENCY_PREFIX = "sync.idempotency."
_OFFERPILOT_REQUEST_IDEMPOTENCY_PREFIX = "request.idempotency."
_CURRENT_BITABLE_SCHEMA_VERSION = "v3"
_SYNC_MAX_ATTEMPTS = 3
_DEFAULT_EXTERNAL_SERVICE = object()


# 根据配置创建默认的 OfferPilot 数据仓库。
def build_default_offerpilot_repository() -> OfferPilotRepository:
    if settings.storage_backend.strip().lower() == "sqlite":
        return SQLiteOfferPilotRepository(settings.sqlite_path)
    return InMemoryOfferPilotRepository()


def build_default_leetcode_repository() -> LeetCodeRepository:
    if settings.storage_backend.strip().lower() == "sqlite":
        return SQLiteLeetCodeRepository(settings.sqlite_path)
    return InMemoryLeetCodeRepository(problems=load_hot100_snapshot().problems)


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
    leetcode_repository: Optional[LeetCodeRepository] = None,
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
    selected_leetcode_repository = leetcode_repository or build_default_leetcode_repository()
    registry = ToolRegistry()

    registry.register(
        AgentActionName.LIST_TODAY_TASKS.value,
        lambda arguments: list_today_tasks(
            selected_repository,
            arguments,
            leetcode_repository=selected_leetcode_repository,
        ),
        description="查看今天的 LeetCode、八股、项目深挖和投递相关任务。",
        mutating=False,
        examples=["今天任务是什么", "查看今日任务", "/today"],
    )
    registry.register(
        AgentActionName.ENABLE_LEETCODE_PLAN.value,
        lambda arguments: enable_leetcode_plan(selected_leetcode_repository, arguments),
        description="开启每天 09:00 的 LeetCode 推荐和 21:00 未反馈提醒。",
        mutating=True,
        examples=["开启每日刷题", "开启 LeetCode 计划"],
    )
    registry.register(
        AgentActionName.DISABLE_LEETCODE_PLAN.value,
        lambda arguments: disable_leetcode_plan(selected_leetcode_repository, arguments),
        description="关闭 LeetCode 主动推送，保留历史训练进度。",
        mutating=True,
        examples=["关闭每日刷题", "停止 LeetCode 推送"],
    )
    registry.register(
        AgentActionName.GET_TODAY_LEETCODE.value,
        lambda arguments: get_today_leetcode(selected_leetcode_repository, arguments),
        description="读取或生成今天的三道 LeetCode 推荐题。",
        mutating=False,
        examples=["今天刷什么", "查看今天的力扣题"],
    )
    registry.register(
        AgentActionName.RECORD_LEETCODE_RESULT.value,
        lambda arguments: record_leetcode_result(selected_leetcode_repository, arguments),
        description="记录明确的 LeetCode 训练结果并计算下次复习日期。",
        mutating=True,
        required_slots=["result"],
        optional_slots=["assignment_id", "problem_index", "problem_title"],
        examples=["第1题独立完成", "LRU 看题解完成", "第2题没做出来"],
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
            "idempotency_key",
            "retry_after_reconciliation",
        ],
        examples=["明天下午三点美团一面", "字节二面过了", "阿里给 offer 了"],
    )
    registry.register(
        AgentActionName.RESCHEDULE_INTERVIEW.value,
        lambda arguments: reschedule_interview(
            selected_repository,
            arguments,
            calendar_service=selected_calendar_service,
            bitable_service=selected_bitable_service,
        ),
        description="按 application_id 或 schedule_id 定位已有面试安排并修改时间；匹配不唯一时返回候选项。",
        mutating=True,
        required_slots=["interview_time"],
        optional_slots=["application_id", "schedule_id", "company", "role", "round", "start_at", "idempotency_key"],
        examples=["把美团一面改到后天下午四点", "将 schedule_1 改到下周一上午十点"],
    )
    registry.register(
        AgentActionName.CANCEL_INTERVIEW.value,
        lambda arguments: cancel_interview(
            selected_repository,
            arguments,
            calendar_service=selected_calendar_service,
            bitable_service=selected_bitable_service,
        ),
        description="按 application_id 或 schedule_id 定位已有面试安排并取消；匹配不唯一时返回候选项。",
        mutating=True,
        optional_slots=["application_id", "schedule_id", "company", "role", "round", "idempotency_key"],
        examples=["取消美团一面", "取消 schedule_1"],
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
def list_today_tasks(
    repository: OfferPilotRepository,
    arguments: Optional[Dict[str, Any]] = None,
    leetcode_repository: Optional[LeetCodeRepository] = None,
) -> ToolResult:
    owner_id = _owner_id_from_arguments(arguments or {})
    tasks = repository.list_today_tasks(owner_id=owner_id)
    task_data = [task.to_dict() for task in tasks]
    recommendations = (
        LeetCodeRecommendationWorkflow(leetcode_repository).get_today(
            owner_id=owner_id,
            today=_today_in_shanghai(),
        )
        if leetcode_repository is not None
        else []
    )
    recommendation_data = [_recommendation_to_dict(item) for item in recommendations]
    if not task_data and not recommendation_data:
        return ToolResult(
            tool_name=AgentActionName.LIST_TODAY_TASKS.value,
            success=True,
            message="今天暂时没有待办任务。",
            data={"tasks": [], "leetcode_recommendations": []},
        )

    sections = []
    if recommendations:
        sections.append(format_leetcode_recommendations(recommendations))
    if tasks:
        lines = [
            f"{index}. {task.title}（{task.task_type.value}，{task.priority.value}）"
            for index, task in enumerate(tasks, start=1)
        ]
        sections.append("其他秋招任务：\n" + "\n".join(lines))
    return ToolResult(
        tool_name=AgentActionName.LIST_TODAY_TASKS.value,
        success=True,
        message="今天的任务：\n\n" + "\n\n".join(sections),
        data={
            "tasks": task_data,
            "leetcode_recommendations": recommendation_data,
        },
    )


def enable_leetcode_plan(
    repository: LeetCodeRepository,
    arguments: Dict[str, Any],
) -> ToolResult:
    owner_id = _owner_id_from_arguments(arguments)
    feishu_open_id = str(arguments.get("attendee_user_id") or "").strip()
    if not feishu_open_id:
        return ToolResult(
            tool_name=AgentActionName.ENABLE_LEETCODE_PLAN.value,
            success=False,
            message="请在飞书中发送“开启每日刷题”，这样我才能保存推送目标。",
            data={"missing_slots": ["attendee_user_id"]},
        )
    repository.save_subscription(
        LeetCodeSubscription(owner_id=owner_id, feishu_open_id=feishu_open_id, enabled=True)
    )
    recommendations = LeetCodeRecommendationWorkflow(repository).get_today(
        owner_id=owner_id,
        today=_today_in_shanghai(),
    )
    return ToolResult(
        tool_name=AgentActionName.ENABLE_LEETCODE_PLAN.value,
        success=True,
        message=(
            "已开启每日刷题：09:00 推送，21:00 提醒未反馈题目。\n\n"
            + format_leetcode_recommendations(recommendations)
        ),
        data={"recommendations": [_recommendation_to_dict(item) for item in recommendations]},
    )


def disable_leetcode_plan(
    repository: LeetCodeRepository,
    arguments: Dict[str, Any],
) -> ToolResult:
    owner_id = _owner_id_from_arguments(arguments)
    subscription = repository.get_subscription(owner_id)
    if subscription is None:
        return ToolResult(
            tool_name=AgentActionName.DISABLE_LEETCODE_PLAN.value,
            success=True,
            message="每日刷题推送当前未开启。",
            data={"enabled": False},
        )
    subscription.enabled = False
    repository.save_subscription(subscription)
    return ToolResult(
        tool_name=AgentActionName.DISABLE_LEETCODE_PLAN.value,
        success=True,
        message="已关闭 LeetCode 主动推送，历史训练进度会继续保留。",
        data={"enabled": False},
    )


def get_today_leetcode(
    repository: LeetCodeRepository,
    arguments: Dict[str, Any],
) -> ToolResult:
    recommendations = LeetCodeRecommendationWorkflow(repository).get_today(
        owner_id=_owner_id_from_arguments(arguments),
        today=_today_in_shanghai(),
    )
    return ToolResult(
        tool_name=AgentActionName.GET_TODAY_LEETCODE.value,
        success=True,
        message=format_leetcode_recommendations(recommendations),
        data={"recommendations": [_recommendation_to_dict(item) for item in recommendations]},
    )


def record_leetcode_result(
    repository: LeetCodeRepository,
    arguments: Dict[str, Any],
) -> ToolResult:
    owner_id = _owner_id_from_arguments(arguments)
    today = _today_in_shanghai()
    workflow = LeetCodeRecommendationWorkflow(repository)
    recommendations = workflow.get_today(owner_id=owner_id, today=today)
    recommendation = _resolve_leetcode_assignment(recommendations, arguments)
    if recommendation is None:
        return ToolResult(
            tool_name=AgentActionName.RECORD_LEETCODE_RESULT.value,
            success=False,
            message="今天有多道 LeetCode 任务，请说明第几题或题目名称。",
            data={
                "missing_slots": ["problem_index"],
                "candidates": [_recommendation_to_dict(item) for item in recommendations],
            },
        )
    try:
        result = LeetCodePracticeResult(str(arguments.get("result") or ""))
    except ValueError:
        return ToolResult(
            tool_name=AgentActionName.RECORD_LEETCODE_RESULT.value,
            success=False,
            message="请说明结果：独立完成、提示后完成、看题解完成、未完成、延期或跳过。",
            data={"missing_slots": ["result"]},
        )
    try:
        feedback = workflow.record_result(
            owner_id=owner_id,
            assignment_id=recommendation.assignment.id,
            result=result,
            practiced_on=today,
        )
    except ValueError:
        return ToolResult(
            tool_name=AgentActionName.RECORD_LEETCODE_RESULT.value,
            success=False,
            message="这道题今天已经记录了其他结果；第一版暂不支持覆盖，请明天继续按计划反馈。",
            data={"assignment": recommendation.assignment.to_dict()},
        )
    next_review = feedback.progress.next_review_on if feedback.progress else None
    message = (
        f"已记录：{recommendation.problem.frontend_id}. {recommendation.problem.title_zh}，"
        f"{_LEETCODE_RESULT_LABELS[result]}。"
    )
    if next_review:
        message += f" 下次复习：{next_review.isoformat()}。"
    elif feedback.progress and feedback.progress.mastery_status.value == "mastered":
        message += " 已通过三次间隔验证，标记为已掌握。"
    return ToolResult(
        tool_name=AgentActionName.RECORD_LEETCODE_RESULT.value,
        success=True,
        message=message,
        data={
            "assignment": feedback.assignment.to_dict(),
            "progress": feedback.progress.to_dict() if feedback.progress else None,
            "problem": recommendation.problem.to_dict(),
        },
    )


_LEETCODE_RESULT_LABELS = {
    LeetCodePracticeResult.INDEPENDENT: "独立完成",
    LeetCodePracticeResult.WITH_HINT: "提示后完成",
    LeetCodePracticeResult.WITH_SOLUTION: "看题解完成",
    LeetCodePracticeResult.FAILED: "尝试但未完成",
    LeetCodePracticeResult.POSTPONED: "延期",
    LeetCodePracticeResult.SKIPPED: "跳过",
}


def _today_in_shanghai() -> date:
    return datetime.now(ZoneInfo("Asia/Shanghai")).date()


def _resolve_leetcode_assignment(
    recommendations: List[LeetCodeRecommendation],
    arguments: Dict[str, Any],
) -> Optional[LeetCodeRecommendation]:
    assignment_id = str(arguments.get("assignment_id") or "").strip()
    if assignment_id:
        return next((item for item in recommendations if item.assignment.id == assignment_id), None)
    problem_index = arguments.get("problem_index")
    if isinstance(problem_index, str) and problem_index.isdigit():
        problem_index = int(problem_index)
    if isinstance(problem_index, int) and 1 <= problem_index <= len(recommendations):
        return recommendations[problem_index - 1]
    title = _normalize_match_value(str(arguments.get("problem_title") or ""))
    if title:
        matches = [
            item
            for item in recommendations
            if title in _normalize_match_value(item.problem.title_zh)
            or title in _normalize_match_value(item.problem.title_en)
        ]
        return matches[0] if len(matches) == 1 else None
    return recommendations[0] if len(recommendations) == 1 else None


def _recommendation_to_dict(item: LeetCodeRecommendation) -> Dict[str, Any]:
    return {
        "assignment": item.assignment.to_dict(),
        "problem": item.problem.to_dict(),
    }


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
    owner_id = _owner_id_from_arguments(arguments)
    application = repository.create_application(
        company=arguments["company"],
        role=arguments["role"],
        interview_time=arguments.get("interview_time"),
        round_name=round_name,
        jd_keywords=arguments.get("jd_keywords", []),
        owner_id=owner_id,
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
            owner_id=owner_id,
        )

    calendar_sync = _sync_interview_schedule_to_calendar(
        repository=repository,
        schedule=schedule,
        calendar_service=calendar_service,
        owner_id=owner_id,
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
    owner_id = _owner_id_from_arguments(arguments)
    query_type = arguments.get("query_type") or "list"
    company = arguments.get("company")
    applications = repository.list_applications(company=company, owner_id=owner_id)
    schedules = []

    if query_type == "upcoming_interviews":
        schedules = repository.list_interview_schedules(company=company, owner_id=owner_id)
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
    application_id = arguments.get("application_id")
    if not company and not application_id:
        return ToolResult(
            tool_name=AgentActionName.UPDATE_APPLICATION.value,
            success=False,
            message="更新投递进度失败，缺少 application_id 或公司。请补充后再试。",
            data={"missing_slots": ["application_id", "company"]},
        )

    status = _resolve_application_status(arguments)
    interview_time = arguments.get("interview_time")
    round_name = arguments.get("round")
    role = arguments.get("role")
    owner_id = _owner_id_from_arguments(arguments)

    if status is None and interview_time is None and round_name is None and role is None:
        return ToolResult(
            tool_name=AgentActionName.UPDATE_APPLICATION.value,
            success=False,
            message="更新投递进度失败，我还没有识别到要更新的状态、轮次或时间。",
            data={},
        )

    if arguments.get("apply_to_all"):
        if not company:
            return ToolResult(
                tool_name=AgentActionName.UPDATE_APPLICATION.value,
                success=False,
                message="批量更新需要提供公司。",
                data={"missing_slots": ["company"]},
            )
        applications = repository.list_applications(company=company, owner_id=owner_id)
        if not applications:
            return ToolResult(
                tool_name=AgentActionName.UPDATE_APPLICATION.value,
                success=False,
                message=f"没有找到 {company} 的投递记录。你可以先说“我投递了{company}的某岗位”。",
                data={"company": company},
            )

        updated_applications = []
        bitable_sync_results = []
        for application_to_update in applications:
            updated_application = repository.update_application_by_id(
                application_id=application_to_update.id,
                status=status,
                interview_time=interview_time,
                round_name=round_name,
                role=role,
                owner_id=owner_id,
            )
            if updated_application is None:
                continue
            updated_applications.append(updated_application)
            bitable_sync_results.append(
                _sync_application_to_bitable(
                    repository=repository,
                    application=updated_application,
                    schedule=None,
                    calendar_sync=None,
                    bitable_service=bitable_service,
                    collaborator_user_id=arguments.get("bitable_collaborator_user_id")
                    or arguments.get("attendee_user_id"),
                    collaborator_user_id_type=arguments.get("bitable_collaborator_user_id_type")
                    or arguments.get("attendee_user_id_type", "open_id"),
                )
            )

        return ToolResult(
            tool_name=AgentActionName.UPDATE_APPLICATION.value,
            success=True,
            message=_format_applications_bulk_updated_message(
                company=company,
                applications=updated_applications,
                bitable_sync_results=bitable_sync_results,
            ),
            data={
                "applications": [application.to_dict() for application in updated_applications],
                "bitable_sync_results": bitable_sync_results,
            },
        )

    candidates = _find_application_candidates(repository, arguments, owner_id=owner_id)
    selection_result = _application_selection_result(
        tool_name=AgentActionName.UPDATE_APPLICATION.value,
        candidates=candidates,
        locator=company or application_id or "指定条件",
    )
    if selection_result is not None:
        return selection_result

    should_create_schedule = _should_create_interview_schedule(arguments, round_name)
    normalized_start_at = (
        arguments.get("start_at") or _normalize_interview_start_at(interview_time)
        if should_create_schedule
        else None
    )
    request_idempotency_key = arguments.get("idempotency_key")
    idempotency_conflict = _claim_request_idempotency_key(
        repository=repository,
        namespace=f"update_application:{owner_id}",
        idempotency_key=request_idempotency_key,
        payload={
            "application_id": candidates[0].id,
            "status": status.value if status else None,
            "round": round_name,
            "start_at": normalized_start_at,
            "role": role,
        },
    )
    if idempotency_conflict:
        return ToolResult(
            tool_name=AgentActionName.UPDATE_APPLICATION.value,
            success=False,
            message=idempotency_conflict,
            data={"idempotency_conflict": True, "application_id": candidates[0].id},
        )

    application = repository.update_application_by_id(
        application_id=candidates[0].id,
        status=status,
        interview_time=interview_time,
        round_name=round_name,
        role=role,
        owner_id=owner_id,
    )
    if application is None:
        return ToolResult(
            tool_name=AgentActionName.UPDATE_APPLICATION.value,
            success=False,
            message=f"没有找到 {company} 的投递记录。你可以先说“我投递了{company}的某岗位”。",
            data={"company": company},
        )

    schedule = None
    if should_create_schedule:
        schedule = _find_existing_interview_schedule(
            repository=repository,
            application_id=application.id,
            round_name=round_name or application.round or "面试",
            start_at=normalized_start_at,
            start_time=interview_time,
            owner_id=owner_id,
        )
        if schedule is None:
            schedule = repository.create_interview_schedule(
                company=application.company,
                round_name=round_name or application.round or "面试",
                application_id=application.id,
                role=application.role,
                start_time=interview_time,
                start_at=normalized_start_at,
                reminder_minutes=30,
                raw_message=arguments.get("raw_message", ""),
                owner_id=owner_id,
            )

    operation_key = (
        f"schedule:{owner_id}:{schedule.id}:{schedule.start_at or schedule.start_time}:"
        f"{request_idempotency_key or 'derived'}"
        if schedule is not None
        else None
    )
    if operation_key and arguments.get("retry_after_reconciliation") is True:
        _clear_sync_state(repository, f"{operation_key}:calendar")
        _clear_sync_state(repository, f"{operation_key}:bitable")

    if schedule and arguments.get("calendar_reminder") is False:
        calendar_sync = {"synced": False, "status": "skipped_by_user"}
    else:
        calendar_sync = _sync_interview_schedule_to_calendar(
            repository=repository,
            schedule=schedule,
            calendar_service=calendar_service,
            owner_id=owner_id,
            attendee_user_id=arguments.get("attendee_user_id"),
            attendee_user_id_type=arguments.get("attendee_user_id_type", "open_id"),
            idempotency_key=f"{operation_key}:calendar" if operation_key else None,
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
        idempotency_key=f"{operation_key}:bitable" if operation_key else None,
    )
    operation_status = _external_write_status(calendar_sync, bitable_sync)

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
            "operation_status": operation_status,
            "retryable": operation_status == "partial_success",
            "idempotency_key": operation_key,
        },
    )


# 修改已有面试安排，不创建新的 schedule。
def reschedule_interview(
    repository: OfferPilotRepository,
    arguments: Dict[str, Any],
    calendar_service: Optional[FeishuCalendarService] = None,
    bitable_service: Optional[FeishuBitableService] = None,
) -> ToolResult:
    interview_time = arguments.get("interview_time")
    if not interview_time:
        return ToolResult(
            tool_name=AgentActionName.RESCHEDULE_INTERVIEW.value,
            success=False,
            message="面试改期失败，缺少新的具体面试时间。",
            data={"missing_slots": ["interview_time"]},
        )

    owner_id = _owner_id_from_arguments(arguments)
    candidates = _find_schedule_candidates(repository, arguments, owner_id=owner_id)
    selection_result = _schedule_selection_result(
        tool_name=AgentActionName.RESCHEDULE_INTERVIEW.value,
        candidates=candidates,
        locator=arguments.get("company") or arguments.get("application_id") or arguments.get("schedule_id") or "指定条件",
    )
    if selection_result is not None:
        return selection_result

    schedule = candidates[0]
    normalized_start_at = _normalize_interview_start_at(interview_time)
    explicit_start_at = arguments.get("start_at")
    if not normalized_start_at or (
        explicit_start_at
        and not _iso_datetimes_match(normalized_start_at, str(explicit_start_at))
    ):
        return ToolResult(
            tool_name=AgentActionName.RESCHEDULE_INTERVIEW.value,
            success=False,
            message="面试改期失败，新的面试时间无法解析。请提供包含日期和具体时刻的时间。",
            data={"missing_slots": ["interview_time"], "schedule_id": schedule.id},
        )
    start_at = normalized_start_at
    request_idempotency_key = arguments.get("idempotency_key")
    idempotency_conflict = _claim_request_idempotency_key(
        repository=repository,
        namespace=f"reschedule:{owner_id}",
        idempotency_key=request_idempotency_key,
        payload={
            "schedule_id": schedule.id,
            "start_at": start_at,
            "round": arguments.get("round") or schedule.round,
        },
    )
    if idempotency_conflict:
        return ToolResult(
            tool_name=AgentActionName.RESCHEDULE_INTERVIEW.value,
            success=False,
            message=idempotency_conflict,
            data={"idempotency_conflict": True, "schedule_id": schedule.id},
        )
    updated_schedule = repository.update_interview_schedule(
        schedule_id=schedule.id,
        start_time=interview_time,
        start_at=start_at,
        round_name=arguments.get("round"),
        owner_id=owner_id,
    )
    if updated_schedule is None:
        return ToolResult(
            tool_name=AgentActionName.RESCHEDULE_INTERVIEW.value,
            success=False,
            message="面试改期失败，目标安排已不存在。",
            data={"schedule_id": schedule.id},
        )

    application = _application_for_schedule(repository, updated_schedule, owner_id=owner_id)
    if application is not None:
        application = repository.update_application_by_id(
            application_id=application.id,
            interview_time=interview_time,
            round_name=arguments.get("round") or updated_schedule.round,
            owner_id=owner_id,
        )

    operation_key = (
        f"reschedule:{owner_id}:{updated_schedule.id}:{updated_schedule.start_at}:"
        f"{request_idempotency_key or 'derived'}"
    )
    calendar_sync = _sync_rescheduled_interview_to_calendar(
        repository=repository,
        schedule=updated_schedule,
        calendar_service=calendar_service,
        owner_id=owner_id,
        idempotency_key=f"{operation_key}:calendar",
    )
    bitable_sync = (
        _sync_application_to_bitable(
            repository=repository,
            application=application,
            schedule=updated_schedule,
            calendar_sync=calendar_sync,
            bitable_service=bitable_service,
            idempotency_key=(
                f"{operation_key}:bitable:"
                f"{calendar_sync.get('calendar_event_id') or updated_schedule.calendar_event_id or 'none'}"
            ),
        )
        if application is not None
        else {"synced": False, "status": "not_applicable", "attempts": 0}
    )
    operation_status = _external_write_status(calendar_sync, bitable_sync)

    return ToolResult(
        tool_name=AgentActionName.RESCHEDULE_INTERVIEW.value,
        success=True,
        message=(
            f"已将 {updated_schedule.company} {updated_schedule.round} 改期到 {updated_schedule.start_time}。"
            + _external_write_summary(operation_status)
        ),
        data={
            "application": application.to_dict() if application else None,
            "interview_schedule": updated_schedule.to_dict(),
            "calendar_sync": calendar_sync,
            "bitable_sync": bitable_sync,
            "operation_status": operation_status,
            "retryable": operation_status == "partial_success",
            "idempotency_key": operation_key,
        },
    )


# 取消已有面试安排，不删除历史记录。
def cancel_interview(
    repository: OfferPilotRepository,
    arguments: Dict[str, Any],
    calendar_service: Optional[FeishuCalendarService] = None,
    bitable_service: Optional[FeishuBitableService] = None,
) -> ToolResult:
    owner_id = _owner_id_from_arguments(arguments)
    candidates = _find_schedule_candidates(
        repository,
        arguments,
        owner_id=owner_id,
        include_cancelled_by_id=True,
    )
    selection_result = _schedule_selection_result(
        tool_name=AgentActionName.CANCEL_INTERVIEW.value,
        candidates=candidates,
        locator=arguments.get("company") or arguments.get("application_id") or arguments.get("schedule_id") or "指定条件",
    )
    if selection_result is not None:
        return selection_result

    selected_schedule = candidates[0]
    request_idempotency_key = arguments.get("idempotency_key")
    idempotency_conflict = _claim_request_idempotency_key(
        repository=repository,
        namespace=f"cancel:{owner_id}",
        idempotency_key=request_idempotency_key,
        payload={"schedule_id": selected_schedule.id},
    )
    if idempotency_conflict:
        return ToolResult(
            tool_name=AgentActionName.CANCEL_INTERVIEW.value,
            success=False,
            message=idempotency_conflict,
            data={"idempotency_conflict": True, "schedule_id": selected_schedule.id},
        )

    schedule = repository.cancel_interview_schedule(selected_schedule.id, owner_id=owner_id)
    if schedule is None:
        return ToolResult(
            tool_name=AgentActionName.CANCEL_INTERVIEW.value,
            success=False,
            message="取消面试失败，目标安排已不存在。",
            data={"schedule_id": candidates[0].id},
        )

    application = _application_for_schedule(repository, schedule, owner_id=owner_id)
    next_schedule: Optional[InterviewSchedule] = None
    if application is not None:
        remaining_schedules = [
            candidate
            for candidate in repository.list_interview_schedules(owner_id=owner_id)
            if candidate.application_id == application.id
            and candidate.status == InterviewScheduleStatus.SCHEDULED
        ]
        if remaining_schedules:
            next_schedule = min(remaining_schedules, key=_interview_schedule_sort_key)
            application = repository.update_application_by_id(
                application_id=application.id,
                status=application_status_from_round(next_schedule.round),
                interview_time=next_schedule.start_time,
                round_name=next_schedule.round,
                owner_id=owner_id,
            )
        else:
            application = repository.update_application_by_id(
                application_id=application.id,
                status=ApplicationStatus.SUBMITTED,
                clear_interview_fields=True,
                owner_id=owner_id,
            )
    operation_key = (
        f"cancel:{owner_id}:{schedule.id}:"
        f"{request_idempotency_key or 'derived'}"
    )
    calendar_sync = _sync_cancelled_interview_to_calendar(
        repository=repository,
        schedule=schedule,
        calendar_service=calendar_service,
        owner_id=owner_id,
        idempotency_key=f"{operation_key}:calendar",
    )
    bitable_sync = (
        _sync_application_to_bitable(
            repository=repository,
            application=application,
            schedule=next_schedule,
            calendar_sync=calendar_sync,
            bitable_service=bitable_service,
            idempotency_key=(
                f"{operation_key}:bitable:"
                f"{calendar_sync.get('calendar_event_id') or 'none'}"
            ),
            clear_interview_fields=next_schedule is None,
        )
        if application is not None
        else {"synced": False, "status": "not_applicable", "attempts": 0}
    )
    operation_status = _external_write_status(calendar_sync, bitable_sync)
    return ToolResult(
        tool_name=AgentActionName.CANCEL_INTERVIEW.value,
        success=True,
        message=(
            f"已取消 {schedule.company} {schedule.round}（原时间：{schedule.start_time or '未记录'}）。"
            + _external_write_summary(operation_status)
        ),
        data={
            "application": application.to_dict() if application else None,
            "interview_schedule": schedule.to_dict(),
            "calendar_sync": calendar_sync,
            "bitable_sync": bitable_sync,
            "operation_status": operation_status,
            "retryable": operation_status == "partial_success",
            "idempotency_key": operation_key,
        },
    )


# 按稳定 ID 或可读字段筛选投递候选；不在这里默认选择任意一条。
def _find_application_candidates(
    repository: OfferPilotRepository,
    arguments: Dict[str, Any],
    owner_id: str,
) -> List[Application]:
    application_id = arguments.get("application_id")
    if application_id:
        return [
            application
            for application in repository.list_applications(owner_id=owner_id)
            if application.id == application_id
        ]

    applications = repository.list_applications(
        company=arguments.get("company"),
        owner_id=owner_id,
    )
    role = arguments.get("role")
    if role:
        normalized_role = _normalize_match_value(str(role))
        applications = [
            application
            for application in applications
            if normalized_role in _normalize_match_value(application.role)
        ]
    return applications


# 按稳定 ID 或投递字段筛选仍有效的面试安排候选。
def _find_schedule_candidates(
    repository: OfferPilotRepository,
    arguments: Dict[str, Any],
    owner_id: str,
    include_cancelled_by_id: bool = False,
) -> List[InterviewSchedule]:
    schedule_id = arguments.get("schedule_id")
    application_id = arguments.get("application_id")
    schedules = repository.list_interview_schedules(owner_id=owner_id)
    if schedule_id:
        return [
            schedule
            for schedule in schedules
            if schedule.id == schedule_id
            and (
                schedule.status == InterviewScheduleStatus.SCHEDULED
                or include_cancelled_by_id
            )
        ]

    if application_id:
        schedules = [schedule for schedule in schedules if schedule.application_id == application_id]
    elif arguments.get("company"):
        normalized_company = _normalize_match_value(str(arguments["company"]))
        schedules = [
            schedule
            for schedule in schedules
            if normalized_company in _normalize_match_value(schedule.company)
        ]
    role = arguments.get("role")
    round_name = arguments.get("round")
    if role:
        normalized_role = _normalize_match_value(str(role))
        schedules = [
            schedule
            for schedule in schedules
            if schedule.role and normalized_role in _normalize_match_value(schedule.role)
        ]
    if round_name:
        normalized_round = _normalize_match_value(str(round_name))
        schedules = [
            schedule
            for schedule in schedules
            if _normalize_match_value(schedule.round) == normalized_round
        ]
    return [schedule for schedule in schedules if schedule.status == InterviewScheduleStatus.SCHEDULED]


# 在零匹配或多匹配时返回明确结果；唯一匹配返回 None 继续执行。
def _application_selection_result(
    tool_name: str,
    candidates: List[Application],
    locator: str,
) -> Optional[ToolResult]:
    if not candidates:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            message=f"没有找到与 {locator} 匹配的投递记录。",
            data={"candidates": []},
        )
    if len(candidates) == 1:
        return None

    candidate_data = [
        {
            "application_id": application.id,
            "company": application.company,
            "role": application.role,
            "status": application.status.value,
            "round": application.round,
            "interview_time": application.interview_time,
        }
        for application in candidates
    ]
    lines = [
        f"{index}. {item['application_id']}｜{item['company']}｜{item['role']}｜{item['round'] or '未记录轮次'}"
        for index, item in enumerate(candidate_data, start=1)
    ]
    return ToolResult(
        tool_name=tool_name,
        success=False,
        message="找到多条投递记录，请回复 application_id 选择后再执行：\n" + "\n".join(lines),
        data={
            "requires_selection": True,
            "selection_slot": "application_id",
            "candidates": candidate_data,
        },
    )


# 在零匹配或多匹配时返回明确结果；唯一匹配返回 None 继续执行。
def _schedule_selection_result(
    tool_name: str,
    candidates: List[InterviewSchedule],
    locator: str,
) -> Optional[ToolResult]:
    if not candidates:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            message=f"没有找到与 {locator} 匹配的有效面试安排。",
            data={"candidates": []},
        )
    if len(candidates) == 1:
        return None

    candidate_data = [
        {
            "schedule_id": schedule.id,
            "application_id": schedule.application_id,
            "company": schedule.company,
            "role": schedule.role,
            "round": schedule.round,
            "interview_time": schedule.start_time,
        }
        for schedule in candidates
    ]
    lines = [
        f"{index}. {item['schedule_id']}｜{item['company']}｜{item['role'] or '岗位未记录'}｜"
        f"{item['round']}｜{item['interview_time'] or '时间未记录'}"
        for index, item in enumerate(candidate_data, start=1)
    ]
    return ToolResult(
        tool_name=tool_name,
        success=False,
        message="找到多条面试安排，请回复 schedule_id 选择后再执行：\n" + "\n".join(lines),
        data={
            "requires_selection": True,
            "selection_slot": "schedule_id",
            "candidates": candidate_data,
        },
    )


# 获取面试安排关联的投递记录。
def _application_for_schedule(
    repository: OfferPilotRepository,
    schedule: InterviewSchedule,
    owner_id: str,
) -> Optional[Application]:
    if not schedule.application_id:
        return None
    return next(
        (
            application
            for application in repository.list_applications(owner_id=owner_id)
            if application.id == schedule.application_id
        ),
        None,
    )


# 标准化候选匹配字段。
def _normalize_match_value(value: str) -> str:
    return "".join(value.split()).lower()


# 确保模型或调用方提供的标准时间没有背离用户表达，允许一分钟内的序列化误差。
def _iso_datetimes_match(first: str, second: str) -> bool:
    try:
        first_value = datetime.fromisoformat(first.replace("Z", "+00:00"))
        second_value = datetime.fromisoformat(second.replace("Z", "+00:00"))
        return abs(first_value.timestamp() - second_value.timestamp()) <= 60
    except (TypeError, ValueError):
        return False


# 优先选择时间上最早的有效面试安排，ID 仅作为相同时间的稳定次序。
def _interview_schedule_sort_key(schedule: InterviewSchedule) -> tuple[float, str]:
    parsed: Optional[datetime] = None
    if schedule.start_at:
        try:
            parsed = datetime.fromisoformat(schedule.start_at.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
    if parsed is None and schedule.start_time:
        parsed = parse_chinese_datetime(schedule.start_time)
    timestamp = parsed.timestamp() if parsed is not None else float("inf")
    return timestamp, schedule.id


# 将指定任务标记为已完成。
def complete_task(repository: OfferPilotRepository, arguments: Dict[str, Any]) -> ToolResult:
    owner_id = _owner_id_from_arguments(arguments)
    task = repository.complete_task(
        task_title=arguments.get("task_title"),
        task_type=arguments.get("task_type"),
        owner_id=owner_id,
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
    owner_id = _owner_id_from_arguments(arguments)
    task = repository.postpone_task(
        task_title=arguments.get("task_title"),
        task_type=arguments.get("task_type"),
        owner_id=owner_id,
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
    owner_id = _owner_id_from_arguments(arguments)
    review = repository.create_interview_review(
        company=arguments.get("company"),
        round_name=arguments.get("round"),
        topics=arguments.get("topics", []),
        raw_message=arguments.get("raw_message", ""),
        owner_id=owner_id,
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
    if update_type == "withdraw":
        return ApplicationStatus.WITHDRAWN
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


# 查找同一投递下相同轮次与时间的有效安排，避免确认重放创建重复记录。
def _find_existing_interview_schedule(
    repository: OfferPilotRepository,
    application_id: str,
    round_name: str,
    start_at: Optional[str],
    start_time: Optional[str],
    owner_id: str,
) -> Optional[InterviewSchedule]:
    for schedule in repository.list_interview_schedules(owner_id=owner_id):
        if schedule.status != InterviewScheduleStatus.SCHEDULED:
            continue
        if schedule.application_id != application_id or schedule.round != round_name:
            continue
        if start_at and schedule.start_at == start_at:
            return schedule
        if not start_at and schedule.start_time == start_time:
            return schedule
    return None


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
    owner_id: str = "local_user",
    attendee_user_id: Optional[str] = None,
    attendee_user_id_type: str = "open_id",
    idempotency_key: Optional[str] = None,
) -> Dict[str, Any]:
    if schedule is None:
        return {"synced": False, "status": "not_applicable"}
    if calendar_service is None:
        return {"synced": False, "status": "disabled"}
    if not calendar_service.is_calendar_sync_enabled():
        return {"synced": False, "status": "disabled"}
    if not schedule.start_at and not schedule.start_time:
        return {"synced": False, "status": "missing_start_time"}

    if idempotency_key:
        cached = _load_sync_state(repository, idempotency_key)
        if cached is not None and (
            cached.get("synced") or cached.get("reconciliation_required")
        ):
            return {**cached, "attempts": 0, "idempotent_replay": True}

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
        failure_result = {
            "synced": False,
            "status": "failed",
            "attempts": 1,
            "reconciliation_required": True,
            "error": _summarize_calendar_sync_error(str(exc)),
        }
        if idempotency_key:
            failure_result["idempotency_key"] = idempotency_key
            _store_sync_result(repository, idempotency_key, failure_result)
        return failure_result

    if result.event_id:
        repository.update_interview_schedule_calendar_event(
            schedule_id=schedule.id,
            calendar_event_id=result.event_id,
            owner_id=owner_id,
        )
        schedule.calendar_event_id = result.event_id

    attendee_sync = _sync_calendar_attendee(
        calendar_service=calendar_service,
        calendar_id=calendar_context.get("calendar_id"),
        event_id=result.event_id,
        attendee_user_id=attendee_user_id,
        attendee_user_id_type=attendee_user_id_type,
    )

    sync_result = {
        "synced": True,
        "status": "synced",
        "attempts": 1,
        "calendar_event_id": result.event_id,
        "attendee_sync": attendee_sync,
        **calendar_context,
    }
    if idempotency_key:
        sync_result["idempotency_key"] = idempotency_key
        _store_sync_result(repository, idempotency_key, sync_result)
    return sync_result


# 更新已有日程；不存在远端事件时补建，并对成功步骤进行幂等缓存。
def _sync_rescheduled_interview_to_calendar(
    repository: OfferPilotRepository,
    schedule: InterviewSchedule,
    calendar_service: Optional[FeishuCalendarService],
    owner_id: str,
    idempotency_key: str,
) -> Dict[str, Any]:
    if calendar_service is None or not calendar_service.is_calendar_sync_enabled():
        return {"synced": False, "status": "disabled", "attempts": 0}

    def operation() -> Dict[str, Any]:
        calendar_context = _ensure_calendar_for_sync(repository, calendar_service)
        calendar_id = calendar_context.get("calendar_id")
        if schedule.calendar_event_id:
            result = calendar_service.update_interview_event(
                calendar_id=calendar_id,
                event_id=schedule.calendar_event_id,
                company=schedule.company,
                role=schedule.role,
                round_name=schedule.round,
                start_time_text=schedule.start_time,
                start_at=schedule.start_at,
                reminder_minutes=schedule.reminder_minutes,
                description=schedule.raw_message,
            )
            operation_name = "updated"
        else:
            result = calendar_service.create_interview_event(
                company=schedule.company,
                role=schedule.role,
                round_name=schedule.round,
                start_time_text=schedule.start_time,
                start_at=schedule.start_at,
                reminder_minutes=schedule.reminder_minutes,
                description=schedule.raw_message,
                calendar_id=calendar_id,
            )
            operation_name = "created"
        if not result.event_id:
            raise FeishuRequestError("Feishu calendar response does not contain event_id.")
        repository.update_interview_schedule_calendar_event(
            schedule_id=schedule.id,
            calendar_event_id=result.event_id,
            owner_id=owner_id,
        )
        schedule.calendar_event_id = result.event_id
        return {
            "synced": True,
            "status": "synced",
            "operation": operation_name,
            "calendar_event_id": result.event_id,
            **calendar_context,
        }

    return _run_idempotent_sync(
        repository=repository,
        idempotency_key=idempotency_key,
        operation=operation,
        error_formatter=_summarize_calendar_sync_error,
        max_attempts=_SYNC_MAX_ATTEMPTS if schedule.calendar_event_id else 1,
        reconciliation_required_on_failure=not bool(schedule.calendar_event_id),
    )


# 删除已有远端日程，成功后清除本地 calendar_event_id。
def _sync_cancelled_interview_to_calendar(
    repository: OfferPilotRepository,
    schedule: InterviewSchedule,
    calendar_service: Optional[FeishuCalendarService],
    owner_id: str,
    idempotency_key: str,
) -> Dict[str, Any]:
    if calendar_service is None or not calendar_service.is_calendar_sync_enabled():
        return {"synced": False, "status": "disabled", "attempts": 0}
    cached = _load_sync_result(repository, idempotency_key)
    if cached is not None:
        return {**cached, "attempts": 0, "idempotent_replay": True}
    if not schedule.calendar_event_id:
        return {"synced": False, "status": "not_applicable", "attempts": 0}

    event_id = schedule.calendar_event_id

    def operation() -> Dict[str, Any]:
        calendar_context = _ensure_calendar_for_sync(repository, calendar_service)
        calendar_service.delete_interview_event(
            calendar_id=calendar_context.get("calendar_id"),
            event_id=event_id,
        )
        repository.update_interview_schedule_calendar_event(
            schedule_id=schedule.id,
            calendar_event_id=None,
            owner_id=owner_id,
        )
        schedule.calendar_event_id = None
        return {
            "synced": True,
            "status": "synced",
            "operation": "deleted",
            "calendar_event_id": event_id,
            **calendar_context,
        }

    return _run_idempotent_sync(
        repository=repository,
        idempotency_key=idempotency_key,
        operation=operation,
        error_formatter=_summarize_calendar_sync_error,
    )


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


# 格式化 applications bulk updated message。
def _format_applications_bulk_updated_message(
    company: str,
    applications: List[Application],
    bitable_sync_results: List[Dict[str, Any]],
) -> str:
    lines = [f"已更新 {len(applications)} 条 {company} 投递记录："]
    for index, application in enumerate(applications, start=1):
        lines.append(
            f"{index}. {application.role}：{application_status_label(application.status)}"
        )

    failed_syncs = [
        sync
        for sync in bitable_sync_results
        if sync and sync.get("status") == "failed"
    ]
    if failed_syncs:
        lines.append(f"其中 {len(failed_syncs)} 条飞书多维表格同步失败。")
        first_error = failed_syncs[0].get("error")
        if first_error:
            lines.append(f"失败原因：{first_error}。")
    elif any(sync and sync.get("synced") for sync in bitable_sync_results):
        lines.append("已同步到飞书多维表格。")
    elif any(sync and sync.get("status") == "disabled" for sync in bitable_sync_results):
        lines.append("飞书多维表格同步未开启，当前只保存到 OfferPilot。")

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
    idempotency_key: Optional[str] = None,
    clear_interview_fields: bool = False,
) -> Dict[str, Any]:
    if bitable_service is None or not bitable_service.is_bitable_sync_enabled():
        return {"synced": False, "status": "disabled", "attempts": 0}

    if idempotency_key:
        cached = _load_sync_state(repository, idempotency_key)
        if cached is not None and (
            cached.get("synced") or cached.get("reconciliation_required")
        ):
            return {**cached, "attempts": 0, "idempotent_replay": True}

    last_result: Dict[str, Any] = {"synced": False, "status": "failed", "error": "unknown"}
    for attempt in range(1, _SYNC_MAX_ATTEMPTS + 1):
        last_result = _sync_application_to_bitable_once(
            repository=repository,
            application=application,
            schedule=schedule,
            calendar_sync=calendar_sync,
            bitable_service=bitable_service,
            collaborator_user_id=collaborator_user_id,
            collaborator_user_id_type=collaborator_user_id_type,
            clear_interview_fields=clear_interview_fields,
        )
        last_result["attempts"] = attempt
        if last_result.get("status") == "failed" and not last_result.get("retry_safe", True):
            last_result["reconciliation_required"] = True
            last_result["idempotency_key"] = idempotency_key
            if idempotency_key:
                _store_sync_result(repository, idempotency_key, last_result)
            return last_result
        if last_result.get("status") != "failed":
            if idempotency_key and last_result.get("synced"):
                last_result["idempotency_key"] = idempotency_key
                _store_sync_result(repository, idempotency_key, last_result)
            return last_result
    if idempotency_key:
        last_result["idempotency_key"] = idempotency_key
        _store_sync_result(repository, idempotency_key, last_result)
    return last_result


# 执行一次 application 多维表格同步。
def _sync_application_to_bitable_once(
    repository: OfferPilotRepository,
    application: Application,
    schedule: Optional[InterviewSchedule],
    calendar_sync: Optional[Dict[str, Any]],
    bitable_service: Optional[FeishuBitableService],
    collaborator_user_id: Optional[str] = None,
    collaborator_user_id_type: str = "open_id",
    clear_interview_fields: bool = False,
) -> Dict[str, Any]:
    if bitable_service is None:
        return {"synced": False, "status": "disabled"}

    retry_safe = False
    try:
        bitable_context = _ensure_bitable_for_sync(repository, bitable_service)
        fields = _build_application_bitable_fields(
            application=application,
            schedule=schedule,
            calendar_sync=calendar_sync,
            clear_interview_fields=clear_interview_fields,
        )
        record_setting_key = _bitable_record_setting_key(
            table_id=bitable_context["table_id"],
            application_id=application.id,
        )
        existing_record_id = repository.get_runtime_setting(record_setting_key)
        retry_safe = bool(existing_record_id)

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
            "retry_safe": retry_safe,
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
        app_token = getattr(bitable_service, "app_token", None)
        table_id = getattr(bitable_service, "table_id", None)
        _migrate_bitable_schema_if_needed(
            repository=repository,
            bitable_service=bitable_service,
            app_token=app_token,
            table_id=table_id,
        )
        return {
            "app_token": app_token,
            "table_id": table_id,
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
    if not table_id:
        table_name = getattr(bitable_service, "table_name", None) or settings.feishu_offerpilot_bitable_table_name
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
    elif stored_schema_version != _CURRENT_BITABLE_SCHEMA_VERSION:
        _migrate_bitable_schema_if_needed(
            repository=repository,
            bitable_service=bitable_service,
            app_token=app_token,
            table_id=table_id,
        )

    return {
        "app_token": app_token,
        "table_id": table_id,
        "bitable_managed": True,
        "bitable_auto_created": auto_created,
    }


# 旧表原地升级，避免每次 schema 变化都创建一张新的 “Pro” 表。
def _migrate_bitable_schema_if_needed(
    repository: OfferPilotRepository,
    bitable_service: FeishuBitableService,
    app_token: Optional[str],
    table_id: Optional[str],
) -> None:
    if not app_token or not table_id:
        return
    stored_schema_version = repository.get_runtime_setting(
        _OFFERPILOT_BITABLE_SCHEMA_VERSION_SETTING
    )
    if stored_schema_version == _CURRENT_BITABLE_SCHEMA_VERSION:
        return
    migrate_schema = getattr(bitable_service, "migrate_application_table_schema", None)
    if not callable(migrate_schema):
        return
    migrate_schema(app_token=app_token, table_id=table_id)
    repository.set_runtime_setting(
        _OFFERPILOT_BITABLE_SCHEMA_VERSION_SETTING,
        _CURRENT_BITABLE_SCHEMA_VERSION,
    )


# 构造 application bitable fields。
def _build_application_bitable_fields(
    application: Application,
    schedule: Optional[InterviewSchedule],
    calendar_sync: Optional[Dict[str, Any]],
    clear_interview_fields: bool = False,
) -> Dict[str, Any]:
    calendar_event_id = application.to_dict().get("calendar_event_id")
    if schedule is not None:
        calendar_event_id = schedule.calendar_event_id
    elif calendar_sync and calendar_sync.get("calendar_event_id"):
        calendar_event_id = calendar_sync["calendar_event_id"]

    fields: Dict[str, Any] = {
        "投递记录": _build_application_bitable_title(application),
        "OfferPilot记录ID": application.id,
        "公司": application.company,
        "岗位": application.role,
        "投递状态": application_status_label(application.status),
        "优先级": _infer_application_priority(application),
        "来源": "飞书助手",
        "下一步": _build_application_next_action(application, schedule),
        "最后同步时间": _datetime_to_bitable_timestamp_ms(datetime.now()),
    }
    if application.owner_id != "local_user":
        fields["OfferPilot用户ID"] = application.owner_id
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
    if schedule is not None:
        fields["日历事件ID"] = calendar_event_id
    elif calendar_event_id:
        fields["日历事件ID"] = calendar_event_id
    if application.jd_keywords:
        fields["JD关键词"] = list(application.jd_keywords)
    if clear_interview_fields:
        fields.update(
            {
                "面试轮次": None,
                "面试时间文本": None,
                "面试开始时间": None,
                "提醒分钟": None,
                "日历事件ID": None,
            }
        )
    return fields


# 生成适合作为多维表格主字段的可读标题，内部 ID 另列保存。
def _build_application_bitable_title(application: Application) -> str:
    company = application.company.strip()
    role = application.role.strip()
    if company and role:
        return f"{company}｜{role}"
    return company or role or "未命名投递"


# 从工具入参中提取数据归属人，缺省兼容本地单用户模式。
def _owner_id_from_arguments(arguments: Dict[str, Any]) -> str:
    owner_id = str(arguments.get("owner_id") or "").strip()
    return owner_id or "local_user"


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


# 带有限重试执行一个外部同步步骤，并缓存成功结果用于幂等回放。
def _run_idempotent_sync(
    repository: OfferPilotRepository,
    idempotency_key: str,
    operation: Callable[[], Dict[str, Any]],
    error_formatter: Callable[[str], str],
    max_attempts: int = _SYNC_MAX_ATTEMPTS,
    reconciliation_required_on_failure: bool = False,
) -> Dict[str, Any]:
    cached = _load_sync_state(repository, idempotency_key)
    if cached is not None and (
        cached.get("synced") or cached.get("reconciliation_required")
    ):
        return {**cached, "attempts": 0, "idempotent_replay": True}

    last_error = "unknown"
    for attempt in range(1, max_attempts + 1):
        try:
            result = operation()
        except (FeishuConfigurationError, FeishuRequestError) as exc:
            last_error = error_formatter(str(exc))
            continue
        result = {
            **result,
            "attempts": attempt,
            "idempotency_key": idempotency_key,
        }
        _store_sync_result(repository, idempotency_key, result)
        return result
    failure_result = {
        "synced": False,
        "status": "failed",
        "attempts": max_attempts,
        "idempotency_key": idempotency_key,
        "error": last_error,
    }
    if reconciliation_required_on_failure:
        failure_result["reconciliation_required"] = True
    if idempotency_key:
        _store_sync_result(repository, idempotency_key, failure_result)
    return failure_result


# 读取已完成同步步骤的缓存结果。
def _load_sync_result(
    repository: OfferPilotRepository,
    idempotency_key: str,
) -> Optional[Dict[str, Any]]:
    value = _load_sync_state(repository, idempotency_key)
    return value if value is not None and value.get("synced") else None


# 读取同步步骤的持久化状态，包括需要人工核对的未知结果。
def _load_sync_state(
    repository: OfferPilotRepository,
    idempotency_key: str,
) -> Optional[Dict[str, Any]]:
    raw_value = repository.get_runtime_setting(_sync_idempotency_setting_key(idempotency_key))
    if not raw_value:
        return None
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


# 保存同步步骤状态；成功用于幂等回放，失败用于恢复与核对。
def _store_sync_result(
    repository: OfferPilotRepository,
    idempotency_key: str,
    result: Dict[str, Any],
) -> None:
    repository.set_runtime_setting(
        _sync_idempotency_setting_key(idempotency_key),
        json.dumps(result, ensure_ascii=False, sort_keys=True),
    )


# 用户完成远端核对后，清除未知结果，允许同一逻辑操作安全恢复。
def _clear_sync_state(
    repository: OfferPilotRepository,
    idempotency_key: str,
) -> None:
    repository.set_runtime_setting(_sync_idempotency_setting_key(idempotency_key), "")


# 将调用方幂等键绑定到请求指纹；相同键不能代表另一项写操作。
def _claim_request_idempotency_key(
    repository: OfferPilotRepository,
    namespace: str,
    idempotency_key: Optional[Any],
    payload: Dict[str, Any],
) -> Optional[str]:
    if idempotency_key is None:
        return None
    normalized_key = str(idempotency_key).strip()
    if not normalized_key:
        return None
    request_fingerprint = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    binding_digest = hashlib.sha256(
        f"{namespace}:{normalized_key}".encode("utf-8")
    ).hexdigest()
    setting_key = f"{_OFFERPILOT_REQUEST_IDEMPOTENCY_PREFIX}{binding_digest}"
    existing_fingerprint = repository.get_runtime_setting(setting_key)
    if existing_fingerprint and existing_fingerprint != request_fingerprint:
        return "幂等键已用于另一组参数，请为新的写操作更换幂等键。"
    if not existing_fingerprint:
        repository.set_runtime_setting(setting_key, request_fingerprint)
    return None


# 将外部幂等键压缩成稳定的 runtime setting key。
def _sync_idempotency_setting_key(idempotency_key: str) -> str:
    digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
    return f"{_OFFERPILOT_SYNC_IDEMPOTENCY_PREFIX}{digest}"


# 汇总本地写入之后的外部同步结果。
def _external_write_status(*sync_results: Dict[str, Any]) -> str:
    if any(result.get("reconciliation_required") for result in sync_results):
        return "reconciliation_required"
    if any(result.get("status") == "failed" for result in sync_results):
        return "partial_success"
    synced_results = [result for result in sync_results if result.get("synced")]
    if synced_results and len(synced_results) == len(sync_results):
        return "completed"
    if synced_results:
        return "completed_with_skips"
    return "completed_local_only"


# 生成用户可理解的外部同步摘要。
def _external_write_summary(operation_status: str) -> str:
    if operation_status == "reconciliation_required":
        return "本地状态已更新，但远端写入结果未知；为避免重复记录，需先核对飞书状态。"
    if operation_status == "partial_success":
        return "本地状态已更新，但部分飞书同步失败，可使用相同幂等键重试。"
    if operation_status == "completed_with_skips":
        return "已完成当前启用的飞书同步；未启用的目标保留本地状态。"
    if operation_status == "completed_local_only":
        return "外部同步未启用，本地状态已更新。"
    return "飞书日历和多维表格均已同步。"


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
_default_leetcode_repository = build_default_leetcode_repository()
_default_registry = build_offerpilot_tool_registry(
    _default_repository,
    leetcode_repository=_default_leetcode_repository,
)


# 返回全局默认工具注册表。
def get_default_tool_registry() -> ToolRegistry:
    return _default_registry


# 返回全局默认数据仓库实例。
def get_default_offerpilot_repository() -> OfferPilotRepository:
    return _default_repository


# 返回全局默认 LeetCode 数据仓库实例。
def get_default_leetcode_repository() -> LeetCodeRepository:
    return _default_leetcode_repository
