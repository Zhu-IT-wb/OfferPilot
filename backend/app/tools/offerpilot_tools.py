from typing import Any, Dict, Optional

from app.core.config import settings
from app.repositories.offerpilot_repository import (
    InMemoryOfferPilotRepository,
    OfferPilotRepository,
)
from app.repositories.sqlite_offerpilot_repository import SQLiteOfferPilotRepository
from app.schemas.agent import AgentActionName
from app.schemas.tool import ToolResult
from app.tools.registry import ToolRegistry


def build_default_offerpilot_repository() -> OfferPilotRepository:
    if settings.storage_backend.strip().lower() == "sqlite":
        return SQLiteOfferPilotRepository(settings.sqlite_path)
    return InMemoryOfferPilotRepository()


def build_offerpilot_tool_registry(repository: Optional[OfferPilotRepository] = None) -> ToolRegistry:
    selected_repository = repository or build_default_offerpilot_repository()
    registry = ToolRegistry()

    registry.register(
        AgentActionName.LIST_TODAY_TASKS.value,
        lambda arguments: list_today_tasks(selected_repository),
    )
    registry.register(
        AgentActionName.CREATE_APPLICATION.value,
        lambda arguments: create_application(selected_repository, arguments),
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


def create_application(repository: OfferPilotRepository, arguments: Dict[str, Any]) -> ToolResult:
    missing = [name for name in ("company", "role") if not arguments.get(name)]
    if missing:
        return ToolResult(
            tool_name=AgentActionName.CREATE_APPLICATION.value,
            success=False,
            message=f"创建投递记录失败，缺少字段：{', '.join(missing)}。",
            data={"missing_slots": missing},
        )

    application = repository.create_application(
        company=arguments["company"],
        role=arguments["role"],
        interview_time=arguments.get("interview_time"),
        round_name=arguments.get("round"),
        jd_keywords=arguments.get("jd_keywords", []),
    )
    application_data = application.to_dict()
    return ToolResult(
        tool_name=AgentActionName.CREATE_APPLICATION.value,
        success=True,
        message=f"已创建投递记录：{application.company} - {application.role}。",
        data={"application": application_data},
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
