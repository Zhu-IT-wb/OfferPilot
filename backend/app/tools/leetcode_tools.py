from datetime import date, datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from app.models.leetcode import (
    LeetCodeAssignmentStatus,
    LeetCodeFeedback,
    LeetCodePracticeResult,
    LeetCodeRecommendation,
    LeetCodeSubscription,
)
from app.repositories.leetcode_repository import LeetCodeRepository
from app.schemas.agent import AgentActionName
from app.schemas.tool import ToolResult
from app.services.leetcode_messages import format_leetcode_recommendations
from app.services.leetcode_recommendation import LeetCodeRecommendationWorkflow
from app.tools.registry import ToolRegistry


def register_leetcode_tools(
    registry: ToolRegistry,
    repository: LeetCodeRepository,
) -> None:
    registry.register(
        AgentActionName.ENABLE_LEETCODE_PLAN.value,
        lambda arguments: enable_leetcode_plan(repository, arguments),
        description="开启每天 09:00 的 LeetCode 推荐和 21:00 未反馈提醒。",
        mutating=True,
        examples=["开启每日刷题", "开启 LeetCode 计划"],
    )
    registry.register(
        AgentActionName.DISABLE_LEETCODE_PLAN.value,
        lambda arguments: disable_leetcode_plan(repository, arguments),
        description="关闭 LeetCode 主动推送，保留历史训练进度。",
        mutating=True,
        examples=["关闭每日刷题", "停止 LeetCode 推送"],
    )
    registry.register(
        AgentActionName.GET_TODAY_LEETCODE.value,
        lambda arguments: get_today_leetcode(repository, arguments),
        description="读取或生成今天的三道 LeetCode 推荐题。",
        mutating=False,
        examples=["今天刷什么", "查看今天的力扣题"],
    )
    registry.register(
        AgentActionName.RECORD_LEETCODE_RESULT.value,
        lambda arguments: record_leetcode_result(repository, arguments),
        description="记录明确的 LeetCode 训练结果并计算下次复习日期。",
        mutating=True,
        required_slots=["result"],
        optional_slots=["assignment_id", "problem_index", "problem_title"],
        examples=["第1题独立完成", "LRU 看题解完成", "第2题没做出来"],
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
    recommendations = get_today_recommendations(repository, owner_id)
    return ToolResult(
        tool_name=AgentActionName.ENABLE_LEETCODE_PLAN.value,
        success=True,
        message=(
            "已开启每日刷题：09:00 推送，21:00 提醒未反馈题目。\n\n"
            + format_leetcode_recommendations(recommendations)
        ),
        data={"recommendations": recommendations_to_dict(recommendations)},
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
    recommendations = get_today_recommendations(
        repository,
        _owner_id_from_arguments(arguments),
    )
    return ToolResult(
        tool_name=AgentActionName.GET_TODAY_LEETCODE.value,
        success=True,
        message=format_leetcode_recommendations(recommendations),
        data={"recommendations": recommendations_to_dict(recommendations)},
    )


def record_leetcode_result(
    repository: LeetCodeRepository,
    arguments: Dict[str, Any],
) -> ToolResult:
    owner_id = _owner_id_from_arguments(arguments)
    today = today_in_shanghai()
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
                "candidates": recommendations_to_dict(recommendations),
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
    message = _feedback_message(recommendation, feedback, result)
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


def get_today_recommendations(
    repository: LeetCodeRepository,
    owner_id: str,
) -> List[LeetCodeRecommendation]:
    return LeetCodeRecommendationWorkflow(repository).get_today(
        owner_id=owner_id,
        today=today_in_shanghai(),
    )


def recommendations_to_dict(
    recommendations: List[LeetCodeRecommendation],
) -> List[Dict[str, Any]]:
    return [
        {
            "assignment": item.assignment.to_dict(),
            "problem": item.problem.to_dict(),
        }
        for item in recommendations
    ]


def today_in_shanghai() -> date:
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


def _feedback_message(
    recommendation: LeetCodeRecommendation,
    feedback: LeetCodeFeedback,
    result: LeetCodePracticeResult,
) -> str:
    labels = {
        LeetCodePracticeResult.INDEPENDENT: "独立完成",
        LeetCodePracticeResult.WITH_HINT: "提示后完成",
        LeetCodePracticeResult.WITH_SOLUTION: "看题解完成",
        LeetCodePracticeResult.FAILED: "尝试但未完成",
        LeetCodePracticeResult.POSTPONED: "延期",
        LeetCodePracticeResult.SKIPPED: "跳过",
    }
    message = (
        f"已记录：{recommendation.problem.frontend_id}. {recommendation.problem.title_zh}，"
        f"{labels[result]}。"
    )
    if result == LeetCodePracticeResult.POSTPONED:
        if feedback.assignment.status == LeetCodeAssignmentStatus.SKIPPED:
            return message + " 已连续延期 3 次，自动跳过并释放新题名额。"
        return message + " 明天会优先继续推荐。"
    next_review = feedback.progress.next_review_on if feedback.progress else None
    if next_review:
        message += f" 下次复习：{next_review.isoformat()}。"
    elif feedback.progress and feedback.progress.mastery_status.value == "mastered":
        message += " 已通过三次间隔验证，标记为已掌握。"
    return message


def _owner_id_from_arguments(arguments: Dict[str, Any]) -> str:
    owner_id = str(arguments.get("owner_id") or "").strip()
    return owner_id or "local_user"


def _normalize_match_value(value: str) -> str:
    return "".join(value.split()).lower()
