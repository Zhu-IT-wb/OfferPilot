"""Deterministic, user-facing presentation for Agent Runtime tools.

Internal tool identifiers remain available to the model, checkpoint, and operation
ledger.  This module is the only place where those identifiers are translated for a
human-facing response.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from app.tools.agent_tool import AgentToolDefinition, ToolEffect, ToolPresentation


_PRESENTATIONS: dict[str, ToolPresentation] = {
    "list_tasks": ToolPresentation("查询今日任务"),
    "list_applications": ToolPresentation("查询投递记录"),
    "get_application_bitable_link": ToolPresentation("获取投递表链接"),
    "create_application": ToolPresentation(
        "新增投递记录",
        ("company", "role", "base_location", "round", "interview_time"),
        "投递信息将保存到 OfferPilot，并同步到已连接的飞书服务。",
    ),
    "update_application": ToolPresentation(
        "更新投递记录",
        (
            "company",
            "role",
            "base_location",
            "round",
            "interview_time",
            "status",
            "update_type",
            "apply_to_all",
        ),
        "现有投递信息将被修改，并同步到已连接的飞书服务。",
    ),
    "list_interviews": ToolPresentation("查询面试安排"),
    "schedule_interview": ToolPresentation(
        "新增面试安排",
        ("company", "role", "round", "interview_time"),
        "面试安排将被保存，并可能同步到飞书日历。",
    ),
    "reschedule_interview": ToolPresentation(
        "调整面试时间",
        ("company", "role", "round", "interview_time"),
        "现有面试时间将被修改，并可能同步到飞书日历。",
    ),
    "cancel_interview": ToolPresentation(
        "取消面试安排",
        ("company", "role", "round"),
        "现有面试安排及其关联日历事件将被取消。",
    ),
    "update_task": ToolPresentation(
        "更新任务状态",
        ("task_title", "operation", "task_type"),
        "任务状态将被修改。",
    ),
    "record_interview_review": ToolPresentation(
        "保存面试复盘",
        ("company", "round", "topics"),
        "复盘内容将保存到 OfferPilot。",
    ),
    "configure_leetcode_plan": ToolPresentation(
        "调整每日刷题计划",
        ("enabled",),
        "每日刷题任务设置将被修改。",
    ),
    "get_today_leetcode": ToolPresentation("查询今日刷题任务"),
    "record_leetcode_result": ToolPresentation(
        "记录刷题结果",
        ("problem_title", "problem_index", "result"),
        "本次刷题结果将被保存。",
    ),
    "start_mock_interview": ToolPresentation(
        "开始模拟面试", ("project", "role")
    ),
    "start_project_training": ToolPresentation(
        "开始项目训练",
        ("project", "role", "difficulty"),
        "新的项目训练会话将被创建。",
    ),
    "resume_project_training": ToolPresentation("继续项目训练", ("project",)),
    "get_project_training_summary": ToolPresentation("查询项目训练总结"),
    "update_execution_plan": ToolPresentation("更新任务计划"),
    "request_user_input": ToolPresentation("补充任务信息"),
    "reconcile_write_operation": ToolPresentation(
        "确认外部写入结果",
        ("outcome", "evidence"),
        "核对结果将用于修正本地执行记录，不会重复发起原写入。",
    ),
    "search_interview_knowledge": ToolPresentation("检索面试知识"),
    "search_project_evidence": ToolPresentation("检索项目证据"),
    "read_evidence": ToolPresentation("读取证据详情"),
    "get_study_preferences": ToolPresentation("读取复习偏好"),
    "save_study_preferences": ToolPresentation(
        "保存复习偏好",
        (
            "timezone",
            "weekday_windows",
            "weekend_windows",
            "daily_max_minutes",
            "session_minutes",
        ),
        "复习时间偏好将保存到 OfferPilot。",
    ),
    "get_calendar_availability": ToolPresentation("查询日历忙闲"),
    "create_study_plan": ToolPresentation(
        "创建复习计划",
        ("goal", "range_start", "range_end", "topics"),
        "排期结果将保存为本地复习计划，暂不会写入飞书日历。",
    ),
    "get_study_plan": ToolPresentation("查询复习计划"),
    "list_study_plans": ToolPresentation("查询复习计划列表"),
    "sync_study_plan_to_calendar": ToolPresentation(
        "同步复习计划到飞书日历",
        (),
        "计划中的复习时段将写入你的飞书日历。",
    ),
    "update_study_session": ToolPresentation(
        "更新复习时段",
        ("status", "start_at", "end_at"),
        "现有复习时段将被修改。",
    ),
    "reconcile_study_calendar_session": ToolPresentation(
        "确认日历同步结果",
        ("outcome",),
        "核对结果将用于修正本地同步状态，不会重复写入日历。",
    ),
    "list_learning_gaps": ToolPresentation("查询薄弱知识点"),
}


_FIELD_LABELS = {
    "company": "公司",
    "role": "岗位",
    "base_location": "工作地点",
    "round": "面试轮次",
    "interview_time": "面试时间",
    "status": "状态",
    "update_type": "更新内容",
    "apply_to_all": "应用到全部匹配记录",
    "task_title": "任务",
    "task_type": "任务类型",
    "operation": "操作",
    "topics": "主题",
    "enabled": "每日计划",
    "problem_title": "题目",
    "problem_index": "题目序号",
    "result": "完成情况",
    "project": "项目",
    "difficulty": "难度",
    "outcome": "核对结果",
    "evidence": "核对依据",
    "timezone": "时区",
    "weekday_windows": "工作日可用时间",
    "weekend_windows": "周末可用时间",
    "daily_max_minutes": "每日上限",
    "session_minutes": "单次时长",
    "goal": "目标",
    "range_start": "开始日期",
    "range_end": "结束日期",
    "start_at": "开始时间",
    "end_at": "结束时间",
}


_VALUE_LABELS = {
    "complete": "完成",
    "postpone": "延期",
    "enabled": "开启",
    "disabled": "关闭",
    "basic": "基础",
    "medium": "中等",
    "advanced": "进阶",
    "applied": "已写入",
    "not_applied": "未写入",
    "created": "已创建",
    "not_created": "未创建",
    "updated": "已更新",
    "not_updated": "未更新",
    "deleted": "已删除",
    "not_deleted": "未删除",
    "planned": "待进行",
    "submitted": "已投递",
    "written_test": "笔试中",
    "written_test_passed": "笔试通过",
    "interview_scheduled": "已安排面试",
    "offer": "已获 Offer",
    "rejected": "未通过",
    "no_response": "暂无回复",
    "withdrawn": "已撤回",
    "completed": "已完成",
    "skipped": "已跳过",
    "cancelled": "已取消",
    "independent": "独立完成",
    "with_hint": "提示后完成",
    "with_solution": "参考题解后完成",
    "failed": "未完成",
}


_INTERNAL_IDENTIFIER_LABELS = {
    "actor_mismatch": "当前会话身份不匹配",
    "approval_missing_or_stale": "确认状态已失效",
    "batch_rejected": "这组操作无法一起执行",
    "cancelled_by_user": "操作已取消",
    "completion_guard_incomplete": "任务结果可能不完整",
    "completion_unverified_budget": "任务结果未能完成核验",
    "completion_verifier_budget_exhausted": "任务结果未能完成核验",
    "completion_verifier_incomplete": "任务结果可能不完整",
    "context_compacted": "较早的会话内容已压缩",
    "context_over_budget": "会话内容超过处理上限",
    "conversation_history_compacted": "较早的会话内容已压缩",
    "deterministic_command_path": "当前使用基础只读模式",
    "duplicate_tool_call_id": "模型返回了重复操作",
    "failed_or_unknown_tool_results_preserved": "失败或待确认的操作结果已保留",
    "idempotency_conflict": "操作状态发生冲突",
    "invalid_arguments": "操作信息不完整或格式不正确",
    "invalid_write_reconciliation": "写入结果核对失败",
    "interaction_conflict": "待处理操作状态已经变化",
    "interaction_deferred": "还有一项信息等待补充",
    "llm_unavailable": "模型服务暂时不可用",
    "mcp_unavailable": "知识检索服务暂时不可用",
    "missing_tool_call_id": "模型返回的操作信息不完整",
    "missing_actor_context": "当前用户信息不完整",
    "model_protocol_error": "模型响应格式异常",
    "prior_write_outcome_unknown": "此前的外部写入结果暂未确认",
    "rejected_by_user": "操作已拒绝",
    "reused_tool_call_id": "模型重复提交了同一操作",
    "revision_requested": "操作内容已请求修改",
    "thread_state_not_found": "未找到当前会话状态",
    "tool_budget_exceeded": "本次任务的操作次数已达上限",
    "tool_effect_metadata_missing": "部分操作信息不完整",
    "tool_not_found": "无法执行请求的操作",
    "unpaired_tool_calls_preserved": "未完成的操作上下文已保留",
    "verifier_unavailable": "结果核验服务暂时不可用",
    "write_confirmed_not_applied": "外部写入确认未生效",
    "write_not_fully_confirmed": "写入结果尚未完全确认",
    "write_outcome_unknown": "外部写入结果暂未确认",
    "write_receipts_preserved": "写入回执已保留",
    "write_reconciliation_state_changed": "写入状态已经变化",
}

_CALL_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:call|toolu|tool_call)_[A-Za-z0-9][A-Za-z0-9_.:-]*(?![A-Za-z0-9_])"
)
_OPERATION_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])agent_op_[A-Fa-f0-9]{8,}(?![A-Za-z0-9_])"
)


def presentation_for(
    tool_name: str,
    definition: AgentToolDefinition | None = None,
) -> ToolPresentation:
    """Return explicit metadata or a non-leaking fallback for an unknown tool."""

    if definition is not None and definition.presentation != ToolPresentation():
        return definition.presentation
    return _PRESENTATIONS.get(tool_name, ToolPresentation())


def render_approval_prompt(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    definition: AgentToolDefinition | None = None,
    effect: str | ToolEffect | None = None,
) -> str:
    presentation = presentation_for(tool_name, definition)
    lines = [f"请确认以下操作：{presentation.display_name}"]
    details = _render_details(arguments, presentation.detail_fields)
    if details:
        lines.extend(("", *details))
    impact = presentation.impact or _default_impact(effect)
    if impact:
        lines.extend(("", f"影响：{impact}"))
    lines.extend(("", "是否确认？"))
    return "\n".join(lines)


def redact_tool_identifiers(
    text: str,
    definitions: Iterable[AgentToolDefinition] = (),
    internal_identifiers: Iterable[str] = (),
) -> str:
    """Remove runtime implementation identifiers from user-facing prose."""

    if not text:
        return text
    replacements: dict[str, str] = dict(_INTERNAL_IDENTIFIER_LABELS)
    for name, presentation in _PRESENTATIONS.items():
        replacements[f"{name}_degraded"] = f"{presentation.display_name}暂时降级"
        replacements[name] = presentation.display_name
    for definition in definitions:
        label = presentation_for(
            definition.name, definition
        ).display_name
        replacements[f"{definition.name}_degraded"] = f"{label}暂时降级"
        replacements[definition.name] = label
    for identifier in internal_identifiers:
        normalized = str(identifier or "").strip()
        if normalized:
            replacements.setdefault(normalized, "内部标识")
    sanitized = text
    for name in sorted(replacements, key=len, reverse=True):
        label = replacements[name]
        sanitized = sanitized.replace(f"`{name}`", label)
        sanitized = sanitized.replace(name, label)
    sanitized = _CALL_IDENTIFIER_PATTERN.sub("内部调用", sanitized)
    sanitized = _OPERATION_IDENTIFIER_PATTERN.sub("内部操作", sanitized)
    sanitized = sanitized.replace("tool_call_id", "内部调用标识")
    sanitized = sanitized.replace("operation_key", "内部操作标识")
    sanitized = sanitized.replace("error_code", "内部错误状态")
    return sanitized


def registered_tool_names() -> frozenset[str]:
    return frozenset(_PRESENTATIONS)


def _render_details(
    arguments: dict[str, Any],
    fields: tuple[str, ...],
) -> list[str]:
    details: list[str] = []
    for field_name in fields:
        value = arguments.get(field_name)
        if value is None or value == "" or value == [] or value == {}:
            continue
        label = _FIELD_LABELS.get(field_name)
        if not label:
            continue
        details.append(f"{label}：{_format_value(field_name, value)}")
    return details


def _format_value(field_name: str, value: Any) -> str:
    if isinstance(value, bool):
        if field_name == "enabled":
            return "开启" if value else "关闭"
        return "是" if value else "否"
    if field_name in {"daily_max_minutes", "session_minutes"}:
        return f"{value} 分钟"
    if field_name == "topics" and isinstance(value, list):
        names = [
            str(item.get("topic"))
            for item in value
            if isinstance(item, dict) and item.get("topic")
        ]
        if names:
            preview = "、".join(names[:3])
            suffix = f"等 {len(names)} 个主题" if len(names) > 3 else ""
            return preview + suffix
    if field_name in {"weekday_windows", "weekend_windows"} and isinstance(value, list):
        windows = [
            f"{item.get('start_time')}-{item.get('end_time')}"
            for item in value
            if isinstance(item, dict)
            and item.get("start_time")
            and item.get("end_time")
        ]
        if windows:
            return "、".join(windows)
    if isinstance(value, list):
        return "、".join(str(item) for item in value[:5])
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    normalized = str(value)
    return _VALUE_LABELS.get(normalized, normalized)


def _default_impact(effect: str | ToolEffect | None) -> str:
    normalized = effect.value if isinstance(effect, ToolEffect) else str(effect or "")
    if normalized == ToolEffect.DESTRUCTIVE.value:
        return "该操作会取消或删除现有数据。"
    if normalized == ToolEffect.EXTERNAL_WRITE.value:
        return "该操作会修改已连接的外部服务。"
    if normalized == ToolEffect.LOCAL_WRITE.value:
        return "该操作会修改你的 OfferPilot 数据。"
    return ""
