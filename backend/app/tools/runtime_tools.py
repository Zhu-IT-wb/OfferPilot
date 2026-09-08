import json
from typing import Any, Callable, Dict, Optional

from app.core.config import Settings, settings as default_settings
from app.schemas.tool import ToolResult, ToolSpec
from app.tools.agent_tool import (
    AgentToolDefinition,
    AgentToolResult,
    FunctionAgentTool,
    ToolApproval,
    ToolEffect,
    ToolInteraction,
    ToolOutcomeStatus,
)
from app.tools.agent_tool_registry import AgentToolRegistry
from app.tools.registry import ToolRegistry
from app.tools.runtime_mcp_tools import build_runtime_mcp_tools
from app.tools.runtime_tool_inputs import RUNTIME_TOOL_INPUT_MODELS
from app.tools.study_tools import StudyCalendarProvider, build_study_agent_tools


_TRUSTED_ARGUMENTS = {
    "owner_id",
    "raw_message",
    "attendee_user_id",
    "attendee_user_id_type",
    "bitable_collaborator_user_id",
    "bitable_collaborator_user_id_type",
    "idempotency_key",
}

_ALIASES: Dict[str, tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = {
    "list_tasks": ("list_today_tasks", lambda value: value),
    "list_applications": (
        "query_application",
        lambda value: {**value, "query_type": "list"},
    ),
    "list_interviews": (
        "query_application",
        lambda value: {**value, "query_type": "upcoming_interviews"},
    ),
    "schedule_interview": (
        "update_application",
        lambda value: {**value, "update_type": "schedule_interview"},
    ),
    "record_interview_review": ("create_interview_review", lambda value: value),
}

_DIRECT_TOOLS = {
    "create_application",
    "get_application_bitable_link",
    "update_application",
    "reschedule_interview",
    "cancel_interview",
    "get_today_leetcode",
    "record_leetcode_result",
    "start_mock_interview",
    "start_project_training",
    "resume_project_training",
    "get_project_training_summary",
}

_EXTERNAL_WRITES = {
    "create_application",
    "update_application",
    "schedule_interview",
    "reschedule_interview",
}
_DESTRUCTIVE = {"cancel_interview"}
_LOCAL_WRITES = {
    "enable_leetcode_plan",
    "disable_leetcode_plan",
    "record_leetcode_result",
    "record_interview_review",
    "start_project_training",
    "get_today_leetcode",
}

REQUIRED_RUNTIME_TOOLS = frozenset(
    {
        "cancel_interview",
        "configure_leetcode_plan",
        "create_application",
        "create_study_plan",
        "get_calendar_availability",
        "get_application_bitable_link",
        "get_project_training_summary",
        "get_study_plan",
        "get_study_preferences",
        "get_today_leetcode",
        "list_applications",
        "list_interviews",
        "list_learning_gaps",
        "list_study_plans",
        "list_tasks",
        "read_evidence",
        "reconcile_write_operation",
        "reconcile_study_calendar_session",
        "record_interview_review",
        "record_leetcode_result",
        "request_user_input",
        "reschedule_interview",
        "resume_project_training",
        "save_study_preferences",
        "schedule_interview",
        "search_interview_knowledge",
        "search_project_evidence",
        "start_mock_interview",
        "start_project_training",
        "sync_study_plan_to_calendar",
        "update_application",
        "update_execution_plan",
        "update_study_session",
        "update_task",
    }
)

_RAG_RUNTIME_TOOLS = frozenset(
    {
        "read_evidence",
        "search_interview_knowledge",
        "search_project_evidence",
    }
)


def build_runtime_tool_registry(
    legacy: ToolRegistry,
    offerpilot_repository: Any = None,
    knowledge_repository: Any = None,
    calendar_provider: Optional[StudyCalendarProvider] = None,
    require_complete: bool = False,
    app_settings: Settings = default_settings,
    mcp_client: Any = None,
) -> AgentToolRegistry:
    specs = {spec.name: spec for spec in legacy.describe_tools()}
    tools = []

    for public_name in sorted(_DIRECT_TOOLS):
        spec = specs.get(public_name)
        if spec is None:
            continue
        tools.append(_legacy_tool(legacy, public_name, public_name, spec, lambda value: value))

    for public_name, (legacy_name, transform) in _ALIASES.items():
        spec = specs.get(legacy_name)
        if spec is None:
            continue
        tools.append(_legacy_tool(legacy, public_name, legacy_name, spec, transform))

    tools.extend(
        [
            FunctionAgentTool(
                AgentToolDefinition(
                    name="update_task",
                    description="完成或延期一个已有任务。operation 必须是 complete 或 postpone。",
                    parameters={
                        "type": "object",
                        "properties": {
                            "operation": {"type": "string", "enum": ["complete", "postpone"]},
                            "task_title": {"type": "string"},
                            "task_type": {"type": "string"},
                        },
                        "required": ["operation"],
                        "additionalProperties": False,
                    },
                    input_model=RUNTIME_TOOL_INPUT_MODELS["update_task"],
                    effect=ToolEffect.LOCAL_WRITE,
                    approval=ToolApproval.IF_IMPLICIT,
                    parallel_safe=False,
                    idempotent=False,
                ),
                _update_task_handler(legacy),
            ),
            FunctionAgentTool(
                AgentToolDefinition(
                    name="configure_leetcode_plan",
                    description="开启或关闭每日 LeetCode 计划。",
                    parameters={
                        "type": "object",
                        "properties": {"enabled": {"type": "boolean"}},
                        "required": ["enabled"],
                        "additionalProperties": False,
                    },
                    input_model=RUNTIME_TOOL_INPUT_MODELS["configure_leetcode_plan"],
                    effect=ToolEffect.LOCAL_WRITE,
                    approval=ToolApproval.IF_IMPLICIT,
                    parallel_safe=False,
                    idempotent=True,
                ),
                _configure_leetcode_handler(legacy),
            ),
            FunctionAgentTool(
                AgentToolDefinition(
                    name="update_execution_plan",
                    description=(
                        "Create or replace a concise user-visible execution plan for a complex task. "
                        "The plan is mutable guidance, not an executor."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "goal": {"type": "string"},
                            "revision": {"type": "integer"},
                            "steps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "description": {"type": "string"},
                                        "status": {
                                            "type": "string",
                                            "enum": [
                                                "pending",
                                                "in_progress",
                                                "completed",
                                                "skipped",
                                                "failed",
                                            ],
                                        },
                                        "required": {"type": "boolean"},
                                    },
                                    "required": ["id", "description", "status"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["goal", "steps"],
                        "additionalProperties": False,
                    },
                    input_model=RUNTIME_TOOL_INPUT_MODELS["update_execution_plan"],
                    effect=ToolEffect.LOCAL_WRITE,
                    approval=ToolApproval.NEVER,
                    parallel_safe=False,
                    idempotent=True,
                    control=True,
                ),
                _plan_handler,
            ),
            FunctionAgentTool(
                AgentToolDefinition(
                    name="request_user_input",
                    description="Pause the current run when required information cannot be obtained from tools.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": ["clarification", "preference_form", "selection"],
                            },
                            "prompt": {"type": "string"},
                            "input_schema": {"type": "object"},
                        },
                        "required": ["kind", "prompt"],
                        "additionalProperties": False,
                    },
                    input_model=RUNTIME_TOOL_INPUT_MODELS["request_user_input"],
                    effect=ToolEffect.READ,
                    approval=ToolApproval.NEVER,
                    parallel_safe=False,
                    idempotent=True,
                    control=True,
                ),
                _request_input_handler,
            ),
            FunctionAgentTool(
                AgentToolDefinition(
                    name="reconcile_write_operation",
                    description=(
                        "Resolve a previously uncertain non-calendar write only after "
                        "obtaining reliable evidence that it was applied or not applied. "
                        "This always requires explicit human approval."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "operation_key": {"type": "string", "minLength": 1},
                            "outcome": {
                                "type": "string",
                                "enum": ["applied", "not_applied"],
                            },
                            "evidence": {"type": "string", "minLength": 1},
                        },
                        "required": ["operation_key", "outcome", "evidence"],
                        "additionalProperties": False,
                    },
                    input_model=RUNTIME_TOOL_INPUT_MODELS[
                        "reconcile_write_operation"
                    ],
                    effect=ToolEffect.LOCAL_WRITE,
                    approval=ToolApproval.ALWAYS,
                    parallel_safe=False,
                    idempotent=True,
                    control=True,
                ),
                _reconcile_write_handler,
            ),
        ]
    )
    tools.extend(
        build_runtime_mcp_tools(
            client=mcp_client,
            app_settings=app_settings,
        )
    )
    if offerpilot_repository is not None:
        tools.extend(
            build_study_agent_tools(
                repository=offerpilot_repository,
                knowledge_repository=knowledge_repository,
                calendar_provider=calendar_provider,
            )
        )
    registry = AgentToolRegistry(tools)
    if require_complete:
        registered = {definition.name for definition in registry.definitions()}
        required = REQUIRED_RUNTIME_TOOLS
        if not app_settings.rag_enabled:
            required = required - _RAG_RUNTIME_TOOLS
        missing = sorted(required - registered)
        if missing:
            raise RuntimeError(
                "Required Agent Runtime tools are missing: " + ", ".join(missing)
            )
    return registry


def _legacy_tool(
    legacy: ToolRegistry,
    public_name: str,
    legacy_name: str,
    spec: ToolSpec,
    transform: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> FunctionAgentTool:
    effect, approval = _effect_policy(public_name, spec)
    definition = AgentToolDefinition(
        name=public_name,
        description=spec.description,
        parameters=_parameters_for_spec(spec),
        input_model=RUNTIME_TOOL_INPUT_MODELS.get(public_name),
        cacheable=effect == ToolEffect.READ,
        effect=effect,
        approval=approval,
        parallel_safe=effect == ToolEffect.READ,
        idempotent=effect == ToolEffect.READ,
    )

    async def handler(arguments: Dict[str, Any]) -> AgentToolResult:
        result = await legacy.run_async(legacy_name, transform(dict(arguments)))
        return _convert_legacy_result(result)

    return FunctionAgentTool(definition, handler)


def _parameters_for_spec(spec: ToolSpec) -> Dict[str, Any]:
    names = [
        name
        for name in [*spec.required_slots, *spec.optional_slots]
        if name not in _TRUSTED_ARGUMENTS
    ]
    properties = {name: {} for name in dict.fromkeys(names)}
    return {
        "type": "object",
        "properties": properties,
        "required": [name for name in spec.required_slots if name in properties],
        "additionalProperties": False,
    }


def _effect_policy(public_name: str, spec: ToolSpec) -> tuple[ToolEffect, ToolApproval]:
    if spec.effect:
        effect = ToolEffect(spec.effect)
        if spec.approval:
            approval = ToolApproval(spec.approval)
        elif effect in {ToolEffect.EXTERNAL_WRITE, ToolEffect.DESTRUCTIVE}:
            approval = ToolApproval.ALWAYS
        elif effect == ToolEffect.LOCAL_WRITE:
            approval = ToolApproval.IF_IMPLICIT
        else:
            approval = ToolApproval.NEVER
        return effect, approval
    if public_name in _DESTRUCTIVE:
        return ToolEffect.DESTRUCTIVE, ToolApproval.ALWAYS
    if public_name in _EXTERNAL_WRITES:
        return ToolEffect.EXTERNAL_WRITE, ToolApproval.ALWAYS
    if public_name in _LOCAL_WRITES or spec.mutating:
        return ToolEffect.LOCAL_WRITE, ToolApproval.IF_IMPLICIT
    return ToolEffect.READ, ToolApproval.NEVER


def _convert_legacy_result(result: ToolResult) -> AgentToolResult:
    data = dict(result.data)
    artifact_refs = _artifact_refs(
        data,
        result_succeeded=bool(result.success),
        tool_name=result.tool_name,
    )
    operation_status = (
        str(data.get("operation_status") or "")
        if isinstance(result.data, dict)
        else ""
    )
    if operation_status == "reconciliation_required":
        status = ToolOutcomeStatus.UNKNOWN
        error_code = "write_outcome_unknown"
    elif operation_status == "partial_success":
        status = ToolOutcomeStatus.PARTIAL
        error_code = "external_sync_partial"
    else:
        interaction = None
        if not operation_status:
            interaction = _legacy_input_interaction(result, data)
        if interaction is not None:
            return AgentToolResult(
                data=data,
                status=ToolOutcomeStatus.NEEDS_INPUT,
                message=result.message,
                interaction=interaction,
                artifact_refs=artifact_refs,
            )
        status = ToolOutcomeStatus.SUCCESS if result.success else ToolOutcomeStatus.ERROR
        error_code = None
    if status == ToolOutcomeStatus.ERROR:
        error = result.data.get("error") if isinstance(result.data, dict) else None
        if isinstance(error, dict):
            error_code = str(error.get("code") or "business_error")
        else:
            error_code = "business_error"
    return AgentToolResult(
        data=data,
        is_error=status != ToolOutcomeStatus.SUCCESS,
        status=status,
        message=result.message,
        error_code=error_code,
        retryable=(
            bool(result.data.get("retryable"))
            if isinstance(result.data, dict)
            else False
        ),
        artifact_refs=artifact_refs,
    )


def _legacy_input_interaction(
    result: ToolResult,
    data: Dict[str, Any],
) -> Optional[ToolInteraction]:
    """Translate explicit pre-execution legacy signals into native interactions."""

    if data.get("requires_selection") is True:
        slot = str(data.get("selection_slot") or "").strip()
        candidates = data.get("candidates")
        if slot and isinstance(candidates, list) and candidates:
            return _legacy_selection_interaction(
                result.message,
                slot,
                candidates,
            )

    missing_slots = data.get("missing_slots")
    if not result.success and isinstance(missing_slots, list):
        slots = []
        for item in missing_slots:
            if not isinstance(item, str):
                continue
            slot = item.strip()
            if slot and slot not in slots:
                slots.append(slot)
        if slots:
            candidates = data.get("candidates")
            if len(slots) == 1 and isinstance(candidates, list) and candidates:
                return _legacy_selection_interaction(
                    result.message,
                    slots[0],
                    candidates,
                )
            return ToolInteraction(
                kind="clarification",
                prompt=result.message or "请补充必要信息。",
                input_schema={
                    "type": "object",
                    "properties": {
                        slot: {"type": "string", "minLength": 1}
                        for slot in slots
                    },
                    "required": slots,
                    "additionalProperties": False,
                },
                allowed_actions=["answer", "cancel"],
                metadata={"missing_slots": slots},
            )
    return None


def _legacy_selection_interaction(
    prompt: str,
    slot: str,
    candidates: list[Any],
) -> ToolInteraction:
    safe_candidates = json.loads(
        json.dumps(candidates, ensure_ascii=False, default=str)
    )
    choices = _legacy_selection_values(slot, safe_candidates)
    property_schema: Dict[str, Any] = {
        "type": _json_schema_scalar_type(choices),
    }
    if property_schema["type"] == "string":
        property_schema["minLength"] = 1
    if choices:
        property_schema["enum"] = choices
    return ToolInteraction(
        kind="selection",
        prompt=prompt or "请选择要继续处理的记录。",
        input_schema={
            "type": "object",
            "properties": {slot: property_schema},
            "required": [slot],
            "additionalProperties": False,
            "x-candidates": safe_candidates,
        },
        allowed_actions=["answer", "cancel"],
        metadata={
            "selection_slot": slot,
            "candidates": safe_candidates,
        },
    )


def _legacy_selection_values(
    slot: str,
    candidates: list[Any],
) -> list[Any]:
    if slot == "problem_index":
        return list(range(1, len(candidates) + 1))

    fallback_keys = {
        "project": ("name", "id"),
        "task_title": ("title", "name", "id"),
    }
    keys = (
        (slot, "id")
        if slot.endswith("_id")
        else (slot, *fallback_keys.get(slot, ()))
    )
    values: list[Any] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        raw_value = next(
            (candidate.get(key) for key in keys if candidate.get(key) is not None),
            None,
        )
        if raw_value is None and slot.endswith("_id"):
            nested = candidate.get(slot.removesuffix("_id"))
            if isinstance(nested, dict):
                raw_value = nested.get("id")
        if raw_value is None:
            continue
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()
        if (
            isinstance(raw_value, (str, int, float, bool))
            and raw_value != ""
            and raw_value not in values
        ):
            values.append(raw_value)
    return values


def _json_schema_scalar_type(values: list[Any]) -> str:
    if values and all(isinstance(value, bool) for value in values):
        return "boolean"
    if values and all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return "integer"
    if values and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in values
    ):
        return "number"
    return "string"


def _artifact_refs(
    data: Dict[str, Any],
    *,
    result_succeeded: bool,
    tool_name: str,
) -> list[Dict[str, Any]]:
    artifacts: list[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    if result_succeeded:
        bitable_urls: list[str] = []
        bitable_sync = data.get("bitable_sync")
        if isinstance(bitable_sync, dict):
            url = _successful_bitable_url(bitable_sync)
            if url:
                bitable_urls.append(url)
        bitable_sync_results = data.get("bitable_sync_results")
        if isinstance(bitable_sync_results, list):
            for sync_result in bitable_sync_results:
                if not isinstance(sync_result, dict):
                    continue
                url = _successful_bitable_url(sync_result)
                if url:
                    bitable_urls.append(url)
        if tool_name == "get_application_bitable_link":
            direct_url = data.get("web_url")
            if isinstance(direct_url, str) and direct_url.strip():
                bitable_urls.append(direct_url.strip())

        for url in bitable_urls:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            artifacts.append(
                {
                    "type": "feishu_bitable",
                    "id": "applications_table",
                    "title": "投递记录",
                    "url": url,
                    "data": {"button_text": "打开多维表格"},
                }
            )

    for key, value in data.items():
        if not isinstance(value, str) or not value:
            continue
        if key.endswith("_url") and value not in seen_urls:
            seen_urls.add(value)
            artifacts.append({"type": "link", "id": key, "title": key, "data": {"url": value}})
    return artifacts


def _successful_bitable_url(sync_result: Dict[str, Any]) -> Optional[str]:
    if sync_result.get("synced") is not True:
        return None
    value = sync_result.get("web_url")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _update_task_handler(legacy: ToolRegistry):
    async def handler(arguments: Dict[str, Any]) -> AgentToolResult:
        operation = arguments.get("operation")
        selected = "complete_task" if operation == "complete" else "postpone_task"
        payload = {key: value for key, value in arguments.items() if key != "operation"}
        return _convert_legacy_result(await legacy.run_async(selected, payload))

    return handler


def _configure_leetcode_handler(legacy: ToolRegistry):
    async def handler(arguments: Dict[str, Any]) -> AgentToolResult:
        selected = "enable_leetcode_plan" if arguments.get("enabled") else "disable_leetcode_plan"
        return _convert_legacy_result(await legacy.run_async(selected, arguments))

    return handler


async def _plan_handler(arguments: Dict[str, Any]) -> AgentToolResult:
    return AgentToolResult(
        data={"plan": json.loads(json.dumps(arguments, ensure_ascii=False))},
        status=ToolOutcomeStatus.SUCCESS,
        message="Execution plan updated.",
    )


async def _request_input_handler(arguments: Dict[str, Any]) -> AgentToolResult:
    interaction = ToolInteraction(
        kind=str(arguments.get("kind") or "clarification"),
        prompt=str(arguments.get("prompt") or "请补充信息。"),
        input_schema=dict(arguments.get("input_schema") or {}),
        allowed_actions=["answer", "cancel"],
    )
    return AgentToolResult(
        data={},
        status=ToolOutcomeStatus.NEEDS_INPUT,
        message=interaction.prompt,
        interaction=interaction,
    )


async def _reconcile_write_handler(arguments: Dict[str, Any]) -> AgentToolResult:
    return AgentToolResult(
        data={
            "operation_key": str(arguments["operation_key"]),
            "reconciled_outcome": str(arguments["outcome"]),
            "evidence": str(arguments["evidence"]),
        },
        status=ToolOutcomeStatus.SUCCESS,
        message="The write reconciliation assertion was accepted for ledger resolution.",
    )
