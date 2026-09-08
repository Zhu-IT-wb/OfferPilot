import asyncio

import pytest

from app.core.config import Settings
from app.models.tool_calling import ModelToolCall
from app.repositories.leetcode_repository import InMemoryLeetCodeRepository
from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository
from app.schemas.tool import ToolResult
from app.tools.agent_tool import ToolApproval, ToolEffect, ToolOutcomeStatus
from app.tools.offerpilot_tools import (
    build_offerpilot_tool_registry,
    list_today_tasks,
    query_application,
)
from app.tools.registry import ToolRegistry
from app.tools.runtime_tools import REQUIRED_RUNTIME_TOOLS, build_runtime_tool_registry
from app.tools.runtime_tool_inputs import RUNTIME_TOOL_INPUT_MODELS


def test_every_registered_runtime_tool_uses_one_strict_input_contract() -> None:
    legacy = ToolRegistry()
    legacy.register(
        "query_application",
        lambda arguments: ToolResult(
            tool_name="query_application", success=True, message="ok", data={}
        ),
        optional_slots=["query_type", "company"],
    )
    registry = build_runtime_tool_registry(
        legacy,
        offerpilot_repository=InMemoryOfferPilotRepository(),
    )

    for definition in registry.definitions():
        assert definition.name in RUNTIME_TOOL_INPUT_MODELS
        assert definition.input_model is RUNTIME_TOOL_INPUT_MODELS[definition.name]
        schema = definition.to_model_schema()["function"]["parameters"]
        assert schema["additionalProperties"] is False


def test_complete_runtime_registry_fails_fast_when_a_required_tool_is_missing() -> None:
    legacy = ToolRegistry()
    legacy.register(
        "query_application",
        lambda arguments: ToolResult(
            tool_name="query_application", success=True, message="ok", data={}
        ),
    )

    with pytest.raises(RuntimeError, match="Required Agent Runtime tools are missing"):
        build_runtime_tool_registry(legacy, require_complete=True)

    assert "list_applications" in REQUIRED_RUNTIME_TOOLS
    assert "get_application_bitable_link" in REQUIRED_RUNTIME_TOOLS
    assert "get_study_plan" in REQUIRED_RUNTIME_TOOLS
    assert "list_study_plans" in REQUIRED_RUNTIME_TOOLS
    assert "reconcile_write_operation" in REQUIRED_RUNTIME_TOOLS
    assert "sync_study_plan_to_calendar" in REQUIRED_RUNTIME_TOOLS


def test_generic_write_reconciliation_is_a_standalone_approved_control_tool() -> None:
    registry = build_runtime_tool_registry(ToolRegistry())
    definition = registry.get_definition("reconcile_write_operation")

    assert definition is not None
    assert definition.approval == ToolApproval.ALWAYS
    assert definition.effect == ToolEffect.LOCAL_WRITE
    assert definition.parallel_safe is False
    assert definition.control is True


def test_application_write_effect_reflects_enabled_external_integrations() -> None:
    repository = InMemoryOfferPilotRepository()
    local_registry = build_runtime_tool_registry(
        build_offerpilot_tool_registry(
            repository,
            calendar_service=None,
            bitable_service=None,
        )
    )
    external_registry = build_runtime_tool_registry(
        build_offerpilot_tool_registry(
            repository,
            calendar_service=object(),
            bitable_service=None,
        )
    )

    local_create = local_registry.get_definition("create_application")
    external_create = external_registry.get_definition("create_application")
    assert local_create.effect == ToolEffect.LOCAL_WRITE
    assert local_create.approval.value == "if_implicit"
    assert external_create.effect == ToolEffect.EXTERNAL_WRITE
    assert external_create.approval.value == "always"


def test_list_interviews_bridge_uses_upcoming_interview_query() -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="美团", role="Java 后端")
    schedule = repository.create_interview_schedule(
        application_id=application.id,
        company=application.company,
        role=application.role,
        round_name="一面",
        start_time="明天下午三点",
    )
    legacy = ToolRegistry()
    legacy.register(
        "query_application",
        lambda arguments: query_application(repository, arguments),
        description="query applications",
        optional_slots=["query_type", "company"],
    )
    registry = build_runtime_tool_registry(legacy)

    result = asyncio.run(
        registry.execute(
            ModelToolCall(id="interviews", name="list_interviews", arguments={})
        )
    )

    assert result.effective_status == ToolOutcomeStatus.SUCCESS
    assert result.data["query_type"] == "upcoming_interviews"
    assert result.data["interview_schedules"][0]["id"] == schedule.id


def test_list_interviews_filters_by_a_trusted_absolute_range() -> None:
    repository = InMemoryOfferPilotRepository()
    in_range_application = repository.create_application(company="Acme", role="Backend")
    out_of_range_application = repository.create_application(company="Beta", role="Backend")
    in_range = repository.create_interview_schedule(
        application_id=in_range_application.id,
        company=in_range_application.company,
        role=in_range_application.role,
        round_name="first",
        start_at="2026-09-03T10:00:00+08:00",
    )
    repository.create_interview_schedule(
        application_id=out_of_range_application.id,
        company=out_of_range_application.company,
        role=out_of_range_application.role,
        round_name="first",
        start_at="2026-09-10T10:00:00+08:00",
    )
    legacy = ToolRegistry()
    legacy.register(
        "query_application",
        lambda arguments: query_application(repository, arguments),
        description="query applications",
        optional_slots=["query_type", "company"],
    )
    registry = build_runtime_tool_registry(legacy)

    result = asyncio.run(
        registry.execute(
            ModelToolCall(
                id="weekly_interviews",
                name="list_interviews",
                arguments={
                    "range_start": "2026-08-31T00:00:00+08:00",
                    "range_end": "2026-09-07T00:00:00+08:00",
                },
            )
        )
    )

    assert result.effective_status == ToolOutcomeStatus.SUCCESS
    assert [item["id"] for item in result.data["interview_schedules"]] == [in_range.id]
    assert [item["id"] for item in result.data["applications"]] == [
        in_range_application.id
    ]


def test_list_interviews_rejects_an_incomplete_or_reversed_range() -> None:
    model = RUNTIME_TOOL_INPUT_MODELS["list_interviews"]

    with pytest.raises(ValueError):
        model.model_validate({"range_start": "2026-09-01T00:00:00+08:00"})
    with pytest.raises(ValueError):
        model.model_validate(
            {
                "range_start": "2026-09-07T00:00:00+08:00",
                "range_end": "2026-09-01T00:00:00+08:00",
            }
        )


def test_list_tasks_does_not_create_a_leetcode_assignment() -> None:
    repository = InMemoryOfferPilotRepository()
    leetcode_repository = InMemoryLeetCodeRepository()
    legacy = ToolRegistry()
    legacy.register(
        "list_today_tasks",
        lambda arguments: list_today_tasks(
            repository,
            arguments,
            leetcode_repository=leetcode_repository,
        ),
        description="list existing tasks",
    )
    registry = build_runtime_tool_registry(legacy)

    result = asyncio.run(
        registry.execute(ModelToolCall(id="tasks", name="list_tasks", arguments={}))
    )

    assert result.effective_status == ToolOutcomeStatus.SUCCESS
    assert result.data["leetcode_recommendations"] == []
    assert leetcode_repository.list_assignments(owner_id="local_user") == []


def test_leetcode_runtime_exposes_three_nonduplicated_capabilities() -> None:
    legacy = ToolRegistry()
    for name, mutating in (
        ("enable_leetcode_plan", True),
        ("disable_leetcode_plan", True),
        ("get_today_leetcode", False),
        ("record_leetcode_result", True),
    ):
        legacy.register(
            name,
            lambda arguments, tool_name=name: ToolResult(
                tool_name=tool_name,
                success=True,
                message="ok",
                data={},
            ),
            description=name,
            mutating=mutating,
        )
    registry = build_runtime_tool_registry(legacy)
    definitions = {item.name: item for item in registry.definitions()}

    assert "enable_leetcode_plan" not in definitions
    assert "disable_leetcode_plan" not in definitions
    assert {
        "configure_leetcode_plan",
        "get_today_leetcode",
        "record_leetcode_result",
    }.issubset(definitions)
    assert definitions["get_today_leetcode"].effect == ToolEffect.LOCAL_WRITE
    assert definitions["get_today_leetcode"].parallel_safe is False


def test_legacy_external_operation_status_maps_to_runtime_outcome() -> None:
    legacy = ToolRegistry()
    legacy.register(
        "update_application",
        lambda arguments: ToolResult(
            tool_name="update_application",
            success=True,
            message="local update completed",
            data={
                "operation_status": arguments["operation_status"],
                "retryable": arguments.get("retryable", False),
            },
        ),
        description="update",
        mutating=True,
        required_slots=["operation_status"],
        optional_slots=["retryable"],
    )
    registry = build_runtime_tool_registry(legacy)

    unknown = asyncio.run(
        registry.execute(
            ModelToolCall(
                id="unknown",
                name="update_application",
                arguments={"operation_status": "reconciliation_required"},
            )
        )
    )
    partial = asyncio.run(
        registry.execute(
            ModelToolCall(
                id="partial",
                name="update_application",
                arguments={"operation_status": "partial_success", "retryable": True},
            )
        )
    )

    assert unknown.effective_status == ToolOutcomeStatus.UNKNOWN
    assert unknown.error_code == "write_outcome_unknown"
    assert unknown.retryable is False
    assert partial.effective_status == ToolOutcomeStatus.PARTIAL
    assert partial.error_code == "external_sync_partial"
    assert partial.retryable is True


def test_successful_bitable_sync_becomes_a_deduplicated_application_table_artifact() -> None:
    url = "https://feishu.cn/base/bascn_offerpilot?table=tbl_applications"
    legacy = ToolRegistry()
    legacy.register(
        "update_application",
        lambda _arguments: ToolResult(
            tool_name="update_application",
            success=True,
            message="updated",
            data={
                "operation_status": "partial_success",
                "bitable_sync_results": [
                    {"synced": True, "status": "synced", "web_url": url},
                    {"synced": True, "status": "synced", "web_url": url},
                    {
                        "synced": False,
                        "status": "failed",
                        "web_url": "https://feishu.cn/base/must-not-be-exposed",
                    },
                ]
            },
        ),
        mutating=True,
    )
    registry = build_runtime_tool_registry(legacy)

    result = asyncio.run(
        registry.execute(
            ModelToolCall(id="bulk_update", name="update_application", arguments={})
        )
    )

    assert result.effective_status == ToolOutcomeStatus.PARTIAL
    assert result.artifact_refs == [
        {
            "type": "feishu_bitable",
            "id": "applications_table",
            "title": "投递记录",
            "url": url,
            "data": {"button_text": "打开多维表格"},
        }
    ]


@pytest.mark.parametrize(
    ("legacy_name", "runtime_name"),
    [
        ("create_application", "create_application"),
        ("update_application", "update_application"),
        ("update_application", "schedule_interview"),
        ("reschedule_interview", "reschedule_interview"),
        ("cancel_interview", "cancel_interview"),
    ],
)
def test_every_application_write_bridge_preserves_a_successful_bitable_link(
    legacy_name: str,
    runtime_name: str,
) -> None:
    url = "https://feishu.cn/base/bascn_offerpilot?table=tbl_applications"
    legacy = ToolRegistry()
    legacy.register(
        legacy_name,
        lambda _arguments: ToolResult(
            tool_name=legacy_name,
            success=True,
            message="done",
            data={
                "bitable_sync": {
                    "synced": True,
                    "status": "synced",
                    "web_url": url,
                }
            },
        ),
        mutating=True,
    )
    registry = build_runtime_tool_registry(legacy)

    result = asyncio.run(
        registry.execute(
            ModelToolCall(id="write_with_link", name=runtime_name, arguments={})
        )
    )

    assert result.artifact_refs == [
        {
            "type": "feishu_bitable",
            "id": "applications_table",
            "title": "投递记录",
            "url": url,
            "data": {"button_text": "打开多维表格"},
        }
    ]


def test_failed_legacy_result_never_exposes_a_bitable_artifact() -> None:
    legacy = ToolRegistry()
    legacy.register(
        "create_application",
        lambda _arguments: ToolResult(
            tool_name="create_application",
            success=False,
            message="failed",
            data={
                "bitable_sync": {
                    "synced": True,
                    "status": "synced",
                    "web_url": "https://feishu.cn/base/must-not-be-exposed",
                }
            },
        ),
        mutating=True,
    )
    registry = build_runtime_tool_registry(legacy)

    result = asyncio.run(
        registry.execute(
            ModelToolCall(
                id="failed_create",
                name="create_application",
                arguments={},
            )
        )
    )

    assert result.effective_status == ToolOutcomeStatus.ERROR
    assert result.artifact_refs == []


def test_application_bitable_link_is_a_strict_cacheable_read_tool() -> None:
    repository = InMemoryOfferPilotRepository()
    repository.set_runtime_setting(
        "feishu.offerpilot_bitable_app_token", "bascn_offerpilot"
    )
    repository.set_runtime_setting(
        "feishu.offerpilot_bitable_table_id", "tbl_applications"
    )
    repository.set_runtime_setting(
        "feishu.offerpilot_bitable_collaborator.bascn_offerpilot.ou_test",
        "ou_test",
    )
    repository.set_runtime_setting(
        "feishu.offerpilot_bitable_owner.bascn_offerpilot.tbl_applications",
        "feishu:ou_test",
    )
    app_settings = Settings(
        feishu_bitable_sync_enabled=True,
        feishu_bitable_web_base_url="https://example.feishu.cn/base",
    )
    legacy = build_offerpilot_tool_registry(
        repository,
        calendar_service=None,
        bitable_service=None,
        app_settings=app_settings,
    )
    registry = build_runtime_tool_registry(legacy, app_settings=app_settings)
    definition = registry.get_definition("get_application_bitable_link")

    assert definition is not None
    assert definition.effect == ToolEffect.READ
    assert definition.approval == ToolApproval.NEVER
    assert definition.parallel_safe is True
    assert definition.cacheable is True
    assert definition.idempotent is True
    assert definition.to_model_schema()["function"]["parameters"] == {
        "additionalProperties": False,
        "properties": {},
        "title": "EmptyInput",
        "type": "object",
    }

    result = asyncio.run(
        registry.execute(
            ModelToolCall(
                id="table_link",
                name="get_application_bitable_link",
                arguments={
                    "owner_id": "feishu:ou_test",
                    "bitable_collaborator_user_id": "ou_test",
                },
            )
        )
    )

    assert result.effective_status == ToolOutcomeStatus.SUCCESS
    assert result.artifact_refs == [
        {
            "type": "feishu_bitable",
            "id": "applications_table",
            "title": "投递记录",
            "url": "https://example.feishu.cn/base/bascn_offerpilot?table=tbl_applications",
            "data": {"button_text": "打开多维表格"},
        }
    ]


@pytest.mark.parametrize(
    (
        "legacy_name",
        "runtime_name",
        "arguments",
        "selection_slot",
        "candidates",
        "choices",
        "success",
    ),
    [
        (
            "update_application",
            "update_application",
            {"application_id": "ambiguous", "status": "screening"},
            "application_id",
            [
                {"application_id": "app_1", "company": "A"},
                {"application_id": "app_2", "company": "A"},
            ],
            ["app_1", "app_2"],
            False,
        ),
        (
            "reschedule_interview",
            "reschedule_interview",
            {"company": "A", "interview_time": "2026-09-03 10:00"},
            "schedule_id",
            [
                {"schedule_id": "schedule_1", "company": "A"},
                {"schedule_id": "schedule_2", "company": "A"},
            ],
            ["schedule_1", "schedule_2"],
            False,
        ),
        (
            "complete_task",
            "update_task",
            {"operation": "complete", "task_type": "interview_question"},
            "task_title",
            [
                {"id": "task_1", "title": "JVM"},
                {"id": "task_2", "title": "Redis"},
            ],
            ["JVM", "Redis"],
            False,
        ),
        (
            "start_project_training",
            "start_project_training",
            {},
            "project",
            [
                {"id": "project_1", "name": "OfferPilot"},
                {"id": "project_2", "name": "Order Service"},
            ],
            ["OfferPilot", "Order Service"],
            True,
        ),
    ],
)
def test_legacy_ambiguous_results_become_native_selection_interactions(
    legacy_name,
    runtime_name,
    arguments,
    selection_slot,
    candidates,
    choices,
    success,
) -> None:
    legacy = ToolRegistry()
    legacy.register(
        legacy_name,
        lambda _arguments: ToolResult(
            tool_name=legacy_name,
            success=success,
            message="请选择一条记录。",
            data={
                "requires_selection": True,
                "selection_slot": selection_slot,
                "candidates": candidates,
            },
        ),
        mutating=True,
    )
    registry = build_runtime_tool_registry(legacy)

    result = asyncio.run(
        registry.execute(
            ModelToolCall(id="selection", name=runtime_name, arguments=arguments)
        )
    )

    assert result.effective_status == ToolOutcomeStatus.NEEDS_INPUT
    assert result.is_error is False
    assert result.interaction is not None
    assert result.interaction.kind == "selection"
    assert result.interaction.allowed_actions == ["answer", "cancel"]
    assert result.interaction.input_schema["required"] == [selection_slot]
    assert result.interaction.input_schema["properties"][selection_slot]["enum"] == choices
    assert result.interaction.input_schema["x-candidates"] == candidates


def test_legacy_missing_slots_become_a_clarification_interaction() -> None:
    legacy = ToolRegistry()
    legacy.register(
        "start_mock_interview",
        lambda _arguments: ToolResult(
            tool_name="start_mock_interview",
            success=False,
            message="请补充面试方向。",
            data={"missing_slots": ["role", "project", "role", ""]},
        ),
    )
    registry = build_runtime_tool_registry(legacy)

    result = asyncio.run(
        registry.execute(
            ModelToolCall(id="clarification", name="start_mock_interview", arguments={})
        )
    )

    assert result.effective_status == ToolOutcomeStatus.NEEDS_INPUT
    assert result.interaction is not None
    assert result.interaction.kind == "clarification"
    assert result.interaction.input_schema["required"] == ["role", "project"]


def test_legacy_task_candidates_with_a_missing_locator_become_selection() -> None:
    legacy = ToolRegistry()
    candidates = [
        {
            "assignment": {"id": "assignment_1"},
            "problem": {"title_zh": "两数之和"},
        },
        {
            "assignment": {"id": "assignment_2"},
            "problem": {"title_zh": "反转链表"},
        },
    ]
    legacy.register(
        "record_leetcode_result",
        lambda _arguments: ToolResult(
            tool_name="record_leetcode_result",
            success=False,
            message="今天有多道题，请选择第几题。",
            data={
                "missing_slots": ["problem_index"],
                "candidates": candidates,
            },
        ),
        mutating=True,
    )
    registry = build_runtime_tool_registry(legacy)

    result = asyncio.run(
        registry.execute(
            ModelToolCall(
                id="task_selection",
                name="record_leetcode_result",
                arguments={"result": "independent", "problem_title": "链表"},
            )
        )
    )

    assert result.effective_status == ToolOutcomeStatus.NEEDS_INPUT
    assert result.interaction is not None
    assert result.interaction.kind == "selection"
    field = result.interaction.input_schema["properties"]["problem_index"]
    assert field == {"type": "integer", "enum": [1, 2]}
    assert result.interaction.input_schema["x-candidates"] == candidates


def test_real_legacy_task_ambiguity_reaches_runtime_selection_protocol() -> None:
    repository = InMemoryOfferPilotRepository()
    legacy = build_offerpilot_tool_registry(
        repository,
        calendar_service=None,
        bitable_service=None,
    )
    registry = build_runtime_tool_registry(legacy)

    result = asyncio.run(
        registry.execute(
            ModelToolCall(
                id="ambiguous_task",
                name="update_task",
                arguments={"operation": "complete"},
            )
        )
    )

    assert result.effective_status == ToolOutcomeStatus.NEEDS_INPUT
    assert result.interaction is not None
    assert result.interaction.kind == "selection"
    assert result.interaction.input_schema["properties"]["task_title"]["enum"] == [
        "HashMap 扩容机制",
        "云聚图库 Caffeine + Redis 两级缓存设计",
    ]
    assert all(task.status.value == "pending" for task in repository.list_today_tasks())


def test_legacy_completed_or_partial_side_effect_is_not_reclassified_as_input() -> None:
    legacy = ToolRegistry()
    responses = iter(
        [
            ToolResult(
                tool_name="create_application",
                success=True,
                message="created",
                data={"application": {"id": "app_1"}, "missing_slots": ["role"]},
            ),
            ToolResult(
                tool_name="create_application",
                success=True,
                message="local write completed",
                data={
                    "application": {"id": "app_1"},
                    "operation_status": "partial_success",
                    "requires_selection": True,
                    "selection_slot": "application_id",
                    "candidates": [{"application_id": "app_1"}],
                },
            ),
            ToolResult(
                tool_name="create_application",
                success=True,
                message="local write completed without external sync",
                data={
                    "application": {"id": "app_1"},
                    "operation_status": "completed_local_only",
                    "requires_selection": True,
                    "selection_slot": "application_id",
                    "candidates": [{"application_id": "app_1"}],
                },
            ),
        ]
    )
    legacy.register(
        "create_application",
        lambda _arguments: next(responses),
        mutating=True,
    )
    registry = build_runtime_tool_registry(legacy)
    call = ModelToolCall(
        id="write",
        name="create_application",
        arguments={"company": "A", "role": "Engineer"},
    )

    completed = asyncio.run(registry.execute(call))
    partial = asyncio.run(
        registry.execute(
            ModelToolCall(
                id="partial",
                name="create_application",
                arguments={"company": "A", "role": "Engineer"},
            )
        )
    )
    completed_local_only = asyncio.run(
        registry.execute(
            ModelToolCall(
                id="completed_local_only",
                name="create_application",
                arguments={"company": "A", "role": "Engineer"},
            )
        )
    )

    assert completed.effective_status == ToolOutcomeStatus.SUCCESS
    assert completed.interaction is None
    assert partial.effective_status == ToolOutcomeStatus.PARTIAL
    assert partial.interaction is None
    assert completed_local_only.effective_status == ToolOutcomeStatus.SUCCESS
    assert completed_local_only.interaction is None
