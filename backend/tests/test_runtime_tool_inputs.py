import pytest
from pydantic import ValidationError

from app.tools.runtime_tool_inputs import (
    CancelInterviewInput,
    CreateStudyPlanInput,
    GetCalendarAvailabilityInput,
    GetStudyPlanInput,
    ListStudyPlansInput,
    RUNTIME_TOOL_INPUT_MODELS,
    RecordLeetCodeResultInput,
    SaveStudyPreferencesInput,
    ScheduleInterviewInput,
    UpdateExecutionPlanInput,
    UpdateApplicationInput,
    UpdateStudySessionInput,
)


EXPECTED_RUNTIME_TOOLS = {
    "list_tasks",
    "list_applications",
    "get_application_bitable_link",
    "create_application",
    "update_application",
    "list_interviews",
    "schedule_interview",
    "reschedule_interview",
    "cancel_interview",
    "update_task",
    "record_interview_review",
    "configure_leetcode_plan",
    "get_today_leetcode",
    "record_leetcode_result",
    "start_mock_interview",
    "start_project_training",
    "resume_project_training",
    "get_project_training_summary",
    "update_execution_plan",
    "request_user_input",
    "search_interview_knowledge",
    "search_project_evidence",
    "read_evidence",
    "reconcile_write_operation",
    "get_study_preferences",
    "save_study_preferences",
    "get_calendar_availability",
    "create_study_plan",
    "get_study_plan",
    "list_study_plans",
    "sync_study_plan_to_calendar",
    "reconcile_study_calendar_session",
    "update_study_session",
    "list_learning_gaps",
}
TRUSTED_FIELDS = {
    "owner_id",
    "raw_message",
    "idempotency_key",
    "attendee_user_id",
    "attendee_user_id_type",
    "bitable_collaborator_user_id",
    "bitable_collaborator_user_id_type",
    "retry_after_reconciliation",
}


def test_mapping_covers_every_public_runtime_tool() -> None:
    assert set(RUNTIME_TOOL_INPUT_MODELS) == EXPECTED_RUNTIME_TOOLS


def test_execution_plan_revision_is_omitted_until_runtime_assigns_it() -> None:
    value = UpdateExecutionPlanInput.model_validate(
        {
            "goal": "完成复习安排",
            "steps": [
                {
                    "id": "step_1",
                    "description": "读取面试",
                    "status": "pending",
                }
            ],
        }
    )

    assert "revision" not in value.model_dump(exclude_none=True)


@pytest.mark.parametrize("tool_name, model", RUNTIME_TOOL_INPUT_MODELS.items())
def test_models_forbid_trusted_and_unknown_fields(tool_name, model) -> None:
    schema_text = str(model.model_json_schema())
    assert not TRUSTED_FIELDS.intersection(model.model_fields)
    assert not any(field in schema_text for field in TRUSTED_FIELDS)
    with pytest.raises(ValidationError):
        model.model_validate({"owner_id": "injected-too-early"})


def test_schedule_requires_a_locator() -> None:
    with pytest.raises(ValidationError, match="application_id or company"):
        ScheduleInterviewInput.model_validate(
            {"round": "一面", "interview_time": "明天下午三点"}
        )
    value = ScheduleInterviewInput.model_validate(
        {
            "application_id": "app_1",
            "round": "一面",
            "interview_time": "明天下午三点",
            "start_at": "2026-09-03T15:00:00+08:00",
        }
    )
    assert value.application_id == "app_1"


def test_update_application_requires_locator_and_actual_change() -> None:
    with pytest.raises(ValidationError, match="application_id or company"):
        UpdateApplicationInput.model_validate({"status": "offer"})
    with pytest.raises(ValidationError, match="at least one application update"):
        UpdateApplicationInput.model_validate({"company": "OpenAI"})
    with pytest.raises(ValidationError, match="requires round"):
        UpdateApplicationInput.model_validate(
            {
                "company": "OpenAI",
                "update_type": "schedule_interview",
                "interview_time": "明天下午三点",
            }
        )


def test_cancel_interview_requires_stable_or_human_locator() -> None:
    with pytest.raises(ValidationError, match="schedule_id, application_id, or company"):
        CancelInterviewInput.model_validate({"round": "一面"})
    assert CancelInterviewInput.model_validate({"schedule_id": "schedule_1"})


def test_record_leetcode_result_requires_one_unambiguous_problem() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        RecordLeetCodeResultInput.model_validate({"result": "independent"})
    with pytest.raises(ValidationError, match="exactly one"):
        RecordLeetCodeResultInput.model_validate(
            {
                "result": "independent",
                "assignment_id": "assignment_1",
                "problem_index": 1,
            }
        )
    result = RecordLeetCodeResultInput.model_validate(
        {"result": "with_hint", "problem_index": 2}
    )
    assert result.problem_index == 2


def test_update_study_session_requires_complete_valid_change() -> None:
    with pytest.raises(ValidationError, match="status or a new"):
        UpdateStudySessionInput.model_validate({"session_id": "session_1"})
    with pytest.raises(ValidationError, match="supplied together"):
        UpdateStudySessionInput.model_validate(
            {"session_id": "session_1", "start_at": "2026-09-03T10:00:00+08:00"}
        )
    with pytest.raises(ValidationError, match="end_at must be after"):
        UpdateStudySessionInput.model_validate(
            {
                "session_id": "session_1",
                "start_at": "2026-09-03T11:00:00+08:00",
                "end_at": "2026-09-03T10:00:00+08:00",
            }
        )


def test_study_time_ranges_and_preferences_are_validated() -> None:
    with pytest.raises(ValidationError, match="timezone offset"):
        GetCalendarAvailabilityInput.model_validate(
            {
                "range_start": "2026-09-03T10:00:00",
                "range_end": "2026-09-03T12:00:00+08:00",
            }
        )
    with pytest.raises(ValidationError, match="after range_start"):
        GetCalendarAvailabilityInput.model_validate(
            {
                "range_start": "2026-09-03T12:00:00+08:00",
                "range_end": "2026-09-03T10:00:00+08:00",
            }
        )
    with pytest.raises(ValidationError, match="availability window"):
        SaveStudyPreferencesInput.model_validate(
            {
                "timezone": "Asia/Shanghai",
                "weekday_windows": [],
                "weekend_windows": [],
                "daily_max_minutes": 120,
                "session_minutes": 30,
            }
        )


def test_create_study_plan_validates_date_range_and_rejects_model_busy_data() -> None:
    payload = {
        "goal": "准备后端面试",
        "range_start": "2026-09-07",
        "range_end": "2026-09-06",
        "topics": [{"topic": "JVM", "duration_minutes": 60, "priority": 3}],
    }
    with pytest.raises(ValidationError, match="must not be before"):
        CreateStudyPlanInput.model_validate(payload)

    payload["range_end"] = "2026-09-10"
    payload["busy_intervals"] = [
        {
            "start_at": "2026-09-08T11:00:00+08:00",
            "end_at": "2026-09-08T10:00:00+08:00",
        }
    ]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CreateStudyPlanInput.model_validate(payload)


def test_study_plan_queries_validate_locator_filter_and_limit() -> None:
    assert GetStudyPlanInput.model_validate({"plan_id": "plan_1"}).plan_id == "plan_1"
    assert ListStudyPlansInput.model_validate({}).limit == 20
    assert ListStudyPlansInput.model_validate({"status": "synced", "limit": 50})
    with pytest.raises(ValidationError):
        GetStudyPlanInput.model_validate({"plan_id": " "})
    with pytest.raises(ValidationError):
        ListStudyPlansInput.model_validate({"status": "not_a_status"})
    with pytest.raises(ValidationError):
        ListStudyPlansInput.model_validate({"limit": 51})


def test_scalar_types_are_strict() -> None:
    with pytest.raises(ValidationError):
        RUNTIME_TOOL_INPUT_MODELS["configure_leetcode_plan"].model_validate(
            {"enabled": "true"}
        )
    with pytest.raises(ValidationError):
        RUNTIME_TOOL_INPUT_MODELS["list_learning_gaps"].model_validate({"limit": "10"})
    with pytest.raises(ValidationError):
        RUNTIME_TOOL_INPUT_MODELS["create_study_plan"].model_validate(
            {
                "goal": "准备面试",
                "range_start": "2026-09-03",
                "range_end": "2026-09-04",
                "topics": [
                    {"topic": "JVM", "duration_minutes": 30, "priority": True}
                ],
            }
        )


def test_write_reconciliation_requires_a_supported_outcome_and_evidence() -> None:
    model = RUNTIME_TOOL_INPUT_MODELS["reconcile_write_operation"]
    with pytest.raises(ValidationError):
        model.model_validate(
            {"operation_key": "agent_op_1", "outcome": "applied"}
        )
    with pytest.raises(ValidationError):
        model.model_validate(
            {
                "operation_key": "agent_op_1",
                "outcome": "maybe",
                "evidence": "checked the destination",
            }
        )
    assert model.model_validate(
        {
            "operation_key": "agent_op_1",
            "outcome": "not_applied",
            "evidence": "The user checked the destination and found no record.",
        }
    ).outcome == "not_applied"
