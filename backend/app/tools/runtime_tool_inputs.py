"""Strict, model-visible input contracts for the unified Agent Runtime tools.

Trusted execution context (actor identity, request metadata, and idempotency keys)
is intentionally absent. The runtime injects that data only after model arguments
have passed validation.
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Annotated, Any, Dict, Literal, Type
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    WithJsonSchema,
    field_validator,
    model_validator,
)


def _aware_datetime_string(value: str) -> str:
    normalized = value.strip()
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("must be a valid ISO 8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("must include a timezone offset")
    return normalized


def _date_string(value: str) -> str:
    normalized = value.strip()
    try:
        parsed = date.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("must be a valid ISO 8601 date") from exc
    if parsed.isoformat() != normalized:
        raise ValueError("must use YYYY-MM-DD format")
    return normalized


NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
]
LongText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4000),
]
AwareDateTimeString = Annotated[
    str,
    AfterValidator(_aware_datetime_string),
    WithJsonSchema(
        {"type": "string", "format": "date-time", "minLength": 1, "maxLength": 64}
    ),
]
DateString = Annotated[
    str,
    AfterValidator(_date_string),
    WithJsonSchema(
        {"type": "string", "format": "date", "minLength": 10, "maxLength": 10}
    ),
]
ClockTimeString = Annotated[
    str,
    StringConstraints(pattern=r"^([01]\d|2[0-3]):[0-5]\d$"),
]

ApplicationStatusValue = Literal[
    "planned",
    "submitted",
    "written_test",
    "written_test_passed",
    "interview_1",
    "interview_1_passed",
    "interview_2",
    "interview_2_passed",
    "interview_3",
    "interview_3_passed",
    "hr",
    "interview_scheduled",
    "offer",
    "rejected",
    "no_response",
    "withdrawn",
]
ApplicationUpdateType = Literal[
    "schedule_interview",
    "pass_round",
    "reject",
    "withdraw",
    "offer",
    "submitted",
    "location_update",
]
TaskTypeValue = Literal[
    "leetcode",
    "interview_question",
    "project_deep_dive",
    "application",
    "review",
    "custom",
]
StudySessionStatusValue = Literal["planned", "completed", "skipped", "cancelled"]
StudyPlanStatusValue = Literal[
    "draft",
    "scheduled",
    "partially_synced",
    "synced",
    "cancelled",
]
LeetCodeResultValue = Literal[
    "independent",
    "with_hint",
    "with_solution",
    "failed",
    "postponed",
    "skipped",
]


class StrictToolInput(BaseModel):
    """Base contract for arguments supplied by the model."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class EmptyInput(StrictToolInput):
    pass


class ListApplicationsInput(StrictToolInput):
    company: NonEmptyString | None = None


class ListInterviewsInput(StrictToolInput):
    company: NonEmptyString | None = None
    range_start: AwareDateTimeString | None = None
    range_end: AwareDateTimeString | None = None

    @model_validator(mode="after")
    def validate_range(self):
        if (self.range_start is None) != (self.range_end is None):
            raise ValueError("range_start and range_end must be provided together")
        if self.range_start is not None and self.range_end is not None:
            start = datetime.fromisoformat(self.range_start.replace("Z", "+00:00"))
            end = datetime.fromisoformat(self.range_end.replace("Z", "+00:00"))
            if end <= start:
                raise ValueError("range_end must be after range_start")
        return self


class CreateApplicationInput(StrictToolInput):
    company: NonEmptyString
    role: NonEmptyString
    base_location: NonEmptyString | None = None
    round: NonEmptyString | None = None
    interview_time: NonEmptyString | None = None
    start_at: AwareDateTimeString | None = None
    jd_keywords: list[NonEmptyString] = Field(default_factory=list, max_length=50)
    calendar_reminder: bool | None = None

    @model_validator(mode="after")
    def validate_optional_interview(self):
        if self.start_at is not None and self.interview_time is None:
            raise ValueError("start_at requires interview_time")
        if self.calendar_reminder is not None and self.interview_time is None:
            raise ValueError("calendar_reminder requires interview_time")
        return self


class UpdateApplicationInput(StrictToolInput):
    application_id: NonEmptyString | None = None
    company: NonEmptyString | None = None
    role: NonEmptyString | None = None
    base_location: NonEmptyString | None = None
    round: NonEmptyString | None = None
    interview_time: NonEmptyString | None = None
    start_at: AwareDateTimeString | None = None
    update_type: ApplicationUpdateType | None = None
    status: ApplicationStatusValue | None = None
    calendar_reminder: bool | None = None
    apply_to_all: bool = False

    @model_validator(mode="after")
    def validate_update(self):
        if self.application_id is None and self.company is None:
            raise ValueError("application_id or company is required")
        if self.apply_to_all and self.company is None:
            raise ValueError("apply_to_all requires company")
        mutable_values = (
            self.role,
            self.base_location,
            self.round,
            self.interview_time,
            self.update_type,
            self.status,
        )
        if all(value is None for value in mutable_values):
            raise ValueError("at least one application update is required")
        if self.start_at is not None and self.interview_time is None:
            raise ValueError("start_at requires interview_time")
        if self.update_type == "schedule_interview":
            if self.round is None:
                raise ValueError("schedule_interview requires round")
            if self.interview_time is None:
                raise ValueError("schedule_interview requires interview_time")
        if self.update_type == "pass_round" and self.round is None:
            raise ValueError("pass_round requires round")
        if self.update_type == "location_update" and self.base_location is None:
            raise ValueError("location_update requires base_location")
        if self.calendar_reminder is not None and self.interview_time is None:
            raise ValueError("calendar_reminder requires an interview update")
        return self


class ScheduleInterviewInput(StrictToolInput):
    application_id: NonEmptyString | None = None
    company: NonEmptyString | None = None
    role: NonEmptyString | None = None
    round: NonEmptyString
    interview_time: NonEmptyString
    start_at: AwareDateTimeString | None = None
    calendar_reminder: bool | None = None

    @model_validator(mode="after")
    def validate_schedule(self):
        if self.application_id is None and self.company is None:
            raise ValueError("application_id or company is required")
        return self


class RescheduleInterviewInput(StrictToolInput):
    schedule_id: NonEmptyString | None = None
    application_id: NonEmptyString | None = None
    company: NonEmptyString | None = None
    role: NonEmptyString | None = None
    round: NonEmptyString | None = None
    interview_time: NonEmptyString
    start_at: AwareDateTimeString | None = None

    @model_validator(mode="after")
    def validate_locator(self):
        if not any((self.schedule_id, self.application_id, self.company)):
            raise ValueError("schedule_id, application_id, or company is required")
        return self


class CancelInterviewInput(StrictToolInput):
    schedule_id: NonEmptyString | None = None
    application_id: NonEmptyString | None = None
    company: NonEmptyString | None = None
    role: NonEmptyString | None = None
    round: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_locator(self):
        if not any((self.schedule_id, self.application_id, self.company)):
            raise ValueError("schedule_id, application_id, or company is required")
        return self


class UpdateTaskInput(StrictToolInput):
    operation: Literal["complete", "postpone"]
    task_title: NonEmptyString | None = None
    task_type: TaskTypeValue | None = None


class RecordInterviewReviewInput(StrictToolInput):
    company: NonEmptyString | None = None
    round: NonEmptyString | None = None
    topics: list[NonEmptyString] = Field(default_factory=list, max_length=50)


class ConfigureLeetCodePlanInput(StrictToolInput):
    enabled: bool


class RecordLeetCodeResultInput(StrictToolInput):
    result: LeetCodeResultValue
    assignment_id: NonEmptyString | None = None
    problem_index: int | None = Field(default=None, ge=1, le=3)
    problem_title: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_problem_locator(self):
        locators = (self.assignment_id, self.problem_index, self.problem_title)
        if sum(value is not None for value in locators) != 1:
            raise ValueError(
                "exactly one of assignment_id, problem_index, or problem_title is required"
            )
        return self


class StartMockInterviewInput(StrictToolInput):
    project: NonEmptyString | None = None
    role: NonEmptyString | None = None


class StartProjectTrainingInput(StrictToolInput):
    project: NonEmptyString | None = None
    role: NonEmptyString | None = None
    difficulty: Literal["basic", "medium", "advanced"] | None = None


class ProjectTrainingLookupInput(StrictToolInput):
    project: NonEmptyString | None = None
    session_id: NonEmptyString | None = None


class ExecutionPlanStepInput(StrictToolInput):
    id: NonEmptyString
    description: NonEmptyString
    status: Literal["pending", "in_progress", "completed", "skipped", "failed"]
    required: bool = True


class UpdateExecutionPlanInput(StrictToolInput):
    goal: NonEmptyString
    revision: int | None = Field(default=None, ge=1)
    steps: list[ExecutionPlanStepInput] = Field(min_length=1, max_length=30)

    @model_validator(mode="after")
    def validate_steps(self):
        step_ids = [step.id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("execution plan step ids must be unique")
        if sum(step.status == "in_progress" for step in self.steps) > 1:
            raise ValueError("only one execution plan step may be in_progress")
        return self


class RequestUserInputInput(StrictToolInput):
    kind: Literal["clarification", "preference_form", "selection"]
    prompt: NonEmptyString
    input_schema: Dict[str, Any] = Field(default_factory=dict)


class ReconcileWriteOperationInput(StrictToolInput):
    operation_key: NonEmptyString
    outcome: Literal["applied", "not_applied"]
    evidence: LongText


class SearchInterviewKnowledgeInput(StrictToolInput):
    query: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]
    top_k: int | None = Field(default=None, ge=1, le=10)


class SearchProjectEvidenceInput(SearchInterviewKnowledgeInput):
    project_id: NonEmptyString | None = None
    project_version: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_project_version(self):
        if self.project_version is not None and self.project_id is None:
            raise ValueError("project_version requires project_id")
        return self


class ReadEvidenceInput(StrictToolInput):
    evidence_id: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)
    ]


class StudyWindowInput(StrictToolInput):
    start_time: ClockTimeString
    end_time: ClockTimeString

    @model_validator(mode="after")
    def validate_window(self):
        if time.fromisoformat(self.end_time) <= time.fromisoformat(self.start_time):
            raise ValueError("study window end_time must be after start_time")
        return self


class SaveStudyPreferencesInput(StrictToolInput):
    timezone: NonEmptyString
    weekday_windows: list[StudyWindowInput] = Field(max_length=20)
    weekend_windows: list[StudyWindowInput] = Field(max_length=20)
    daily_max_minutes: int = Field(ge=15, le=720)
    session_minutes: int = Field(ge=10, le=240)

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone must be a valid IANA timezone") from exc
        return value

    @model_validator(mode="after")
    def validate_windows(self):
        if not self.weekday_windows and not self.weekend_windows:
            raise ValueError("at least one study availability window is required")
        return self


class GetCalendarAvailabilityInput(StrictToolInput):
    range_start: AwareDateTimeString
    range_end: AwareDateTimeString

    @model_validator(mode="after")
    def validate_range(self):
        if _as_datetime(self.range_end) <= _as_datetime(self.range_start):
            raise ValueError("range_end must be after range_start")
        return self


class StudyTopicInput(StrictToolInput):
    topic: NonEmptyString
    duration_minutes: int = Field(ge=10, le=1440)
    priority: int = Field(strict=True, ge=1, le=3)
    deadline: AwareDateTimeString | None = None
    source_refs: list[NonEmptyString] = Field(default_factory=list, max_length=100)
    rationale: LongText | None = None


class CreateStudyPlanInput(StrictToolInput):
    goal: NonEmptyString
    range_start: DateString
    range_end: DateString
    topics: list[StudyTopicInput] = Field(min_length=1, max_length=100)
    source_interview_ids: list[NonEmptyString] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def validate_range(self):
        if date.fromisoformat(self.range_end) < date.fromisoformat(self.range_start):
            raise ValueError("range_end must not be before range_start")
        return self


class GetStudyPlanInput(StrictToolInput):
    plan_id: NonEmptyString


class ListStudyPlansInput(StrictToolInput):
    status: StudyPlanStatusValue | None = None
    limit: int = Field(default=20, ge=1, le=50)


class SyncStudyPlanToCalendarInput(StrictToolInput):
    plan_id: NonEmptyString


class ReconcileStudyCalendarSessionInput(StrictToolInput):
    session_id: NonEmptyString
    operation_key: NonEmptyString
    outcome: Literal[
        "created",
        "not_created",
        "updated",
        "not_updated",
        "deleted",
        "not_deleted",
    ]
    event_id: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_outcome(self):
        if self.outcome == "created" and self.event_id is None:
            raise ValueError("outcome=created requires event_id")
        if self.outcome in {"not_created", "deleted"} and self.event_id is not None:
            raise ValueError(f"outcome={self.outcome} must not include event_id")
        return self


class UpdateStudySessionInput(StrictToolInput):
    session_id: NonEmptyString
    status: StudySessionStatusValue | None = None
    start_at: AwareDateTimeString | None = None
    end_at: AwareDateTimeString | None = None

    @model_validator(mode="after")
    def validate_update(self):
        has_start = self.start_at is not None
        has_end = self.end_at is not None
        if self.status is None and not has_start and not has_end:
            raise ValueError("status or a new start_at/end_at pair is required")
        if has_start != has_end:
            raise ValueError("start_at and end_at must be supplied together")
        if has_start and _as_datetime(self.end_at) <= _as_datetime(self.start_at):
            raise ValueError("end_at must be after start_at")
        return self


class ListLearningGapsInput(StrictToolInput):
    limit: int = Field(default=10, ge=1, le=30)


def _as_datetime(value: str | None) -> datetime:
    if value is None:
        raise ValueError("datetime value is required")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


RUNTIME_TOOL_INPUT_MODELS: Dict[str, Type[BaseModel]] = {
    "list_tasks": EmptyInput,
    "list_applications": ListApplicationsInput,
    "get_application_bitable_link": EmptyInput,
    "create_application": CreateApplicationInput,
    "update_application": UpdateApplicationInput,
    "list_interviews": ListInterviewsInput,
    "schedule_interview": ScheduleInterviewInput,
    "reschedule_interview": RescheduleInterviewInput,
    "cancel_interview": CancelInterviewInput,
    "update_task": UpdateTaskInput,
    "record_interview_review": RecordInterviewReviewInput,
    "configure_leetcode_plan": ConfigureLeetCodePlanInput,
    "get_today_leetcode": EmptyInput,
    "record_leetcode_result": RecordLeetCodeResultInput,
    "start_mock_interview": StartMockInterviewInput,
    "start_project_training": StartProjectTrainingInput,
    "resume_project_training": ProjectTrainingLookupInput,
    "get_project_training_summary": ProjectTrainingLookupInput,
    "update_execution_plan": UpdateExecutionPlanInput,
    "request_user_input": RequestUserInputInput,
    "reconcile_write_operation": ReconcileWriteOperationInput,
    "search_interview_knowledge": SearchInterviewKnowledgeInput,
    "search_project_evidence": SearchProjectEvidenceInput,
    "read_evidence": ReadEvidenceInput,
    "get_study_preferences": EmptyInput,
    "save_study_preferences": SaveStudyPreferencesInput,
    "get_calendar_availability": GetCalendarAvailabilityInput,
    "create_study_plan": CreateStudyPlanInput,
    "get_study_plan": GetStudyPlanInput,
    "list_study_plans": ListStudyPlansInput,
    "sync_study_plan_to_calendar": SyncStudyPlanToCalendarInput,
    "reconcile_study_calendar_session": ReconcileStudyCalendarSessionInput,
    "update_study_session": UpdateStudySessionInput,
    "list_learning_gaps": ListLearningGapsInput,
}

# Short alias for callers that do not need to know the runtime module name.
TOOL_INPUT_MODELS = RUNTIME_TOOL_INPUT_MODELS
