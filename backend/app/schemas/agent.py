from copy import deepcopy
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.tools.tool_presentation import redact_tool_identifiers


class _AgentRequestModel(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)


class AgentRunStatus(str, Enum):
    COMPLETED = "completed"
    WAITING_FOR_INPUT = "waiting_for_input"
    PARTIAL = "partial"
    FAILED = "failed"
    DEGRADED = "degraded"


class AgentResumeDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ANSWER = "answer"
    REVISE = "revise"
    CANCEL = "cancel"


class AgentInteractionType(str, Enum):
    APPROVAL = "approval"
    CLARIFICATION = "clarification"
    PREFERENCE_FORM = "preference_form"
    SELECTION = "selection"


class AgentPlanStepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class AgentToolExecutionStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    NEEDS_INPUT = "needs_input"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class AgentRunRequest(_AgentRequestModel):
    message: str = Field(..., min_length=1, max_length=4000)
    # Deprecated compatibility hints. Authentication comes from the Bearer token.
    user_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    source: Optional[str] = Field(default=None, min_length=1, max_length=64)
    conversation_scope: Optional[str] = Field(default=None, min_length=1, max_length=512)
    request_id: Optional[str] = Field(default=None, min_length=1, max_length=128)


class AgentResumeRequest(_AgentRequestModel):
    interaction_id: str = Field(..., min_length=1, max_length=128)
    decision: AgentResumeDecision
    value: Optional[Any] = None
    text: Optional[str] = Field(default=None, min_length=1, max_length=4000)
    # Deprecated compatibility hints. Authentication comes from the Bearer token.
    user_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    source: Optional[str] = Field(default=None, min_length=1, max_length=64)
    request_id: Optional[str] = Field(default=None, min_length=1, max_length=128)


class AgentPlanStepView(BaseModel):
    id: str = Field(..., min_length=1, max_length=128)
    description: str = Field(..., min_length=1, max_length=1000)
    status: AgentPlanStepStatus = AgentPlanStepStatus.PENDING
    operation: Optional[str] = Field(default=None, max_length=200)
    tool_name: Optional[str] = Field(default=None, max_length=128)


class AgentPlanView(BaseModel):
    id: Optional[str] = Field(default=None, max_length=128)
    goal: str = Field(..., min_length=1, max_length=2000)
    revision: int = Field(default=1, ge=1)
    steps: List[AgentPlanStepView] = Field(default_factory=list)


class AgentInteraction(BaseModel):
    id: str = Field(..., min_length=1, max_length=128)
    type: AgentInteractionType
    prompt: str = Field(..., min_length=1, max_length=4000)
    allowed_actions: List[AgentResumeDecision] = Field(default_factory=list)
    operation: Optional[str] = Field(default=None, max_length=200)
    tool_name: Optional[str] = Field(default=None, max_length=128)
    arguments: Dict[str, Any] = Field(default_factory=dict)
    required_fields: List[str] = Field(default_factory=list)
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    expires_at: Optional[datetime] = None


class AgentToolExecution(BaseModel):
    call_id: str = Field(..., min_length=1, max_length=128)
    name: str = Field(..., min_length=1, max_length=128)
    operation: Optional[str] = Field(default=None, max_length=200)
    status: AgentToolExecutionStatus
    summary: str = Field(default="", max_length=4000)
    error_code: Optional[str] = Field(default=None, max_length=128)
    retryable: bool = False
    result_refs: List[str] = Field(default_factory=list)


class AgentArtifact(BaseModel):
    type: str = Field(..., min_length=1, max_length=128)
    id: Optional[str] = Field(default=None, max_length=128)
    title: Optional[str] = Field(default=None, max_length=500)
    url: Optional[str] = Field(default=None, max_length=4000)
    data: Dict[str, Any] = Field(default_factory=dict)


class AgentUsage(BaseModel):
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    verifier_calls: int = Field(default=0, ge=0)
    input_tokens: Optional[int] = Field(default=None, ge=0)
    output_tokens: Optional[int] = Field(default=None, ge=0)


class AgentRunResponse(BaseModel):
    thread_id: str = Field(..., min_length=1, max_length=512)
    run_id: str = Field(..., min_length=1, max_length=128)
    status: AgentRunStatus
    reply: str = ""
    plan: Optional[AgentPlanView] = None
    interaction: Optional[AgentInteraction] = None
    tool_executions: List[AgentToolExecution] = Field(default_factory=list)
    artifacts: List[AgentArtifact] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    usage: AgentUsage = Field(default_factory=AgentUsage)


class PublicAgentPlanStep(BaseModel):
    id: str = Field(..., min_length=1, max_length=128)
    description: str = Field(..., min_length=1, max_length=1000)
    status: AgentPlanStepStatus = AgentPlanStepStatus.PENDING
    operation: Optional[str] = Field(default=None, max_length=200)

    @field_validator("description", "operation", mode="before")
    @classmethod
    def sanitize_public_text(cls, value: Any) -> Any:
        return redact_tool_identifiers(value) if isinstance(value, str) else value


class PublicAgentPlan(BaseModel):
    id: Optional[str] = Field(default=None, max_length=128)
    goal: str = Field(..., min_length=1, max_length=2000)
    revision: int = Field(default=1, ge=1)
    steps: List[PublicAgentPlanStep] = Field(default_factory=list)

    @field_validator("goal", mode="before")
    @classmethod
    def sanitize_public_text(cls, value: Any) -> Any:
        return redact_tool_identifiers(value) if isinstance(value, str) else value


class PublicAgentInteraction(BaseModel):
    id: str = Field(..., min_length=1, max_length=128)
    type: AgentInteractionType
    prompt: str = Field(..., min_length=1, max_length=4000)
    allowed_actions: List[AgentResumeDecision] = Field(default_factory=list)
    operation: Optional[str] = Field(default=None, max_length=200)
    required_fields: List[str] = Field(default_factory=list)
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    expires_at: Optional[datetime] = None

    @field_validator("prompt", "operation", mode="before")
    @classmethod
    def sanitize_public_text(cls, value: Any) -> Any:
        return redact_tool_identifiers(value) if isinstance(value, str) else value


class PublicAgentToolExecution(BaseModel):
    operation: Optional[str] = Field(default=None, max_length=200)
    status: AgentToolExecutionStatus
    summary: str = Field(default="", max_length=4000)
    retryable: bool = False
    result_refs: List[str] = Field(default_factory=list)

    @field_validator("operation", "summary", mode="before")
    @classmethod
    def sanitize_public_text(cls, value: Any) -> Any:
        return redact_tool_identifiers(value) if isinstance(value, str) else value


class AgentRunPublicResponse(BaseModel):
    """External contract without model/runtime implementation identifiers."""

    thread_id: str = Field(..., min_length=1, max_length=512)
    run_id: str = Field(..., min_length=1, max_length=128)
    status: AgentRunStatus
    reply: str = ""
    plan: Optional[PublicAgentPlan] = None
    interaction: Optional[PublicAgentInteraction] = None
    tool_executions: List[PublicAgentToolExecution] = Field(default_factory=list)
    artifacts: List[AgentArtifact] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    usage: AgentUsage = Field(default_factory=AgentUsage)

    @model_validator(mode="before")
    @classmethod
    def sanitize_public_payload(cls, value: Any) -> Any:
        if isinstance(value, BaseModel):
            payload = value.model_dump(mode="json")
        elif isinstance(value, dict):
            payload = deepcopy(value)
        else:
            return value

        internal_identifiers: set[str] = set()
        executions = payload.get("tool_executions") or []
        for execution in executions:
            if not isinstance(execution, dict):
                continue
            for key in ("call_id", "error_code", "operation_key", "target_key"):
                identifier = str(execution.get(key) or "").strip()
                if identifier:
                    internal_identifiers.add(identifier)
        interaction = payload.get("interaction")
        if isinstance(interaction, dict):
            arguments = interaction.get("arguments")
            if isinstance(arguments, dict):
                for key in ("operation_key", "idempotency_key"):
                    identifier = str(arguments.get(key) or "").strip()
                    if identifier:
                        internal_identifiers.add(identifier)

        identifiers = tuple(sorted(internal_identifiers, key=len, reverse=True))

        def sanitize(text: Any) -> Any:
            if not isinstance(text, str):
                return text
            return redact_tool_identifiers(
                text,
                internal_identifiers=identifiers,
            )

        payload["reply"] = sanitize(payload.get("reply"))
        payload["warnings"] = [sanitize(item) for item in payload.get("warnings") or []]
        plan = payload.get("plan")
        if isinstance(plan, dict):
            plan["goal"] = sanitize(plan.get("goal"))
            for step in plan.get("steps") or []:
                if isinstance(step, dict):
                    step["description"] = sanitize(step.get("description"))
                    step["operation"] = sanitize(step.get("operation"))
        if isinstance(interaction, dict):
            interaction["prompt"] = sanitize(interaction.get("prompt"))
            interaction["operation"] = sanitize(interaction.get("operation"))
        for execution in executions:
            if isinstance(execution, dict):
                execution["operation"] = sanitize(execution.get("operation"))
                execution["summary"] = sanitize(execution.get("summary"))
        return payload

    @field_validator("reply", mode="before")
    @classmethod
    def sanitize_public_reply(cls, value: Any) -> Any:
        return redact_tool_identifiers(value) if isinstance(value, str) else value

    @field_validator("warnings", mode="before")
    @classmethod
    def sanitize_public_warnings(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [
            redact_tool_identifiers(item) if isinstance(item, str) else item
            for item in value
        ]


class DebugAgentRequest(AgentRunRequest):
    user_id: str = Field(default="local_user", min_length=1, max_length=128)


# Runtime identifiers remain on the in-process models for checkpoint recovery and
# diagnostics, but public HTTP responses expose only human-readable operation data.
AGENT_RUN_PUBLIC_EXCLUDE: Dict[str, Any] = {
    "plan": {"steps": {"__all__": {"tool_name"}}},
    "interaction": {"tool_name", "arguments"},
    "tool_executions": {"__all__": {"call_id", "name", "error_code"}},
}
