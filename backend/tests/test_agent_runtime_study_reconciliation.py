import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any, Dict, List, Sequence
from zoneinfo import ZoneInfo

from langgraph.checkpoint.memory import InMemorySaver

from app.agents.runtime import AgentRuntime
from app.core.config import settings as default_settings
from app.models.study import (
    StudyPlan,
    StudyPlanStatus,
    StudyPriority,
    StudySession,
    StudySessionSyncStatus,
)
from app.models.tool_calling import ModelToolCall, ModelTurn, TokenUsage
from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository
from app.services.agent_runtime_store import InMemoryAgentRuntimeStore
from app.tools.agent_tool_registry import AgentToolRegistry
from app.tools.study_tools import CalendarSyncReceipt, build_study_agent_tools


class ReconciliationCalendarProvider:
    def __init__(self, *, fail_first: bool) -> None:
        self.fail_first = fail_first
        self.attempts: List[tuple[str, str, str]] = []

    async def list_busy(self, owner_id, start_at, end_at):
        return []

    async def create_study_event(self, owner_id, session, operation_key):
        self.attempts.append((owner_id, session.id, operation_key))
        if self.fail_first and len(self.attempts) == 1:
            raise RuntimeError("connection dropped after calendar dispatch")
        return CalendarSyncReceipt(
            event_id=f"event_{session.id}",
            calendar_id="primary",
        )

    def authorization_url(self, owner_id):
        return None


class SlowCalendarProvider(ReconciliationCalendarProvider):
    def __init__(self) -> None:
        super().__init__(fail_first=False)

    async def create_study_event(self, owner_id, session, operation_key):
        self.attempts.append((owner_id, session.id, operation_key))
        await asyncio.sleep(1)
        return CalendarSyncReceipt(
            event_id=f"event_{session.id}",
            calendar_id="primary",
        )


class StudyReconciliationModel:
    def __init__(self, *, plan_id: str, session_id: str, outcome: str) -> None:
        self.plan_id = plan_id
        self.session_id = session_id
        self.outcome = outcome
        self.operation_key: str | None = None
        self.calls: List[List[Dict[str, Any]]] = []

    async def complete(self, messages, tools, options) -> ModelTurn:
        history = [dict(message) for message in messages]
        self.calls.append(history)
        stage = len(self.calls) - 1
        if stage == 0:
            return _tool_turn(
                _call(
                    "initial_sync",
                    "sync_study_plan_to_calendar",
                    plan_id=self.plan_id,
                )
            )
        if stage == 1:
            unknown = _tool_payload(history, "initial_sync")
            assert unknown["status"] == "unknown"
            self.operation_key = unknown["data"]["operation_key"]
            arguments = {
                "session_id": self.session_id,
                "operation_key": self.operation_key,
                "outcome": self.outcome,
            }
            if self.outcome == "created":
                arguments["event_id"] = "verified_event_1"
            return _tool_turn(
                _call(
                    "reconcile_sync",
                    "reconcile_study_calendar_session",
                    **arguments,
                )
            )
        if stage == 2:
            reconciled = _tool_payload(history, "reconcile_sync")
            assert reconciled["status"] == "success"
            assert reconciled["data"]["operation_key"] == self.operation_key
            return _tool_turn(
                _call(
                    "sync_after_reconciliation",
                    "sync_study_plan_to_calendar",
                    plan_id=self.plan_id,
                )
            )
        if stage == 3:
            retried = _tool_payload(history, "sync_after_reconciliation")
            assert retried["status"] == "success"
            assert retried["data"]["operation_key"] == self.operation_key
            return _answer_turn(
                "Calendar write reconciliation completed and sync is confirmed."
            )
        raise AssertionError(f"Unexpected reconciliation model stage: {stage}")


class ForgedReconciliationModel:
    def __init__(self, *, session_id: str) -> None:
        self.session_id = session_id
        self.rejection: Dict[str, Any] | None = None

    async def complete(self, messages, tools, options) -> ModelTurn:
        history = [dict(message) for message in messages]
        if self.rejection is None:
            if any(message.get("role") == "tool" for message in history):
                self.rejection = _tool_payload(history, "forged_reconcile")
                assert self.rejection["status"] == "rejected"
                assert (
                    self.rejection["error_code"]
                    == "invalid_reconciliation_operation"
                )
                return _answer_turn("The forged operation key was rejected.")
            return _tool_turn(
                _call(
                    "forged_reconcile",
                    "reconcile_study_calendar_session",
                    session_id=self.session_id,
                    operation_key="agent_op_from_another_run",
                    outcome="created",
                    event_id="forged_event",
                )
            )
        raise AssertionError("Unexpected forged reconciliation model call.")


class TimeoutObservationModel:
    def __init__(self, *, plan_id: str) -> None:
        self.plan_id = plan_id
        self.operation_key: str | None = None
        self.stage = 0

    async def complete(self, messages, tools, options) -> ModelTurn:
        history = [dict(message) for message in messages]
        if self.stage == 0:
            self.stage = 1
            return _tool_turn(
                _call(
                    "timed_out_sync",
                    "sync_study_plan_to_calendar",
                    plan_id=self.plan_id,
                )
            )
        if self.stage == 1:
            self.stage = 2
            unknown = _tool_payload(history, "timed_out_sync")
            assert unknown["status"] == "unknown"
            self.operation_key = unknown["data"]["operation_key"]
            return _answer_turn("The timed-out write requires reconciliation.")
        raise AssertionError("Unexpected timeout observation model call.")


class CrossRunReconciliationModel:
    def __init__(self, *, session_id: str, operation_key: str) -> None:
        self.session_id = session_id
        self.operation_key = operation_key
        self.stage = 0

    async def complete(self, messages, tools, options) -> ModelTurn:
        history = [dict(message) for message in messages]
        if self.stage == 0:
            self.stage = 1
            return _tool_turn(
                _call(
                    "cross_run_reconcile",
                    "reconcile_study_calendar_session",
                    session_id=self.session_id,
                    operation_key=self.operation_key,
                    outcome="created",
                    event_id="verified_cross_run_event",
                )
            )
        if self.stage == 1:
            self.stage = 2
            reconciled = _tool_payload(history, "cross_run_reconcile")
            assert reconciled["status"] == "success"
            return _answer_turn("The earlier unknown write is now confirmed.")
        raise AssertionError("Unexpected cross-run reconciliation model call.")


def test_created_reconciliation_resolves_unknown_without_second_write() -> None:
    async def scenario():
        calendar = ReconciliationCalendarProvider(fail_first=True)
        runtime, store, model, repository = _reconciliation_runtime(
            outcome="created",
            calendar=calendar,
            owner_user_id="reconcile_created",
        )
        completed = await _drive_reconciliation(runtime, "reconcile_created")
        snapshot = await runtime.graph.aget_state(runtime._config(completed.thread_id))
        return completed, calendar, store, model, repository, dict(snapshot.values)

    completed, calendar, store, model, repository, state = asyncio.run(scenario())

    assert completed.status.value == "completed"
    assert len(calendar.attempts) == 1
    assert model.operation_key is not None
    operation = store.get_operation(model.operation_key)
    assert operation is not None
    assert operation.status == "success"
    session = repository.list_study_sessions(owner_id="api:reconcile_created")[0]
    assert session.sync_status == StudySessionSyncStatus.SYNCED
    assert session.calendar_event_id == "verified_event_1"
    assert _sync_statuses_for_operation(state, model.operation_key) == [
        "unknown",
        "success",
        "success",
    ]
    _assert_tool_messages_are_paired(state)


def test_not_created_reconciliation_allows_same_run_sync_retry() -> None:
    async def scenario():
        calendar = ReconciliationCalendarProvider(fail_first=True)
        runtime, store, model, repository = _reconciliation_runtime(
            outcome="not_created",
            calendar=calendar,
            owner_user_id="reconcile_not_created",
        )
        completed = await _drive_reconciliation(runtime, "reconcile_not_created")
        snapshot = await runtime.graph.aget_state(runtime._config(completed.thread_id))
        return completed, calendar, store, model, repository, dict(snapshot.values)

    completed, calendar, store, model, repository, state = asyncio.run(scenario())

    assert completed.status.value == "completed"
    assert len(calendar.attempts) == 2
    assert model.operation_key is not None
    operation = store.get_operation(model.operation_key)
    assert operation is not None
    assert operation.status == "success"
    session = repository.list_study_sessions(owner_id="api:reconcile_not_created")[0]
    assert session.sync_status == StudySessionSyncStatus.SYNCED
    assert session.calendar_event_id == f"event_{session.id}"
    assert _sync_statuses_for_operation(state, model.operation_key) == [
        "unknown",
        "error",
        "success",
    ]
    _assert_tool_messages_are_paired(state)


def test_pending_session_can_reconcile_a_started_crash_window_operation() -> None:
    async def scenario():
        calendar = ReconciliationCalendarProvider(fail_first=False)
        runtime, store, model, repository = _reconciliation_runtime(
            outcome="created",
            calendar=calendar,
            owner_user_id="reconcile_crash",
        )
        paused = await runtime.start(
            "Sync the study plan to my calendar.",
            user_id="reconcile_crash",
            request_id="req_reconcile_crash_start",
        )
        snapshot = await runtime.graph.aget_state(runtime._config(paused.thread_id))
        state = dict(snapshot.values)
        call = dict(state["validated_calls"][0])
        injected = runtime._inject_actor_context(state, call)
        operation_key = runtime._operation_key(state, call)
        store.begin_operation(
            operation_key,
            runtime._call_fingerprint(call["name"], injected),
        )

        completed = await _resume_three_approvals(
            runtime,
            paused,
            "reconcile_crash",
            request_prefix="req_reconcile_crash",
        )
        final_snapshot = await runtime.graph.aget_state(
            runtime._config(completed.thread_id)
        )
        return (
            completed,
            calendar,
            store,
            model,
            repository,
            operation_key,
            dict(final_snapshot.values),
        )

    (
        completed,
        calendar,
        store,
        model,
        repository,
        operation_key,
        state,
    ) = asyncio.run(scenario())

    assert completed.status.value == "completed"
    assert calendar.attempts == []
    assert model.operation_key == operation_key
    operation = store.get_operation(operation_key)
    assert operation is not None
    assert operation.status == "success"
    session = repository.list_study_sessions(owner_id="api:reconcile_crash")[0]
    assert session.sync_status == StudySessionSyncStatus.SYNCED
    assert session.calendar_event_id == "verified_event_1"
    assert _sync_statuses_for_operation(state, operation_key) == [
        "unknown",
        "success",
        "success",
    ]
    _assert_tool_messages_are_paired(state)


def test_reconciliation_rejects_operation_not_created_in_current_run() -> None:
    async def scenario():
        calendar = ReconciliationCalendarProvider(fail_first=False)
        runtime, store, _, repository = _reconciliation_runtime(
            outcome="created",
            calendar=calendar,
            owner_user_id="reconcile_forged",
        )
        session = repository.list_study_sessions(owner_id="api:reconcile_forged")[0]
        model = ForgedReconciliationModel(session_id=session.id)
        runtime.model = model
        completed = await runtime.start(
            "Reconcile this calendar write.",
            user_id="reconcile_forged",
            request_id="req_reconcile_forged",
        )
        snapshot = await runtime.graph.aget_state(runtime._config(completed.thread_id))
        return completed, calendar, store, model, repository, dict(snapshot.values)

    completed, calendar, store, model, repository, state = asyncio.run(scenario())

    assert completed.status.value == "completed"
    assert completed.interaction is None
    assert model.rejection is not None
    assert calendar.attempts == []
    assert store.get_operation("agent_op_from_another_run") is None
    session = repository.list_study_sessions(owner_id="api:reconcile_forged")[0]
    assert session.sync_status == StudySessionSyncStatus.PENDING
    assert session.calendar_event_id is None
    assert completed.tool_executions[-1].status.value == "rejected"
    assert (
        completed.tool_executions[-1].error_code
        == "invalid_reconciliation_operation"
    )
    _assert_tool_messages_are_paired(state)


def test_runtime_timeout_unknown_exposes_its_reconciliation_operation_key() -> None:
    async def scenario():
        calendar = SlowCalendarProvider()
        runtime, store, _, repository = _reconciliation_runtime(
            outcome="created",
            calendar=calendar,
            owner_user_id="reconcile_timeout",
        )
        session = repository.list_study_sessions(owner_id="api:reconcile_timeout")[0]
        model = TimeoutObservationModel(plan_id=session.plan_id)
        runtime.model = model
        sync_tool = runtime.tool_registry._tools["sync_study_plan_to_calendar"]
        sync_tool._definition = replace(
            sync_tool.definition,
            timeout_seconds=0.01,
        )
        paused = await runtime.start(
            "Sync the study plan to my calendar.",
            user_id="reconcile_timeout",
            request_id="req_reconcile_timeout_start",
        )
        completed = await runtime.resume(
            paused.thread_id,
            paused.interaction.id,
            "approve",
            user_id="reconcile_timeout",
            request_id="req_reconcile_timeout_approve",
        )
        return completed, calendar, store, model, repository

    completed, calendar, store, model, repository = asyncio.run(scenario())

    assert completed.status.value == "partial"
    assert model.operation_key is not None
    operation = store.get_operation(model.operation_key)
    assert operation is not None
    assert operation.status == "unknown"
    assert operation.outcome is not None
    assert operation.outcome["data"]["operation_key"] == model.operation_key
    assert len(calendar.attempts) == 1
    session = repository.list_study_sessions(owner_id="api:reconcile_timeout")[0]
    assert session.sync_status == StudySessionSyncStatus.PENDING


def test_unknown_calendar_write_can_be_reconciled_in_a_later_run() -> None:
    async def scenario():
        calendar = SlowCalendarProvider()
        runtime, store, _, repository = _reconciliation_runtime(
            outcome="created",
            calendar=calendar,
            owner_user_id="reconcile_cross_run",
        )
        session = repository.list_study_sessions(
            owner_id="api:reconcile_cross_run"
        )[0]
        timeout_model = TimeoutObservationModel(plan_id=session.plan_id)
        runtime.model = timeout_model
        sync_tool = runtime.tool_registry._tools["sync_study_plan_to_calendar"]
        sync_tool._definition = replace(
            sync_tool.definition,
            timeout_seconds=0.01,
        )
        paused = await runtime.start(
            "Sync the study plan to my calendar.",
            user_id="reconcile_cross_run",
            request_id="req_cross_run_start",
        )
        partial = await runtime.resume(
            paused.thread_id,
            paused.interaction.id,
            "approve",
            user_id="reconcile_cross_run",
            request_id="req_cross_run_sync",
        )
        assert partial.status.value == "partial"
        assert timeout_model.operation_key is not None

        runtime.model = CrossRunReconciliationModel(
            session_id=session.id,
            operation_key=timeout_model.operation_key,
        )
        reconcile_paused = await runtime.start(
            "I checked Feishu: that calendar event was created.",
            user_id="reconcile_cross_run",
            request_id="req_cross_run_reconcile_start",
        )
        completed = await runtime.resume(
            reconcile_paused.thread_id,
            reconcile_paused.interaction.id,
            "approve",
            user_id="reconcile_cross_run",
            request_id="req_cross_run_reconcile_approve",
        )
        return completed, store, repository, timeout_model.operation_key, partial.run_id

    completed, store, repository, operation_key, original_run_id = asyncio.run(
        scenario()
    )

    assert completed.status.value == "completed"
    assert completed.run_id != original_run_id
    operation = store.get_operation(operation_key)
    assert operation is not None
    assert operation.status == "success"
    assert operation.owner_id == "api:reconcile_cross_run"
    assert operation.thread_id == completed.thread_id
    assert operation.run_id == original_run_id
    assert operation.tool_name == "sync_study_plan_to_calendar"
    session = repository.list_study_sessions(owner_id="api:reconcile_cross_run")[0]
    assert session.sync_status == StudySessionSyncStatus.SYNCED
    assert session.calendar_event_id == "verified_cross_run_event"


def _reconciliation_runtime(
    *,
    outcome: str,
    calendar: ReconciliationCalendarProvider,
    owner_user_id: str,
):
    zone = ZoneInfo("Asia/Shanghai")
    starts_at = datetime.now(zone) + timedelta(days=2)
    starts_at = starts_at.replace(hour=19, minute=0, second=0, microsecond=0)
    owner_id = f"api:{owner_user_id}"
    plan = StudyPlan(
        id=f"study_plan_{owner_user_id}",
        owner_id=owner_id,
        goal="Reconcile calendar write",
        range_start=starts_at.date(),
        range_end=starts_at.date(),
        status=StudyPlanStatus.SCHEDULED,
    )
    session = StudySession(
        id=f"study_session_{owner_user_id}",
        owner_id=owner_id,
        plan_id=plan.id,
        topic="JVM",
        start_at=starts_at,
        end_at=starts_at + timedelta(minutes=45),
        priority=StudyPriority.HIGH,
    )
    repository = InMemoryOfferPilotRepository()
    repository.create_study_plan(plan)
    repository.create_study_session(session)
    model = StudyReconciliationModel(
        plan_id=plan.id,
        session_id=session.id,
        outcome=outcome,
    )
    store = InMemoryAgentRuntimeStore()
    runtime = AgentRuntime(
        model=model,
        tool_registry=AgentToolRegistry(
            build_study_agent_tools(repository, calendar_provider=calendar)
        ),
        checkpointer=InMemorySaver(),
        runtime_store=store,
        settings=replace(
            default_settings,
            agent_max_model_turns=12,
            agent_max_tool_calls=20,
            agent_no_progress_limit=3,
            agent_max_verifier_passes=0,
            context_max_input_tokens=40000,
        ),
    )
    return runtime, store, model, repository


async def _drive_reconciliation(runtime: AgentRuntime, user_id: str):
    paused = await runtime.start(
        "Sync the study plan to my calendar.",
        user_id=user_id,
        request_id=f"req_{user_id}_start",
    )
    return await _resume_three_approvals(
        runtime,
        paused,
        user_id,
        request_prefix=f"req_{user_id}",
    )


async def _resume_three_approvals(
    runtime: AgentRuntime,
    paused,
    user_id: str,
    *,
    request_prefix: str,
):
    expected_tools = [
        "sync_study_plan_to_calendar",
        "reconcile_study_calendar_session",
        "sync_study_plan_to_calendar",
    ]
    current = paused
    for index, tool_name in enumerate(expected_tools, start=1):
        assert current.status.value == "waiting_for_input"
        assert current.interaction is not None
        assert current.interaction.tool_name == tool_name
        current = await runtime.resume(
            current.thread_id,
            current.interaction.id,
            "approve",
            user_id=user_id,
            request_id=f"{request_prefix}_approval_{index}",
        )
    return current


def _sync_statuses_for_operation(
    state: Dict[str, Any], operation_key: str
) -> List[str]:
    return [
        str(item["status"])
        for item in state.get("tool_executions") or []
        if item.get("name") == "sync_study_plan_to_calendar"
        and item.get("operation_key") == operation_key
    ]


def _assert_tool_messages_are_paired(state: Dict[str, Any]) -> None:
    messages = list(state.get("messages") or [])
    assistant_call_ids = [
        str(call["id"])
        for message in messages
        for call in message.get("tool_calls") or []
    ]
    tool_message_ids = [
        str(message["tool_call_id"])
        for message in messages
        if message.get("role") == "tool"
    ]
    assert sorted(tool_message_ids) == sorted(assistant_call_ids)


def _call(call_id: str, name: str, **arguments: Any) -> ModelToolCall:
    return ModelToolCall(id=call_id, name=name, arguments=arguments)


def _tool_turn(*calls: ModelToolCall) -> ModelTurn:
    return ModelTurn(
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in calls
            ],
            "reasoning_content": "hidden",
        },
        tool_calls=list(calls),
        content="",
        reasoning_content="hidden",
        finish_reason="tool_calls",
        usage=TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
        provider="fake",
        model="study-reconciliation",
    )


def _answer_turn(content: str) -> ModelTurn:
    return ModelTurn(
        assistant_message={
            "role": "assistant",
            "content": content,
            "reasoning_content": "hidden",
        },
        tool_calls=[],
        content=content,
        reasoning_content="hidden",
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
        provider="fake",
        model="study-reconciliation",
    )


def _tool_payload(messages: Sequence[Dict[str, Any]], call_id: str) -> Dict[str, Any]:
    for message in reversed(messages):
        if message.get("role") == "tool" and message.get("tool_call_id") == call_id:
            return json.loads(str(message["content"]))
    raise AssertionError(f"ToolMessage not found for {call_id}")
