import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

from app.models.study import StudySessionSyncStatus
from app.models.tool_calling import ModelToolCall
from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository
from app.services.study_calendar_provider import (
    StudyCalendarAuthorizationRequired,
    StudyCalendarCredentialRefreshError,
)
from app.tools.agent_tool import ToolOutcomeStatus
from app.tools.agent_tool_registry import AgentToolRegistry
from app.tools.study_tools import (
    CalendarSyncReceipt,
    build_study_agent_tools,
)


class FakeCalendarProvider:
    def __init__(self) -> None:
        self.created: List[tuple[str, str, str]] = []
        self.updated: List[tuple[str, str, str]] = []
        self.deleted: List[tuple[str, str, str]] = []

    async def list_busy(self, owner_id, start_at, end_at):
        return []

    async def create_study_event(self, owner_id, session, operation_key):
        self.created.append((owner_id, session.id, operation_key))
        return CalendarSyncReceipt(event_id=f"event_{session.id}", calendar_id="primary")

    async def update_study_event(self, owner_id, session, event_id, operation_key):
        self.updated.append((owner_id, event_id, operation_key))
        return CalendarSyncReceipt(event_id=event_id, calendar_id="primary")

    async def delete_study_event(self, owner_id, event_id, operation_key):
        self.deleted.append((owner_id, event_id, operation_key))
        return CalendarSyncReceipt(event_id=event_id, calendar_id="primary")

    def authorization_url(self, owner_id):
        return "https://example.test/calendar/authorize"


class UnauthorizedCalendarProvider(FakeCalendarProvider):
    async def create_study_event(self, owner_id, session, operation_key):
        raise StudyCalendarAuthorizationRequired(self.authorization_url(owner_id))


class UnauthorizedFreebusyCalendarProvider(FakeCalendarProvider):
    async def list_busy(self, owner_id, start_at, end_at):
        raise StudyCalendarAuthorizationRequired(self.authorization_url(owner_id))


class FailedFreebusyCalendarProvider(FakeCalendarProvider):
    async def list_busy(self, owner_id, start_at, end_at):
        raise RuntimeError("freebusy endpoint unavailable")


class UncertainCalendarProvider(FakeCalendarProvider):
    async def create_study_event(self, owner_id, session, operation_key):
        raise RuntimeError("connection dropped after calendar request")


class UncertainUpdateCalendarProvider(FakeCalendarProvider):
    async def update_study_event(self, owner_id, session, event_id, operation_key):
        raise RuntimeError("connection dropped after calendar update")


class UncertainDeleteCalendarProvider(FakeCalendarProvider):
    async def delete_study_event(self, owner_id, event_id, operation_key):
        raise RuntimeError("connection dropped after calendar delete")


class RefreshFailedCalendarProvider(FakeCalendarProvider):
    async def create_study_event(self, owner_id, session, operation_key):
        raise StudyCalendarCredentialRefreshError("refresh token rejected")


async def _execute(
    registry: AgentToolRegistry,
    name: str,
    arguments: Dict[str, Any],
):
    return await registry.execute(
        ModelToolCall(id=f"call_{name}", name=name, arguments=arguments)
    )


def _future_range() -> tuple[datetime, datetime]:
    zone = ZoneInfo("Asia/Shanghai")
    start = datetime.now(zone) + timedelta(days=1)
    start = start.replace(hour=9, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=2)


def _preferences(owner_id: str) -> Dict[str, Any]:
    return {
        "owner_id": owner_id,
        "timezone": "Asia/Shanghai",
        "weekday_windows": [{"start_time": "09:00", "end_time": "18:00"}],
        "weekend_windows": [{"start_time": "09:00", "end_time": "18:00"}],
        "daily_max_minutes": 120,
        "session_minutes": 45,
    }


def test_study_tools_close_preferences_plan_and_calendar_loop() -> None:
    async def scenario():
        owner_id = "api:user_study"
        repository = InMemoryOfferPilotRepository()
        calendar = FakeCalendarProvider()
        registry = AgentToolRegistry(
            build_study_agent_tools(repository, calendar_provider=calendar)
        )
        missing = await _execute(
            registry,
            "get_study_preferences",
            {"owner_id": owner_id},
        )
        saved = await _execute(
            registry,
            "save_study_preferences",
            _preferences(owner_id),
        )
        start, end = _future_range()
        availability = await _execute(
            registry,
            "get_calendar_availability",
            {
                "owner_id": owner_id,
                "range_start": start.isoformat(),
                "range_end": end.isoformat(),
            },
        )
        created = await _execute(
            registry,
            "create_study_plan",
            {
                "owner_id": owner_id,
                "goal": "准备 JVM 面试",
                "range_start": start.date().isoformat(),
                "range_end": end.date().isoformat(),
                "topics": [
                    {
                        "topic": "JVM GC",
                        "duration_minutes": 90,
                        "priority": 3,
                        "deadline": end.isoformat(),
                        "source_refs": ["gap:jvm"],
                    }
                ],
            },
        )
        plan_id = created.data["plan"]["id"]
        synced = await _execute(
            registry,
            "sync_study_plan_to_calendar",
            {
                "owner_id": owner_id,
                "plan_id": plan_id,
                "idempotency_key": "operation_study_1",
            },
        )
        return repository, calendar, missing, saved, availability, created, synced

    repository, calendar, missing, saved, availability, created, synced = asyncio.run(
        scenario()
    )

    assert missing.status == ToolOutcomeStatus.NEEDS_INPUT
    assert missing.interaction.kind == "preference_form"
    assert saved.status == ToolOutcomeStatus.SUCCESS
    assert availability.status == ToolOutcomeStatus.SUCCESS
    assert availability.data["provider"] == "local_and_feishu"
    assert created.status == ToolOutcomeStatus.SUCCESS
    assert len(created.data["sessions"]) == 2
    plan_artifact = next(
        item for item in created.artifact_refs if item["type"] == "study_plan"
    )
    assert len(plan_artifact["data"]["sessions"]) == 2
    assert plan_artifact["data"]["sessions"][0]["topic"] == "JVM GC"
    assert plan_artifact["data"]["sessions"][0]["duration_minutes"] == 45
    assert plan_artifact["data"]["unscheduled"] == []
    assert synced.status == ToolOutcomeStatus.SUCCESS
    assert len(calendar.created) == 2
    sessions = repository.list_study_sessions(owner_id="api:user_study")
    assert all(item.sync_status == StudySessionSyncStatus.SYNCED for item in sessions)
    assert all(item.calendar_event_id for item in sessions)


def test_study_plan_read_tools_return_persisted_schedule_and_isolate_owner() -> None:
    async def scenario():
        owner_id = "api:plan_reader"
        repository = InMemoryOfferPilotRepository()
        registry = AgentToolRegistry(build_study_agent_tools(repository))
        await _execute(registry, "save_study_preferences", _preferences(owner_id))
        start, _ = _future_range()
        created = await _execute(
            registry,
            "create_study_plan",
            {
                "owner_id": owner_id,
                "goal": "容量不足的复习计划",
                "range_start": start.date().isoformat(),
                "range_end": start.date().isoformat(),
                "topics": [
                    {"topic": "JVM", "duration_minutes": 180, "priority": 3}
                ],
            },
        )
        plan_id = created.data["plan"]["id"]
        fetched = await _execute(
            registry,
            "get_study_plan",
            {"owner_id": owner_id, "plan_id": plan_id},
        )
        forbidden = await _execute(
            registry,
            "get_study_plan",
            {"owner_id": "api:another_owner", "plan_id": plan_id},
        )
        listed = await _execute(
            registry,
            "list_study_plans",
            {"owner_id": owner_id, "status": "scheduled", "limit": 10},
        )
        other_list = await _execute(
            registry,
            "list_study_plans",
            {"owner_id": "api:another_owner"},
        )
        return created, fetched, forbidden, listed, other_list

    created, fetched, forbidden, listed, other_list = asyncio.run(scenario())

    assert created.status == ToolOutcomeStatus.PARTIAL
    assert created.data["unscheduled"][0]["remaining_minutes"] == 60
    assert fetched.status == ToolOutcomeStatus.SUCCESS
    assert len(fetched.data["sessions"]) == 3
    assert fetched.data["unscheduled"] == created.data["unscheduled"]
    assert fetched.artifact_refs[0]["data"]["sessions"][0]["topic"] == "JVM"
    assert fetched.artifact_refs[0]["data"]["unscheduled"] == created.data["unscheduled"]
    assert forbidden.error_code == "study_plan_not_found"
    assert listed.data["count"] == 1
    assert listed.data["plans"][0]["session_count"] == 3
    assert listed.data["plans"][0]["unscheduled"] == created.data["unscheduled"]
    assert other_list.data == {"plans": [], "count": 0}


def test_create_plan_reloads_local_interviews_instead_of_trusting_model_context() -> None:
    async def scenario():
        owner_id = "api:deterministic_availability"
        repository = InMemoryOfferPilotRepository()
        calendar = FakeCalendarProvider()
        registry = AgentToolRegistry(
            build_study_agent_tools(repository, calendar_provider=calendar)
        )
        await _execute(registry, "save_study_preferences", _preferences(owner_id))
        start, end = _future_range()
        repository.create_interview_schedule(
            company="Example Tech",
            round_name="一面",
            start_at=start.isoformat(),
            owner_id=owner_id,
        )
        created = await _execute(
            registry,
            "create_study_plan",
            {
                "owner_id": owner_id,
                "goal": "避开面试安排复习",
                "range_start": start.date().isoformat(),
                "range_end": start.date().isoformat(),
                "topics": [
                    {
                        "topic": "JVM",
                        "duration_minutes": 45,
                        "priority": 3,
                    }
                ],
            },
        )
        return created

    created = asyncio.run(scenario())

    assert created.status == ToolOutcomeStatus.SUCCESS
    session_start = datetime.fromisoformat(created.data["sessions"][0]["start_at"])
    assert session_start.hour == 10
    assert created.data["availability"]["provider"] == "local_and_feishu"


def test_changed_synced_session_is_updated_then_cancelled_remotely() -> None:
    async def scenario():
        owner_id = "api:calendar_update"
        repository = InMemoryOfferPilotRepository()
        calendar = FakeCalendarProvider()
        registry = AgentToolRegistry(
            build_study_agent_tools(repository, calendar_provider=calendar)
        )
        await _execute(registry, "save_study_preferences", _preferences(owner_id))
        start, _ = _future_range()
        created = await _execute(
            registry,
            "create_study_plan",
            {
                "owner_id": owner_id,
                "goal": "同步变更",
                "range_start": start.date().isoformat(),
                "range_end": start.date().isoformat(),
                "topics": [
                    {"topic": "Redis", "duration_minutes": 45, "priority": 2}
                ],
            },
        )
        plan_id = created.data["plan"]["id"]
        session_id = created.data["sessions"][0]["id"]
        await _execute(
            registry,
            "sync_study_plan_to_calendar",
            {"owner_id": owner_id, "plan_id": plan_id, "idempotency_key": "create"},
        )
        session = repository.get_study_session(session_id, owner_id=owner_id)
        shifted_start = session.start_at + timedelta(hours=1)
        shifted_end = session.end_at + timedelta(hours=1)
        changed = await _execute(
            registry,
            "update_study_session",
            {
                "owner_id": owner_id,
                "session_id": session_id,
                "start_at": shifted_start.isoformat(),
                "end_at": shifted_end.isoformat(),
            },
        )
        updated = await _execute(
            registry,
            "sync_study_plan_to_calendar",
            {"owner_id": owner_id, "plan_id": plan_id, "idempotency_key": "update"},
        )
        await _execute(
            registry,
            "update_study_session",
            {"owner_id": owner_id, "session_id": session_id, "status": "cancelled"},
        )
        deleted = await _execute(
            registry,
            "sync_study_plan_to_calendar",
            {"owner_id": owner_id, "plan_id": plan_id, "idempotency_key": "delete"},
        )
        final_session = repository.get_study_session(session_id, owner_id=owner_id)
        return calendar, changed, updated, deleted, final_session

    calendar, changed, updated, deleted, final_session = asyncio.run(scenario())

    assert changed.data["session"]["sync_status"] == "pending"
    assert len(calendar.updated) == 1
    assert len(updated.data["updated"]) == 1
    assert len(calendar.deleted) == 1
    assert len(deleted.data["deleted"]) == 1
    assert final_session.calendar_event_id is None


def test_missing_calendar_authorization_degrades_with_authorization_artifact() -> None:
    async def scenario():
        owner_id = "api:availability_auth"
        repository = InMemoryOfferPilotRepository()
        registry = AgentToolRegistry(
            build_study_agent_tools(
                repository,
                calendar_provider=UnauthorizedFreebusyCalendarProvider(),
            )
        )
        start, end = _future_range()
        return await _execute(
            registry,
            "get_calendar_availability",
            {
                "owner_id": owner_id,
                "range_start": start.isoformat(),
                "range_end": end.isoformat(),
            },
        )

    result = asyncio.run(scenario())

    assert result.status == ToolOutcomeStatus.SUCCESS
    assert result.data["provider"] == "local_only"
    assert result.data["degraded"] is True
    assert result.data["authorization_required"] is True
    assert result.artifact_refs == [
        {
            "type": "authorization",
            "id": "feishu_calendar_authorization",
            "title": "授权飞书日历",
            "url": "https://example.test/calendar/authorize",
            "data": {"url": "https://example.test/calendar/authorize"},
        }
    ]


def test_freebusy_transport_failure_degrades_without_authorization_artifact() -> None:
    async def scenario():
        owner_id = "api:availability_network"
        repository = InMemoryOfferPilotRepository()
        registry = AgentToolRegistry(
            build_study_agent_tools(
                repository,
                calendar_provider=FailedFreebusyCalendarProvider(),
            )
        )
        start, end = _future_range()
        return await _execute(
            registry,
            "get_calendar_availability",
            {
                "owner_id": owner_id,
                "range_start": start.isoformat(),
                "range_end": end.isoformat(),
            },
        )

    result = asyncio.run(scenario())

    assert result.status == ToolOutcomeStatus.SUCCESS
    assert result.data["provider"] == "local_only"
    assert result.data["degraded"] is True
    assert result.data["authorization_required"] is False
    assert result.artifact_refs == []


def test_calendar_authorization_failure_returns_artifact_without_unknown_session() -> None:
    async def scenario():
        owner_id = "feishu:ou_study"
        repository = InMemoryOfferPilotRepository()
        registry = AgentToolRegistry(
            build_study_agent_tools(
                repository,
                calendar_provider=UnauthorizedCalendarProvider(),
            )
        )
        await _execute(registry, "save_study_preferences", _preferences(owner_id))
        start, end = _future_range()
        created = await _execute(
            registry,
            "create_study_plan",
            {
                "owner_id": owner_id,
                "goal": "准备面试",
                "range_start": start.date().isoformat(),
                "range_end": end.date().isoformat(),
                "topics": [
                    {
                        "topic": "Redis",
                        "duration_minutes": 45,
                        "priority": 2,
                        "deadline": end.isoformat(),
                    }
                ],
            },
        )
        result = await _execute(
            registry,
            "sync_study_plan_to_calendar",
            {
                "owner_id": owner_id,
                "plan_id": created.data["plan"]["id"],
                "idempotency_key": "operation_auth",
            },
        )
        sessions = repository.list_study_sessions(owner_id=owner_id)
        return result, sessions

    result, sessions = asyncio.run(scenario())

    assert result.status == ToolOutcomeStatus.ERROR
    assert result.error_code == "calendar_authorization_required"
    assert any(item["type"] == "authorization" for item in result.artifact_refs)
    assert all(item.sync_status == StudySessionSyncStatus.PENDING for item in sessions)


def test_unknown_calendar_write_requires_explicit_reconciliation() -> None:
    async def scenario():
        owner_id = "api:calendar_unknown"
        repository = InMemoryOfferPilotRepository()
        registry = AgentToolRegistry(
            build_study_agent_tools(
                repository,
                calendar_provider=UncertainCalendarProvider(),
            )
        )
        await _execute(registry, "save_study_preferences", _preferences(owner_id))
        start, end = _future_range()
        created = await _execute(
            registry,
            "create_study_plan",
            {
                "owner_id": owner_id,
                "goal": "协调未知日历写",
                "range_start": start.date().isoformat(),
                "range_end": end.date().isoformat(),
                "topics": [
                    {
                        "topic": "JVM",
                        "duration_minutes": 45,
                        "priority": 3,
                        "deadline": end.isoformat(),
                    }
                ],
            },
        )
        synced = await _execute(
            registry,
            "sync_study_plan_to_calendar",
            {
                "owner_id": owner_id,
                "plan_id": created.data["plan"]["id"],
                "idempotency_key": "operation_unknown",
            },
        )
        session = repository.list_study_sessions(owner_id=owner_id)[0]
        reconciled = await _execute(
            registry,
            "reconcile_study_calendar_session",
            {
                "owner_id": owner_id,
                "session_id": session.id,
                "operation_key": synced.data["operation_key"],
                "outcome": "created",
                "event_id": "verified_event_1",
            },
        )
        updated = repository.list_study_sessions(owner_id=owner_id)[0]
        return registry, synced, session, reconciled, updated

    registry, synced, session, reconciled, updated = asyncio.run(scenario())

    assert synced.status == ToolOutcomeStatus.UNKNOWN
    assert session.sync_status == StudySessionSyncStatus.UNKNOWN
    assert reconciled.status == ToolOutcomeStatus.SUCCESS
    assert updated.sync_status == StudySessionSyncStatus.SYNCED
    assert updated.calendar_event_id == "verified_event_1"
    definition = registry.get_definition("reconcile_study_calendar_session")
    assert definition.approval.value == "always"


def test_unknown_calendar_update_requires_update_specific_reconciliation() -> None:
    async def scenario():
        owner_id = "api:calendar_update_unknown"
        repository = InMemoryOfferPilotRepository()
        registry = AgentToolRegistry(
            build_study_agent_tools(
                repository,
                calendar_provider=UncertainUpdateCalendarProvider(),
            )
        )
        await _execute(registry, "save_study_preferences", _preferences(owner_id))
        start, _ = _future_range()
        created = await _execute(
            registry,
            "create_study_plan",
            {
                "owner_id": owner_id,
                "goal": "更新未知协调",
                "range_start": start.date().isoformat(),
                "range_end": start.date().isoformat(),
                "topics": [{"topic": "JVM", "duration_minutes": 45, "priority": 3}],
            },
        )
        plan_id = created.data["plan"]["id"]
        session_id = created.data["sessions"][0]["id"]
        await _execute(
            registry,
            "sync_study_plan_to_calendar",
            {"owner_id": owner_id, "plan_id": plan_id, "idempotency_key": "create"},
        )
        session = repository.get_study_session(session_id, owner_id=owner_id)
        await _execute(
            registry,
            "update_study_session",
            {
                "owner_id": owner_id,
                "session_id": session_id,
                "start_at": (session.start_at + timedelta(hours=1)).isoformat(),
                "end_at": (session.end_at + timedelta(hours=1)).isoformat(),
            },
        )
        unknown = await _execute(
            registry,
            "sync_study_plan_to_calendar",
            {"owner_id": owner_id, "plan_id": plan_id, "idempotency_key": "update"},
        )
        mismatch = await _execute(
            registry,
            "reconcile_study_calendar_session",
            {
                "owner_id": owner_id,
                "session_id": session_id,
                "operation_key": unknown.data["operation_key"],
                "outcome": "not_created",
            },
        )
        reconciled = await _execute(
            registry,
            "reconcile_study_calendar_session",
            {
                "owner_id": owner_id,
                "session_id": session_id,
                "operation_key": unknown.data["operation_key"],
                "outcome": "updated",
            },
        )
        return unknown, mismatch, reconciled

    unknown, mismatch, reconciled = asyncio.run(scenario())

    assert unknown.status == ToolOutcomeStatus.UNKNOWN
    assert unknown.data["failed"][0]["operation"] == "update"
    assert mismatch.error_code == "calendar_reconciliation_operation_mismatch"
    assert reconciled.status == ToolOutcomeStatus.SUCCESS
    assert reconciled.data["session"]["sync_status"] == "synced"


def test_unknown_calendar_delete_can_be_confirmed_deleted() -> None:
    async def scenario():
        owner_id = "api:calendar_delete_unknown"
        repository = InMemoryOfferPilotRepository()
        registry = AgentToolRegistry(
            build_study_agent_tools(
                repository,
                calendar_provider=UncertainDeleteCalendarProvider(),
            )
        )
        await _execute(registry, "save_study_preferences", _preferences(owner_id))
        start, _ = _future_range()
        created = await _execute(
            registry,
            "create_study_plan",
            {
                "owner_id": owner_id,
                "goal": "删除未知协调",
                "range_start": start.date().isoformat(),
                "range_end": start.date().isoformat(),
                "topics": [{"topic": "Redis", "duration_minutes": 45, "priority": 2}],
            },
        )
        plan_id = created.data["plan"]["id"]
        session_id = created.data["sessions"][0]["id"]
        await _execute(
            registry,
            "sync_study_plan_to_calendar",
            {"owner_id": owner_id, "plan_id": plan_id, "idempotency_key": "create"},
        )
        await _execute(
            registry,
            "update_study_session",
            {"owner_id": owner_id, "session_id": session_id, "status": "cancelled"},
        )
        unknown = await _execute(
            registry,
            "sync_study_plan_to_calendar",
            {"owner_id": owner_id, "plan_id": plan_id, "idempotency_key": "delete"},
        )
        reconciled = await _execute(
            registry,
            "reconcile_study_calendar_session",
            {
                "owner_id": owner_id,
                "session_id": session_id,
                "operation_key": unknown.data["operation_key"],
                "outcome": "deleted",
            },
        )
        return unknown, reconciled

    unknown, reconciled = asyncio.run(scenario())

    assert unknown.status == ToolOutcomeStatus.UNKNOWN
    assert unknown.data["failed"][0]["operation"] == "delete"
    assert reconciled.status == ToolOutcomeStatus.SUCCESS
    assert reconciled.data["session"]["calendar_event_id"] is None


def test_credential_refresh_failure_is_known_error_not_unknown_write() -> None:
    async def scenario():
        owner_id = "api:calendar_refresh"
        repository = InMemoryOfferPilotRepository()
        registry = AgentToolRegistry(
            build_study_agent_tools(
                repository,
                calendar_provider=RefreshFailedCalendarProvider(),
            )
        )
        await _execute(registry, "save_study_preferences", _preferences(owner_id))
        start, end = _future_range()
        created = await _execute(
            registry,
            "create_study_plan",
            {
                "owner_id": owner_id,
                "goal": "刷新失败测试",
                "range_start": start.date().isoformat(),
                "range_end": end.date().isoformat(),
                "topics": [
                    {
                        "topic": "Redis",
                        "duration_minutes": 45,
                        "priority": 2,
                        "deadline": end.isoformat(),
                    }
                ],
            },
        )
        result = await _execute(
            registry,
            "sync_study_plan_to_calendar",
            {
                "owner_id": owner_id,
                "plan_id": created.data["plan"]["id"],
                "idempotency_key": "operation_refresh_failed",
            },
        )
        session = repository.list_study_sessions(owner_id=owner_id)[0]
        return result, session

    result, session = asyncio.run(scenario())

    assert result.status == ToolOutcomeStatus.ERROR
    assert session.sync_status == StudySessionSyncStatus.FAILED
