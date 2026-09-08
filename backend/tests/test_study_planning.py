import sqlite3
from dataclasses import replace
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from app.models.study import (
    StudyPlanStatus,
    StudyPriority,
    StudySessionStatus,
    StudySessionSyncStatus,
    StudyWindow,
)
from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository
from app.repositories.sqlite_offerpilot_repository import SQLiteOfferPilotRepository
from app.services.study_plan_service import (
    MissingStudyPreferencesError,
    StudyPlanService,
)
from app.services.study_scheduler import (
    BusyInterval,
    DeterministicStudyScheduler,
    StudySchedulingError,
    StudyTopic,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")


def _at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 9, day, hour, minute, tzinfo=SHANGHAI)


def _save_default_preferences(service: StudyPlanService, owner_id: str = "user_1"):
    return service.save_preferences(
        owner_id=owner_id,
        timezone="Asia/Shanghai",
        weekday_windows=[StudyWindow("19:00", "22:00")],
        weekend_windows=[StudyWindow("09:00", "12:00")],
        daily_max_minutes=120,
        session_minutes=60,
        now=_at(2, 18),
    )


def test_scheduler_splits_topics_around_busy_intervals_and_daily_cap() -> None:
    repository = InMemoryOfferPilotRepository()
    service = StudyPlanService(repository)
    preferences = _save_default_preferences(service)

    result = DeterministicStudyScheduler().schedule(
        preferences=preferences,
        topics=[
            StudyTopic(
                topic="JVM GC",
                duration_minutes=180,
                priority=StudyPriority.HIGH,
                deadline=_at(3, 22),
            )
        ],
        range_start=date(2026, 9, 2),
        range_end=date(2026, 9, 3),
        busy_intervals=[BusyInterval(_at(2, 20), _at(2, 21))],
        now=_at(2, 18),
    )

    assert [(block.start_at, block.end_at) for block in result.blocks] == [
        (_at(2, 19), _at(2, 20)),
        (_at(2, 21), _at(2, 22)),
        (_at(3, 19), _at(3, 20)),
    ]
    assert result.unscheduled_items == []


def test_scheduler_orders_by_deadline_then_priority_and_reports_remaining_work() -> None:
    repository = InMemoryOfferPilotRepository()
    service = StudyPlanService(repository)
    preferences = service.save_preferences(
        owner_id="user_1",
        timezone="Asia/Shanghai",
        weekday_windows=[StudyWindow("19:00", "20:00")],
        weekend_windows=[],
        daily_max_minutes=60,
        session_minutes=60,
        now=_at(2, 18),
    )

    result = DeterministicStudyScheduler().schedule(
        preferences=preferences,
        topics=[
            StudyTopic("系统设计", 60, StudyPriority.HIGH),
            StudyTopic("Redis", 60, StudyPriority.LOW, deadline=_at(2, 20)),
        ],
        range_start=date(2026, 9, 2),
        range_end=date(2026, 9, 2),
        now=_at(2, 18),
    )

    assert [block.topic for block in result.blocks] == ["Redis"]
    assert len(result.unscheduled_items) == 1
    assert result.unscheduled_items[0].topic == "系统设计"
    assert result.unscheduled_items[0].remaining_minutes == 60


def test_scheduler_counts_existing_study_time_toward_daily_cap() -> None:
    repository = InMemoryOfferPilotRepository()
    service = StudyPlanService(repository)
    preferences = _save_default_preferences(service)

    result = DeterministicStudyScheduler().schedule(
        preferences=preferences,
        topics=[StudyTopic("并发编程", 120, StudyPriority.HIGH)],
        range_start=date(2026, 9, 2),
        range_end=date(2026, 9, 2),
        busy_intervals=[
            BusyInterval(_at(2, 19), _at(2, 20), counts_toward_daily_cap=True)
        ],
        now=_at(2, 18),
    )

    assert [(block.start_at, block.end_at) for block in result.blocks] == [
        (_at(2, 20), _at(2, 21))
    ]
    assert result.unscheduled_items[0].remaining_minutes == 60


def test_save_preferences_rejects_invalid_or_empty_windows() -> None:
    service = StudyPlanService(InMemoryOfferPilotRepository())

    with pytest.raises(StudySchedulingError, match="At least one"):
        service.save_preferences(
            owner_id="user_1",
            timezone="Asia/Shanghai",
            weekday_windows=[],
            weekend_windows=[],
            daily_max_minutes=120,
            session_minutes=60,
            now=_at(2, 18),
        )

    with pytest.raises(StudySchedulingError, match="cannot cross midnight"):
        service.save_preferences(
            owner_id="user_1",
            timezone="Asia/Shanghai",
            weekday_windows=[StudyWindow("22:00", "20:00")],
            weekend_windows=[],
            daily_max_minutes=120,
            session_minutes=60,
            now=_at(2, 18),
        )


def test_create_plan_requires_preferences() -> None:
    service = StudyPlanService(InMemoryOfferPilotRepository())

    with pytest.raises(MissingStudyPreferencesError):
        service.create_plan(
            owner_id="user_1",
            goal="准备本周面试",
            range_start=date(2026, 9, 2),
            range_end=date(2026, 9, 3),
            topics=[StudyTopic("JVM", 60)],
            now=_at(2, 18),
        )


def test_service_creates_plan_avoids_existing_sessions_and_updates_sync_status() -> None:
    repository = InMemoryOfferPilotRepository()
    service = StudyPlanService(repository)
    _save_default_preferences(service)
    repository.create_interview_schedule(
        company="Example Tech",
        round_name="一面",
        start_at=_at(2, 22).isoformat(),
        owner_id="user_1",
    )
    first = service.create_plan(
        owner_id="user_1",
        goal="JVM 专项",
        range_start=date(2026, 9, 2),
        range_end=date(2026, 9, 2),
        topics=[StudyTopic("JVM", 60, StudyPriority.HIGH)],
        source_interview_ids=["schedule_1", "schedule_1"],
        now=_at(2, 18),
    )
    second = service.create_plan(
        owner_id="user_1",
        goal="Redis 专项",
        range_start=date(2026, 9, 2),
        range_end=date(2026, 9, 2),
        topics=[StudyTopic("Redis", 120, StudyPriority.MEDIUM)],
        now=_at(2, 18),
    )

    assert first.plan.source_interview_ids == ["schedule_1"]
    assert first.sessions[0].start_at == _at(2, 19)
    assert [session.start_at for session in second.sessions] == [_at(2, 20)]
    assert second.unscheduled_items[0].remaining_minutes == 60

    service.update_session(
        owner_id="user_1",
        session_id=first.sessions[0].id,
        sync_status=StudySessionSyncStatus.SYNCED,
        calendar_event_id="event_1",
        now=_at(2, 18, 5),
    )
    assert service.get_plan("user_1", first.plan.id).status == StudyPlanStatus.SYNCED

    changed = service.update_session(
        owner_id="user_1",
        session_id=first.sessions[0].id,
        start_at=_at(2, 20),
        end_at=_at(2, 21),
        now=_at(2, 18, 10),
    )
    assert changed.sync_status == StudySessionSyncStatus.PENDING
    assert changed.calendar_event_id == "event_1"


def test_source_interview_deterministically_caps_topic_deadline() -> None:
    repository = InMemoryOfferPilotRepository()
    service = StudyPlanService(repository)
    _save_default_preferences(service)
    interview = repository.create_interview_schedule(
        company="Example Tech",
        round_name="二面",
        start_at=_at(2, 20).isoformat(),
        owner_id="user_1",
    )

    result = service.create_plan(
        owner_id="user_1",
        goal="面试前复习",
        range_start=date(2026, 9, 2),
        range_end=date(2026, 9, 2),
        topics=[StudyTopic("JVM", 120, deadline=_at(3, 22))],
        source_interview_ids=[interview.id],
        now=_at(2, 18),
    )

    assert [(item.start_at, item.end_at) for item in result.sessions] == [
        (_at(2, 19), _at(2, 20))
    ]
    assert result.unscheduled_items[0].remaining_minutes == 60


def test_plan_status_tracks_partial_and_complete_calendar_sync() -> None:
    service = StudyPlanService(InMemoryOfferPilotRepository())
    _save_default_preferences(service)
    created = service.create_plan(
        owner_id="user_1",
        goal="两阶段复习",
        range_start=date(2026, 9, 2),
        range_end=date(2026, 9, 2),
        topics=[StudyTopic("网络", 120)],
        now=_at(2, 18),
    )

    service.update_session(
        owner_id="user_1",
        session_id=created.sessions[0].id,
        sync_status=StudySessionSyncStatus.SYNCED,
        calendar_event_id="event_1",
        now=_at(2, 18, 5),
    )
    assert service.get_plan("user_1", created.plan.id).status == StudyPlanStatus.PARTIALLY_SYNCED

    service.update_session(
        owner_id="user_1",
        session_id=created.sessions[1].id,
        sync_status=StudySessionSyncStatus.SYNCED,
        calendar_event_id="event_2",
        now=_at(2, 18, 10),
    )
    assert service.get_plan("user_1", created.plan.id).status == StudyPlanStatus.SYNCED


def test_in_memory_repository_deletes_plan_sessions_and_isolates_owners() -> None:
    repository = InMemoryOfferPilotRepository()
    service = StudyPlanService(repository)
    _save_default_preferences(service, "user_1")
    created = service.create_plan(
        owner_id="user_1",
        goal="面试复习",
        range_start=date(2026, 9, 2),
        range_end=date(2026, 9, 2),
        topics=[StudyTopic("MySQL", 60)],
        now=_at(2, 18),
    )

    assert repository.get_study_plan(created.plan.id, owner_id="user_2") is None
    assert repository.delete_study_plan(created.plan.id, owner_id="user_2") is False
    assert repository.delete_study_plan(created.plan.id, owner_id="user_1") is True
    assert repository.list_study_sessions(owner_id="user_1") == []


@pytest.mark.parametrize("repository_kind", ["memory", "sqlite"])
def test_study_sessions_cannot_reference_another_owners_plan(
    repository_kind: str,
    tmp_path,
) -> None:
    repository = (
        InMemoryOfferPilotRepository()
        if repository_kind == "memory"
        else SQLiteOfferPilotRepository(str(tmp_path / "offerpilot.db"))
    )
    service = StudyPlanService(repository)
    for owner_id in ("user_1", "user_2"):
        _save_default_preferences(service, owner_id)

    first = service.create_plan(
        owner_id="user_1",
        goal="用户一复习",
        range_start=date(2026, 9, 2),
        range_end=date(2026, 9, 2),
        topics=[StudyTopic("MySQL", 60)],
        now=_at(2, 18),
    )
    second = service.create_plan(
        owner_id="user_2",
        goal="用户二复习",
        range_start=date(2026, 9, 2),
        range_end=date(2026, 9, 2),
        topics=[StudyTopic("Redis", 60)],
        now=_at(2, 18),
    )

    with pytest.raises(ValueError, match="Study plan does not exist"):
        repository.create_study_session(
            replace(
                first.sessions[0],
                id="cross_owner_session",
                owner_id="user_2",
            )
        )

    with pytest.raises(ValueError, match="current owner"):
        repository.update_study_session(
            replace(second.sessions[0], plan_id=first.plan.id)
        )


def test_sqlite_repository_persists_study_domain_and_cascades_manual_delete(tmp_path) -> None:
    db_path = tmp_path / "offerpilot.db"
    service = StudyPlanService(SQLiteOfferPilotRepository(str(db_path)))
    preferences = _save_default_preferences(service)
    source_interview = service.repository.create_interview_schedule(
        company="Example Tech",
        round_name="技术面",
        start_at=_at(3, 21).isoformat(),
        owner_id="user_1",
    )
    created = service.create_plan(
        owner_id="user_1",
        goal="准备字节一面",
        range_start=date(2026, 9, 2),
        range_end=date(2026, 9, 3),
        topics=[
            StudyTopic(
                "Agent 记忆",
                300,
                StudyPriority.HIGH,
                deadline=_at(3, 21),
                source_refs=["gap_1"],
                rationale="最近复盘得分偏低",
            )
        ],
        source_interview_ids=[source_interview.id],
        now=_at(2, 18),
    )

    reopened = SQLiteOfferPilotRepository(str(db_path))
    persisted_preferences = reopened.get_study_preferences(owner_id="user_1")
    persisted_plan = reopened.get_study_plan(created.plan.id, owner_id="user_1")
    persisted_sessions = reopened.list_study_sessions(
        owner_id="user_1",
        plan_id=created.plan.id,
    )

    assert persisted_preferences == preferences
    assert persisted_plan is not None
    assert persisted_plan.goal == "准备字节一面"
    assert persisted_plan.source_interview_ids == [source_interview.id]
    assert [session.topic for session in persisted_sessions] == [
        "Agent 记忆",
        "Agent 记忆",
        "Agent 记忆",
        "Agent 记忆",
    ]
    assert persisted_plan.unscheduled_items == created.unscheduled_items
    assert persisted_plan.unscheduled_items[0].remaining_minutes == 60
    assert persisted_sessions[0].source_refs == ["gap_1"]
    assert reopened.list_study_plans(owner_id="user_2") == []

    persisted_sessions[0].status = StudySessionStatus.COMPLETED
    assert reopened.update_study_session(persisted_sessions[0]) is not None
    assert SQLiteOfferPilotRepository(str(db_path)).get_study_session(
        persisted_sessions[0].id,
        owner_id="user_1",
    ).status == StudySessionStatus.COMPLETED

    assert reopened.delete_study_plan(created.plan.id, owner_id="user_1") is True
    assert reopened.list_study_sessions(owner_id="user_1") == []


def test_sqlite_migrates_legacy_study_plan_table_with_empty_unscheduled_items(
    tmp_path,
) -> None:
    db_path = tmp_path / "legacy_offerpilot.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE study_plans (
                id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                goal TEXT NOT NULL,
                range_start TEXT NOT NULL,
                range_end TEXT NOT NULL,
                source_interview_ids TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO study_plans (
                id, owner_id, goal, range_start, range_end,
                source_interview_ids, status, version, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy_plan",
                "user_1",
                "旧复习计划",
                "2026-09-02",
                "2026-09-03",
                "[]",
                "draft",
                1,
                _at(2, 18).isoformat(),
                _at(2, 18).isoformat(),
            ),
        )

    repository = SQLiteOfferPilotRepository(str(db_path))
    plan = repository.get_study_plan("legacy_plan", owner_id="user_1")

    assert plan is not None
    assert plan.unscheduled_items == []
    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(study_plans)")
        }
    assert "unscheduled_items" in columns
