from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from typing import List, Optional, Sequence
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.models.study import (
    StudyPlan,
    StudyPlanStatus,
    StudyPreferences,
    StudySession,
    StudySessionStatus,
    StudySessionSyncStatus,
    StudyWindow,
    UnscheduledStudyItem,
)
from app.models.interview_schedule import InterviewScheduleStatus
from app.repositories.offerpilot_repository import OfferPilotRepository
from app.services.study_scheduler import (
    BusyInterval,
    DeterministicStudyScheduler,
    StudySchedulingError,
    StudyTopic,
)


class MissingStudyPreferencesError(RuntimeError):
    pass


class StudyPlanNotFoundError(LookupError):
    pass


class StudySessionNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class StudyPlanCreationResult:
    plan: StudyPlan
    sessions: List[StudySession]
    unscheduled_items: List[UnscheduledStudyItem]

    def to_dict(self) -> dict:
        return {
            "plan": self.plan.to_dict(),
            "sessions": [session.to_dict() for session in self.sessions],
            "unscheduled": [item.to_dict() for item in self.unscheduled_items],
        }


class StudyPlanService:
    def __init__(
        self,
        repository: OfferPilotRepository,
        scheduler: Optional[DeterministicStudyScheduler] = None,
    ) -> None:
        self.repository = repository
        self.scheduler = scheduler or DeterministicStudyScheduler()

    def get_preferences(
        self,
        owner_id: str = "local_user",
    ) -> Optional[StudyPreferences]:
        return self.repository.get_study_preferences(owner_id=self._owner_id(owner_id))

    def save_preferences(
        self,
        owner_id: str,
        timezone: str,
        weekday_windows: Sequence[StudyWindow],
        weekend_windows: Sequence[StudyWindow],
        daily_max_minutes: int,
        session_minutes: int,
        now: Optional[datetime] = None,
    ) -> StudyPreferences:
        try:
            zone = ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise StudySchedulingError(f"Unknown timezone: {timezone}") from exc
        updated_at = now or datetime.now(zone)
        if updated_at.tzinfo is None:
            raise StudySchedulingError("now must be timezone-aware")
        preferences = StudyPreferences(
            owner_id=self._owner_id(owner_id),
            timezone=timezone,
            weekday_windows=list(weekday_windows),
            weekend_windows=list(weekend_windows),
            daily_max_minutes=daily_max_minutes,
            session_minutes=session_minutes,
            updated_at=updated_at.astimezone(zone),
        )
        if not preferences.weekday_windows and not preferences.weekend_windows:
            raise StudySchedulingError("At least one study window is required")
        self.scheduler.schedule(
            preferences=preferences,
            topics=(),
            range_start=updated_at.astimezone(zone).date(),
            range_end=updated_at.astimezone(zone).date(),
            now=updated_at,
        )
        return self.repository.save_study_preferences(preferences)

    def create_plan(
        self,
        owner_id: str,
        goal: str,
        range_start: date,
        range_end: date,
        topics: Sequence[StudyTopic],
        source_interview_ids: Sequence[str] = (),
        busy_intervals: Sequence[BusyInterval] = (),
        now: Optional[datetime] = None,
    ) -> StudyPlanCreationResult:
        normalized_owner = self._owner_id(owner_id)
        if not goal.strip():
            raise StudySchedulingError("Study plan goal must not be empty")
        if not topics:
            raise StudySchedulingError("At least one study topic is required")
        preferences = self.get_preferences(owner_id=normalized_owner)
        if preferences is None:
            raise MissingStudyPreferencesError(
                "Study preferences are required before creating a study plan"
            )
        zone = ZoneInfo(preferences.timezone)
        created_at = now or datetime.now(zone)
        if created_at.tzinfo is None:
            raise StudySchedulingError("now must be timezone-aware")
        created_at = created_at.astimezone(zone)

        normalized_interview_ids = self._unique_nonempty(source_interview_ids)
        interview_deadline = self._source_interview_deadline(
            normalized_owner,
            normalized_interview_ids,
            zone,
        )
        constrained_topics = [
            replace(
                topic,
                deadline=(
                    interview_deadline
                    if topic.deadline is None
                    else min(topic.deadline.astimezone(zone), interview_deadline)
                ),
            )
            if interview_deadline is not None
            else topic
            for topic in topics
        ]

        query_start = datetime.combine(range_start, time.min, zone)
        query_end = datetime.combine(range_end + timedelta(days=1), time.min, zone)
        existing_sessions = self.repository.list_study_sessions(
            owner_id=normalized_owner,
            range_start=query_start,
            range_end=query_end,
        )
        existing_study_busy = [
            BusyInterval(
                start_at=session.start_at,
                end_at=session.end_at,
                counts_toward_daily_cap=True,
            )
            for session in existing_sessions
            if session.status
            not in {StudySessionStatus.SKIPPED, StudySessionStatus.CANCELLED}
        ]
        schedule = self.scheduler.schedule(
            preferences=preferences,
            topics=constrained_topics,
            range_start=range_start,
            range_end=range_end,
            busy_intervals=[*busy_intervals, *existing_study_busy],
            now=created_at,
        )

        plan = StudyPlan(
            id=f"study_plan_{uuid4().hex}",
            owner_id=normalized_owner,
            goal=goal.strip(),
            range_start=range_start,
            range_end=range_end,
            source_interview_ids=normalized_interview_ids,
            unscheduled_items=list(schedule.unscheduled_items),
            status=(StudyPlanStatus.SCHEDULED if schedule.blocks else StudyPlanStatus.DRAFT),
            created_at=created_at,
            updated_at=created_at,
        )
        sessions = [
            StudySession(
                id=f"study_session_{uuid4().hex}",
                owner_id=normalized_owner,
                plan_id=plan.id,
                topic=block.topic,
                start_at=block.start_at,
                end_at=block.end_at,
                priority=block.priority,
                source_refs=self._unique_nonempty(block.source_refs),
                rationale=block.rationale,
                created_at=created_at,
                updated_at=created_at,
            )
            for block in schedule.blocks
        ]

        self.repository.create_study_plan(plan)
        try:
            for session in sessions:
                self.repository.create_study_session(session)
        except Exception:
            self.repository.delete_study_plan(plan.id, owner_id=normalized_owner)
            raise
        return StudyPlanCreationResult(
            plan=plan,
            sessions=sessions,
            unscheduled_items=schedule.unscheduled_items,
        )

    def _source_interview_deadline(
        self,
        owner_id: str,
        source_interview_ids: Sequence[str],
        zone: ZoneInfo,
    ) -> Optional[datetime]:
        if not source_interview_ids:
            return None
        schedules = {
            schedule.id: schedule
            for schedule in self.repository.list_interview_schedules(
                owner_id=owner_id
            )
        }
        deadlines: List[datetime] = []
        for schedule_id in source_interview_ids:
            schedule = schedules.get(schedule_id)
            if schedule is None:
                raise StudySchedulingError(
                    f"Source interview does not belong to this owner: {schedule_id}"
                )
            if (
                schedule.status != InterviewScheduleStatus.SCHEDULED
                or not schedule.start_at
            ):
                raise StudySchedulingError(
                    f"Source interview is not an active absolute schedule: {schedule_id}"
                )
            try:
                start_at = datetime.fromisoformat(
                    str(schedule.start_at).replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise StudySchedulingError(
                    f"Source interview has an invalid start_at: {schedule_id}"
                ) from exc
            if start_at.tzinfo is None or start_at.utcoffset() is None:
                raise StudySchedulingError(
                    f"Source interview must have a timezone: {schedule_id}"
                )
            deadlines.append(start_at.astimezone(zone))
        return min(deadlines)

    def get_plan(
        self,
        owner_id: str,
        plan_id: str,
    ) -> Optional[StudyPlan]:
        return self.repository.get_study_plan(plan_id, owner_id=self._owner_id(owner_id))

    def list_plans(
        self,
        owner_id: str,
        status: Optional[StudyPlanStatus] = None,
    ) -> List[StudyPlan]:
        return self.repository.list_study_plans(
            owner_id=self._owner_id(owner_id),
            status=status,
        )

    def list_sessions(
        self,
        owner_id: str,
        plan_id: Optional[str] = None,
        range_start: Optional[datetime] = None,
        range_end: Optional[datetime] = None,
        status: Optional[StudySessionStatus] = None,
    ) -> List[StudySession]:
        return self.repository.list_study_sessions(
            owner_id=self._owner_id(owner_id),
            plan_id=plan_id,
            range_start=range_start,
            range_end=range_end,
            status=status,
        )

    def update_session(
        self,
        owner_id: str,
        session_id: str,
        status: Optional[StudySessionStatus] = None,
        sync_status: Optional[StudySessionSyncStatus] = None,
        calendar_event_id: Optional[str] = None,
        clear_calendar_event: bool = False,
        start_at: Optional[datetime] = None,
        end_at: Optional[datetime] = None,
        now: Optional[datetime] = None,
    ) -> StudySession:
        normalized_owner = self._owner_id(owner_id)
        existing = self.repository.get_study_session(
            session_id,
            owner_id=normalized_owner,
        )
        if existing is None:
            raise StudySessionNotFoundError(f"Study session not found: {session_id}")
        next_start = start_at or existing.start_at
        next_end = end_at or existing.end_at
        if next_start.tzinfo is None or next_end.tzinfo is None:
            raise StudySchedulingError("Study session times must be timezone-aware")
        if next_end <= next_start:
            raise StudySchedulingError("Study session end must be after start")
        next_event_id = (
            None
            if clear_calendar_event
            else calendar_event_id or existing.calendar_event_id
        )
        next_status = status or existing.status
        calendar_visible_change = bool(
            existing.calendar_event_id
            and (
                next_start != existing.start_at
                or next_end != existing.end_at
                or (
                    next_status != existing.status
                    and next_status
                    in {StudySessionStatus.SKIPPED, StudySessionStatus.CANCELLED}
                )
            )
        )
        next_sync_status = (
            sync_status
            or (
                StudySessionSyncStatus.PENDING
                if calendar_visible_change
                else existing.sync_status
            )
        )
        if next_sync_status == StudySessionSyncStatus.SYNCED and not next_event_id:
            raise StudySchedulingError(
                "A synced study session must have a calendar_event_id"
            )
        updated = replace(
            existing,
            start_at=next_start,
            end_at=next_end,
            status=next_status,
            sync_status=next_sync_status,
            calendar_event_id=next_event_id,
            updated_at=now or datetime.now().astimezone(),
        )
        if updated.updated_at.tzinfo is None:
            raise StudySchedulingError("now must be timezone-aware")
        persisted = self.repository.update_study_session(updated)
        if persisted is None:
            raise StudySessionNotFoundError(f"Study session not found: {session_id}")
        self.refresh_plan_status(normalized_owner, updated.plan_id, now=updated.updated_at)
        return persisted

    def refresh_plan_status(
        self,
        owner_id: str,
        plan_id: str,
        now: Optional[datetime] = None,
    ) -> StudyPlan:
        normalized_owner = self._owner_id(owner_id)
        plan = self.repository.get_study_plan(plan_id, owner_id=normalized_owner)
        if plan is None:
            raise StudyPlanNotFoundError(f"Study plan not found: {plan_id}")
        if plan.status == StudyPlanStatus.CANCELLED:
            return plan
        sessions = self.repository.list_study_sessions(
            owner_id=normalized_owner,
            plan_id=plan_id,
        )
        syncable_sessions = [
            session
            for session in sessions
            if session.status
            not in {StudySessionStatus.SKIPPED, StudySessionStatus.CANCELLED}
        ]
        if not sessions:
            next_status = StudyPlanStatus.DRAFT
        elif not syncable_sessions:
            next_status = StudyPlanStatus.SCHEDULED
        elif all(
            session.sync_status == StudySessionSyncStatus.SYNCED
            for session in syncable_sessions
        ):
            next_status = StudyPlanStatus.SYNCED
        elif any(
            session.sync_status == StudySessionSyncStatus.SYNCED
            for session in syncable_sessions
        ):
            next_status = StudyPlanStatus.PARTIALLY_SYNCED
        else:
            next_status = StudyPlanStatus.SCHEDULED
        updated = replace(
            plan,
            status=next_status,
            updated_at=now or datetime.now().astimezone(),
        )
        persisted = self.repository.update_study_plan(updated)
        if persisted is None:
            raise StudyPlanNotFoundError(f"Study plan not found: {plan_id}")
        return persisted

    @staticmethod
    def _owner_id(owner_id: str) -> str:
        return (owner_id or "").strip() or "local_user"

    @staticmethod
    def _unique_nonempty(values: Sequence[str]) -> List[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))
