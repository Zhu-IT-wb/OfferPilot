from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from math import inf
from typing import List, Optional, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.models.study import (
    StudyPreferences,
    StudyPriority,
    StudyWindow,
    UnscheduledStudyItem,
)


class StudySchedulingError(ValueError):
    pass


@dataclass(frozen=True)
class BusyInterval:
    start_at: datetime
    end_at: datetime
    counts_toward_daily_cap: bool = False


@dataclass
class StudyTopic:
    topic: str
    duration_minutes: int
    priority: StudyPriority = StudyPriority.MEDIUM
    deadline: Optional[datetime] = None
    source_refs: List[str] = field(default_factory=list)
    rationale: str = ""


@dataclass(frozen=True)
class ScheduledStudyBlock:
    topic: str
    start_at: datetime
    end_at: datetime
    priority: StudyPriority
    source_refs: List[str]
    rationale: str

    @property
    def duration_minutes(self) -> int:
        return int((self.end_at - self.start_at).total_seconds() // 60)


@dataclass(frozen=True)
class StudyScheduleResult:
    blocks: List[ScheduledStudyBlock]
    unscheduled_items: List[UnscheduledStudyItem]


class DeterministicStudyScheduler:
    def schedule(
        self,
        preferences: StudyPreferences,
        topics: Sequence[StudyTopic],
        range_start: date,
        range_end: date,
        busy_intervals: Sequence[BusyInterval] = (),
        now: Optional[datetime] = None,
    ) -> StudyScheduleResult:
        timezone = self._validate_inputs(
            preferences=preferences,
            topics=topics,
            range_start=range_start,
            range_end=range_end,
            busy_intervals=busy_intervals,
            now=now,
        )
        current_time = (now or datetime.now(timezone)).astimezone(timezone)
        normalized_busy = [
            BusyInterval(
                start_at=interval.start_at.astimezone(timezone),
                end_at=interval.end_at.astimezone(timezone),
                counts_toward_daily_cap=interval.counts_toward_daily_cap,
            )
            for interval in busy_intervals
        ]
        all_busy = list(normalized_busy)
        daily_study_minutes = self._daily_study_minutes(
            normalized_busy,
            range_start,
            range_end,
            timezone,
        )
        blocks: List[ScheduledStudyBlock] = []
        unscheduled: List[UnscheduledStudyItem] = []

        ordered_topics = sorted(
            enumerate(topics),
            key=lambda indexed: (
                indexed[1].deadline.timestamp() if indexed[1].deadline else inf,
                -int(indexed[1].priority),
                indexed[0],
            ),
        )
        for _, topic in ordered_topics:
            remaining = topic.duration_minutes
            deadline = topic.deadline.astimezone(timezone) if topic.deadline else None
            day_cursor = range_start
            while day_cursor <= range_end and remaining > 0:
                if deadline is not None and self._day_start(day_cursor, timezone) >= deadline:
                    break
                used_today = daily_study_minutes.get(day_cursor, 0)
                available_today = max(preferences.daily_max_minutes - used_today, 0)
                if available_today == 0:
                    day_cursor += timedelta(days=1)
                    continue

                windows = (
                    preferences.weekend_windows
                    if day_cursor.weekday() >= 5
                    else preferences.weekday_windows
                )
                for window in sorted(windows, key=lambda item: item.start_time):
                    if remaining == 0 or available_today == 0:
                        break
                    window_start, window_end = self._window_bounds(day_cursor, window, timezone)
                    window_start = max(window_start, current_time)
                    if deadline is not None:
                        window_end = min(window_end, deadline)
                    if window_start >= window_end:
                        continue

                    while remaining > 0 and available_today > 0:
                        block_minutes = min(
                            preferences.session_minutes,
                            remaining,
                            available_today,
                        )
                        slot_start = self._find_slot(
                            window_start,
                            window_end,
                            block_minutes,
                            all_busy,
                        )
                        if slot_start is None:
                            break
                        slot_end = slot_start + timedelta(minutes=block_minutes)
                        block = ScheduledStudyBlock(
                            topic=topic.topic.strip(),
                            start_at=slot_start,
                            end_at=slot_end,
                            priority=StudyPriority(topic.priority),
                            source_refs=list(topic.source_refs),
                            rationale=topic.rationale,
                        )
                        blocks.append(block)
                        all_busy.append(BusyInterval(slot_start, slot_end, True))
                        remaining -= block_minutes
                        available_today -= block_minutes
                        daily_study_minutes[day_cursor] = (
                            daily_study_minutes.get(day_cursor, 0) + block_minutes
                        )
                        window_start = slot_end

                day_cursor += timedelta(days=1)

            if remaining > 0:
                reason = (
                    "deadline_outside_available_range"
                    if deadline is not None and deadline <= self._day_start(range_start, timezone)
                    else "insufficient_capacity_before_deadline"
                )
                unscheduled.append(
                    UnscheduledStudyItem(
                        topic=topic.topic.strip(),
                        requested_minutes=topic.duration_minutes,
                        remaining_minutes=remaining,
                        priority=StudyPriority(topic.priority),
                        deadline=deadline,
                        reason=reason,
                    )
                )

        return StudyScheduleResult(
            blocks=sorted(blocks, key=lambda block: (block.start_at, block.topic)),
            unscheduled_items=unscheduled,
        )

    @classmethod
    def _validate_inputs(
        cls,
        preferences: StudyPreferences,
        topics: Sequence[StudyTopic],
        range_start: date,
        range_end: date,
        busy_intervals: Sequence[BusyInterval],
        now: Optional[datetime],
    ) -> ZoneInfo:
        try:
            timezone = ZoneInfo(preferences.timezone)
        except ZoneInfoNotFoundError as exc:
            raise StudySchedulingError(f"Unknown timezone: {preferences.timezone}") from exc
        if range_end < range_start:
            raise StudySchedulingError("range_end must not be before range_start")
        if preferences.daily_max_minutes <= 0:
            raise StudySchedulingError("daily_max_minutes must be positive")
        if preferences.session_minutes <= 0:
            raise StudySchedulingError("session_minutes must be positive")
        if now is not None and now.tzinfo is None:
            raise StudySchedulingError("now must be timezone-aware")
        for window in (*preferences.weekday_windows, *preferences.weekend_windows):
            cls._parse_window(window)
        for topic in topics:
            if not topic.topic.strip():
                raise StudySchedulingError("Study topic must not be empty")
            if topic.duration_minutes <= 0:
                raise StudySchedulingError("duration_minutes must be positive")
            try:
                StudyPriority(topic.priority)
            except ValueError as exc:
                raise StudySchedulingError("priority must be low, medium, or high") from exc
            if topic.deadline is not None and topic.deadline.tzinfo is None:
                raise StudySchedulingError("Topic deadline must be timezone-aware")
        for interval in busy_intervals:
            if interval.start_at.tzinfo is None or interval.end_at.tzinfo is None:
                raise StudySchedulingError("Busy intervals must be timezone-aware")
            if interval.end_at <= interval.start_at:
                raise StudySchedulingError("Busy interval end must be after start")
        return timezone

    @staticmethod
    def _parse_window(window: StudyWindow) -> tuple[time, time]:
        try:
            start = time.fromisoformat(window.start_time)
            end = time.fromisoformat(window.end_time)
        except ValueError as exc:
            raise StudySchedulingError("Study windows must use HH:MM time values") from exc
        if start.tzinfo is not None or end.tzinfo is not None:
            raise StudySchedulingError("Study window times must not contain a timezone")
        if end <= start:
            raise StudySchedulingError("Study windows cannot cross midnight")
        return start, end

    @classmethod
    def _window_bounds(
        cls,
        day: date,
        window: StudyWindow,
        timezone: ZoneInfo,
    ) -> tuple[datetime, datetime]:
        start, end = cls._parse_window(window)
        return datetime.combine(day, start, timezone), datetime.combine(day, end, timezone)

    @staticmethod
    def _day_start(day: date, timezone: ZoneInfo) -> datetime:
        return datetime.combine(day, time.min, timezone)

    @classmethod
    def _daily_study_minutes(
        cls,
        busy_intervals: Sequence[BusyInterval],
        range_start: date,
        range_end: date,
        timezone: ZoneInfo,
    ) -> dict[date, int]:
        result: dict[date, int] = {}
        day = range_start
        while day <= range_end:
            day_start = cls._day_start(day, timezone)
            day_end = cls._day_start(day + timedelta(days=1), timezone)
            minutes = 0
            for interval in busy_intervals:
                if not interval.counts_toward_daily_cap:
                    continue
                overlap_start = max(interval.start_at, day_start)
                overlap_end = min(interval.end_at, day_end)
                if overlap_end > overlap_start:
                    minutes += int((overlap_end - overlap_start).total_seconds() // 60)
            result[day] = minutes
            day += timedelta(days=1)
        return result

    @staticmethod
    def _find_slot(
        window_start: datetime,
        window_end: datetime,
        duration_minutes: int,
        busy_intervals: Sequence[BusyInterval],
    ) -> Optional[datetime]:
        duration = timedelta(minutes=duration_minutes)
        candidate = window_start
        relevant_busy = sorted(
            (
                interval
                for interval in busy_intervals
                if interval.end_at > window_start and interval.start_at < window_end
            ),
            key=lambda interval: interval.start_at,
        )
        for interval in relevant_busy:
            if candidate + duration <= interval.start_at:
                return candidate
            if interval.end_at > candidate:
                candidate = interval.end_at
            if candidate + duration > window_end:
                return None
        return candidate if candidate + duration <= window_end else None
