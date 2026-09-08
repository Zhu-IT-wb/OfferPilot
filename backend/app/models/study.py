from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional


class StudyPlanStatus(str, Enum):
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    PARTIALLY_SYNCED = "partially_synced"
    SYNCED = "synced"
    CANCELLED = "cancelled"


class StudySessionStatus(str, Enum):
    PLANNED = "planned"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class StudySessionSyncStatus(str, Enum):
    PENDING = "pending"
    SYNCED = "synced"
    FAILED = "failed"
    UNKNOWN = "unknown"


class StudyPriority(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass(frozen=True)
class UnscheduledStudyItem:
    topic: str
    requested_minutes: int
    remaining_minutes: int
    priority: StudyPriority
    deadline: Optional[datetime]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "requested_minutes": self.requested_minutes,
            "remaining_minutes": self.remaining_minutes,
            "priority": int(self.priority),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class StudyWindow:
    start_time: str
    end_time: str

    def to_dict(self) -> Dict[str, str]:
        return {"start_time": self.start_time, "end_time": self.end_time}


@dataclass
class StudyPreferences:
    owner_id: str
    timezone: str
    weekday_windows: List[StudyWindow]
    weekend_windows: List[StudyWindow]
    daily_max_minutes: int
    session_minutes: int
    updated_at: datetime = field(default_factory=lambda: datetime.now().astimezone())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "owner_id": self.owner_id,
            "timezone": self.timezone,
            "weekday_windows": [window.to_dict() for window in self.weekday_windows],
            "weekend_windows": [window.to_dict() for window in self.weekend_windows],
            "daily_max_minutes": self.daily_max_minutes,
            "session_minutes": self.session_minutes,
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class StudyPlan:
    id: str
    owner_id: str
    goal: str
    range_start: date
    range_end: date
    source_interview_ids: List[str] = field(default_factory=list)
    unscheduled_items: List[UnscheduledStudyItem] = field(default_factory=list)
    status: StudyPlanStatus = StudyPlanStatus.DRAFT
    version: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    updated_at: datetime = field(default_factory=lambda: datetime.now().astimezone())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "goal": self.goal,
            "range_start": self.range_start.isoformat(),
            "range_end": self.range_end.isoformat(),
            "source_interview_ids": list(self.source_interview_ids),
            "unscheduled": [item.to_dict() for item in self.unscheduled_items],
            "status": self.status.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class StudySession:
    id: str
    owner_id: str
    plan_id: str
    topic: str
    start_at: datetime
    end_at: datetime
    priority: StudyPriority = StudyPriority.MEDIUM
    source_refs: List[str] = field(default_factory=list)
    rationale: str = ""
    status: StudySessionStatus = StudySessionStatus.PLANNED
    sync_status: StudySessionSyncStatus = StudySessionSyncStatus.PENDING
    calendar_event_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    updated_at: datetime = field(default_factory=lambda: datetime.now().astimezone())

    @property
    def duration_minutes(self) -> int:
        return int((self.end_at - self.start_at).total_seconds() // 60)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "plan_id": self.plan_id,
            "topic": self.topic,
            "start_at": self.start_at.isoformat(),
            "end_at": self.end_at.isoformat(),
            "duration_minutes": self.duration_minutes,
            "priority": int(self.priority),
            "source_refs": list(self.source_refs),
            "rationale": self.rationale,
            "status": self.status.value,
            "sync_status": self.sync_status.value,
            "calendar_event_id": self.calendar_event_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
