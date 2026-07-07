from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


# 枚举 InterviewScheduleStatus 的可选值。
class InterviewScheduleStatus(str, Enum):
    SCHEDULED = "scheduled"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# 定义 InterviewSchedule 相关的数据结构或领域对象。
@dataclass
class InterviewSchedule:
    id: str
    company: str
    round: str
    owner_id: str = "local_user"
    application_id: Optional[str] = None
    role: Optional[str] = None
    start_time: Optional[str] = None
    start_at: Optional[str] = None
    reminder_minutes: int = 30
    status: InterviewScheduleStatus = InterviewScheduleStatus.SCHEDULED
    calendar_event_id: Optional[str] = None
    raw_message: str = ""

    # 将当前领域对象转换为可序列化字典。
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "application_id": self.application_id,
            "company": self.company,
            "role": self.role,
            "round": self.round,
            "start_time": self.start_time,
            "start_at": self.start_at,
            "reminder_minutes": self.reminder_minutes,
            "status": self.status.value,
            "calendar_event_id": self.calendar_event_id,
            "raw_message": self.raw_message,
        }
