from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class TaskType(str, Enum):
    LEETCODE = "leetcode"
    INTERVIEW_QUESTION = "interview_question"
    PROJECT_DEEP_DIVE = "project_deep_dive"
    APPLICATION = "application"
    REVIEW = "review"
    CUSTOM = "custom"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    POSTPONED = "postponed"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Task:
    id: str
    title: str
    task_type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "priority": self.priority.value,
        }
