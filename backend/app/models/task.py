from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


# 枚举 TaskType 的可选值。
class TaskType(str, Enum):
    LEETCODE = "leetcode"
    INTERVIEW_QUESTION = "interview_question"
    PROJECT_DEEP_DIVE = "project_deep_dive"
    APPLICATION = "application"
    REVIEW = "review"
    CUSTOM = "custom"


# 枚举 TaskStatus 的可选值。
class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    POSTPONED = "postponed"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


# 枚举 TaskPriority 的可选值。
class TaskPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# 定义 Task 相关的数据结构或领域对象。
@dataclass
class Task:
    id: str
    title: str
    task_type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM

    # 将当前领域对象转换为可序列化字典。
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "priority": self.priority.value,
        }
