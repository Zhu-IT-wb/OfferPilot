from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class LeetCodeDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class LeetCodeAssignmentType(str, Enum):
    NEW = "new"
    REVIEW = "review"
    CARRYOVER = "carryover"


class LeetCodeAssignmentStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    POSTPONED = "postponed"
    SKIPPED = "skipped"


class LeetCodePracticeResult(str, Enum):
    INDEPENDENT = "independent"
    WITH_HINT = "with_hint"
    WITH_SOLUTION = "with_solution"
    FAILED = "failed"
    POSTPONED = "postponed"
    SKIPPED = "skipped"


class LeetCodeDeliveryType(str, Enum):
    MORNING = "morning"
    EVENING = "evening"


class LeetCodeMasteryStatus(str, Enum):
    LEARNING = "learning"
    REVIEWING = "reviewing"
    MASTERED = "mastered"


@dataclass(frozen=True)
class LeetCodeProblem:
    id: str
    frontend_id: str
    title_zh: str
    title_en: str
    slug: str
    difficulty: LeetCodeDifficulty
    topics: List[str]
    category: str
    category_order: int
    problem_order: int
    url: str
    source: str = "leetcode_hot_100"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "frontend_id": self.frontend_id,
            "title_zh": self.title_zh,
            "title_en": self.title_en,
            "slug": self.slug,
            "difficulty": self.difficulty.value,
            "topics": list(self.topics),
            "category": self.category,
            "category_order": self.category_order,
            "problem_order": self.problem_order,
            "url": self.url,
            "source": self.source,
        }


@dataclass
class LeetCodeAssignment:
    id: str
    owner_id: str
    problem_id: str
    assigned_on: date
    assignment_type: LeetCodeAssignmentType
    recommendation_reason: str
    status: LeetCodeAssignmentStatus = LeetCodeAssignmentStatus.PENDING
    result: Optional[LeetCodePracticeResult] = None
    postpone_count: int = 0
    feedback_reminded_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "problem_id": self.problem_id,
            "assigned_on": self.assigned_on.isoformat(),
            "assignment_type": self.assignment_type.value,
            "recommendation_reason": self.recommendation_reason,
            "status": self.status.value,
            "result": self.result.value if self.result else None,
            "postpone_count": self.postpone_count,
            "feedback_reminded_at": self.feedback_reminded_at.isoformat() if self.feedback_reminded_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class LeetCodeProgress:
    owner_id: str
    problem_id: str
    mastery_status: LeetCodeMasteryStatus = LeetCodeMasteryStatus.LEARNING
    independent_streak: int = 0
    attempt_count: int = 0
    last_result: Optional[LeetCodePracticeResult] = None
    last_practiced_on: Optional[date] = None
    next_review_on: Optional[date] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "owner_id": self.owner_id,
            "problem_id": self.problem_id,
            "mastery_status": self.mastery_status.value,
            "independent_streak": self.independent_streak,
            "attempt_count": self.attempt_count,
            "last_result": self.last_result.value if self.last_result else None,
            "last_practiced_on": self.last_practiced_on.isoformat() if self.last_practiced_on else None,
            "next_review_on": self.next_review_on.isoformat() if self.next_review_on else None,
        }


@dataclass
class LeetCodeSubscription:
    owner_id: str
    feishu_open_id: str
    enabled: bool = True
    timezone: str = "Asia/Shanghai"
    morning_time: str = "09:00"
    evening_time: str = "21:00"


@dataclass(frozen=True)
class LeetCodeRecommendation:
    assignment: LeetCodeAssignment
    problem: LeetCodeProblem


@dataclass(frozen=True)
class LeetCodeFeedback:
    assignment: LeetCodeAssignment
    progress: Optional[LeetCodeProgress]
