from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class KnowledgeDifficulty(str, Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class KnowledgeRubricKind(str, Enum):
    REQUIRED = "required"
    BONUS = "bonus"
    MISCONCEPTION = "misconception"


class KnowledgeAnswerSource(str, Enum):
    TEXT = "text"
    VOICE_TRANSCRIPT = "voice_transcript"
    SKIPPED = "skipped"


class KnowledgeFactualErrorSeverity(str, Enum):
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


class KnowledgeAssignmentType(str, Enum):
    NEW = "new"
    DUE_REVIEW = "due_review"
    WEAKNESS = "weakness"
    MANUAL = "manual"


class KnowledgeAssignmentStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    POSTPONED = "postponed"


class KnowledgeMasteryStatus(str, Enum):
    UNSEEN = "unseen"
    LEARNING = "learning"
    REVIEWING = "reviewing"
    MASTERED = "mastered"


class KnowledgeDeliveryType(str, Enum):
    MORNING = "morning"
    NOON = "noon"
    EVENING = "evening"


@dataclass(frozen=True)
class KnowledgeRubricPoint:
    id: str
    kind: KnowledgeRubricKind
    label: str
    description: str
    weight: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "label": self.label,
            "description": self.description,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class KnowledgeFactualError:
    quote: str
    explanation: str
    severity: KnowledgeFactualErrorSeverity

    def to_dict(self) -> Dict[str, str]:
        return {
            "quote": self.quote,
            "explanation": self.explanation,
            "severity": self.severity.value,
        }


@dataclass(frozen=True)
class KnowledgeQuestion:
    id: str
    module_id: str
    module_title: str
    module_order: int
    chapter_id: str
    chapter_title: str
    chapter_order: int
    question_order: int
    prompt: str
    difficulty: KnowledgeDifficulty
    frequency: int
    short_reference_answer: str
    full_reference_answer: str
    rubric_points: List[KnowledgeRubricPoint]
    hint: str = ""
    source_title: str = ""
    source_url: str = ""
    source_chapter: str = ""
    source_page_start: Optional[int] = None
    source_page_end: Optional[int] = None
    enabled: bool = True
    source_file: str = ""
    source_heading: str = ""
    content_hash: str = ""
    keywords: List[str] = field(default_factory=list)

    @property
    def required_points(self) -> List[KnowledgeRubricPoint]:
        return [point for point in self.rubric_points if point.kind == KnowledgeRubricKind.REQUIRED]

    @property
    def bonus_points(self) -> List[KnowledgeRubricPoint]:
        return [point for point in self.rubric_points if point.kind == KnowledgeRubricKind.BONUS]

    @property
    def misconception_points(self) -> List[KnowledgeRubricPoint]:
        return [
            point for point in self.rubric_points if point.kind == KnowledgeRubricKind.MISCONCEPTION
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "module_id": self.module_id,
            "module_title": self.module_title,
            "module_order": self.module_order,
            "chapter_id": self.chapter_id,
            "chapter_title": self.chapter_title,
            "chapter_order": self.chapter_order,
            "question_order": self.question_order,
            "prompt": self.prompt,
            "difficulty": self.difficulty.value,
            "frequency": self.frequency,
            "short_reference_answer": self.short_reference_answer,
            "full_reference_answer": self.full_reference_answer,
            "rubric_points": [point.to_dict() for point in self.rubric_points],
            "hint": self.hint,
            "source_title": self.source_title,
            "source_url": self.source_url,
            "source_chapter": self.source_chapter,
            "source_page_start": self.source_page_start,
            "source_page_end": self.source_page_end,
            "enabled": self.enabled,
            "source_file": self.source_file,
            "source_heading": self.source_heading,
            "content_hash": self.content_hash,
            "keywords": list(self.keywords),
        }


@dataclass
class KnowledgeAssignment:
    id: str
    owner_id: str
    question_id: str
    assigned_on: date
    assignment_type: KnowledgeAssignmentType
    recommendation_reason: str
    status: KnowledgeAssignmentStatus = KnowledgeAssignmentStatus.PENDING
    attempt_id: Optional[str] = None
    completed_at: Optional[datetime] = None
    active_attempt_id: Optional[str] = None
    active_attempt_started_at: Optional[datetime] = None


@dataclass
class KnowledgeAttempt:
    id: str
    owner_id: str
    question_id: str
    assignment_id: Optional[str]
    answer_text: str
    answer_source: KnowledgeAnswerSource
    submitted_at: datetime
    submission_id: Optional[str] = None
    evaluation_status: str = "pending"
    evaluation_payload: Dict[str, Any] = field(default_factory=dict)
    score: Optional[int] = None


@dataclass
class KnowledgeProgress:
    owner_id: str
    question_id: str
    mastery_status: KnowledgeMasteryStatus = KnowledgeMasteryStatus.UNSEEN
    mastery_score: int = 0
    attempt_count: int = 0
    high_score_streak: int = 0
    last_score: Optional[int] = None
    last_attempt_at: Optional[datetime] = None
    next_review_on: Optional[date] = None
    last_detected_gaps: List[str] = field(default_factory=list)


@dataclass
class KnowledgeSubscription:
    owner_id: str
    feishu_open_id: str
    enabled: bool = True
    timezone: str = "Asia/Shanghai"
    morning_time: str = "08:00"
    noon_time: str = "12:00"
    evening_time: str = "18:00"
