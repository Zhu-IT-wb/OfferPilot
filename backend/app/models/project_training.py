from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


PROJECT_TRAINING_THEME_ORDER = (
    "project_overview",
    "ownership",
    "architecture",
    "technical_depth",
    "tradeoff",
    "reliability",
    "metrics",
)
PROJECT_TRAINING_THEMES = frozenset(
    (*PROJECT_TRAINING_THEME_ORDER, "troubleshooting", "communication")
)


class ProjectProfileStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class ProjectTrainingDifficulty(str, Enum):
    BASIC = "basic"
    MEDIUM = "medium"
    ADVANCED = "advanced"


class ProjectTrainingSessionStatus(str, Enum):
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class ProjectTrainingTurnStatus(str, Enum):
    PENDING = "pending"
    EVALUATING = "evaluating"
    COMPLETED = "completed"


class ProjectTrainingAnswerSource(str, Enum):
    TEXT = "text"
    VOICE_TRANSCRIPT = "voice_transcript"


class ProjectTrainingEvaluationStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProjectProfile:
    id: str
    owner_id: str
    name: str
    target_role: str = ""
    background: str = ""
    responsibilities: List[str] = field(default_factory=list)
    tech_stack: List[str] = field(default_factory=list)
    architecture: str = ""
    key_decisions: List[str] = field(default_factory=list)
    technical_challenges: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)
    resume_description: str = ""
    supplemental_text: str = ""
    status: ProjectProfileStatus = ProjectProfileStatus.ACTIVE
    version: int = 1
    content_hash: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    updated_at: datetime = field(default_factory=lambda: datetime.now().astimezone())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "target_role": self.target_role,
            "background": self.background,
            "responsibilities": list(self.responsibilities),
            "tech_stack": list(self.tech_stack),
            "architecture": self.architecture,
            "key_decisions": list(self.key_decisions),
            "technical_challenges": list(self.technical_challenges),
            "metrics": list(self.metrics),
            "outcomes": list(self.outcomes),
            "resume_description": self.resume_description,
            "supplemental_text": self.supplemental_text,
            "status": self.status.value,
            "version": self.version,
            "content_hash": self.content_hash,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "missing_fields": self.missing_fields(),
        }

    def missing_fields(self) -> List[str]:
        missing = []
        if not self.responsibilities:
            missing.append("responsibilities")
        if not self.technical_challenges:
            missing.append("technical_challenges")
        if not self.metrics:
            missing.append("metrics")
        return missing


@dataclass(frozen=True)
class ProjectProfileVersion:
    project: ProjectProfile
    version: int
    content_hash: str
    created_at: datetime


@dataclass(frozen=True)
class ProjectEvidence:
    id: str
    owner_id: str
    project_id: str
    project_version: int
    source_field: str
    heading: str
    content: str
    topic_tags: List[str]
    content_hash: str
    order: int


@dataclass
class ProjectTrainingSession:
    id: str
    owner_id: str
    project_id: str
    project_version: int
    creation_id: str
    target_role: str
    goal: str
    difficulty: ProjectTrainingDifficulty
    max_turns: int
    focus_topics: List[str]
    status: ProjectTrainingSessionStatus
    current_turn_id: Optional[str]
    created_at: datetime
    started_at: datetime
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "project_version": self.project_version,
            "target_role": self.target_role,
            "goal": self.goal,
            "difficulty": self.difficulty.value,
            "max_turns": self.max_turns,
            "focus_topics": list(self.focus_topics),
            "status": self.status.value,
            "current_turn_id": self.current_turn_id,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class ProjectTrainingTurn:
    id: str
    owner_id: str
    session_id: str
    parent_turn_id: Optional[str]
    sequence: int
    theme: str
    question_kind: str
    question_text: str
    generation_reason: str
    source_evidence_ids: List[str]
    hypothetical: bool
    status: ProjectTrainingTurnStatus
    created_at: datetime
    answer_id: Optional[str] = None
    completed_at: Optional[datetime] = None
    active_submission_id: Optional[str] = None
    active_submission_started_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "parent_turn_id": self.parent_turn_id,
            "sequence": self.sequence,
            "theme": self.theme,
            "question_kind": self.question_kind,
            "question_text": self.question_text,
            "generation_reason": self.generation_reason,
            "source_evidence_ids": list(self.source_evidence_ids),
            "hypothetical": self.hypothetical,
            "status": self.status.value,
            "answer_id": self.answer_id,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class ProjectTrainingAnswer:
    id: str
    owner_id: str
    session_id: str
    turn_id: str
    submission_id: str
    answer_text: str
    answer_source: ProjectTrainingAnswerSource
    submitted_at: datetime
    evaluation_status: ProjectTrainingEvaluationStatus
    evaluation_payload: Dict[str, Any] = field(default_factory=dict)
    overall_score: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "submission_id": self.submission_id,
            "answer_text": self.answer_text,
            "answer_source": self.answer_source.value,
            "submitted_at": self.submitted_at.isoformat(),
            "evaluation_status": self.evaluation_status.value,
            "overall_score": self.overall_score,
        }


@dataclass
class ProjectTopicProgress:
    owner_id: str
    project_id: str
    topic: str
    attempt_count: int = 0
    mastery_score: int = 0
    last_score: Optional[int] = None
    last_practiced_at: Optional[datetime] = None
    last_detected_gaps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "topic": self.topic,
            "attempt_count": self.attempt_count,
            "mastery_score": self.mastery_score,
            "last_score": self.last_score,
            "last_practiced_at": (
                self.last_practiced_at.isoformat() if self.last_practiced_at else None
            ),
            "last_detected_gaps": list(self.last_detected_gaps),
        }


@dataclass
class ProjectTrainingSummary:
    session_id: str
    owner_id: str
    project_id: str
    project_version: int
    overall_score: int
    topic_scores: Dict[str, int]
    strengths: List[str]
    improvement_areas: List[str]
    unsupported_claims: List[str]
    contradictions: List[str]
    recommended_topics: List[str]
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "project_id": self.project_id,
            "project_version": self.project_version,
            "overall_score": self.overall_score,
            "topic_scores": dict(self.topic_scores),
            "strengths": list(self.strengths),
            "improvement_areas": list(self.improvement_areas),
            "unsupported_claims": list(self.unsupported_claims),
            "contradictions": list(self.contradictions),
            "recommended_topics": list(self.recommended_topics),
            "created_at": self.created_at.isoformat(),
        }
