from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ProjectDiscoveryStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    NEEDS_INPUT = "needs_input"
    READY = "ready"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProjectDiscoveryStage(str, Enum):
    CLONING = "cloning"
    INVENTORY = "inventory"
    SELECTING = "selecting"
    ANALYZING = "analyzing"
    SYNTHESIZING = "synthesizing"
    VALIDATING = "validating"


ACTIVE_DISCOVERY_STATUSES = {
    ProjectDiscoveryStatus.QUEUED,
    ProjectDiscoveryStatus.RUNNING,
    ProjectDiscoveryStatus.NEEDS_INPUT,
    ProjectDiscoveryStatus.READY,
}


@dataclass(frozen=True)
class ProjectDiscoveryQuestion:
    id: str
    target_field: str
    prompt: str
    required: bool
    answered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "target_field": self.target_field,
            "prompt": self.prompt,
            "required": self.required,
            "answered": self.answered,
        }


@dataclass(frozen=True)
class ProjectDiscoveryEvidence:
    id: str
    owner_id: str
    job_id: str
    source_type: str
    file_path: str
    start_line: Optional[int]
    end_line: Optional[int]
    excerpt: str
    content_hash: str
    topic: str
    target_field: str
    confidence: float
    commit_sha: str = ""
    claim: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_type": self.source_type,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "excerpt": self.excerpt,
            "content_hash": self.content_hash,
            "topic": self.topic,
            "target_field": self.target_field,
            "confidence": self.confidence,
            "commit_sha": self.commit_sha,
            "claim": self.claim,
        }


@dataclass(frozen=True)
class ProjectDiscoveryAnswer:
    id: str
    owner_id: str
    job_id: str
    question_id: str
    submission_id: str
    answer_text: str
    submitted_at: datetime


@dataclass
class ProjectDiscoveryJob:
    id: str
    owner_id: str
    request_id: str
    repository_url: str
    project_id: Optional[str]
    status: ProjectDiscoveryStatus
    stage: Optional[ProjectDiscoveryStage]
    progress: int
    commit_sha: str
    draft: Dict[str, Any]
    questions: List[ProjectDiscoveryQuestion]
    warnings: List[str]
    stats: Dict[str, Any]
    error_code: str
    error_message: str
    retry_count: int
    confirmation_id: str
    confirmed_project_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    lease_expires_at: Optional[datetime] = None
    lease_token: str = ""

    def to_dict(self, evidence: Optional[List[ProjectDiscoveryEvidence]] = None):
        result = {
            "id": self.id,
            "repository_url": self.repository_url,
            "project_id": self.project_id,
            "status": self.status.value,
            "stage": self.stage.value if self.stage else None,
            "progress": self.progress,
            "commit_sha": self.commit_sha,
            "draft": dict(self.draft),
            "questions": [item.to_dict() for item in self.questions],
            "warnings": list(self.warnings),
            "stats": dict(self.stats),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "confirmed_project_id": self.confirmed_project_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if evidence is not None:
            result["evidence"] = [item.to_dict() for item in evidence]
        return result
