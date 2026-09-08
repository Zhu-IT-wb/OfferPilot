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


_PROJECT_ANALYSIS_TOOL_LABELS = {
    "list_files": "查看项目文件",
    "read_file": "读取项目文件",
    "search_code": "检索项目代码",
    "run_repository_command": "执行项目只读分析",
    "submit_analysis": "提交项目分析结果",
}
_PUBLIC_ANALYSIS_FAILURE_MESSAGE = "项目分析未能生成有效结果，请重新分析。"


def _public_error_message(message: str) -> str:
    if any(tool_name in message for tool_name in _PROJECT_ANALYSIS_TOOL_LABELS):
        return _PUBLIC_ANALYSIS_FAILURE_MESSAGE
    return message


def _public_warning_message(message: str) -> str:
    public_message = message
    for tool_name, label in _PROJECT_ANALYSIS_TOOL_LABELS.items():
        public_message = public_message.replace(tool_name, label)
    return public_message


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
        public_stats = {
            key: value
            for key, value in self.stats.items()
            if key != "runtime_trace"
        }
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
            "warnings": [_public_warning_message(item) for item in self.warnings],
            "stats": public_stats,
            "error_code": "analysis_failed" if self.error_code else "",
            "error_message": _public_error_message(self.error_message),
            "retry_count": self.retry_count,
            "confirmed_project_id": self.confirmed_project_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if evidence is not None:
            result["evidence"] = [item.to_dict() for item in evidence]
        return result
