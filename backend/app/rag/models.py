from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RAGDocument:
    evidence_id: str
    domain: str
    scope: str
    title: str
    text: str
    content_hash: str
    visibility: str = "public"
    owner_id: str = ""
    project_id: str = ""
    project_version: Optional[int] = None
    source_type: str = ""
    source_path: str = ""
    source_url: str = ""
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    topic_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "domain": self.domain,
            "scope": self.scope,
            "title": self.title,
            "text": self.text,
            "content_hash": self.content_hash,
            "visibility": self.visibility,
            "owner_id": self.owner_id,
            "project_id": self.project_id,
            "project_version": self.project_version,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "source_url": self.source_url,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "topic_tags": list(self.topic_tags),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RetrievalHit:
    evidence_id: str
    domain: str
    title: str
    text: str
    score: float
    source_type: str = ""
    source_path: str = ""
    source_url: str = ""
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    project_id: str = ""
    project_version: Optional[int] = None
    topic_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "domain": self.domain,
            "title": self.title,
            "text": self.text,
            "score": round(self.score, 6),
            "source_type": self.source_type,
            "source_path": self.source_path,
            "source_url": self.source_url,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "project_id": self.project_id,
            "project_version": self.project_version,
            "topic_tags": list(self.topic_tags),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class IndexReport:
    scope: str
    discovered: int
    upserted: int
    deleted: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "discovered": self.discovered,
            "upserted": self.upserted,
            "deleted": self.deleted,
        }
