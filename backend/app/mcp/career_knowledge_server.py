import atexit
import json
import threading
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from app.core.config import settings
from app.rag.qdrant_store import QdrantHybridStore
from app.rag.service import CareerKnowledgeRAGService
from app.repositories.interview_knowledge_repository import (
    InMemoryInterviewKnowledgeRepository,
    InterviewKnowledgeRepository,
)
from app.repositories.sqlite_interview_knowledge_repository import (
    SQLiteInterviewKnowledgeRepository,
)
from app.services.knowledge_catalog import load_knowledge_catalog


mcp = FastMCP(
    "OfferPilot Career Knowledge",
    instructions=(
        "Read-only retrieval over public interview knowledge and actor-scoped "
        "project evidence. Preserve evidence_id values in downstream answers."
    ),
)
_service: Optional[CareerKnowledgeRAGService] = None
_service_lock = threading.Lock()
_knowledge_repository: Optional[InterviewKnowledgeRepository] = None
_knowledge_repository_lock = threading.Lock()


class SearchResponse(BaseModel):
    query: str
    hits: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)


class EvidenceResponse(BaseModel):
    evidence: Optional[Dict[str, Any]] = None


def get_knowledge_repository() -> InterviewKnowledgeRepository:
    global _knowledge_repository
    if _knowledge_repository is not None:
        return _knowledge_repository
    with _knowledge_repository_lock:
        if _knowledge_repository is None:
            if settings.storage_backend.strip().lower() == "sqlite":
                _knowledge_repository = SQLiteInterviewKnowledgeRepository(
                    settings.sqlite_path, read_only=True
                )
            else:
                _knowledge_repository = InMemoryInterviewKnowledgeRepository(
                    questions=load_knowledge_catalog().questions
                )
    return _knowledge_repository


def get_rag_service() -> CareerKnowledgeRAGService:
    global _service
    if _service is not None:
        return _service
    with _service_lock:
        if _service is None:
            knowledge_repository = get_knowledge_repository()
            from app.services.project_training_dependencies import (
                get_default_project_training_repository,
            )

            store = QdrantHybridStore(
                path=settings.rag_qdrant_path,
                collection_name=settings.rag_collection_name,
                dense_model=settings.rag_dense_model,
                sparse_model=settings.rag_sparse_model,
                cache_dir=settings.rag_fastembed_cache_path,
            )
            _service = CareerKnowledgeRAGService(
                knowledge_repository=knowledge_repository,
                project_repository=get_default_project_training_repository(),
                store=store,
            )
    return _service


@mcp.tool(
    name="search_knowledge",
    description=(
        "Hybrid-search public interview knowledge and the current actor's private "
        "project evidence. Returns source-grounded evidence with stable IDs."
    ),
    structured_output=True,
)
def search_knowledge(
    query: str,
    owner_id: str,
    domains: Optional[List[str]] = None,
    project_id: str = "",
    top_k: int = 5,
) -> SearchResponse:
    _require_owner(owner_id)
    hits = get_rag_service().search(
        query=query,
        owner_id=owner_id,
        domains=domains,
        project_id=project_id,
        top_k=top_k,
    )
    return SearchResponse(
        query=query,
        hits=[hit.to_dict() for hit in hits],
        evidence_ids=[hit.evidence_id for hit in hits],
    )


@mcp.tool(
    name="search_project_evidence",
    description=(
        "Hybrid-search only the current actor's project profile and source-code "
        "evidence. Optionally restrict to a project and version."
    ),
    structured_output=True,
)
def search_project_evidence(
    query: str,
    owner_id: str,
    project_id: str = "",
    project_version: Optional[int] = None,
    top_k: int = 5,
) -> SearchResponse:
    _require_owner(owner_id)
    hits = get_rag_service().search_project_evidence(
        query=query,
        owner_id=owner_id,
        project_id=project_id,
        project_version=project_version,
        top_k=top_k,
    )
    return SearchResponse(
        query=query,
        hits=[hit.to_dict() for hit in hits],
        evidence_ids=[hit.evidence_id for hit in hits],
    )


@mcp.tool(
    name="read_evidence",
    description=(
        "Read one evidence item by its stable ID. Private project evidence is "
        "returned only when it belongs to the current actor."
    ),
    structured_output=True,
)
def read_evidence(evidence_id: str, owner_id: str) -> EvidenceResponse:
    _require_owner(owner_id)
    hit = get_rag_service().read_evidence(evidence_id=evidence_id, owner_id=owner_id)
    return EvidenceResponse(evidence=hit.to_dict() if hit is not None else None)


@mcp.resource(
    "offerpilot://knowledge/{question_id}",
    name="Interview knowledge question",
    description="Read one public interview-knowledge item by question ID.",
    mime_type="application/json",
)
def read_knowledge_resource(question_id: str) -> str:
    question = get_knowledge_repository().get_question(question_id)
    return json.dumps(
        question.to_dict() if question is not None else {"error": "not_found"},
        ensure_ascii=False,
    )


def _require_owner(owner_id: str) -> None:
    if not owner_id or not owner_id.strip():
        raise ValueError("owner_id is required for actor-scoped retrieval")


def _close_service() -> None:
    global _service
    if _service is not None:
        _service.close()
        _service = None


atexit.register(_close_service)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
