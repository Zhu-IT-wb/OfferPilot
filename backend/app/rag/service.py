import threading
from typing import Dict, List, Optional, Sequence, Tuple

from app.rag.models import IndexReport, RAGDocument, RetrievalHit
from app.rag.qdrant_store import QdrantHybridStore
from app.rag.source_loader import (
    INTERVIEW_DOMAIN,
    PROJECT_DOMAIN,
    load_interview_documents,
    load_project_documents,
)
from app.repositories.interview_knowledge_repository import InterviewKnowledgeRepository
from app.repositories.project_training_repository import ProjectTrainingRepository


class CareerKnowledgeRAGService:
    def __init__(
        self,
        knowledge_repository: InterviewKnowledgeRepository,
        project_repository: ProjectTrainingRepository,
        store: QdrantHybridStore,
    ) -> None:
        self.knowledge_repository = knowledge_repository
        self.project_repository = project_repository
        self.store = store
        self._signatures: Dict[str, Tuple[Tuple[str, str], ...]] = {}
        self._sync_lock = threading.RLock()

    def close(self) -> None:
        self.store.close()

    def sync_for_owner(self, owner_id: str) -> List[IndexReport]:
        reports = [
            self._sync_documents(
                "public:interview_knowledge",
                load_interview_documents(self.knowledge_repository),
            )
        ]
        if owner_id:
            reports.append(
                self._sync_documents(
                    f"owner:{owner_id}:project_evidence",
                    load_project_documents(self.project_repository, owner_id),
                )
            )
        return reports

    def search(
        self,
        query: str,
        owner_id: str,
        domains: Optional[Sequence[str]] = None,
        project_id: str = "",
        project_version: Optional[int] = None,
        top_k: int = 5,
    ) -> List[RetrievalHit]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        selected_domains = _normalize_domains(domains)
        self.sync_for_owner(owner_id)
        return self.store.search(
            query=normalized_query,
            owner_id=owner_id,
            domains=selected_domains,
            project_id=project_id.strip(),
            project_version=project_version,
            limit=max(1, min(int(top_k), 10)),
        )

    def search_project_evidence(
        self,
        query: str,
        owner_id: str,
        project_id: str = "",
        project_version: Optional[int] = None,
        top_k: int = 5,
    ) -> List[RetrievalHit]:
        return self.search(
            query=query,
            owner_id=owner_id,
            domains=[PROJECT_DOMAIN],
            project_id=project_id,
            project_version=project_version,
            top_k=top_k,
        )

    def read_evidence(
        self,
        evidence_id: str,
        owner_id: str,
    ) -> Optional[RetrievalHit]:
        self.sync_for_owner(owner_id)
        return self.store.read_evidence(evidence_id.strip(), owner_id)

    def _sync_documents(
        self,
        scope: str,
        documents: List[RAGDocument],
    ) -> IndexReport:
        signature = tuple(
            sorted((document.evidence_id, document.content_hash) for document in documents)
        )
        with self._sync_lock:
            if self._signatures.get(scope) == signature:
                return IndexReport(
                    scope=scope,
                    discovered=len(documents),
                    upserted=0,
                    deleted=0,
                )
            report = self.store.sync_scope(scope, documents)
            self._signatures[scope] = signature
            return report


def _normalize_domains(domains: Optional[Sequence[str]]) -> List[str]:
    if not domains:
        return [INTERVIEW_DOMAIN, PROJECT_DOMAIN]
    allowed = {INTERVIEW_DOMAIN, PROJECT_DOMAIN}
    selected = []
    for domain in domains:
        normalized = str(domain).strip().lower()
        if normalized in allowed and normalized not in selected:
            selected.append(normalized)
    if not selected:
        raise ValueError("domains must contain interview_knowledge or project_evidence")
    return selected
