import threading
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from qdrant_client import QdrantClient, models

from app.rag.models import IndexReport, RAGDocument, RetrievalHit
from app.rag.source_loader import INTERVIEW_DOMAIN, PROJECT_DOMAIN


class QdrantHybridStore:
    DENSE_VECTOR = "dense"
    SPARSE_VECTOR = "sparse"

    def __init__(
        self,
        path: str,
        collection_name: str,
        dense_model: str,
        sparse_model: str,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.path = path
        self.collection_name = collection_name
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        if path != ":memory:":
            Path(path).mkdir(parents=True, exist_ok=True)
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(
            path=path,
            cache_dir=cache_dir,
            local_inference_batch_size=16,
        )
        self._lock = threading.RLock()
        self._ensure_collection()

    def close(self) -> None:
        self.client.close()

    def sync_scope(
        self,
        scope: str,
        documents: Sequence[RAGDocument],
    ) -> IndexReport:
        deduplicated = {document.evidence_id: document for document in documents}
        with self._lock:
            existing = self._scope_points(scope)
            changed = [
                document
                for evidence_id, document in deduplicated.items()
                if existing.get(evidence_id, {}).get("content_hash")
                != document.content_hash
            ]
            stale_ids = [
                payload["point_id"]
                for evidence_id, payload in existing.items()
                if evidence_id not in deduplicated
            ]
            if changed:
                self.client.upload_collection(
                    collection_name=self.collection_name,
                    vectors=[
                        {
                            self.DENSE_VECTOR: models.Document(
                                text=document.text,
                                model=self.dense_model,
                            ),
                            self.SPARSE_VECTOR: models.Document(
                                text=document.text,
                                model=self.sparse_model,
                            ),
                        }
                        for document in changed
                    ],
                    payload=[document.to_payload() for document in changed],
                    ids=[self._point_id(document.evidence_id) for document in changed],
                    wait=True,
                )
            if stale_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=stale_ids),
                    wait=True,
                )
        return IndexReport(
            scope=scope,
            discovered=len(deduplicated),
            upserted=len(changed),
            deleted=len(stale_ids),
        )

    def search(
        self,
        query: str,
        owner_id: str,
        domains: Sequence[str],
        project_id: str = "",
        project_version: Optional[int] = None,
        limit: int = 5,
    ) -> List[RetrievalHit]:
        query_filter = self._access_filter(
            owner_id=owner_id,
            domains=domains,
            project_id=project_id,
            project_version=project_version,
        )
        candidate_limit = max(limit * 4, 20)
        with self._lock:
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=models.Document(text=query, model=self.dense_model),
                        using=self.DENSE_VECTOR,
                        filter=query_filter,
                        limit=candidate_limit,
                    ),
                    models.Prefetch(
                        query=models.Document(text=query, model=self.sparse_model),
                        using=self.SPARSE_VECTOR,
                        filter=query_filter,
                        limit=candidate_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        return [self._hit(point.payload or {}, point.score) for point in response.points]

    def read_evidence(
        self,
        evidence_id: str,
        owner_id: str,
    ) -> Optional[RetrievalHit]:
        access = self._access_filter(
            owner_id=owner_id,
            domains=[INTERVIEW_DOMAIN, PROJECT_DOMAIN],
        )
        query_filter = models.Filter(
            must=[
                access,
                models.FieldCondition(
                    key="evidence_id",
                    match=models.MatchValue(value=evidence_id),
                ),
            ]
        )
        with self._lock:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=1,
                with_payload=True,
            )
        if not points:
            return None
        return self._hit(points[0].payload or {}, 1.0)

    def _ensure_collection(self) -> None:
        with self._lock:
            if self.client.collection_exists(self.collection_name):
                return
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.DENSE_VECTOR: models.VectorParams(
                        size=self.client.get_embedding_size(self.dense_model),
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    self.SPARSE_VECTOR: models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                },
            )

    def _scope_points(self, scope: str) -> Dict[str, Dict[str, object]]:
        result: Dict[str, Dict[str, object]] = {}
        offset = None
        scope_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="scope",
                    match=models.MatchValue(value=scope),
                )
            ]
        )
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scope_filter,
                limit=256,
                offset=offset,
                with_payload=["evidence_id", "content_hash"],
            )
            for point in points:
                payload = dict(point.payload or {})
                evidence_id = str(payload.get("evidence_id") or "")
                if evidence_id:
                    result[evidence_id] = {
                        "point_id": point.id,
                        "content_hash": payload.get("content_hash"),
                    }
            if offset is None:
                break
        return result

    @staticmethod
    def _access_filter(
        owner_id: str,
        domains: Sequence[str],
        project_id: str = "",
        project_version: Optional[int] = None,
    ) -> models.Filter:
        allowed: List[models.Filter] = []
        if INTERVIEW_DOMAIN in domains:
            allowed.append(
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="domain",
                            match=models.MatchValue(value=INTERVIEW_DOMAIN),
                        ),
                        models.FieldCondition(
                            key="visibility",
                            match=models.MatchValue(value="public"),
                        ),
                    ]
                )
            )
        if PROJECT_DOMAIN in domains and owner_id:
            conditions: List[models.Condition] = [
                models.FieldCondition(
                    key="domain",
                    match=models.MatchValue(value=PROJECT_DOMAIN),
                ),
                models.FieldCondition(
                    key="visibility",
                    match=models.MatchValue(value="private"),
                ),
                models.FieldCondition(
                    key="owner_id",
                    match=models.MatchValue(value=owner_id),
                ),
            ]
            if project_id:
                conditions.append(
                    models.FieldCondition(
                        key="project_id",
                        match=models.MatchValue(value=project_id),
                    )
                )
            if project_version is not None:
                conditions.append(
                    models.FieldCondition(
                        key="project_version",
                        match=models.MatchValue(value=project_version),
                    )
                )
            allowed.append(models.Filter(must=conditions))
        if not allowed:
            return models.Filter(
                must=[
                    models.FieldCondition(
                        key="domain",
                        match=models.MatchValue(value="__access_denied__"),
                    )
                ]
            )
        if len(allowed) == 1:
            return allowed[0]
        return models.Filter(should=allowed)

    @staticmethod
    def _point_id(evidence_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"offerpilot:{evidence_id}"))

    @staticmethod
    def _hit(payload: Dict[str, object], score: float) -> RetrievalHit:
        return RetrievalHit(
            evidence_id=str(payload.get("evidence_id") or ""),
            domain=str(payload.get("domain") or ""),
            title=str(payload.get("title") or ""),
            text=str(payload.get("text") or ""),
            score=float(score),
            source_type=str(payload.get("source_type") or ""),
            source_path=str(payload.get("source_path") or ""),
            source_url=str(payload.get("source_url") or ""),
            start_line=_optional_int(payload.get("start_line")),
            end_line=_optional_int(payload.get("end_line")),
            project_id=str(payload.get("project_id") or ""),
            project_version=_optional_int(payload.get("project_version")),
            topic_tags=[str(value) for value in payload.get("topic_tags") or []],
            metadata=dict(payload.get("metadata") or {}),
        )


def _optional_int(value: object) -> Optional[int]:
    return int(value) if isinstance(value, (int, float)) else None
