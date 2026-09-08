import asyncio
import os
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from mcp import ClientSession
from mcp.types import CallToolResult
from qdrant_client import models

from app.mcp import client as mcp_client_module
from app.core.config import Settings
from app.mcp.client import (
    MCPClientError,
    MCPToolError,
    PersistentMCPClient,
    _structured_result,
)
from app.models.interview_knowledge import (
    KnowledgeDifficulty,
    KnowledgeQuestion,
    KnowledgeRubricKind,
    KnowledgeRubricPoint,
)
from app.models.project_training import ProjectEvidence, ProjectProfile
from app.rag.evaluation import RetrievalEvaluationCase, evaluate_retrieval
from app.rag.models import IndexReport, RAGDocument, RetrievalHit
from app.rag.qdrant_store import QdrantHybridStore
from app.rag.service import CareerKnowledgeRAGService
from app.rag.source_loader import (
    INTERVIEW_DOMAIN,
    PROJECT_DOMAIN,
    load_interview_documents,
    load_project_documents,
)
from app.tools.mcp_rag_tools import CareerKnowledgeToolAdapter
from app.tools.tool_names import AgentActionName


class _KnowledgeRepository:
    def __init__(self, questions):
        self.questions = questions

    def list_questions(self):
        return list(self.questions)


class _ProjectRepository:
    def __init__(self, project, evidence):
        self.project = project
        self.evidence = evidence

    def list_projects(self, owner_id):
        return [self.project] if owner_id == self.project.owner_id else []

    def list_project_evidence(self, owner_id, project_id, version):
        if (
            owner_id == self.project.owner_id
            and project_id == self.project.id
            and version == self.project.version
        ):
            return [self.evidence]
        return []


class _FakeStore:
    def __init__(self):
        self.sync_calls = []
        self.search_calls = []

    def sync_scope(self, scope, documents):
        self.sync_calls.append((scope, list(documents)))
        return IndexReport(scope, len(documents), len(documents), 0)

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return []

    def read_evidence(self, evidence_id, owner_id):
        return None

    def close(self):
        return None


def _question():
    return KnowledgeQuestion(
        id="q1",
        module_id="database",
        module_title="Database",
        module_order=1,
        chapter_id="index",
        chapter_title="Index",
        chapter_order=1,
        question_order=1,
        prompt="Why does a B+ tree fit database indexes?",
        difficulty=KnowledgeDifficulty.INTERMEDIATE,
        frequency=5,
        short_reference_answer="It reduces disk IO.",
        full_reference_answer="High fan-out keeps the tree shallow.",
        rubric_points=[
            KnowledgeRubricPoint(
                id="r1",
                kind=KnowledgeRubricKind.REQUIRED,
                label="fan-out",
                description="Explain high fan-out.",
            )
        ],
        content_hash="question-hash",
        keywords=["B+ tree", "index"],
    )


def _project():
    return ProjectProfile(
        id="p1",
        owner_id="feishu:ou_a",
        name="OfferPilot",
        target_role="Agent engineer",
        version=3,
    )


def _evidence():
    return ProjectEvidence(
        id="e1",
        owner_id="feishu:ou_a",
        project_id="p1",
        project_version=3,
        source_field="architecture",
        heading="Conversation memory",
        content="SQLite stores ordered conversation events.",
        topic_tags=["memory", "sqlite"],
        content_hash="evidence-hash",
        order=1,
        source_type="code",
        source_path="app/agents/conversation.py",
        start_line=10,
        end_line=30,
    )


def test_source_loaders_preserve_evidence_and_tenant_metadata():
    knowledge = load_interview_documents(_KnowledgeRepository([_question()]))
    projects = load_project_documents(
        _ProjectRepository(_project(), _evidence()),
        "feishu:ou_a",
    )

    assert knowledge[0].evidence_id == "knowledge:q1"
    assert knowledge[0].visibility == "public"
    assert "High fan-out" in knowledge[0].text
    assert projects[0].evidence_id == "project:e1"
    assert projects[0].visibility == "private"
    assert projects[0].owner_id == "feishu:ou_a"
    assert projects[0].source_path == "app/agents/conversation.py"


def test_service_incrementally_syncs_and_clears_empty_owner_scope():
    store = _FakeStore()
    project_repository = _ProjectRepository(_project(), _evidence())
    service = CareerKnowledgeRAGService(
        knowledge_repository=_KnowledgeRepository([_question()]),
        project_repository=project_repository,
        store=store,
    )

    service.sync_for_owner("feishu:ou_a")
    service.sync_for_owner("feishu:ou_a")
    project_repository.project.owner_id = "feishu:ou_b"
    service.sync_for_owner("feishu:ou_a")

    scopes = [scope for scope, _ in store.sync_calls]
    assert scopes.count("public:interview_knowledge") == 1
    assert scopes.count("owner:feishu:ou_a:project_evidence") == 2
    assert store.sync_calls[-1][1] == []


def test_hybrid_prefetches_apply_actor_filter_to_both_branches():
    captured = {}

    class FakeClient:
        def query_points(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(points=[])

    store = QdrantHybridStore.__new__(QdrantHybridStore)
    store.collection_name = "career_knowledge"
    store.dense_model = "dense-model"
    store.sparse_model = "sparse-model"
    store.client = FakeClient()
    store._lock = threading.RLock()

    store.search(
        query="memory",
        owner_id="feishu:ou_a",
        domains=[PROJECT_DOMAIN],
    )

    assert len(captured["prefetch"]) == 2
    for prefetch in captured["prefetch"]:
        assert isinstance(prefetch.filter, models.Filter)
        owner_conditions = [
            condition
            for condition in prefetch.filter.must
            if isinstance(condition, models.FieldCondition)
            and condition.key == "owner_id"
        ]
        assert owner_conditions[0].match.value == "feishu:ou_a"


def test_access_filter_never_grants_private_domain_without_actor():
    query_filter = QdrantHybridStore._access_filter(
        owner_id="",
        domains=[PROJECT_DOMAIN],
    )

    condition = query_filter.must[0]
    assert isinstance(condition, models.FieldCondition)
    assert condition.match.value == "__access_denied__"


def test_qdrant_point_identity_is_scoped_per_owner_collection_partition():
    evidence_id = "project:e1"

    owner_a_point = QdrantHybridStore._point_id(
        "owner:feishu:ou_a:project_evidence",
        evidence_id,
    )
    owner_b_point = QdrantHybridStore._point_id(
        "owner:feishu:ou_b:project_evidence",
        evidence_id,
    )

    assert owner_a_point != owner_b_point
    assert owner_a_point == QdrantHybridStore._point_id(
        "owner:feishu:ou_a:project_evidence",
        evidence_id,
    )


def test_mcp_adapter_injects_owner_and_returns_traceable_hits():
    class FakeClient:
        def __init__(self):
            self.arguments = None

        async def call_tool(self, name, arguments=None):
            assert name == "search_project_evidence"
            self.arguments = arguments
            return {
                "hits": [
                    {
                        "evidence_id": "project:e1",
                        "title": "Conversation memory",
                        "text": "SQLite stores ordered events.",
                        "source_path": "app/agents/conversation.py",
                        "start_line": 10,
                        "end_line": 30,
                    }
                ]
            }

    client = FakeClient()
    adapter = CareerKnowledgeToolAdapter(client=client)
    result = asyncio.run(
        adapter.search_project_evidence(
            {
                "raw_message": "How is memory persisted?",
                "owner_id": "feishu:ou_a",
            }
        )
    )

    assert client.arguments["owner_id"] == "feishu:ou_a"
    assert result.success is True
    assert result.data["evidence_ids"] == ["project:e1"]
    assert result.data["citations"][0]["evidence_id"] == "project:e1"
    assert "project:e1" not in result.message


def test_retrieval_evaluation_reports_recall_mrr_and_latency():
    cases = [
        RetrievalEvaluationCase("memory", ["project:e1"]),
        RetrievalEvaluationCase("index", ["knowledge:q1"]),
    ]

    def search(case):
        evidence_id = "project:e1" if case.query == "memory" else "knowledge:q1"
        return [
            RetrievalHit(
                evidence_id=evidence_id,
                domain=PROJECT_DOMAIN if case.query == "memory" else INTERVIEW_DOMAIN,
                title="title",
                text="text",
                score=1.0,
            )
        ]

    report = evaluate_retrieval(cases, search)

    assert report["recall_at_k"] == 1.0
    assert report["mrr"] == 1.0
    assert report["p95_latency_ms"] >= 0


def test_persistent_mcp_client_discovers_read_only_tools():
    client = PersistentMCPClient(
        Settings(
            storage_backend="memory",
            rag_qdrant_path=":memory:",
        )
    )

    async def exercise():
        tools = await client.list_tools()
        await client.close()
        return tools

    tools = asyncio.run(exercise())
    names = {tool["name"] for tool in tools}

    assert names == {
        "search_knowledge",
        "search_project_evidence",
        "read_evidence",
    }


@pytest.mark.parametrize(
    ("structured", "text"),
    [([], ""), ("not an object", ""), (None, "[]"), (None, "invalid JSON")],
)
def test_mcp_rejects_invalid_results_instead_of_reporting_empty_hits(structured, text):
    response = SimpleNamespace(
        structuredContent=structured,
        content=[SimpleNamespace(text=text)],
    )
    with pytest.raises(MCPToolError):
        _structured_result(response)


@pytest.mark.parametrize("structured", [True, False])
def test_mcp_preserves_valid_object_results(structured):
    expected = {"hits": [], "evidence_ids": []}
    response = SimpleNamespace(
        structuredContent=expected if structured else None,
        content=[SimpleNamespace(text='{"hits": [], "evidence_ids": []}')],
    )
    assert _structured_result(response) == expected


@pytest.mark.parametrize(
    ("failure", "error_type"),
    [
        ("tool", MCPToolError),
        ("schema", MCPToolError),
        ("validation", MCPToolError),
        ("transport", MCPClientError),
    ],
)
def test_mcp_distinguishes_tool_failure_from_transport_and_keeps_session(
    monkeypatch, failure, error_type,
):
    @asynccontextmanager
    async def fake_stdio(parameters):
        yield None, None

    class FakeSession:
        _validate_tool_result = ClientSession._validate_tool_result

        def __init__(self, *args):
            self.calls = 0
            self._tool_output_schemas = {
                "search_knowledge": {
                    "type": "object",
                    "properties": {"hits": {"type": "array"}},
                    "required": ["hits"],
                }
            }

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            pass

        async def call_tool(self, name, arguments):
            self.calls += 1
            if self.calls == 1:
                if failure == "transport":
                    raise ConnectionError("connection interrupted")
                if failure == "schema":
                    await self._validate_tool_result(
                        name,
                        CallToolResult(content=[], structuredContent={"hits": "invalid"}),
                    )
                if failure == "validation":
                    CallToolResult.model_validate({"content": "invalid"})
                return SimpleNamespace(
                    isError=True,
                    structuredContent=None,
                    content=[SimpleNamespace(text="domains parameter rejected")],
                )
            return SimpleNamespace(
                isError=False,
                structuredContent={"hits": []},
                content=[],
            )

    monkeypatch.setattr(mcp_client_module, "stdio_client", fake_stdio)
    monkeypatch.setattr(mcp_client_module, "ClientSession", FakeSession)
    client = PersistentMCPClient(Settings(storage_backend="memory"))

    async def exercise():
        try:
            with pytest.raises(error_type) as caught:
                await client.call_tool("search_knowledge", {"query": "index"})
            assert type(caught.value) is error_type
            assert await client.call_tool("search_knowledge", {"query": "index"}) == {
                "hits": [],
            }
        finally:
            await client.close()

    asyncio.run(exercise())


def test_mcp_child_uses_explicit_settings_without_loading_main_env(monkeypatch):
    monkeypatch.setenv("OFFERPILOT_ENV_FILE", "private-application.env")
    monkeypatch.setenv("FEISHU_APP_SECRET", "must-not-reach-child")
    environment = PersistentMCPClient(
        Settings(storage_backend="memory", rag_qdrant_path=":memory:")
    )._server_environment()

    assert environment["OFFERPILOT_ENV_FILE"] == os.devnull
    assert environment["OFFERPILOT_STORAGE_BACKEND"] == "memory"
    assert environment["OFFERPILOT_RAG_QDRANT_PATH"] == ":memory:"
    assert "FEISHU_APP_SECRET" not in environment
