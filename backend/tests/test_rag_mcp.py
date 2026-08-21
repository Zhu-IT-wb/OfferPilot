import asyncio
import json
import threading
from types import SimpleNamespace

from qdrant_client import models

from app.core.config import Settings
from app.agents.conversation import InMemoryConversationStore
from app.agents.orchestrator import AgentOrchestrator
from app.agents.planner import AgentPlanner
from app.mcp.client import PersistentMCPClient
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
from app.schemas.agent import AgentActionName
from app.schemas.tool import ToolResult
from app.services.llm_service import LLMResult
from app.tools.mcp_rag_tools import CareerKnowledgeToolAdapter
from app.tools.registry import ToolRegistry


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

    class NoSynthesis:
        async def compose(self, query, hits):
            return None

    client = FakeClient()
    adapter = CareerKnowledgeToolAdapter(client=client, answer_composer=NoSynthesis())
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
    assert "[project:e1]" in result.message


def test_agent_planner_executes_async_grounded_retrieval_tool():
    class FakeLLMService:
        async def generate_text(self, **kwargs):
            return LLMResult(
                provider="fake",
                model="fake",
                content=json.dumps(
                    {
                        "intent": "ask_help",
                        "confidence": 0.93,
                        "action": "search_project_evidence",
                        "reply": "Searching project evidence.",
                        "need_confirmation": False,
                        "slots": {
                            "query": "How is memory persisted?",
                            "owner_id": "feishu:ou_attacker_selected",
                        },
                        "missing_slots": [],
                        "steps": [],
                        "reason": "project implementation question",
                    }
                ),
                raw_response={},
            )

    async def search(arguments):
        assert arguments["owner_id"] == "local_user"
        return ToolResult(
            tool_name="search_project_evidence",
            success=True,
            message="SQLite events [project:e1]",
            data={"evidence_ids": ["project:e1"], "grounded": True},
        )

    registry = ToolRegistry()
    registry.register(
        "search_project_evidence",
        search,
        optional_slots=["query"],
    )
    orchestrator = AgentOrchestrator(
        planner=AgentPlanner(
            llm_service=FakeLLMService(),
            llm_planner_enabled=True,
        ),
        tool_registry=registry,
        conversation_store=InMemoryConversationStore(),
    )

    response = asyncio.run(
        orchestrator.handle_message(
            "How is memory persisted?",
            user_id="local_user",
            source="api",
        )
    )

    assert response.action == AgentActionName.SEARCH_PROJECT_EVIDENCE
    assert response.tool_result is not None
    assert response.tool_result.data["evidence_ids"] == ["project:e1"]


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
