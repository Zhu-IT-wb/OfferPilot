import asyncio
import json
import os
import sqlite3
import sys
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.core.config import Settings
from app.models.interview_knowledge import (
    KnowledgeAnswerSource,
    KnowledgeAssignmentType,
    KnowledgeMasteryStatus,
    KnowledgeProgress,
)
from app.repositories.interview_knowledge_repository import (
    InMemoryInterviewKnowledgeRepository,
)
from app.repositories.sqlite_interview_knowledge_repository import (
    SQLiteInterviewKnowledgeRepository,
)
from app.services.knowledge_catalog import load_knowledge_catalog


def _database_snapshot(database: Path) -> tuple[str, ...]:
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as connection:
        return tuple(connection.iterdump())


@pytest.fixture
def stored_corpus(tmp_path):
    database = tmp_path / "knowledge corpus.db"
    template = load_knowledge_catalog().questions[0]
    questions = [
        replace(
            template,
            id=f"custom_question_{index:04d}",
            question_order=index,
            content_hash=f"custom-hash-{index}",
            enabled=index < 814,
        )
        for index in range(815)
    ]
    repository = SQLiteInterviewKnowledgeRepository(str(database), questions=questions)
    assignment = repository.create_assignment(
        owner_id="feishu:test_owner",
        question_id=questions[0].id,
        assigned_on=date(2026, 9, 7),
        assignment_type=KnowledgeAssignmentType.NEW,
        recommendation_reason="Test corpus",
    )
    attempt = repository.create_attempt(
        owner_id=assignment.owner_id,
        question_id=assignment.question_id,
        assignment_id=assignment.id,
        answer_text="Persisted answer",
        answer_source=KnowledgeAnswerSource.TEXT,
        submitted_at=datetime(2026, 9, 7, 10),
        submission_id="persisted-submission",
    )
    attempt.score = 88
    attempt.evaluation_status = "completed"
    attempt.evaluation_payload = {"score": 88}
    repository.save_attempt(attempt)
    repository.save_progress(
        KnowledgeProgress(
            owner_id=assignment.owner_id,
            question_id=assignment.question_id,
            mastery_status=KnowledgeMasteryStatus.REVIEWING,
            mastery_score=88,
            attempt_count=1,
            last_score=88,
            last_attempt_at=attempt.submitted_at,
            next_review_on=date(2026, 9, 10),
            last_detected_gaps=["Persisted gap"],
        )
    )
    return database


def _settings(tmp_path, database=None, storage_backend="sqlite") -> Settings:
    return Settings(
        storage_backend=storage_backend,
        sqlite_path=str(database or tmp_path / "missing" / "knowledge.db"),
        knowledge_source_path=str(tmp_path / "missing-markdown"),
        knowledge_markdown_sync_enabled=False,
        rag_qdrant_path=str(tmp_path / "qdrant"),
        rag_fastembed_cache_path=str(tmp_path / "fastembed"),
    )


def _main_repository_builder(monkeypatch, tmp_path):
    import app.core.config as config

    monkeypatch.setattr(config, "settings", _settings(tmp_path, storage_backend="memory"))
    from app.services.knowledge_dependencies import build_default_knowledge_repository

    return build_default_knowledge_repository


def test_reopening_existing_corpus_never_replaces_it_with_seed_questions(
    stored_corpus, tmp_path, monkeypatch
):
    before = _database_snapshot(stored_corpus)
    build_repository = _main_repository_builder(monkeypatch, tmp_path)

    for _ in range(2):
        direct = SQLiteInterviewKnowledgeRepository(str(stored_corpus))
        configured = build_repository(_settings(tmp_path, stored_corpus))
        assert len(direct.list_questions()) == 814
        assert len(configured.list_questions()) == 814
        assert _database_snapshot(stored_corpus) == before


def test_disabled_only_corpus_is_not_treated_as_an_empty_database(tmp_path):
    database = tmp_path / "disabled.db"
    question = replace(load_knowledge_catalog().questions[0], enabled=False)
    SQLiteInterviewKnowledgeRepository(str(database), questions=[question])
    before = _database_snapshot(database)

    reopened = SQLiteInterviewKnowledgeRepository(str(database))

    assert reopened.list_questions() == []
    assert _database_snapshot(database) == before


def test_main_repository_seeds_empty_storage_and_memory_fallback(tmp_path, monkeypatch):
    build_repository = _main_repository_builder(monkeypatch, tmp_path)
    database = tmp_path / "empty.db"
    SQLiteInterviewKnowledgeRepository(str(database), initialize_questions=False)
    expected = [question.id for question in load_knowledge_catalog().questions]

    for backend in ("sqlite", "memory"):
        repository = build_repository(_settings(tmp_path, database, backend))
        assert [question.id for question in repository.list_questions()] == expected


def test_read_only_repository_rejects_catalog_and_progress_writes(stored_corpus):
    before = _database_snapshot(stored_corpus)
    repository = SQLiteInterviewKnowledgeRepository(str(stored_corpus), read_only=True)

    assert len(repository.list_questions()) == 814
    progress = repository.get_progress("feishu:test_owner", "custom_question_0000")
    assert progress is not None
    assert progress.last_score == 88
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        repository.upsert_questions(load_knowledge_catalog().questions)
    progress.last_score = 0
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        repository.save_progress(progress)

    assert _database_snapshot(stored_corpus) == before


def test_read_only_repository_does_not_create_missing_database(tmp_path):
    database = tmp_path / "missing-parent" / "knowledge.db"

    with pytest.raises(sqlite3.OperationalError, match="unable to open"):
        SQLiteInterviewKnowledgeRepository(str(database), read_only=True)

    assert not database.parent.exists()


def test_read_only_repository_does_not_initialize_or_seed_empty_database(tmp_path):
    database = tmp_path / "uninitialized.db"
    sqlite3.connect(database).close()
    before = _database_snapshot(database)

    repository = SQLiteInterviewKnowledgeRepository(str(database), read_only=True)
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        repository.list_questions()

    assert _database_snapshot(database) == before


def test_mcp_memory_repository_uses_catalog_without_creating_sqlite(tmp_path, monkeypatch):
    from app.mcp import career_knowledge_server as server

    app_settings = _settings(tmp_path, storage_backend="memory")
    monkeypatch.setattr(server, "settings", app_settings)
    monkeypatch.setattr(server, "_knowledge_repository", None)

    repository = server.get_knowledge_repository()

    assert isinstance(repository, InMemoryInterviewKnowledgeRepository)
    assert len(repository.list_questions()) == len(load_knowledge_catalog().questions)
    assert server.get_knowledge_repository() is repository
    assert not Path(app_settings.sqlite_path).exists()


def test_mcp_missing_database_fails_before_creating_storage(tmp_path, monkeypatch):
    from app.mcp import career_knowledge_server as server

    app_settings = _settings(tmp_path)
    monkeypatch.setattr(server, "settings", app_settings)
    monkeypatch.setattr(server, "_knowledge_repository", None)
    monkeypatch.setattr(server, "_service", None)

    with pytest.raises(sqlite3.OperationalError, match="unable to open"):
        server.get_rag_service()

    assert not Path(app_settings.sqlite_path).parent.exists()
    assert not Path(app_settings.rag_qdrant_path).exists()
    assert not Path(app_settings.rag_fastembed_cache_path).exists()


def test_mcp_startup_discovery_and_reads_preserve_existing_corpus_twice(
    stored_corpus, tmp_path
):
    before = _database_snapshot(stored_corpus)
    environment = {
        "PYTHONUTF8": "1",
        "OFFERPILOT_ENV_FILE": os.devnull,
        "OFFERPILOT_STORAGE_BACKEND": "sqlite",
        "OFFERPILOT_SQLITE_PATH": str(stored_corpus),
        "OFFERPILOT_KNOWLEDGE_SOURCE_PATH": str(tmp_path / "missing-markdown"),
        "OFFERPILOT_KNOWLEDGE_MARKDOWN_SYNC_ENABLED": "false",
        "OFFERPILOT_RAG_QDRANT_PATH": str(tmp_path / "unused-qdrant"),
        "OFFERPILOT_RAG_FASTEMBED_CACHE_PATH": str(tmp_path / "unused-fastembed"),
    }
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp.career_knowledge_server"],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=environment,
    )

    async def exercise():
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                response = await session.list_tools()
                assert {tool.name for tool in response.tools} == {
                    "search_knowledge", "search_project_evidence", "read_evidence"
                }
                resource = await session.read_resource(
                    "offerpilot://knowledge/custom_question_0000"
                )
                assert json.loads(resource.contents[0].text)["id"] == "custom_question_0000"
                disabled = await session.read_resource(
                    "offerpilot://knowledge/custom_question_0814"
                )
                assert json.loads(disabled.contents[0].text) == {"error": "not_found"}

    for _ in range(2):
        asyncio.run(exercise())
        assert _database_snapshot(stored_corpus) == before

    assert not (tmp_path / "unused-qdrant").exists()
    assert not (tmp_path / "unused-fastembed").exists()
