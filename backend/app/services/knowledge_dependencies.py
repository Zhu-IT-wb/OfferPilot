import logging
from pathlib import Path

from app.core.config import Settings, settings
from app.repositories.interview_knowledge_repository import (
    InMemoryInterviewKnowledgeRepository,
    InterviewKnowledgeRepository,
)
from app.repositories.sqlite_interview_knowledge_repository import (
    SQLiteInterviewKnowledgeRepository,
)
from app.services.knowledge_catalog import load_knowledge_catalog
from app.services.knowledge_corpus import KnowledgeCorpus


logger = logging.getLogger(__name__)


def build_default_knowledge_repository(
    app_settings: Settings = settings,
    sync_markdown: bool = True,
) -> InterviewKnowledgeRepository:
    seed_questions = load_knowledge_catalog().questions
    source_root = Path(app_settings.knowledge_source_path)
    should_sync_markdown = (
        app_settings.knowledge_markdown_sync_enabled and source_root.is_dir()
    )
    if app_settings.storage_backend.strip().lower() == "sqlite":
        repository: InterviewKnowledgeRepository = SQLiteInterviewKnowledgeRepository(
            app_settings.sqlite_path,
            questions=seed_questions,
            initialize_questions=not should_sync_markdown,
        )
    else:
        repository = InMemoryInterviewKnowledgeRepository(questions=seed_questions)
    if should_sync_markdown and sync_markdown:
        report = KnowledgeCorpus(repository, source_root).sync_markdown()
        if report.imported_questions == 0:
            repository.upsert_questions(seed_questions)
        logger.info(
            "Knowledge Markdown sync completed: files=%s questions=%s new=%s "
            "updated=%s unchanged=%s removed=%s skipped=%s",
            report.discovered_files,
            report.imported_questions,
            report.new_questions,
            report.updated_questions,
            report.unchanged_questions,
            report.removed_questions,
            report.skipped_files,
        )
    return repository


_default_knowledge_repository = build_default_knowledge_repository(
    sync_markdown=False
)


def get_default_knowledge_repository() -> InterviewKnowledgeRepository:
    return _default_knowledge_repository


def sync_default_knowledge_repository(
    app_settings: Settings = settings,
) -> None:
    sync_knowledge_repository(_default_knowledge_repository, app_settings)


def sync_knowledge_repository(
    repository: InterviewKnowledgeRepository,
    app_settings: Settings = settings,
) -> None:
    source_root = Path(app_settings.knowledge_source_path)
    if not app_settings.knowledge_markdown_sync_enabled or not source_root.is_dir():
        return
    report = KnowledgeCorpus(
        repository,
        source_root,
    ).sync_markdown()
    if report.imported_questions == 0:
        repository.upsert_questions(load_knowledge_catalog().questions)
    logger.info(
        "Knowledge Markdown startup sync completed: files=%s questions=%s new=%s "
        "updated=%s unchanged=%s removed=%s skipped=%s",
        report.discovered_files,
        report.imported_questions,
        report.new_questions,
        report.updated_questions,
        report.unchanged_questions,
        report.removed_questions,
        report.skipped_files,
    )
