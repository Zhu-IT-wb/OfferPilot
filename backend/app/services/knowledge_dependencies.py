from app.core.config import settings
from app.repositories.interview_knowledge_repository import (
    InMemoryInterviewKnowledgeRepository,
    InterviewKnowledgeRepository,
)
from app.repositories.sqlite_interview_knowledge_repository import (
    SQLiteInterviewKnowledgeRepository,
)
from app.services.knowledge_catalog import load_knowledge_catalog


def build_default_knowledge_repository() -> InterviewKnowledgeRepository:
    if settings.storage_backend.strip().lower() == "sqlite":
        return SQLiteInterviewKnowledgeRepository(settings.sqlite_path)
    return InMemoryInterviewKnowledgeRepository(
        questions=load_knowledge_catalog().questions
    )


_default_knowledge_repository = build_default_knowledge_repository()


def get_default_knowledge_repository() -> InterviewKnowledgeRepository:
    return _default_knowledge_repository
