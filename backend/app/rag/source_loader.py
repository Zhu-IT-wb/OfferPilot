import hashlib
from typing import Iterable, List

from app.models.interview_knowledge import KnowledgeQuestion
from app.models.project_training import ProjectEvidence, ProjectProfile
from app.repositories.interview_knowledge_repository import InterviewKnowledgeRepository
from app.repositories.project_training_repository import ProjectTrainingRepository
from app.rag.models import RAGDocument


INTERVIEW_DOMAIN = "interview_knowledge"
PROJECT_DOMAIN = "project_evidence"


def load_interview_documents(
    repository: InterviewKnowledgeRepository,
) -> List[RAGDocument]:
    return [_knowledge_document(question) for question in repository.list_questions()]


def load_project_documents(
    repository: ProjectTrainingRepository,
    owner_id: str,
) -> List[RAGDocument]:
    documents: List[RAGDocument] = []
    for project in repository.list_projects(owner_id):
        evidence = repository.list_project_evidence(
            owner_id,
            project.id,
            project.version,
        )
        documents.extend(_project_document(project, item) for item in evidence)
    return documents


def _knowledge_document(question: KnowledgeQuestion) -> RAGDocument:
    rubric = "\n".join(
        f"- {point.kind.value}: {point.label}: {point.description}"
        for point in question.rubric_points
    )
    text = _join_sections(
        (
            ("Question", question.prompt),
            ("Short answer", question.short_reference_answer),
            ("Full answer", question.full_reference_answer),
            ("Rubric", rubric),
            ("Hint", question.hint),
            ("Keywords", ", ".join(question.keywords)),
        )
    )
    content_hash = question.content_hash or _content_hash(text)
    source_path = question.source_file
    return RAGDocument(
        evidence_id=f"knowledge:{question.id}",
        domain=INTERVIEW_DOMAIN,
        scope="public:interview_knowledge",
        title=f"{question.module_title} / {question.chapter_title}: {question.prompt}",
        text=text,
        content_hash=content_hash,
        visibility="public",
        source_type="interview_question",
        source_path=source_path,
        source_url=question.source_url,
        topic_tags=[question.module_id, question.chapter_id, *question.keywords],
        metadata={
            "question_id": question.id,
            "module_id": question.module_id,
            "module_title": question.module_title,
            "chapter_id": question.chapter_id,
            "chapter_title": question.chapter_title,
            "difficulty": question.difficulty.value,
            "frequency": question.frequency,
            "source_heading": question.source_heading,
            "source_title": question.source_title,
            "source_chapter": question.source_chapter,
            "source_page_start": question.source_page_start,
            "source_page_end": question.source_page_end,
        },
    )


def _project_document(
    project: ProjectProfile,
    evidence: ProjectEvidence,
) -> RAGDocument:
    text = _join_sections(
        (
            ("Project", project.name),
            ("Target role", project.target_role),
            ("Evidence heading", evidence.heading),
            ("Grounded claim", evidence.grounded_claim),
            ("Evidence", evidence.content),
            ("Topics", ", ".join(evidence.topic_tags)),
        )
    )
    return RAGDocument(
        evidence_id=f"project:{evidence.id}",
        domain=PROJECT_DOMAIN,
        scope=f"owner:{project.owner_id}:project_evidence",
        title=f"{project.name}: {evidence.heading or evidence.source_field}",
        text=text,
        content_hash=evidence.content_hash or _content_hash(text),
        visibility="private",
        owner_id=project.owner_id,
        project_id=project.id,
        project_version=project.version,
        source_type=evidence.source_type,
        source_path=evidence.source_path,
        start_line=evidence.start_line,
        end_line=evidence.end_line,
        topic_tags=list(evidence.topic_tags),
        metadata={
            "source_field": evidence.source_field,
            "confidence": evidence.confidence,
            "source_commit_sha": evidence.source_commit_sha,
            "source_evidence_id": evidence.source_evidence_id,
            "grounded_claim": evidence.grounded_claim,
        },
    )


def _join_sections(sections: Iterable[tuple[str, str]]) -> str:
    return "\n\n".join(
        f"{title}:\n{value.strip()}"
        for title, value in sections
        if value and value.strip()
    )


def _content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
