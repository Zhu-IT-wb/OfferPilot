from datetime import date, datetime

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


def test_catalog_exposes_ordered_questions_with_scoring_rubrics() -> None:
    catalog = load_knowledge_catalog()

    assert len(catalog.questions) >= 12
    assert catalog.modules[:4] == ["network", "operating_system", "mysql", "redis"]
    first = catalog.questions[0]
    assert first.id == "knowledge_network_http_001"
    assert first.prompt == "HTTP/1.1 相比 HTTP/1.0 有哪些主要改进？"
    assert first.required_points
    assert first.short_reference_answer
    assert first.source_url.startswith("https://")


def test_in_memory_repository_preserves_owner_scoped_practice_state() -> None:
    repository = InMemoryInterviewKnowledgeRepository(
        questions=load_knowledge_catalog().questions
    )
    today = date(2026, 7, 27)

    assignment = repository.create_assignment(
        owner_id="feishu:ou_owner",
        question_id="knowledge_network_http_001",
        assigned_on=today,
        assignment_type=KnowledgeAssignmentType.NEW,
        recommendation_reason="当前章节新题",
    )
    duplicate = repository.create_assignment(
        owner_id="feishu:ou_owner",
        question_id="knowledge_network_http_001",
        assigned_on=today,
        assignment_type=KnowledgeAssignmentType.WEAKNESS,
        recommendation_reason="不应覆盖",
    )
    attempt = repository.create_attempt(
        owner_id="feishu:ou_owner",
        question_id=assignment.question_id,
        assignment_id=assignment.id,
        answer_text="默认使用持久连接，并增加 Host 请求头。",
        answer_source=KnowledgeAnswerSource.TEXT,
        submitted_at=datetime(2026, 7, 27, 9, 0),
    )
    progress = KnowledgeProgress(
        owner_id="feishu:ou_owner",
        question_id=assignment.question_id,
        mastery_status=KnowledgeMasteryStatus.REVIEWING,
        mastery_score=72,
        attempt_count=1,
        last_score=72,
        last_attempt_at=attempt.submitted_at,
        next_review_on=date(2026, 7, 30),
        last_detected_gaps=["分块传输"],
    )
    repository.save_progress(progress)

    assert duplicate.id == assignment.id
    assert duplicate.assignment_type == KnowledgeAssignmentType.NEW
    assert repository.get_attempt("feishu:ou_owner", attempt.id) == attempt
    assert repository.get_attempt("feishu:ou_other", attempt.id) is None
    assert repository.get_progress("feishu:ou_owner", assignment.question_id) == progress


def test_sqlite_repository_restores_assignments_attempts_and_progress(tmp_path) -> None:
    database = tmp_path / "offerpilot.db"
    today = date(2026, 7, 27)
    repository = SQLiteInterviewKnowledgeRepository(str(database))
    assignment = repository.create_assignment(
        owner_id="feishu:ou_owner",
        question_id="knowledge_network_http_001",
        assigned_on=today,
        assignment_type=KnowledgeAssignmentType.NEW,
        recommendation_reason="当前章节新题",
    )
    attempt = repository.create_attempt(
        owner_id="feishu:ou_owner",
        question_id=assignment.question_id,
        assignment_id=assignment.id,
        answer_text="持久连接、Host、缓存控制和分块传输。",
        answer_source=KnowledgeAnswerSource.TEXT,
        submitted_at=datetime(2026, 7, 27, 9, 0),
    )
    attempt.score = 92
    attempt.evaluation_status = "completed"
    attempt.evaluation_payload = {"matched_required_point_ids": ["http11_required_1"]}
    repository.save_attempt(attempt)
    repository.save_progress(
        KnowledgeProgress(
            owner_id="feishu:ou_owner",
            question_id=assignment.question_id,
            mastery_status=KnowledgeMasteryStatus.REVIEWING,
            mastery_score=92,
            attempt_count=1,
            high_score_streak=1,
            last_score=92,
            last_attempt_at=attempt.submitted_at,
            next_review_on=date(2026, 8, 10),
        )
    )

    reopened = SQLiteInterviewKnowledgeRepository(str(database))

    restored_assignment = reopened.list_assignments("feishu:ou_owner", today)[0]
    restored_attempt = reopened.get_attempt("feishu:ou_owner", attempt.id)
    restored_progress = reopened.get_progress("feishu:ou_owner", assignment.question_id)
    assert restored_assignment.id == assignment.id
    assert restored_attempt is not None
    assert restored_attempt.score == 92
    assert restored_attempt.evaluation_payload == {
        "matched_required_point_ids": ["http11_required_1"]
    }
    assert restored_progress is not None
    assert restored_progress.next_review_on == date(2026, 8, 10)
    assert reopened.get_attempt("feishu:ou_other", attempt.id) is None


def test_sqlite_catalog_sync_disables_questions_removed_from_source(tmp_path) -> None:
    questions = load_knowledge_catalog().questions[:2]
    repository = SQLiteInterviewKnowledgeRepository(
        str(tmp_path / "offerpilot.db"), questions=questions
    )

    repository.upsert_questions([questions[0]])

    assert [item.id for item in repository.list_questions()] == [questions[0].id]
