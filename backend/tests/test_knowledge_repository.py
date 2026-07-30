import sqlite3
from datetime import date, datetime, timedelta, timezone

from app.models.interview_knowledge import (
    KnowledgeAnswerSource,
    KnowledgeAssignmentStatus,
    KnowledgeAssignmentType,
    KnowledgeMasteryStatus,
    KnowledgeProgress,
)
from app.repositories.interview_knowledge_repository import (
    ConcurrentKnowledgeProgressUpdateError,
    InMemoryInterviewKnowledgeRepository,
    KnowledgeAssignmentInProgressError,
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


def test_sqlite_answer_completion_rolls_back_all_writes_on_progress_conflict(
    tmp_path,
) -> None:
    repository = SQLiteInterviewKnowledgeRepository(str(tmp_path / "offerpilot.db"))
    assignment = repository.create_assignment(
        owner_id="feishu:ou_owner",
        question_id="knowledge_network_http_001",
        assigned_on=date(2026, 7, 27),
        assignment_type=KnowledgeAssignmentType.NEW,
        recommendation_reason="验证原子提交",
    )
    attempt = repository.begin_attempt(
        owner_id="feishu:ou_owner",
        question_id=assignment.question_id,
        assignment_id=assignment.id,
        answer_text="默认使用持久连接。",
        answer_source=KnowledgeAnswerSource.TEXT,
        submitted_at=datetime(2026, 7, 27, 9, 0),
        submission_id="submission-atomic-finalize",
    )
    repository.save_progress(
        KnowledgeProgress(
            owner_id="feishu:ou_owner",
            question_id=assignment.question_id,
            mastery_status=KnowledgeMasteryStatus.REVIEWING,
            mastery_score=70,
            attempt_count=1,
            last_score=70,
        )
    )
    attempt.evaluation_status = "completed"
    attempt.score = 90
    attempt.evaluation_payload = {"score": 90}
    assignment.status = KnowledgeAssignmentStatus.COMPLETED
    assignment.attempt_id = attempt.id
    assignment.completed_at = attempt.submitted_at
    stale_progress = KnowledgeProgress(
        owner_id="feishu:ou_owner",
        question_id=assignment.question_id,
        mastery_status=KnowledgeMasteryStatus.REVIEWING,
        mastery_score=90,
        attempt_count=1,
        last_score=90,
    )

    try:
        repository.complete_attempt(
            attempt=attempt,
            assignment=assignment,
            progress=stale_progress,
            expected_progress_attempt_count=0,
        )
        raise AssertionError("stale progress update should have failed")
    except ConcurrentKnowledgeProgressUpdateError:
        pass

    stored_attempt = repository.get_attempt("feishu:ou_owner", attempt.id)
    stored_assignment = repository.list_assignments("feishu:ou_owner")[0]
    stored_progress = repository.get_progress(
        "feishu:ou_owner", assignment.question_id
    )
    assert stored_attempt is not None
    assert stored_attempt.evaluation_status == "pending"
    assert stored_attempt.score is None
    assert stored_assignment.status == KnowledgeAssignmentStatus.PENDING
    assert stored_assignment.attempt_id is None
    assert stored_assignment.active_attempt_id == attempt.id
    assert stored_progress is not None
    assert stored_progress.attempt_count == 1
    assert stored_progress.last_score == 70


def test_sqlite_assignment_lease_rejects_a_second_concurrent_attempt(tmp_path) -> None:
    database = str(tmp_path / "offerpilot.db")
    first_repository = SQLiteInterviewKnowledgeRepository(database)
    second_repository = SQLiteInterviewKnowledgeRepository(database)
    assignment = first_repository.create_assignment(
        owner_id="feishu:ou_owner",
        question_id="knowledge_network_http_001",
        assigned_on=date(2026, 7, 27),
        assignment_type=KnowledgeAssignmentType.NEW,
        recommendation_reason="验证跨实例并发",
    )
    first_attempt = first_repository.begin_attempt(
        owner_id="feishu:ou_owner",
        question_id=assignment.question_id,
        assignment_id=assignment.id,
        answer_text="默认使用持久连接。",
        answer_source=KnowledgeAnswerSource.TEXT,
        submitted_at=datetime(2026, 7, 27, 9, 0),
        submission_id="submission-first-worker",
    )

    try:
        second_repository.begin_attempt(
            owner_id="feishu:ou_owner",
            question_id=assignment.question_id,
            assignment_id=assignment.id,
            answer_text="Host 是必须的请求头。",
            answer_source=KnowledgeAnswerSource.TEXT,
            submitted_at=datetime(2026, 7, 27, 9, 0, 1),
            submission_id="submission-second-worker",
        )
        raise AssertionError("a second active attempt should have been rejected")
    except KnowledgeAssignmentInProgressError:
        pass

    attempts = first_repository.list_attempts("feishu:ou_owner")
    assert [item.id for item in attempts] == [first_attempt.id]
    stored_assignment = first_repository.list_assignments("feishu:ou_owner")[0]
    assert stored_assignment.active_attempt_id == first_attempt.id


def test_sqlite_expired_lease_can_resume_the_same_submission(tmp_path) -> None:
    database = str(tmp_path / "offerpilot.db")
    repository = SQLiteInterviewKnowledgeRepository(database)
    assignment = repository.create_assignment(
        owner_id="feishu:ou_owner",
        question_id="knowledge_network_http_001",
        assigned_on=date(2026, 7, 27),
        assignment_type=KnowledgeAssignmentType.NEW,
        recommendation_reason="验证崩溃恢复",
    )
    attempt = repository.begin_attempt(
        owner_id="feishu:ou_owner",
        question_id=assignment.question_id,
        assignment_id=assignment.id,
        answer_text="默认使用持久连接。",
        answer_source=KnowledgeAnswerSource.TEXT,
        submitted_at=datetime.now(timezone.utc),
        submission_id="submission-resume-after-crash",
    )
    expired_at = datetime.now(timezone.utc) - timedelta(minutes=16)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """UPDATE knowledge_assignments SET active_attempt_started_at=?
               WHERE id=?""",
            (expired_at.isoformat(), assignment.id),
        )
        connection.commit()

    resumed = repository.begin_attempt(
        owner_id="feishu:ou_owner",
        question_id=assignment.question_id,
        assignment_id=assignment.id,
        answer_text="默认使用持久连接。",
        answer_source=KnowledgeAnswerSource.TEXT,
        submitted_at=datetime.now(timezone.utc),
        submission_id="submission-resume-after-crash",
        existing_attempt_id=attempt.id,
    )

    assert resumed.id == attempt.id
    assert len(repository.list_attempts("feishu:ou_owner")) == 1
    stored_assignment = repository.list_assignments("feishu:ou_owner")[0]
    assert stored_assignment.active_attempt_id == attempt.id
