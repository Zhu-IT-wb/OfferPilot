from datetime import date, datetime

from app.models.interview_knowledge import (
    KnowledgeAssignmentType,
    KnowledgeMasteryStatus,
    KnowledgeProgress,
)
from app.repositories.interview_knowledge_repository import (
    InMemoryInterviewKnowledgeRepository,
)
from app.services.knowledge_catalog import load_knowledge_catalog
from app.services.knowledge_recommendation import KnowledgeRecommendationWorkflow


def test_first_daily_plan_contains_five_deterministic_new_questions() -> None:
    repository = InMemoryInterviewKnowledgeRepository(
        questions=load_knowledge_catalog().questions
    )
    workflow = KnowledgeRecommendationWorkflow(repository)

    first = workflow.get_today("feishu:ou_owner", date(2026, 7, 27))
    reopened = workflow.get_today("feishu:ou_owner", date(2026, 7, 27))

    assert len(first) == 5
    assert [item.assignment.id for item in reopened] == [
        item.assignment.id for item in first
    ]
    assert all(
        item.assignment.assignment_type == KnowledgeAssignmentType.NEW
        for item in first
    )
    assert [item.question.id for item in first[:3]] == [
        "knowledge_network_http_001",
        "knowledge_network_http_002",
        "knowledge_network_tcp_001",
    ]


def test_daily_plan_prioritizes_due_and_weak_questions_before_new_questions() -> None:
    repository = InMemoryInterviewKnowledgeRepository(
        questions=load_knowledge_catalog().questions
    )
    workflow = KnowledgeRecommendationWorkflow(repository)
    owner_id = "feishu:ou_owner"
    workflow.get_today(owner_id, date(2026, 7, 27))
    repository.save_progress(
        KnowledgeProgress(
            owner_id=owner_id,
            question_id="knowledge_network_http_001",
            mastery_status=KnowledgeMasteryStatus.LEARNING,
            mastery_score=35,
            attempt_count=1,
            last_score=35,
            last_attempt_at=datetime(2026, 7, 27, 9, 0),
            next_review_on=date(2026, 7, 28),
            last_detected_gaps=["缓存增强"],
        )
    )
    repository.save_progress(
        KnowledgeProgress(
            owner_id=owner_id,
            question_id="knowledge_network_http_002",
            mastery_status=KnowledgeMasteryStatus.LEARNING,
            mastery_score=50,
            attempt_count=1,
            last_score=50,
            last_attempt_at=datetime(2026, 7, 27, 9, 5),
            next_review_on=date(2026, 8, 1),
            last_detected_gaps=["身份认证"],
        )
    )

    plan = workflow.get_today(owner_id, date(2026, 7, 28))

    assert len(plan) == 5
    assert plan[0].question.id == "knowledge_network_http_001"
    assert plan[0].assignment.assignment_type == KnowledgeAssignmentType.DUE_REVIEW
    assert plan[1].question.id == "knowledge_network_http_002"
    assert plan[1].assignment.assignment_type == KnowledgeAssignmentType.WEAKNESS
    assert len({item.question.id for item in plan}) == 5


def test_mastered_questions_are_not_recommended_before_review_date() -> None:
    questions = load_knowledge_catalog().questions
    repository = InMemoryInterviewKnowledgeRepository(questions=questions)
    owner_id = "feishu:ou_owner"
    learned_on = date(2026, 7, 27)
    for question in questions:
        repository.create_assignment(
            owner_id=owner_id,
            question_id=question.id,
            assigned_on=learned_on,
            assignment_type=KnowledgeAssignmentType.NEW,
            recommendation_reason="已学完",
        )
        repository.save_progress(
            KnowledgeProgress(
                owner_id=owner_id,
                question_id=question.id,
                mastery_status=KnowledgeMasteryStatus.MASTERED,
                mastery_score=95,
                attempt_count=2,
                high_score_streak=2,
                last_score=95,
                last_attempt_at=datetime(2026, 7, 27, 9, 0),
                next_review_on=date(2026, 8, 27),
            )
        )

    plan = KnowledgeRecommendationWorkflow(repository).get_today(
        owner_id, date(2026, 7, 28)
    )

    assert plan == []
