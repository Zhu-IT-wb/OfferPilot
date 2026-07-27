import asyncio
import json
from datetime import date, datetime

from app.models.interview_knowledge import (
    KnowledgeAnswerSource,
    KnowledgeAssignmentType,
    KnowledgeMasteryStatus,
)
from app.repositories.interview_knowledge_repository import (
    InMemoryInterviewKnowledgeRepository,
)
from app.services.knowledge_catalog import load_knowledge_catalog
from app.services.knowledge_evaluation import KnowledgeEvaluationService
from app.services.knowledge_practice import KnowledgePracticeWorkflow
from app.services.llm_service import LLMResult


class FakeLLMService:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def generate_text(self, *args, **kwargs) -> LLMResult:
        return LLMResult(
            provider="fake",
            model="fake-model",
            content=f"```json\n{json.dumps(self.payload, ensure_ascii=False)}\n```",
            raw_response={},
        )


def test_evaluation_uses_known_rubric_points_and_server_side_score() -> None:
    question = load_knowledge_catalog().questions[0]
    answer = "HTTP/1.1 默认复用 TCP 连接，并要求 Host 请求头。"
    service = KnowledgeEvaluationService(
        FakeLLMService(
            {
                "matched_required_point_ids": [
                    "http11_required_1",
                    "http11_required_2",
                    "invented_point",
                ],
                "matched_bonus_point_ids": ["http11_bonus_1"],
                "triggered_misconception_ids": [],
                "evidence": [
                    {
                        "point_id": "http11_required_1",
                        "quote": "默认复用 TCP 连接",
                    },
                    {
                        "point_id": "http11_required_2",
                        "quote": "这段文字不在用户回答中",
                    },
                ],
                "organization_score": 10,
                "clarity_score": 10,
                "feedback": "覆盖了连接与 Host，但遗漏缓存和分块传输。",
                "suggested_improvement": "按连接、请求头、缓存、传输四部分回答。",
            }
        )
    )

    result = asyncio.run(service.evaluate(question, answer))

    assert result.score == 55
    assert result.matched_required_point_ids == ["http11_required_1"]
    assert result.matched_bonus_point_ids == []
    assert result.missing_required_labels == ["Host", "缓存增强", "分块传输"]
    assert result.evidence == [
        {
            "point_id": "http11_required_1",
            "quote": "默认复用 TCP 连接",
        }
    ]


def test_practice_workflow_saves_result_and_schedules_review() -> None:
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
    evaluator = KnowledgeEvaluationService(
        FakeLLMService(
            {
                "matched_required_point_ids": [
                    "http11_required_1",
                    "http11_required_2",
                ],
                "matched_bonus_point_ids": ["http11_bonus_1"],
                "triggered_misconception_ids": [],
                "evidence": [
                    {"point_id": "http11_required_1", "quote": "持久连接"},
                    {"point_id": "http11_required_2", "quote": "Host"},
                ],
                "organization_score": 10,
                "clarity_score": 10,
                "feedback": "基本正确。",
                "suggested_improvement": "补充缓存与分块传输。",
            }
        )
    )
    workflow = KnowledgePracticeWorkflow(repository, evaluator)

    outcome = asyncio.run(
        workflow.submit_answer(
            owner_id="feishu:ou_owner",
            assignment_id=assignment.id,
            answer_text="默认持久连接，并增加 Host 请求头。",
            answer_source=KnowledgeAnswerSource.TEXT,
            submitted_at=datetime(2026, 7, 27, 9, 0),
        )
    )

    assert outcome.evaluation.score == 70
    assert outcome.progress.mastery_status == KnowledgeMasteryStatus.REVIEWING
    assert outcome.progress.next_review_on == date(2026, 7, 30)
    assert outcome.assignment.status.value == "completed"
    assert outcome.assignment.attempt_id == outcome.attempt.id
    assert repository.get_attempt("feishu:ou_owner", outcome.attempt.id).score == 70


def test_skipped_answer_is_recorded_without_calling_llm() -> None:
    class ExplodingLLMService:
        async def generate_text(self, *args, **kwargs):
            raise AssertionError("LLM should not be called for a skipped answer")

    repository = InMemoryInterviewKnowledgeRepository(
        questions=load_knowledge_catalog().questions
    )
    assignment = repository.create_assignment(
        owner_id="feishu:ou_owner",
        question_id="knowledge_network_http_001",
        assigned_on=date(2026, 7, 27),
        assignment_type=KnowledgeAssignmentType.NEW,
        recommendation_reason="当前章节新题",
    )
    workflow = KnowledgePracticeWorkflow(
        repository,
        KnowledgeEvaluationService(ExplodingLLMService()),
    )

    outcome = asyncio.run(
        workflow.submit_answer(
            owner_id="feishu:ou_owner",
            assignment_id=assignment.id,
            answer_text="",
            answer_source=KnowledgeAnswerSource.SKIPPED,
            submitted_at=datetime(2026, 7, 27, 9, 0),
        )
    )

    assert outcome.evaluation.score == 0
    assert outcome.progress.mastery_status == KnowledgeMasteryStatus.LEARNING
    assert outcome.progress.next_review_on == date(2026, 7, 28)


def test_two_high_scores_are_required_before_mastery() -> None:
    repository = InMemoryInterviewKnowledgeRepository(
        questions=load_knowledge_catalog().questions
    )
    evaluator = KnowledgeEvaluationService(
        FakeLLMService(
            {
                "matched_required_point_ids": [
                    "http11_required_1",
                    "http11_required_2",
                    "http11_required_3",
                    "http11_required_4",
                ],
                "matched_bonus_point_ids": ["http11_bonus_1"],
                "triggered_misconception_ids": [],
                "evidence": [
                    {"point_id": "http11_required_1", "quote": "持久连接"},
                    {"point_id": "http11_required_2", "quote": "Host"},
                    {"point_id": "http11_required_3", "quote": "缓存增强"},
                    {"point_id": "http11_required_4", "quote": "分块传输"},
                    {"point_id": "http11_bonus_1", "quote": "管线化限制"},
                ],
                "organization_score": 10,
                "clarity_score": 10,
                "feedback": "完整准确。",
                "suggested_improvement": "保持当前结构。",
            }
        )
    )
    workflow = KnowledgePracticeWorkflow(repository, evaluator)

    for index, practiced_on in enumerate((date(2026, 7, 27), date(2026, 8, 10))):
        assignment = repository.create_assignment(
            owner_id="feishu:ou_owner",
            question_id="knowledge_network_http_001",
            assigned_on=practiced_on,
            assignment_type=(
                KnowledgeAssignmentType.NEW
                if index == 0
                else KnowledgeAssignmentType.DUE_REVIEW
            ),
            recommendation_reason="验证掌握度",
        )
        outcome = asyncio.run(
            workflow.submit_answer(
                owner_id="feishu:ou_owner",
                assignment_id=assignment.id,
                answer_text="持久连接、Host、缓存增强、分块传输和管线化限制。",
                answer_source=KnowledgeAnswerSource.TEXT,
                submitted_at=datetime.combine(practiced_on, datetime.min.time()),
            )
        )

    assert outcome.evaluation.score == 100
    assert outcome.progress.high_score_streak == 2
    assert outcome.progress.mastery_status == KnowledgeMasteryStatus.MASTERED
    assert outcome.progress.next_review_on == date(2026, 9, 9)


def test_same_day_retry_does_not_count_as_spaced_mastery() -> None:
    repository = InMemoryInterviewKnowledgeRepository(
        questions=load_knowledge_catalog().questions
    )
    assignment = repository.create_assignment(
        owner_id="feishu:ou_owner",
        question_id="knowledge_network_http_001",
        assigned_on=date(2026, 7, 27),
        assignment_type=KnowledgeAssignmentType.NEW,
        recommendation_reason="验证同日重答",
    )
    evaluator = KnowledgeEvaluationService(
        FakeLLMService(
            {
                "matched_required_point_ids": [
                    "http11_required_1",
                    "http11_required_2",
                    "http11_required_3",
                    "http11_required_4",
                ],
                "matched_bonus_point_ids": ["http11_bonus_1"],
                "triggered_misconception_ids": [],
                "evidence": [
                    {"point_id": "http11_required_1", "quote": "持久连接"},
                    {"point_id": "http11_required_2", "quote": "Host"},
                    {"point_id": "http11_required_3", "quote": "缓存增强"},
                    {"point_id": "http11_required_4", "quote": "分块传输"},
                    {"point_id": "http11_bonus_1", "quote": "管线化限制"},
                ],
                "organization_score": 10,
                "clarity_score": 10,
                "feedback": "完整准确。",
                "suggested_improvement": "保持当前结构。",
            }
        )
    )
    workflow = KnowledgePracticeWorkflow(repository, evaluator)

    for hour in (9, 10):
        outcome = asyncio.run(
            workflow.submit_answer(
                owner_id="feishu:ou_owner",
                assignment_id=assignment.id,
                answer_text="持久连接、Host、缓存增强、分块传输和管线化限制。",
                answer_source=KnowledgeAnswerSource.TEXT,
                submitted_at=datetime(2026, 7, 27, hour, 0),
            )
        )

    outcome = asyncio.run(
        workflow.submit_answer(
            owner_id="feishu:ou_owner",
            assignment_id=assignment.id,
            answer_text="持久连接、Host、缓存增强、分块传输和管线化限制。",
            answer_source=KnowledgeAnswerSource.TEXT,
            submitted_at=datetime(2026, 7, 28, 9, 0),
        )
    )

    assert outcome.progress.attempt_count == 3
    assert outcome.progress.high_score_streak == 1
    assert outcome.progress.mastery_status == KnowledgeMasteryStatus.REVIEWING
    assert outcome.progress.next_review_on == date(2026, 8, 10)
