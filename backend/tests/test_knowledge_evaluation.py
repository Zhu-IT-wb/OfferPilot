import asyncio
import json
from datetime import date, datetime

from app.models.interview_knowledge import (
    KnowledgeAnswerSource,
    KnowledgeAssignmentType,
    KnowledgeDifficulty,
    KnowledgeMasteryStatus,
    KnowledgeQuestion,
    KnowledgeRubricKind,
    KnowledgeRubricPoint,
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


class SequencedLLMService:
    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.call_count = 0
        self.calls = []

    async def generate_text(self, *args, **kwargs) -> LLMResult:
        self.calls.append(kwargs)
        content = self.contents[self.call_count]
        self.call_count += 1
        return LLMResult("fake", "fake-model", content, {})


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


def test_evaluation_accepts_deepseek_evidence_object() -> None:
    question = load_knowledge_catalog().questions[0]
    answer = (
        "默认复用 TCP 连接，支持虚拟主机，采用更细粒度缓存控制，"
        "并支持 chunked 传输。"
    )
    service = KnowledgeEvaluationService(
        FakeLLMService(
            {
                "matched_required_point_ids": [
                    "http11_required_1",
                    "http11_required_2",
                    "http11_required_3",
                    "http11_required_4",
                ],
                "matched_bonus_point_ids": [],
                "triggered_misconception_ids": [],
                "evidence": {
                    "http11_required_1": "默认复用 TCP 连接",
                    "http11_required_2": "支持虚拟主机",
                    "http11_required_3": "更细粒度缓存控制",
                    "http11_required_4": "支持 chunked 传输",
                },
                "organization_score": 9,
                "clarity_score": 10,
                "feedback": "完整覆盖核心改进。",
                "suggested_improvement": "可以补充队头阻塞限制。",
            }
        )
    )

    result = asyncio.run(service.evaluate(question, answer))

    assert result.score == 99
    assert result.matched_required_point_ids == [
        "http11_required_1",
        "http11_required_2",
        "http11_required_3",
        "http11_required_4",
    ]
    assert len(result.evidence) == 4


def test_evaluation_accepts_semantic_evidence_from_a_markdown_table_row() -> None:
    point = KnowledgeRubricPoint(
        id="tree_required_1",
        kind=KnowledgeRubricKind.REQUIRED,
        label="节点容量",
        description=(
            "节点容量：每个节点存1个键值对；每个节点存多个键值对；"
            "非叶节点存多个键，仅叶节点存值"
        ),
    )
    answer = (
        "| 特性 | 红黑树 | B树 | B+树 |\n"
        "| --- | --- | --- | --- |\n"
        "| 节点容量 | 每个节点存1个键值对 | 每个节点存多个键值对 | "
        "非叶节点存多个键，仅叶节点存值 |"
    )
    question = KnowledgeQuestion(
        id="tree_question",
        module_id="data_structures",
        module_title="数据结构",
        module_order=1,
        chapter_id="tree",
        chapter_title="树",
        chapter_order=1,
        question_order=1,
        prompt="红黑树、B树、B+树有什么区别？",
        difficulty=KnowledgeDifficulty.INTERMEDIATE,
        frequency=3,
        short_reference_answer=answer,
        full_reference_answer=answer,
        rubric_points=[point],
    )
    service = KnowledgeEvaluationService(
        FakeLLMService(
            {
                "matched_required_point_ids": [point.id],
                "matched_bonus_point_ids": [],
                "triggered_misconception_ids": [],
                "evidence": {
                    point.id: (
                        "每个节点存1个键值对；每个节点存多个键值对；"
                        "非叶节点存多个键，仅叶节点存值"
                    )
                },
                "organization_score": 10,
                "clarity_score": 10,
                "feedback": "完整。",
                "suggested_improvement": "保持。",
            }
        )
    )

    result = asyncio.run(service.evaluate(question, answer))

    assert result.score == 100
    assert result.matched_required_point_ids == [point.id]


def test_evaluation_retries_once_after_invalid_model_json() -> None:
    question = load_knowledge_catalog().questions[0]
    answer = "默认复用 TCP 连接。"
    valid_payload = {
        "matched_required_point_ids": ["http11_required_1"],
        "matched_bonus_point_ids": [],
        "triggered_misconception_ids": [],
        "evidence": {
            "http11_required_1": "默认复用 TCP 连接",
        },
        "organization_score": 5,
        "clarity_score": 8,
        "feedback": "命中持久连接。",
        "suggested_improvement": "继续补充其他改进。",
    }
    llm_service = SequencedLLMService(
        ["{\"matched_required_point_ids\": [", json.dumps(valid_payload)]
    )
    service = KnowledgeEvaluationService(llm_service)

    result = asyncio.run(service.evaluate(question, answer))

    assert llm_service.call_count == 2
    assert all(
        call["response_format"] == {"type": "json_object"}
        for call in llm_service.calls
    )
    assert all(
        call["thinking"] == {"type": "disabled"} for call in llm_service.calls
    )
    assert result.score == 48
    assert result.matched_required_point_ids == ["http11_required_1"]


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
