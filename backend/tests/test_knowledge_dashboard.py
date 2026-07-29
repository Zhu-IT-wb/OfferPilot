import json
from datetime import date, datetime

from fastapi.testclient import TestClient

from app.api.routes import knowledge_dashboard
from app.core.config import Settings
from app.main import create_app
from app.models.interview_knowledge import KnowledgeAnswerSource
from app.repositories.interview_knowledge_repository import (
    InMemoryInterviewKnowledgeRepository,
)
from app.services.knowledge_catalog import load_knowledge_catalog
from app.services.knowledge_evaluation import KnowledgeEvaluationService
from app.services.leetcode_dashboard_auth import DashboardTokenSigner
from app.services.llm_service import LLMConfigurationError, LLMResult


class FakeLLMService:
    async def generate_text(self, *args, **kwargs) -> LLMResult:
        payload = {
            "matched_required_point_ids": [
                "http11_required_1",
                "http11_required_2",
            ],
            "matched_bonus_point_ids": [],
            "triggered_misconception_ids": [],
            "evidence": {
                "http11_required_1": "持久连接",
                "http11_required_2": "Host",
            },
            "organization_score": 8,
            "clarity_score": 8,
            "feedback": "连接和 Host 回答正确，但遗漏缓存与分块传输。",
            "suggested_improvement": "按四个方面回答。",
        }
        return LLMResult("fake", "fake", json.dumps(payload), {})


def _client(monkeypatch, open_id: str = "ou_owner") -> tuple[TestClient, object]:
    repository = InMemoryInterviewKnowledgeRepository(
        questions=load_knowledge_catalog().questions
    )
    monkeypatch.setattr(
        knowledge_dashboard, "get_default_knowledge_repository", lambda: repository
    )
    monkeypatch.setattr(
        knowledge_dashboard,
        "get_knowledge_evaluation_service",
        lambda: KnowledgeEvaluationService(FakeLLMService()),
    )
    settings = Settings(
        debug_routes_enabled=False,
        feishu_app_id="cli_dashboard_test",
        feishu_app_secret="dashboard-secret",
    )
    client = TestClient(create_app(settings), follow_redirects=False)
    client.cookies.set(
        "offerpilot_dashboard_session",
        DashboardTokenSigner("dashboard-secret").issue(
            purpose="dashboard_session",
            claims={"open_id": open_id},
            ttl_seconds=3600,
        ),
    )
    return client, repository


def test_authenticated_user_can_open_knowledge_practice_page(monkeypatch) -> None:
    client, _ = _client(monkeypatch)

    response = client.get("/study/knowledge")

    assert response.status_code == 200
    assert "OfferPilot 八股复习" in response.text
    assert "/api/study/knowledge/today" in response.text
    assert "/api/study/knowledge/answers" in response.text
    assert "逐题练习" in response.text
    assert "原文资料" in response.text
    assert "window.open" not in response.text
    assert "submission_id" in response.text
    assert 'findIndex(item=>item.status!=="completed")' in response.text
    assert "/api/study/knowledge/practice" in response.text
    assert "/api/study/knowledge/catalog/questions" in response.text
    assert "加载更多" in response.text


def test_today_api_returns_navigation_and_hides_answers_before_submission(
    monkeypatch,
) -> None:
    client, _ = _client(monkeypatch)

    response = client.get("/api/study/knowledge/today")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"] == {"handled": 0, "total": 5}
    assert len(payload["items"]) == 5
    assert payload["catalog"][0]["id"] == "network"
    first_chapter = payload["catalog"][0]["chapters"][0]
    assert first_chapter["question_count"] == 2
    assert "questions" not in first_chapter
    assert payload["items"][0]["question"]["prompt"]
    assert "short_reference_answer" not in response.text
    assert "full_reference_answer" not in response.text


def test_catalog_questions_are_loaded_by_chapter_with_progress(monkeypatch) -> None:
    client, _ = _client(monkeypatch)
    client.get("/api/study/knowledge/today")

    response = client.get(
        "/api/study/knowledge/catalog/questions",
        params={"chapter_id": "index", "limit": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert payload["has_more"] is True
    assert len(payload["questions"]) == 1
    assert payload["questions"][0]["id"] == "knowledge_mysql_index_001"
    assert payload["questions"][0]["status"] == "unseen"
    assert payload["questions"][0]["score"] is None


def test_materials_api_exposes_all_in_app_reference_answers(monkeypatch) -> None:
    client, _ = _client(monkeypatch)

    response = client.get("/api/study/knowledge/materials")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["questions"]) >= 12
    assert payload["total"] == 12
    assert payload["has_more"] is False
    first = payload["questions"][0]
    assert first["id"] == "knowledge_network_http_001"
    assert "默认使用持久连接" in first["short_reference_answer"]
    assert first["full_reference_answer"]
    assert first["source"]["url"].startswith("https://")


def test_materials_api_supports_pagination_for_large_corpus(monkeypatch) -> None:
    client, _ = _client(monkeypatch)

    first_page = client.get(
        "/api/study/knowledge/materials", params={"limit": 2, "offset": 0}
    ).json()
    second_page = client.get(
        "/api/study/knowledge/materials", params={"limit": 2, "offset": 2}
    ).json()

    assert first_page["total"] == 12
    assert first_page["has_more"] is True
    assert len(first_page["questions"]) == 2
    assert len(second_page["questions"]) == 2
    assert first_page["questions"][0]["id"] != second_page["questions"][0]["id"]


def test_chapter_navigation_creates_manual_practice_assignments(monkeypatch) -> None:
    client, _ = _client(monkeypatch)
    client.get("/api/study/knowledge/today")

    response = client.post(
        "/api/study/knowledge/practice",
        json={"chapter_id": "index"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["focus_question_id"] == "knowledge_mysql_index_001"
    manual_items = [
        item
        for item in payload["items"]
        if item["question"]["chapter_id"] == "index"
    ]
    assert len(manual_items) == 2
    assert all(item["assignment_type"] == "manual" for item in manual_items)
    assert payload["summary"] == {"handled": 0, "total": 2}
    daily = client.get("/api/study/knowledge/today").json()
    assert daily["summary"] == {"handled": 0, "total": 5}


def test_answer_api_returns_evaluation_reference_answer_and_updated_progress(
    monkeypatch,
) -> None:
    client, _ = _client(monkeypatch)
    today = client.get("/api/study/knowledge/today").json()
    assignment_id = today["items"][0]["assignment_id"]

    response = client.post(
        "/api/study/knowledge/answers",
        json={
            "assignment_id": assignment_id,
            "submission_id": "submission-first-answer",
            "answer_text": "默认持久连接，并增加 Host 请求头。",
            "answer_source": "text",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["attempt"]["score"] == 66
    assert payload["evaluation"]["missing_required_labels"] == [
        "缓存增强",
        "分块传输",
    ]
    assert payload["evaluation"]["matched_required_labels"] == ["持久连接", "Host"]
    assert "默认使用持久连接" in payload["reference_answer"]["short"]
    assert payload["progress"]["next_review_on"]
    updated = client.get("/api/study/knowledge/today").json()
    assert updated["summary"] == {"handled": 1, "total": 5}


def test_completed_assignment_can_be_answered_again_without_double_counting(
    monkeypatch,
) -> None:
    client, _ = _client(monkeypatch)
    assignment_id = client.get("/api/study/knowledge/today").json()["items"][0][
        "assignment_id"
    ]
    answer = {
        "assignment_id": assignment_id,
        "submission_id": "submission-idempotent-answer",
        "answer_text": "默认持久连接，并增加 Host 请求头。",
        "answer_source": "text",
    }

    first = client.post("/api/study/knowledge/answers", json=answer)
    second = client.post("/api/study/knowledge/answers", json=answer)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["attempt"]["id"] == second.json()["attempt"]["id"]
    assert second.json()["progress"]["attempt_count"] == 1
    answer["submission_id"] = "submission-deliberate-retry"
    retry = client.post("/api/study/knowledge/answers", json=answer)
    assert retry.status_code == 200
    assert retry.json()["attempt"]["id"] != first.json()["attempt"]["id"]
    assert retry.json()["progress"]["attempt_count"] == 2
    updated = client.get("/api/study/knowledge/today").json()
    assert updated["summary"] == {"handled": 1, "total": 5}


def test_llm_configuration_failure_returns_retryable_service_error(monkeypatch) -> None:
    client, _ = _client(monkeypatch)
    assignment_id = client.get("/api/study/knowledge/today").json()["items"][0][
        "assignment_id"
    ]

    class UnconfiguredLLMService:
        async def generate_text(self, *args, **kwargs):
            raise LLMConfigurationError("missing test key")

    monkeypatch.setattr(
        knowledge_dashboard,
        "get_knowledge_evaluation_service",
        lambda: KnowledgeEvaluationService(UnconfiguredLLMService()),
    )

    response = client.post(
        "/api/study/knowledge/answers",
        json={
            "assignment_id": assignment_id,
            "submission_id": "submission-llm-failure",
            "answer_text": "这是需要保留并允许重试的回答。",
            "answer_source": "text",
        },
    )

    assert response.status_code == 503
    assert "AI 评价暂时不可用" in response.json()["detail"]
    updated = client.get("/api/study/knowledge/today").json()
    assert updated["summary"] == {"handled": 0, "total": 5}


def test_pending_duplicate_submission_returns_conflict_without_second_evaluation(
    monkeypatch,
) -> None:
    client, repository = _client(monkeypatch)
    assignment_id = client.get("/api/study/knowledge/today").json()["items"][0][
        "assignment_id"
    ]
    repository.create_attempt(
        owner_id="feishu:ou_owner",
        question_id="knowledge_network_http_001",
        assignment_id=assignment_id,
        answer_text="正在评价的回答",
        answer_source=KnowledgeAnswerSource.TEXT,
        submitted_at=datetime(2026, 7, 27, 9, 0),
        submission_id="submission-already-pending",
    )

    response = client.post(
        "/api/study/knowledge/answers",
        json={
            "assignment_id": assignment_id,
            "submission_id": "submission-already-pending",
            "answer_text": "正在评价的回答",
            "answer_source": "text",
        },
    )

    assert response.status_code == 409
    assert "already being evaluated" in response.json()["detail"]


def test_user_cannot_answer_another_users_assignment(monkeypatch) -> None:
    owner_client, repository = _client(monkeypatch, "ou_owner")
    assignment_id = owner_client.get("/api/study/knowledge/today").json()["items"][0][
        "assignment_id"
    ]
    other_client, _ = _client(monkeypatch, "ou_other")
    monkeypatch.setattr(
        knowledge_dashboard, "get_default_knowledge_repository", lambda: repository
    )

    response = other_client.post(
        "/api/study/knowledge/answers",
        json={
            "assignment_id": assignment_id,
            "submission_id": "submission-other-owner",
            "answer_text": "尝试修改他人的题目",
            "answer_source": "text",
        },
    )

    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()


def test_anonymous_knowledge_api_request_is_rejected() -> None:
    app = create_app(
        Settings(
            debug_routes_enabled=False,
            feishu_app_id="cli_dashboard_test",
            feishu_app_secret="dashboard-secret",
        )
    )
    client = TestClient(app)

    response = client.get("/api/study/knowledge/today")

    assert response.status_code == 401
