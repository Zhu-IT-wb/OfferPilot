import json
import asyncio

import pytest
from fastapi.testclient import TestClient

from app.api.routes import project_training_dashboard
from app.core.config import Settings
from app.main import create_app
from app.repositories.project_training_repository import (
    InMemoryProjectTrainingRepository,
    ProjectTrainingSubmissionInProgressError,
)
from app.models.project_training import (
    ProjectTopicProgress,
    ProjectTrainingAnswerSource,
    ProjectTrainingSessionStatus,
)
from app.repositories.sqlite_project_training_repository import (
    SQLiteProjectTrainingRepository,
)
from app.services.leetcode_dashboard_auth import DashboardTokenSigner
from app.services.llm_service import LLMResult
from app.services.project_training import ProjectTrainingWorkflow
from app.services.speech_transcription import SpeechTranscriptionResult
from app.agents.orchestrator import AgentOrchestrator
from app.agents.interview_agent import _validated_decision
from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository
from app.schemas.agent import AgentActionName
from app.schemas.intent import IntentClassification, IntentName
from app.tools.offerpilot_tools import build_offerpilot_tool_registry


class FakeProjectInterviewLLM:
    async def generate_text(self, *args, **kwargs) -> LLMResult:
        payload = {
            "dimension_scores": {
                "ownership": 85,
                "technical_depth": 80,
                "evidence": 55,
                "tradeoff": 70,
                "reliability": 65,
                "structure": 80,
                "clarity": 85,
            },
            "strengths": ["说明了个人负责的核心链路"],
            "missing_details": ["缺少优化前后的延迟数据"],
            "unsupported_claims": [],
            "contradictions": [],
            "answer_evidence": [
                {"dimension": "ownership", "quote": "我负责订单创建链路"}
            ],
            "follow_up_signals": ["missing_metric"],
            "feedback": "职责清楚，但性能结果需要量化。",
            "improved_outline": ["背景", "个人职责", "方案", "指标"],
            "candidate_next_turn": {
                "theme": "metrics",
                "question_kind": "evidence",
                "question_text": "你提到性能有提升，优化前后的 P95 延迟分别是多少？",
                "generation_reason": "回答缺少量化指标",
                "source_evidence_ids": [],
                "hypothetical": False,
            },
        }
        return LLMResult("fake", "fake-project-interviewer", json.dumps(payload), {})


def _client(monkeypatch, open_id: str = "ou_project_owner"):
    repository = InMemoryProjectTrainingRepository()
    workflow = ProjectTrainingWorkflow(
        repository=repository,
        llm_service=FakeProjectInterviewLLM(),
    )
    monkeypatch.setattr(
        project_training_dashboard,
        "get_default_project_training_repository",
        lambda: repository,
    )
    monkeypatch.setattr(
        project_training_dashboard,
        "get_project_training_workflow",
        lambda: workflow,
    )
    settings = Settings(
        debug_routes_enabled=False,
        feishu_app_id="cli_project_test",
        feishu_app_secret="project-secret",
        dashboard_public_base_url="http://testserver",
    )
    client = TestClient(create_app(settings), follow_redirects=False)
    client.cookies.set(
        "offerpilot_dashboard_session",
        DashboardTokenSigner("project-secret").issue(
            purpose="dashboard_session",
            claims={"open_id": open_id},
            ttl_seconds=3600,
        ),
    )
    return client, repository


def _project_payload(name: str = "高并发订单系统"):
    return {
        "name": name,
        "target_role": "Java 后端工程师",
        "background": "电商大促期间订单链路需要承受突发流量。",
        "responsibilities": ["负责订单创建链路", "设计幂等与降级方案"],
        "tech_stack": ["Java", "Redis", "MySQL", "Kafka"],
        "architecture": "网关、订单服务、Redis、MySQL 和 Kafka。",
        "key_decisions": ["使用 Redis 处理热点数据"],
        "technical_challenges": ["重复下单与流量突增"],
        "metrics": ["P95 延迟从 320ms 降至 120ms"],
        "outcomes": ["大促期间保持稳定"],
        "resume_description": "负责高并发订单系统核心链路。",
        "supplemental_text": "## 故障处理\nKafka 不可用时进入本地补偿队列。",
    }


def test_user_can_create_update_and_archive_a_versioned_project(monkeypatch) -> None:
    client, _ = _client(monkeypatch)

    created = client.post("/api/study/projects", json=_project_payload())

    assert created.status_code == 201
    first = created.json()["project"]
    assert first["version"] == 1
    assert first["name"] == "高并发订单系统"
    assert first["missing_fields"] == []

    updated_payload = _project_payload()
    updated_payload["metrics"] = ["P95 延迟从 320ms 降至 95ms"]
    updated = client.put(
        f"/api/study/projects/{first['id']}", json=updated_payload
    )
    assert updated.status_code == 200
    assert updated.json()["project"]["version"] == 2

    fetched = client.get(f"/api/study/projects/{first['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["project"]["metrics"] == ["P95 延迟从 320ms 降至 95ms"]
    assert fetched.json()["versions"] == [1, 2]

    archived = client.post(f"/api/study/projects/{first['id']}/archive")
    assert archived.status_code == 200
    assert archived.json()["project"]["status"] == "archived"


def test_project_resources_are_hidden_from_other_users(monkeypatch) -> None:
    owner_client, repository = _client(monkeypatch, "ou_owner")
    project_id = owner_client.post(
        "/api/study/projects", json=_project_payload()
    ).json()["project"]["id"]

    other_workflow = ProjectTrainingWorkflow(
        repository=repository,
        llm_service=FakeProjectInterviewLLM(),
    )
    monkeypatch.setattr(
        project_training_dashboard,
        "get_project_training_workflow",
        lambda: other_workflow,
    )
    settings = Settings(
        debug_routes_enabled=False,
        feishu_app_secret="project-secret",
        dashboard_public_base_url="http://testserver",
    )
    other_client = TestClient(create_app(settings), follow_redirects=False)
    other_client.cookies.set(
        "offerpilot_dashboard_session",
        DashboardTokenSigner("project-secret").issue(
            purpose="dashboard_session",
            claims={"open_id": "ou_other"},
            ttl_seconds=3600,
        ),
    )

    assert other_client.get(f"/api/study/projects/{project_id}").status_code == 404
    hidden_session = other_client.post(
        "/api/study/project-training/sessions",
        json={
            "project_id": project_id,
            "creation_id": "creation-hidden-owner",
            "max_turns": 6,
        },
    )
    assert hidden_session.status_code == 404


def test_session_creation_is_idempotent_and_first_turn_survives_refresh(
    monkeypatch,
) -> None:
    client, _ = _client(monkeypatch)
    project_id = client.post(
        "/api/study/projects", json=_project_payload()
    ).json()["project"]["id"]
    request = {
        "project_id": project_id,
        "creation_id": "creation-order-project-001",
        "difficulty": "medium",
        "max_turns": 6,
        "goal": "训练项目深挖表达",
    }

    first = client.post("/api/study/project-training/sessions", json=request)
    duplicate = client.post("/api/study/project-training/sessions", json=request)

    assert first.status_code == 201
    assert duplicate.status_code == 200
    assert duplicate.json()["session"]["id"] == first.json()["session"]["id"]
    assert duplicate.json()["resumed"] is True
    session_id = first.json()["session"]["id"]
    first_turn = first.json()["current_turn"]
    assert first_turn["sequence"] == 1
    assert first_turn["theme"] == "project_overview"
    assert first_turn["question_kind"] == "opening"
    assert "高并发订单系统" in first_turn["question_text"]

    refreshed = client.get(
        f"/api/study/project-training/sessions/{session_id}/current"
    )
    assert refreshed.status_code == 200
    assert refreshed.json()["current_turn"]["id"] == first_turn["id"]


def test_archived_project_cannot_start_a_new_session(monkeypatch) -> None:
    client, _ = _client(monkeypatch)
    project_id = client.post(
        "/api/study/projects", json=_project_payload()
    ).json()["project"]["id"]
    client.post(f"/api/study/projects/{project_id}/archive")

    response = client.post(
        "/api/study/project-training/sessions",
        json={
            "project_id": project_id,
            "creation_id": "creation-archived-project",
            "difficulty": "medium",
            "max_turns": 6,
        },
    )

    assert response.status_code == 400
    assert "归档" in response.json()["detail"]


def test_answer_is_evaluated_by_interview_graph_and_creates_one_next_turn(
    monkeypatch,
) -> None:
    client, _ = _client(monkeypatch)
    project_id = client.post(
        "/api/study/projects", json=_project_payload()
    ).json()["project"]["id"]
    started = client.post(
        "/api/study/project-training/sessions",
        json={
            "project_id": project_id,
            "creation_id": "creation-answer-flow-001",
            "difficulty": "medium",
            "max_turns": 6,
        },
    ).json()
    session_id = started["session"]["id"]
    turn_id = started["current_turn"]["id"]
    request = {
        "turn_id": turn_id,
        "submission_id": "submission-project-answer-001",
        "answer_text": "我负责订单创建链路，使用 Redis 做幂等，并把结果写入 MySQL。",
        "answer_source": "text",
    }

    answered = client.post(
        f"/api/study/project-training/sessions/{session_id}/answers",
        json=request,
    )
    duplicate = client.post(
        f"/api/study/project-training/sessions/{session_id}/answers",
        json=request,
    )

    assert answered.status_code == 200
    payload = answered.json()
    assert 0 <= payload["answer"]["overall_score"] <= 100
    assert payload["evaluation"]["dimension_scores"]["ownership"] == 85
    assert payload["evaluation"]["answer_evidence"] == [
        {"dimension": "ownership", "quote": "我负责订单创建链路"}
    ]
    assert payload["current_turn"]["sequence"] == 2
    assert payload["current_turn"]["parent_turn_id"] == turn_id
    assert "P95" in payload["current_turn"]["question_text"]
    assert duplicate.status_code == 200
    assert duplicate.json()["answer"]["id"] == payload["answer"]["id"]
    assert duplicate.json()["current_turn"]["id"] == payload["current_turn"]["id"]


def test_interview_decision_discards_unverifiable_negative_judgments() -> None:
    payload = json.loads(
        asyncio.run(FakeProjectInterviewLLM().generate_text()).content
    )
    payload["unsupported_claims"] = [
        {"quote": "不存在的原文", "explanation": "缺少指标"},
        {"quote": "Redis 做幂等", "explanation": "没有说明失效策略"},
    ]
    payload["contradictions"] = [
        {
            "quote": "Redis 做幂等",
            "explanation": "与项目档案冲突",
            "evidence_id": "unknown-evidence",
        }
    ]

    decision = _validated_decision(
        payload,
        answer_text="我负责订单链路，并使用 Redis 做幂等。",
        allowed_evidence_ids={"evidence-known"},
    )

    assert decision.unsupported_claims == ["Redis 做幂等：没有说明失效策略"]
    assert decision.contradictions == []


def test_session_payload_contains_project_snapshot_and_remaining_turns(monkeypatch) -> None:
    client, _ = _client(monkeypatch)
    project_id = client.post(
        "/api/study/projects", json=_project_payload()
    ).json()["project"]["id"]

    started = client.post(
        "/api/study/project-training/sessions",
        json={
            "project_id": project_id,
            "creation_id": "creation-project-context",
            "max_turns": 6,
        },
    )

    assert started.status_code == 201
    assert started.json()["project"]["name"] == "高并发订单系统"
    assert started.json()["remaining_turns"] == 6


def test_submission_id_cannot_be_reused_for_another_answer(monkeypatch) -> None:
    client, _ = _client(monkeypatch)
    project_id = client.post(
        "/api/study/projects", json=_project_payload()
    ).json()["project"]["id"]
    started = client.post(
        "/api/study/project-training/sessions",
        json={
            "project_id": project_id,
            "creation_id": "creation-conflict-flow",
            "difficulty": "medium",
            "max_turns": 6,
        },
    ).json()
    session_id = started["session"]["id"]
    turn_id = started["current_turn"]["id"]
    first = {
        "turn_id": turn_id,
        "submission_id": "submission-conflict-project",
        "answer_text": "我负责订单创建链路，使用 Redis 做幂等。",
        "answer_source": "text",
    }
    assert client.post(
        f"/api/study/project-training/sessions/{session_id}/answers", json=first
    ).status_code == 200

    first["answer_text"] = "换一个完全不同的回答"
    conflict = client.post(
        f"/api/study/project-training/sessions/{session_id}/answers", json=first
    )

    assert conflict.status_code == 400
    assert "submission_id" in conflict.json()["detail"]


def test_sqlite_project_and_session_survive_repository_restart(
    monkeypatch, tmp_path
) -> None:
    database = tmp_path / "project-training.db"
    first_repository = SQLiteProjectTrainingRepository(str(database))
    first_workflow = ProjectTrainingWorkflow(
        first_repository, FakeProjectInterviewLLM()
    )
    monkeypatch.setattr(
        project_training_dashboard,
        "get_default_project_training_repository",
        lambda: first_repository,
    )
    monkeypatch.setattr(
        project_training_dashboard,
        "get_project_training_workflow",
        lambda: first_workflow,
    )
    settings = Settings(
        debug_routes_enabled=False,
        feishu_app_secret="project-secret",
        dashboard_public_base_url="http://testserver",
    )
    client = TestClient(create_app(settings), follow_redirects=False)
    client.cookies.set(
        "offerpilot_dashboard_session",
        DashboardTokenSigner("project-secret").issue(
            purpose="dashboard_session",
            claims={"open_id": "ou_sqlite"},
            ttl_seconds=3600,
        ),
    )
    project_id = client.post(
        "/api/study/projects", json=_project_payload()
    ).json()["project"]["id"]
    started = client.post(
        "/api/study/project-training/sessions",
        json={
            "project_id": project_id,
            "creation_id": "creation-sqlite-restart",
            "difficulty": "medium",
            "max_turns": 6,
        },
    ).json()
    answered = client.post(
        f"/api/study/project-training/sessions/{started['session']['id']}/answers",
        json={
            "turn_id": started["current_turn"]["id"],
            "submission_id": "submission-sqlite-restart",
            "answer_text": "我负责订单创建链路，使用 Redis 做幂等。",
            "answer_source": "text",
        },
    ).json()

    restarted_repository = SQLiteProjectTrainingRepository(str(database))
    restarted_workflow = ProjectTrainingWorkflow(
        restarted_repository, FakeProjectInterviewLLM()
    )
    monkeypatch.setattr(
        project_training_dashboard,
        "get_default_project_training_repository",
        lambda: restarted_repository,
    )
    monkeypatch.setattr(
        project_training_dashboard,
        "get_project_training_workflow",
        lambda: restarted_workflow,
    )

    fetched = client.get(f"/api/study/projects/{project_id}")
    resumed = client.get(
        f"/api/study/project-training/sessions/{started['session']['id']}/current"
    )
    assert fetched.status_code == 200
    assert fetched.json()["project"]["version"] == 1
    assert resumed.status_code == 200
    assert resumed.json()["current_turn"]["id"] == answered["current_turn"]["id"]
    assert resumed.json()["current_turn"]["sequence"] == 2


def test_user_can_finish_training_and_read_summary_history(monkeypatch) -> None:
    client, _ = _client(monkeypatch)
    project_id = client.post(
        "/api/study/projects", json=_project_payload()
    ).json()["project"]["id"]
    started = client.post(
        "/api/study/project-training/sessions",
        json={
            "project_id": project_id,
            "creation_id": "creation-summary-flow",
            "difficulty": "medium",
            "max_turns": 6,
        },
    ).json()
    session_id = started["session"]["id"]
    answered = client.post(
        f"/api/study/project-training/sessions/{session_id}/answers",
        json={
            "turn_id": started["current_turn"]["id"],
            "submission_id": "submission-summary-answer",
            "answer_text": "我负责订单创建链路，使用 Redis 做幂等，并把结果写入 MySQL。",
            "answer_source": "text",
        },
    )
    assert answered.status_code == 200

    finished = client.post(
        f"/api/study/project-training/sessions/{session_id}/finish"
    )
    fetched = client.get(
        f"/api/study/project-training/sessions/{session_id}/summary"
    )
    history = client.get("/api/study/project-training/sessions")

    assert finished.status_code == 200
    assert fetched.status_code == 200
    summary = fetched.json()["summary"]
    assert summary["overall_score"] > 0
    assert summary["project_version"] == 1
    assert summary["improvement_areas"] == ["缺少优化前后的延迟数据"]
    assert history.json()["sessions"][0]["status"] == "completed"


def test_abandoned_training_has_no_formal_summary(monkeypatch) -> None:
    client, _ = _client(monkeypatch)
    project_id = client.post(
        "/api/study/projects", json=_project_payload()
    ).json()["project"]["id"]
    session_id = client.post(
        "/api/study/project-training/sessions",
        json={
            "project_id": project_id,
            "creation_id": "creation-abandon-flow",
            "difficulty": "medium",
            "max_turns": 6,
        },
    ).json()["session"]["id"]

    abandoned = client.post(
        f"/api/study/project-training/sessions/{session_id}/abandon"
    )
    summary = client.get(
        f"/api/study/project-training/sessions/{session_id}/summary"
    )

    assert abandoned.status_code == 200
    assert abandoned.json()["session"]["status"] == "abandoned"
    assert summary.status_code == 404


def test_inflight_answer_cannot_revive_finished_session() -> None:
    repository = InMemoryProjectTrainingRepository()
    workflow = ProjectTrainingWorkflow(repository, FakeProjectInterviewLLM())
    owner_id = "feishu:ou_race"
    project = repository.create_project(owner_id, _project_payload())
    outcome = workflow.start(
        owner_id,
        project.id,
        creation_id="creation-finish-race",
        max_turns=6,
    )
    answer, _ = repository.begin_answer(
        owner_id,
        outcome.session.id,
        outcome.current_turn.id,
        "submission-finish-race",
        "我负责订单链路",
        ProjectTrainingAnswerSource.TEXT,
        outcome.session.started_at,
    )
    finished_session = outcome.session.__class__(
        **{
            **outcome.session.__dict__,
            "status": ProjectTrainingSessionStatus.COMPLETED,
            "current_turn_id": None,
            "completed_at": outcome.session.started_at,
        }
    )
    repository.sessions[(owner_id, outcome.session.id)] = finished_session

    with pytest.raises(ProjectTrainingSubmissionInProgressError):
        repository.complete_answer(
            answer,
            outcome.current_turn,
            outcome.session,
            None,
            ProjectTopicProgress(owner_id, project.id, "project_overview"),
        )


def test_project_voice_transcription_uses_safe_turn_context(monkeypatch) -> None:
    client, _ = _client(monkeypatch)
    project_id = client.post(
        "/api/study/projects", json=_project_payload()
    ).json()["project"]["id"]
    started = client.post(
        "/api/study/project-training/sessions",
        json={
            "project_id": project_id,
            "creation_id": "creation-voice-flow",
            "difficulty": "medium",
            "max_turns": 6,
        },
    ).json()
    captured = {}

    class FakeSpeechService:
        async def transcribe(self, **kwargs):
            captured.update(kwargs)
            return SpeechTranscriptionResult(
                transcript="我负责订单创建链路，并用 Redis 实现幂等。",
                provider="fake-asr",
                model="fake-voice",
                duration_seconds=2.0,
            )

    monkeypatch.setattr(
        project_training_dashboard,
        "get_speech_transcription_service",
        lambda request: FakeSpeechService(),
    )
    response = client.post(
        "/api/study/project-training/transcriptions",
        params={"turn_id": started["current_turn"]["id"]},
        content=b"fake-webm-audio",
        headers={
            "content-type": "audio/webm",
            "x-audio-duration-ms": "2000",
        },
    )

    assert response.status_code == 200
    assert response.json()["transcript"].startswith("我负责")
    assert captured["audio_format"] == "webm"
    assert "高并发订单系统" in captured["context"]
    assert "Redis" in captured["context"]
    assert "P95 延迟从" not in captured["context"]


def test_main_agent_graph_creates_and_resumes_project_training_session() -> None:
    project_repository = InMemoryProjectTrainingRepository()
    project = project_repository.create_project(
        "feishu:ou_agent", _project_payload("Agent 求职助手")
    )

    class ProjectTrainingClassifier:
        async def classify(self, message):
            assert message == "开始 Agent 求职助手项目训练"
            return IntentClassification(
                intent=IntentName.START_PROJECT_TRAINING,
                confidence=0.94,
                slots={"project": "Agent 求职助手"},
            )

    registry = build_offerpilot_tool_registry(
        InMemoryOfferPilotRepository(),
        calendar_service=None,
        bitable_service=None,
        project_training_repository=project_repository,
        dashboard_public_base_url="https://offerpilot.example.com",
    )
    orchestrator = AgentOrchestrator(
        intent_classifier=ProjectTrainingClassifier(),
        tool_registry=registry,
    )

    first = asyncio.run(
        orchestrator.handle_message(
            "开始 Agent 求职助手项目训练",
            user_id="ou_agent",
            source="feishu",
        )
    )
    second = asyncio.run(
        orchestrator.handle_message(
            "开始 Agent 求职助手项目训练",
            user_id="ou_agent",
            source="feishu",
        )
    )

    assert first.action == AgentActionName.START_PROJECT_TRAINING
    assert first.tool_result is not None
    assert first.tool_result.data["project_id"] == project.id
    assert first.tool_result.data["launch_url"].startswith(
        "https://offerpilot.example.com/study/projects/training?session_id="
    )
    assert second.tool_result.data["session_id"] == first.tool_result.data["session_id"]
    assert second.tool_result.data["resumed"] is True


def test_project_pages_expose_profile_training_and_voice_flows(monkeypatch) -> None:
    client, _ = _client(monkeypatch)

    projects_page = client.get("/study/projects")
    training_page = client.get(
        "/study/projects/training?session_id=project_session_placeholder"
    )
    voice_asset = client.get("/study/projects/assets/voice_recorder.js")

    assert projects_page.status_code == 200
    assert "新建项目档案" in projects_page.text
    assert "/api/study/project-training/sessions" in client.get(
        "/study/projects/assets/project_profiles.js"
    ).text
    assert training_page.status_code == 200
    assert "项目训练房间" in training_page.text
    assert "voice_recorder.js" in training_page.text
    assert voice_asset.status_code == 200
    assert "MediaRecorder" in voice_asset.text
    assert "getRecorderManager" in voice_asset.text


def test_session_auto_completes_at_max_turns_and_limits_same_theme_followups(
    monkeypatch,
) -> None:
    client, _ = _client(monkeypatch)
    project_id = client.post(
        "/api/study/projects", json=_project_payload()
    ).json()["project"]["id"]
    state = client.post(
        "/api/study/project-training/sessions",
        json={
            "project_id": project_id,
            "creation_id": "creation-max-turns",
            "difficulty": "medium",
            "max_turns": 3,
        },
    ).json()
    themes = [state["current_turn"]["theme"]]
    for index in range(3):
        response = client.post(
            f"/api/study/project-training/sessions/{state['session']['id']}/answers",
            json={
                "turn_id": state["current_turn"]["id"],
                "submission_id": f"submission-max-turns-{index}",
                "answer_text": "我负责订单创建链路，使用 Redis 做幂等。",
                "answer_source": "text",
            },
        )
        assert response.status_code == 200
        state = response.json()
        if state["current_turn"]:
            themes.append(state["current_turn"]["theme"])

    assert state["session"]["status"] == "completed"
    assert state["current_turn"] is None
    assert themes == ["project_overview", "metrics", "metrics"]
    summary = client.get(
        f"/api/study/project-training/sessions/{state['session']['id']}/summary"
    )
    assert summary.status_code == 200

    resumed = client.post(
        "/api/study/project-training/sessions",
        json={
            "project_id": project_id,
            "creation_id": "creation-max-turns",
            "difficulty": "medium",
            "max_turns": 3,
        },
    )
    assert resumed.status_code == 200
    assert resumed.json()["session"]["id"] == state["session"]["id"]
    assert resumed.json()["current_turn"] is None


def test_closing_session_and_checking_active_evaluation_is_atomic() -> None:
    repository = InMemoryProjectTrainingRepository()
    owner_id = "feishu:ou_close_race"
    project = repository.create_project(owner_id, _project_payload())
    outcome = ProjectTrainingWorkflow(repository, FakeProjectInterviewLLM()).start(
        owner_id,
        project.id,
        creation_id="creation-close-race",
        max_turns=6,
    )
    repository.begin_answer(
        owner_id,
        outcome.session.id,
        outcome.current_turn.id,
        "submission-close-race",
        "我负责订单链路",
        ProjectTrainingAnswerSource.TEXT,
        outcome.session.started_at,
    )

    with pytest.raises(ProjectTrainingSubmissionInProgressError):
        repository.close_session(
            owner_id,
            outcome.session.id,
            ProjectTrainingSessionStatus.ABANDONED,
            outcome.session.started_at,
        )

    assert repository.get_session(owner_id, outcome.session.id).status.value == "in_progress"


def test_failed_ai_evaluation_releases_turn_and_same_submission_can_retry(
    monkeypatch,
) -> None:
    repository = InMemoryProjectTrainingRepository()

    class RecoveringLLM:
        available = False

        async def generate_text(self, *args, **kwargs):
            if not self.available:
                return LLMResult("fake", "fake", "not-json", {})
            return await FakeProjectInterviewLLM().generate_text(*args, **kwargs)

    llm = RecoveringLLM()
    workflow = ProjectTrainingWorkflow(repository, llm)
    monkeypatch.setattr(
        project_training_dashboard,
        "get_default_project_training_repository",
        lambda: repository,
    )
    monkeypatch.setattr(
        project_training_dashboard,
        "get_project_training_workflow",
        lambda: workflow,
    )
    settings = Settings(
        debug_routes_enabled=False,
        feishu_app_secret="project-secret",
        dashboard_public_base_url="http://testserver",
    )
    client = TestClient(create_app(settings), follow_redirects=False)
    client.cookies.set(
        "offerpilot_dashboard_session",
        DashboardTokenSigner("project-secret").issue(
            purpose="dashboard_session",
            claims={"open_id": "ou_recover"},
            ttl_seconds=3600,
        ),
    )
    project_id = client.post(
        "/api/study/projects", json=_project_payload()
    ).json()["project"]["id"]
    started = client.post(
        "/api/study/project-training/sessions",
        json={
            "project_id": project_id,
            "creation_id": "creation-ai-recovery",
            "difficulty": "medium",
            "max_turns": 3,
        },
    ).json()
    answer = {
        "turn_id": started["current_turn"]["id"],
        "submission_id": "submission-ai-recovery",
        "answer_text": "我负责订单创建链路，使用 Redis 做幂等。",
        "answer_source": "text",
    }

    failed = client.post(
        f"/api/study/project-training/sessions/{started['session']['id']}/answers",
        json=answer,
    )
    llm.available = True
    recovered = client.post(
        f"/api/study/project-training/sessions/{started['session']['id']}/answers",
        json=answer,
    )

    assert failed.status_code == 503
    assert "稍后重试" in failed.json()["detail"]
    assert recovered.status_code == 200
    assert recovered.json()["answer"]["submission_id"] == "submission-ai-recovery"
