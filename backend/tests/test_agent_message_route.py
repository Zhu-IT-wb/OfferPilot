import json

import pytest
from fastapi.testclient import TestClient

from app.agents.runtime import (
    AgentActorMismatch,
    AgentInteractionError,
    AgentThreadNotFound,
)
from app.core.config import Settings
from app.main import create_app
from app.schemas.agent import (
    AgentInteraction,
    AgentInteractionType,
    AgentRunResponse,
    AgentRunStatus,
    AgentToolExecution,
    AgentToolExecutionStatus,
)
from app.services.agent_api_auth import issue_agent_api_actor_token


_AUTH_SECRET = "agent-api-test-secret-with-at-least-32-bytes"


def _settings(**overrides) -> Settings:
    return Settings(
        debug_routes_enabled=False,
        agent_api_signing_secret=_AUTH_SECRET,
        **overrides,
    )


def _auth_headers(user_id: str = "local_user") -> dict[str, str]:
    token = issue_agent_api_actor_token(_AUTH_SECRET, user_id)
    return {"Authorization": f"Bearer {token}"}


def _response(*, status: AgentRunStatus = AgentRunStatus.COMPLETED) -> AgentRunResponse:
    return AgentRunResponse(
        thread_id="agent_0123456789abcdef",
        run_id="run-1",
        status=status,
        reply="本周有两场面试。",
    )


def test_start_agent_run_uses_shared_runtime() -> None:
    calls = []

    class FakeRuntime:
        async def start(self, **kwargs):
            calls.append(kwargs)
            return _response()

    app = create_app(_settings())
    app.state.agent_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.post(
        "/api/agent/runs",
        json={
            "message": "查询本周面试",
            "user_id": "will",
            "source": "api",
            "conversation_scope": "autumn-recruiting",
            "request_id": "request-1",
        },
        headers=_auth_headers("will"),
    )

    assert response.status_code == 200
    assert calls == [
        {
            "message": "查询本周面试",
            "user_id": "will",
            "source": "api",
            "conversation_scope": "autumn-recruiting",
            "request_id": "request-1",
        }
    ]
    assert response.json() == {
        "thread_id": "agent_0123456789abcdef",
        "run_id": "run-1",
        "status": "completed",
        "reply": "本周有两场面试。",
        "tool_executions": [],
        "artifacts": [],
        "warnings": [],
        "usage": {
            "model_calls": 0,
            "tool_calls": 0,
            "verifier_calls": 0,
        },
    }


def test_agent_http_response_hides_internal_tool_identifiers() -> None:
    class FakeRuntime:
        async def start(self, **_kwargs):
            return AgentRunResponse(
                thread_id="agent_public",
                run_id="run-public",
                status=AgentRunStatus.WAITING_FOR_INPUT,
                reply=(
                    "请确认 create_application；调用 call-private，"
                    "状态 write_outcome_unknown"
                ),
                interaction=AgentInteraction(
                    id="interaction-public",
                    type=AgentInteractionType.APPROVAL,
                    prompt="请确认以下操作：新增投递记录",
                    operation="新增投递记录",
                    tool_name="create_application",
                    arguments={"company": "百度", "role": "AI 应用工程师"},
                ),
                tool_executions=[
                    AgentToolExecution(
                        call_id="call-private",
                        name="create_application",
                        operation="新增投递记录",
                        status=AgentToolExecutionStatus.NEEDS_INPUT,
                        summary="等待确认",
                        error_code="approval_required",
                    )
                ],
                warnings=[
                    "create_application_degraded",
                    "write_outcome_unknown",
                ],
            )

    app = create_app(_settings())
    app.state.agent_runtime = FakeRuntime()
    response = TestClient(app).post(
        "/api/agent/runs",
        json={"message": "我投递了百度的 AI 应用工程师"},
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["interaction"]["operation"] == "新增投递记录"
    assert "tool_name" not in body["interaction"]
    assert "arguments" not in body["interaction"]
    assert body["tool_executions"][0]["operation"] == "新增投递记录"
    assert "name" not in body["tool_executions"][0]
    assert "call_id" not in body["tool_executions"][0]
    assert "error_code" not in body["tool_executions"][0]
    serialized = json.dumps(body, ensure_ascii=False)
    assert "create_application" not in serialized
    assert "call-private" not in serialized
    assert "write_outcome_unknown" not in serialized
    assert body["warnings"] == [
        "新增投递记录暂时降级",
        "外部写入结果暂未确认",
    ]

    schemas = app.openapi()["components"]["schemas"]
    interaction_fields = schemas["PublicAgentInteraction"]["properties"]
    execution_fields = schemas["PublicAgentToolExecution"]["properties"]
    assert "tool_name" not in interaction_fields
    assert "arguments" not in interaction_fields
    assert "name" not in execution_fields
    assert "call_id" not in execution_fields


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("message", "   "),
        ("user_id", "   "),
        ("source", "   "),
        ("conversation_scope", "   "),
        ("request_id", "   "),
    ],
)
def test_start_agent_run_rejects_whitespace_only_identifiers(field, value) -> None:
    class FakeRuntime:
        async def start(self, **_kwargs):
            return _response()

    app = create_app(_settings())
    app.state.agent_runtime = FakeRuntime()
    client = TestClient(app)
    payload = {"message": "查询本周面试", field: value}

    response = client.post(
        "/api/agent/runs",
        json=payload,
        headers=_auth_headers(),
    )

    assert response.status_code == 422


def test_start_agent_run_strips_surrounding_whitespace() -> None:
    calls = []

    class FakeRuntime:
        async def start(self, **kwargs):
            calls.append(kwargs)
            return _response()

    app = create_app(_settings())
    app.state.agent_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.post(
        "/api/agent/runs",
        json={
            "message": "  查询本周面试  ",
            "user_id": "  will  ",
            "source": "  api  ",
            "conversation_scope": "  autumn-recruiting  ",
            "request_id": "  request-1  ",
        },
        headers=_auth_headers("will"),
    )

    assert response.status_code == 200
    assert calls[0] == {
        "message": "查询本周面试",
        "user_id": "will",
        "source": "api",
        "conversation_scope": "autumn-recruiting",
        "request_id": "request-1",
    }


def test_resume_agent_run_passes_explicit_interaction() -> None:
    calls = []

    class FakeRuntime:
        async def resume(self, **kwargs):
            calls.append(kwargs)
            return _response()

    app = create_app(_settings())
    app.state.agent_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.post(
        "/api/agent/runs/agent_0123456789abcdef/resume",
        json={
            "interaction_id": "interaction-1",
            "decision": "approve",
            "value": {"timezone": "Asia/Shanghai"},
            "text": "确认同步",
            "user_id": "will",
            "source": "api",
            "request_id": "request-2",
        },
        headers=_auth_headers("will"),
    )

    assert response.status_code == 200
    assert calls == [
        {
            "thread_id": "agent_0123456789abcdef",
            "interaction_id": "interaction-1",
            "decision": "approve",
            "value": {"timezone": "Asia/Shanghai"},
            "text": "确认同步",
            "user_id": "will",
            "source": "api",
            "request_id": "request-2",
        }
    ]


def test_resume_agent_run_contract_remains_structured() -> None:
    calls = []

    class FakeRuntime:
        async def resume(self, **kwargs):
            calls.append(("resume", kwargs))
            return _response()

        async def resume_message(self, **kwargs):
            calls.append(("resume_message", kwargs))
            return _response()

    app = create_app(_settings())
    app.state.agent_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.post(
        "/api/agent/runs/agent_0123456789abcdef/resume",
        json={"message": "确认"},
        headers=_auth_headers(),
    )

    assert response.status_code == 422
    assert calls == []
    resume_schema = app.openapi()["components"]["schemas"]["AgentResumeRequest"]
    assert set(resume_schema["required"]) == {"interaction_id", "decision"}
    assert "message" not in resume_schema["properties"]


@pytest.mark.parametrize(
    "field",
    ["interaction_id", "text", "user_id", "source", "request_id"],
)
def test_resume_agent_run_rejects_whitespace_only_fields(field) -> None:
    class FakeRuntime:
        async def resume(self, **_kwargs):
            return _response()

    app = create_app(_settings())
    app.state.agent_runtime = FakeRuntime()
    client = TestClient(app)
    payload = {
        "interaction_id": "interaction-1",
        "decision": "approve",
        field: "   ",
    }

    response = client.post(
        "/api/agent/runs/agent_0123456789abcdef/resume",
        json=payload,
        headers=_auth_headers(),
    )

    assert response.status_code == 422


def test_get_agent_run_does_not_advance_runtime() -> None:
    calls = []

    class FakeRuntime:
        async def get_state(self, **kwargs):
            calls.append(kwargs)
            return _response(status=AgentRunStatus.WAITING_FOR_INPUT)

    app = create_app(_settings())
    app.state.agent_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.get(
        "/api/agent/runs/agent_0123456789abcdef",
        params={"user_id": "will", "source": "api"},
        headers=_auth_headers("will"),
    )

    assert response.status_code == 200
    assert response.json()["status"] == "waiting_for_input"
    assert calls == [
        {
            "thread_id": "agent_0123456789abcdef",
            "user_id": "will",
            "source": "api",
        }
    ]


def test_legacy_message_route_is_removed() -> None:
    app = create_app(_settings())
    client = TestClient(app)

    response = client.post("/api/agent/message", json={"message": "今天任务是什么？"})

    assert response.status_code == 404


def test_agent_route_reports_unavailable_runtime() -> None:
    app = create_app(_settings())
    client = TestClient(app)

    response = client.post(
        "/api/agent/runs",
        json={"message": "今天任务是什么？"},
        headers=_auth_headers(),
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Agent runtime is not available."}


def test_get_agent_run_maps_missing_thread_to_not_found() -> None:
    calls = []

    class FakeRuntime:
        async def get_state(self, **kwargs):
            calls.append(kwargs)
            raise AgentThreadNotFound("missing")

    app = create_app(_settings())
    app.state.agent_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.get(
        "/api/agent/runs/agent_missing",
        headers=_auth_headers(),
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Agent thread was not found."}
    assert calls == [
        {
            "thread_id": "agent_missing",
            "user_id": "local_user",
            "source": "api",
        }
    ]


def test_resume_agent_run_maps_stale_interaction_to_conflict() -> None:
    class FakeRuntime:
        async def resume(self, **_kwargs):
            raise AgentInteractionError("The interaction id is stale.")

    app = create_app(_settings())
    app.state.agent_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.post(
        "/api/agent/runs/agent_existing/resume",
        json={"interaction_id": "stale", "decision": "approve"},
        headers=_auth_headers(),
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "The interaction id is stale."}


def test_resume_agent_run_rejects_actor_mismatch() -> None:
    class FakeRuntime:
        async def resume(self, **_kwargs):
            raise AgentActorMismatch("The thread belongs to a different actor.")

    app = create_app(_settings())
    app.state.agent_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.post(
        "/api/agent/runs/agent_private/resume",
        json={"interaction_id": "interaction-1", "decision": "approve"},
        headers=_auth_headers(),
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "The thread belongs to a different actor."}


def test_get_agent_run_rejects_actor_mismatch() -> None:
    class FakeRuntime:
        async def get_state(self, **_kwargs):
            raise AgentActorMismatch("The thread belongs to a different actor.")

    app = create_app(_settings())
    app.state.agent_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.get(
        "/api/agent/runs/agent_private",
        headers=_auth_headers("other-user"),
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "The thread belongs to a different actor."}


def test_agent_api_requires_actor_token_before_runtime_execution() -> None:
    calls = []

    class FakeRuntime:
        async def start(self, **kwargs):
            calls.append(kwargs)
            return _response()

    app = create_app(_settings())
    app.state.agent_runtime = FakeRuntime()

    response = TestClient(app).post(
        "/api/agent/runs",
        json={"message": "查询本周面试"},
    )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert calls == []


@pytest.mark.parametrize(
    "authorization",
    ["Bearer invalid-token", "Basic credentials"],
)
def test_agent_api_rejects_invalid_actor_credentials(authorization: str) -> None:
    class FakeRuntime:
        async def start(self, **_kwargs):
            raise AssertionError("runtime must not execute")

    app = create_app(_settings())
    app.state.agent_runtime = FakeRuntime()

    response = TestClient(app).post(
        "/api/agent/runs",
        json={"message": "查询本周面试"},
        headers={"Authorization": authorization},
    )

    assert response.status_code == 401


def test_agent_api_rejects_expired_actor_token() -> None:
    token = issue_agent_api_actor_token(
        _AUTH_SECRET,
        "will",
        ttl_seconds=1,
        now=1,
    )
    app = create_app(_settings())
    app.state.agent_runtime = object()

    response = TestClient(app).get(
        "/api/agent/runs/agent_private",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401


def test_agent_api_is_unavailable_without_signing_secret() -> None:
    app = create_app(
        Settings(
            debug_routes_enabled=False,
            agent_api_signing_secret="",
        )
    )
    app.state.agent_runtime = object()

    response = TestClient(app).post(
        "/api/agent/runs",
        json={"message": "查询本周面试"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Agent API authentication is not configured."
    }


@pytest.mark.parametrize(
    ("identity_hint", "value"),
    [("user_id", "another-user"), ("source", "feishu")],
)
def test_agent_api_rejects_spoofed_legacy_identity_hints(
    identity_hint: str,
    value: str,
) -> None:
    calls = []

    class FakeRuntime:
        async def start(self, **kwargs):
            calls.append(kwargs)
            return _response()

    app = create_app(_settings())
    app.state.agent_runtime = FakeRuntime()

    response = TestClient(app).post(
        "/api/agent/runs",
        json={"message": "查询本周面试", identity_hint: value},
        headers=_auth_headers("will"),
    )

    assert response.status_code == 403
    assert response.json() == {
        "detail": "Request identity does not match the authenticated actor."
    }
    assert calls == []


def test_agent_api_uses_token_identity_when_legacy_hints_are_omitted() -> None:
    calls = []

    class FakeRuntime:
        async def start(self, **kwargs):
            calls.append(kwargs)
            return _response()

    app = create_app(_settings())
    app.state.agent_runtime = FakeRuntime()

    response = TestClient(app).post(
        "/api/agent/runs",
        json={"message": "查询本周面试", "request_id": "trusted-actor-1"},
        headers=_auth_headers("token-owner"),
    )

    assert response.status_code == 200
    assert calls == [
        {
            "message": "查询本周面试",
            "user_id": "token-owner",
            "source": "api",
            "conversation_scope": None,
            "request_id": "trusted-actor-1",
        }
    ]


def test_get_agent_run_rejects_query_identity_spoofing() -> None:
    class FakeRuntime:
        async def get_state(self, **_kwargs):
            raise AssertionError("runtime must not execute")

    app = create_app(_settings())
    app.state.agent_runtime = FakeRuntime()

    response = TestClient(app).get(
        "/api/agent/runs/agent_private",
        params={"user_id": "other-user"},
        headers=_auth_headers("token-owner"),
    )

    assert response.status_code == 403
