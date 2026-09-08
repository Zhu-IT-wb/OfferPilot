from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.schemas.agent import AgentRunResponse, AgentRunStatus


def test_debug_routes_are_disabled_when_configured_off() -> None:
    app = create_app(Settings(environment="production", debug_routes_enabled=False))
    client = TestClient(app)

    health_response = client.get("/api/health")
    debug_response = client.post("/api/debug/agent", json={"message": "今天任务是什么？"})

    assert health_response.status_code == 200
    assert debug_response.status_code == 404


def test_debug_routes_are_enabled_when_configured_on() -> None:
    class FakeRuntime:
        async def start(self, **kwargs):
            assert kwargs["source"] == "debug"
            return AgentRunResponse(
                thread_id="agent_debug",
                run_id="run-debug",
                status=AgentRunStatus.COMPLETED,
                reply="今天没有待办任务。",
            )

    app = create_app(Settings(environment="local", debug_routes_enabled=True))
    app.state.agent_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.post("/api/debug/agent", json={"message": "今天任务是什么？"})

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["reply"] == "今天没有待办任务。"
