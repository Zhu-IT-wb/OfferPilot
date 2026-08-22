from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import main as main_module
from app.core.config import Settings
from app.main import app


def test_health_check_returns_ok() -> None:
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_application_lifespan_starts_and_stops_services(monkeypatch) -> None:
    events = []

    def service_type(name):
        class FakeService:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def start(self):
                events.append(f"{name}.start")

            async def stop(self):
                events.append(f"{name}.stop")

        return FakeService

    class FakeRunner:
        def start(self):
            events.append("discovery.start")

        async def stop(self):
            events.append("discovery.stop")

    def sync_knowledge(_settings):
        events.append("knowledge.sync")

    def ensure_subscription(**_kwargs):
        events.append("subscription.ensure")
        return SimpleNamespace(
            subscribed=False,
            status="disabled",
            app_token=None,
            error=None,
        )

    async def close_mcp_client():
        events.append("mcp.close")

    discovery_services = SimpleNamespace(workflow=object(), runner=FakeRunner())
    monkeypatch.setattr(main_module, "sync_default_knowledge_repository", sync_knowledge)
    monkeypatch.setattr(
        main_module,
        "ensure_bitable_event_subscription",
        ensure_subscription,
    )
    monkeypatch.setattr(
        main_module,
        "build_project_discovery_services",
        lambda *_args, **_kwargs: discovery_services,
    )
    monkeypatch.setattr(
        main_module,
        "BitablePullSyncService",
        service_type("bitable"),
    )
    monkeypatch.setattr(
        main_module,
        "LeetCodePushService",
        service_type("leetcode"),
    )
    monkeypatch.setattr(
        main_module,
        "KnowledgePushService",
        service_type("knowledge"),
    )
    monkeypatch.setattr(main_module, "close_default_mcp_client", close_mcp_client)

    test_app = main_module.create_app(
        Settings(debug_routes_enabled=False, project_discovery_enabled=True)
    )
    with TestClient(test_app) as client:
        assert client.get("/api/health").status_code == 200
        assert events == [
            "knowledge.sync",
            "subscription.ensure",
            "bitable.start",
            "leetcode.start",
            "knowledge.start",
            "discovery.start",
        ]

    assert events[-5:] == [
        "mcp.close",
        "bitable.stop",
        "leetcode.stop",
        "knowledge.stop",
        "discovery.stop",
    ]

