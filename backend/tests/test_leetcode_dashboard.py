from urllib.parse import parse_qs, urlparse

import asyncio
import httpx
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.leetcode_dashboard_auth import (
    DashboardTokenSigner,
    FeishuDashboardAuthService,
)
from app.api.routes import leetcode_dashboard
from app.repositories.leetcode_repository import InMemoryLeetCodeRepository
from app.repositories.sqlite_leetcode_repository import SQLiteLeetCodeRepository
from app.services.leetcode_catalog import load_hot100_snapshot


def test_dashboard_redirects_anonymous_user_to_feishu_login() -> None:
    app = create_app(
        Settings(
            debug_routes_enabled=False,
            feishu_app_id="cli_dashboard_test",
            feishu_app_secret="dashboard-secret",
            dashboard_public_base_url="",
        )
    )
    client = TestClient(app, follow_redirects=False)

    response = client.get("/leetcode/dashboard")

    assert response.status_code == 307
    assert response.headers["location"] == "/leetcode/dashboard/auth/start"


def test_dashboard_auth_start_redirects_to_feishu_with_signed_state() -> None:
    app = create_app(
        Settings(
            debug_routes_enabled=False,
            feishu_app_id="cli_dashboard_test",
            feishu_app_secret="dashboard-secret",
            dashboard_public_base_url="",
        )
    )
    client = TestClient(app, follow_redirects=False)

    response = client.get("/leetcode/dashboard/auth/start")

    assert response.status_code == 307
    location = urlparse(response.headers["location"])
    query = parse_qs(location.query)
    assert f"{location.scheme}://{location.netloc}{location.path}" == (
        "https://accounts.feishu.cn/open-apis/authen/v1/authorize"
    )
    assert query["client_id"] == ["cli_dashboard_test"]
    assert query["scope"] == ["auth:user.id:read"]
    assert "app_id" not in query
    assert query["redirect_uri"] == [
        "http://testserver/leetcode/dashboard/auth/callback"
    ]
    assert query["state"][0]
    assert "dashboard-secret" not in response.headers["location"]


def test_dashboard_auth_callback_sets_an_http_only_session_cookie(monkeypatch) -> None:
    class FakeFeishuDashboardAuthService:
        async def exchange_code(self, code: str, redirect_uri: str) -> str:
            assert code == "temporary-code"
            assert redirect_uri == "http://testserver/leetcode/dashboard/auth/callback"
            return "ou_dashboard_user"

    monkeypatch.setattr(
        leetcode_dashboard,
        "get_dashboard_auth_service",
        lambda request: FakeFeishuDashboardAuthService(),
    )
    settings = Settings(
        debug_routes_enabled=False,
        feishu_app_id="cli_dashboard_test",
        feishu_app_secret="dashboard-secret",
        dashboard_public_base_url="",
    )
    state = DashboardTokenSigner("dashboard-secret").issue(
        purpose="feishu_oauth_state",
        claims={"redirect": "/leetcode/dashboard"},
        ttl_seconds=600,
    )
    client = TestClient(create_app(settings), follow_redirects=False)

    response = client.get(
        "/leetcode/dashboard/auth/callback",
        params={"code": "temporary-code", "state": state},
    )

    assert response.status_code == 307
    assert response.headers["location"] == "/leetcode/dashboard"
    cookie = response.headers["set-cookie"]
    assert "offerpilot_dashboard_session=" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    assert "ou_dashboard_user" not in cookie


def test_feishu_oauth_exchanges_code_with_the_current_v2_contract() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/open-apis/authen/v2/oauth/token":
            return httpx.Response(
                200,
                json={"code": 0, "access_token": "u-test-token"},
            )
        if request.url.path == "/open-apis/authen/v1/user_info":
            assert request.headers["authorization"] == "Bearer u-test-token"
            return httpx.Response(
                200,
                json={"code": 0, "data": {"open_id": "ou_dashboard_user"}},
            )
        raise AssertionError(f"Unexpected OAuth request: {request.url}")

    service = FeishuDashboardAuthService(
        app_id="cli_dashboard_test",
        app_secret="dashboard-secret",
        api_base_url="https://open.feishu.cn/open-apis",
        timeout_seconds=5,
        transport=httpx.MockTransport(handler),
    )

    open_id = asyncio.run(
        service.exchange_code(
            "temporary-code",
            "https://offerpilot.example/leetcode/dashboard/auth/callback",
        )
    )

    assert open_id == "ou_dashboard_user"
    token_body = requests[0].content.decode("utf-8")
    assert '"client_id":"cli_dashboard_test"' in token_body
    assert '"client_secret":"dashboard-secret"' in token_body
    assert '"code":"temporary-code"' in token_body
    assert (
        '"redirect_uri":"https://offerpilot.example/leetcode/dashboard/auth/callback"'
        in token_body
    )


def test_authenticated_user_can_open_the_dashboard_page() -> None:
    settings = Settings(
        debug_routes_enabled=False,
        feishu_app_id="cli_dashboard_test",
        feishu_app_secret="dashboard-secret",
    )
    session = DashboardTokenSigner("dashboard-secret").issue(
        purpose="dashboard_session",
        claims={"open_id": "ou_dashboard_user"},
        ttl_seconds=3600,
    )
    client = TestClient(create_app(settings), follow_redirects=False)
    client.cookies.set("offerpilot_dashboard_session", session)

    response = client.get("/leetcode/dashboard")

    assert response.status_code == 200
    assert "OfferPilot 刷题计划" in response.text
    assert "/api/leetcode/dashboard/today" in response.text
    assert "/api/leetcode/dashboard/results" in response.text
    assert "独立完成" in response.text
    assert "明天继续" in response.text
    assert "ou_dashboard_user" not in response.text


def test_today_api_returns_three_assignments_for_the_authenticated_user(
    monkeypatch,
) -> None:
    repository = InMemoryLeetCodeRepository(
        problems=load_hot100_snapshot().problems
    )
    monkeypatch.setattr(
        leetcode_dashboard,
        "get_default_leetcode_repository",
        lambda: repository,
    )
    settings = Settings(
        debug_routes_enabled=False,
        feishu_app_id="cli_dashboard_test",
        feishu_app_secret="dashboard-secret",
    )
    session = DashboardTokenSigner("dashboard-secret").issue(
        purpose="dashboard_session",
        claims={"open_id": "ou_dashboard_user"},
        ttl_seconds=3600,
    )
    client = TestClient(create_app(settings))
    client.cookies.set("offerpilot_dashboard_session", session)

    response = client.get("/api/leetcode/dashboard/today")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"] == {"handled": 0, "total": 3}
    assert len(payload["items"]) == 3
    assert payload["items"][0]["assignment_id"]
    assert payload["items"][0]["problem"]["url"].startswith(
        "https://leetcode.cn/problems/"
    )
    assert "owner_id" not in response.text


def test_result_api_records_feedback_and_returns_the_updated_dashboard(
    monkeypatch,
) -> None:
    repository = InMemoryLeetCodeRepository(
        problems=load_hot100_snapshot().problems
    )
    monkeypatch.setattr(
        leetcode_dashboard,
        "get_default_leetcode_repository",
        lambda: repository,
    )
    settings = Settings(
        debug_routes_enabled=False,
        feishu_app_id="cli_dashboard_test",
        feishu_app_secret="dashboard-secret",
    )
    session = DashboardTokenSigner("dashboard-secret").issue(
        purpose="dashboard_session",
        claims={"open_id": "ou_dashboard_user"},
        ttl_seconds=3600,
    )
    client = TestClient(create_app(settings))
    client.cookies.set("offerpilot_dashboard_session", session)
    assignment_id = client.get("/api/leetcode/dashboard/today").json()["items"][0][
        "assignment_id"
    ]

    response = client.post(
        "/api/leetcode/dashboard/results",
        json={"assignment_id": assignment_id, "result": "independent"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "独立完成" in payload["message"]
    assert payload["dashboard"]["summary"] == {"handled": 1, "total": 3}
    updated = next(
        item
        for item in payload["dashboard"]["items"]
        if item["assignment_id"] == assignment_id
    )
    assert updated["result"] == "independent"
    assert updated["status"] == "completed"
    assert updated["progress"]["next_review_on"]
    reopened = client.get("/api/leetcode/dashboard/today").json()
    persisted = next(
        item
        for item in reopened["items"]
        if item["assignment_id"] == assignment_id
    )
    assert persisted["result"] == "independent"
    assert reopened["summary"] == {"handled": 1, "total": 3}


def test_result_api_cannot_modify_another_users_assignment(monkeypatch) -> None:
    repository = InMemoryLeetCodeRepository(
        problems=load_hot100_snapshot().problems
    )
    monkeypatch.setattr(
        leetcode_dashboard,
        "get_default_leetcode_repository",
        lambda: repository,
    )
    settings = Settings(
        debug_routes_enabled=False,
        feishu_app_id="cli_dashboard_test",
        feishu_app_secret="dashboard-secret",
    )
    app = create_app(settings)
    owner_client = TestClient(app)
    other_client = TestClient(app)
    owner_client.cookies.set(
        "offerpilot_dashboard_session",
        DashboardTokenSigner("dashboard-secret").issue(
            purpose="dashboard_session",
            claims={"open_id": "ou_owner"},
            ttl_seconds=3600,
        ),
    )
    other_client.cookies.set(
        "offerpilot_dashboard_session",
        DashboardTokenSigner("dashboard-secret").issue(
            purpose="dashboard_session",
            claims={"open_id": "ou_other"},
            ttl_seconds=3600,
        ),
    )
    assignment_id = owner_client.get("/api/leetcode/dashboard/today").json()["items"][
        0
    ]["assignment_id"]

    response = other_client.post(
        "/api/leetcode/dashboard/results",
        json={"assignment_id": assignment_id, "result": "independent"},
    )

    assert response.status_code == 400
    owner_dashboard = owner_client.get("/api/leetcode/dashboard/today").json()
    owner_assignment = next(
        item
        for item in owner_dashboard["items"]
        if item["assignment_id"] == assignment_id
    )
    assert owner_assignment["result"] is None
    assert owner_assignment["status"] == "pending"


def test_dashboard_result_survives_sqlite_repository_restart(
    monkeypatch,
    tmp_path,
) -> None:
    database = tmp_path / "offerpilot.db"
    repositories = {"current": SQLiteLeetCodeRepository(str(database))}
    monkeypatch.setattr(
        leetcode_dashboard,
        "get_default_leetcode_repository",
        lambda: repositories["current"],
    )
    settings = Settings(
        debug_routes_enabled=False,
        feishu_app_id="cli_dashboard_test",
        feishu_app_secret="dashboard-secret",
        dashboard_public_base_url="",
    )
    client = TestClient(create_app(settings))
    client.cookies.set(
        "offerpilot_dashboard_session",
        DashboardTokenSigner("dashboard-secret").issue(
            purpose="dashboard_session",
            claims={"open_id": "ou_dashboard_user"},
            ttl_seconds=3600,
        ),
    )
    assignment_id = client.get("/api/leetcode/dashboard/today").json()["items"][0][
        "assignment_id"
    ]
    response = client.post(
        "/api/leetcode/dashboard/results",
        json={"assignment_id": assignment_id, "result": "with_solution"},
    )
    assert response.status_code == 200

    repositories["current"] = SQLiteLeetCodeRepository(str(database))
    reopened = client.get("/api/leetcode/dashboard/today").json()

    persisted = next(
        item
        for item in reopened["items"]
        if item["assignment_id"] == assignment_id
    )
    assert persisted["result"] == "with_solution"
    assert persisted["progress"]["next_review_on"]
