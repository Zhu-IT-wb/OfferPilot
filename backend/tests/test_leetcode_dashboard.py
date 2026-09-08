from urllib.parse import parse_qs, urlparse

import asyncio
import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.leetcode_dashboard_auth import (
    DashboardTokenSigner,
    FEISHU_OAUTH_STATE_PURPOSE,
    FeishuDashboardAuthService,
    STUDY_CALENDAR_OWNER_TOKEN_PURPOSE,
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
    assert set(query["scope"][0].split()) == {
        "auth:user.id:read",
    }
    assert "app_id" not in query
    assert query["redirect_uri"] == [
        "http://testserver/leetcode/dashboard/auth/callback"
    ]
    assert query["state"][0]
    assert "dashboard-secret" not in response.headers["location"]
    state_payload = DashboardTokenSigner("dashboard-secret").verify(
        query["state"][0],
        purpose=FEISHU_OAUTH_STATE_PURPOSE,
    )
    oauth_nonce = response.cookies.get(leetcode_dashboard._OAUTH_STATE_COOKIE)
    assert oauth_nonce == state_payload["nonce"]
    state_cookie = response.headers["set-cookie"]
    assert "HttpOnly" in state_cookie
    assert "SameSite=lax" in state_cookie
    assert f"Path={leetcode_dashboard._OAUTH_STATE_COOKIE_PATH}" in state_cookie


@pytest.mark.parametrize(
    "page", ["/leetcode/dashboard", "/study/knowledge", "/study/projects"]
)
def test_learning_page_login_does_not_request_legacy_calendar_scopes(page: str) -> None:
    settings = Settings(
        debug_routes_enabled=False,
        feishu_app_id="cli_dashboard_test",
        feishu_app_secret="dashboard-secret",
        dashboard_public_base_url="",
        dashboard_oauth_scope=(
            "auth:user.id:read offline_access calendar:calendar:read calendar:calendar"
        ),
    )
    client = TestClient(create_app(settings), follow_redirects=False)

    page_response = client.get(page)
    auth_response = client.get(page_response.headers["location"])

    assert auth_response.status_code == 307
    query = parse_qs(urlparse(auth_response.headers["location"]).query)
    assert query["scope"] == ["auth:user.id:read"]
    state = DashboardTokenSigner("dashboard-secret").verify(
        query["state"][0], purpose=FEISHU_OAUTH_STATE_PURPOSE
    )
    assert state["redirect"] == page
    assert "calendar_owner_id" not in state


def test_dashboard_auth_start_carries_only_a_signed_calendar_owner() -> None:
    settings = Settings(
        debug_routes_enabled=False,
        feishu_app_id="cli_dashboard_test",
        feishu_app_secret="dashboard-secret",
        dashboard_session_secret="session-secret",
        dashboard_public_base_url="",
    )
    signer = DashboardTokenSigner("session-secret")
    owner_token = signer.issue(
        purpose=STUDY_CALENDAR_OWNER_TOKEN_PURPOSE,
        claims={"owner_id": "api:local_user"},
        ttl_seconds=600,
    )
    client = TestClient(create_app(settings), follow_redirects=False)

    response = client.get(
        "/leetcode/dashboard/auth/start",
        params={"calendar_owner_token": owner_token},
    )

    assert response.status_code == 307
    oauth_query = parse_qs(urlparse(response.headers["location"]).query)
    state_payload = signer.verify(
        oauth_query["state"][0],
        purpose=FEISHU_OAUTH_STATE_PURPOSE,
    )
    assert state_payload["calendar_owner_id"] == "api:local_user"
    assert set(oauth_query["scope"][0].split()) == {
        "auth:user.id:read",
        "offline_access",
        "calendar:calendar:read",
        "calendar:calendar",
    }
    assert "calendar_owner_token" not in oauth_query
    assert "api:local_user" not in response.headers["location"]


def test_dashboard_auth_start_rejects_tampered_calendar_owner_token() -> None:
    settings = Settings(
        debug_routes_enabled=False,
        feishu_app_id="cli_dashboard_test",
        feishu_app_secret="dashboard-secret",
        dashboard_session_secret="session-secret",
        dashboard_public_base_url="",
    )
    owner_token = DashboardTokenSigner("session-secret").issue(
        purpose=STUDY_CALENDAR_OWNER_TOKEN_PURPOSE,
        claims={"owner_id": "api:local_user"},
        ttl_seconds=600,
    )
    replacement = "A" if owner_token[-1] != "A" else "B"
    tampered = owner_token[:-1] + replacement
    client = TestClient(create_app(settings), follow_redirects=False)

    response = client.get(
        "/leetcode/dashboard/auth/start",
        params={"calendar_owner_token": tampered},
    )

    assert response.status_code == 401


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
    oauth_nonce = "callback-browser-nonce"
    state = DashboardTokenSigner("dashboard-secret").issue(
        purpose=FEISHU_OAUTH_STATE_PURPOSE,
        claims={"redirect": "/leetcode/dashboard", "nonce": oauth_nonce},
        ttl_seconds=600,
    )
    client = TestClient(create_app(settings), follow_redirects=False)
    client.cookies.set(
        leetcode_dashboard._OAUTH_STATE_COOKIE,
        oauth_nonce,
        path=leetcode_dashboard._OAUTH_STATE_COOKIE_PATH,
    )

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
    assert any(
        leetcode_dashboard._OAUTH_STATE_COOKIE in value and "Max-Age=0" in value
        for value in response.headers.get_list("set-cookie")
    )


def test_learning_page_login_does_not_replace_calendar_credentials(monkeypatch) -> None:
    class FakeAuthService:
        async def exchange_code(self, code, redirect_uri):
            assert code == "identity-only-code"
            return "ou_dashboard_user"

    def unexpected_credential_access(request):
        raise AssertionError("Dashboard login must not access calendar credentials.")

    monkeypatch.setattr(
        leetcode_dashboard,
        "get_dashboard_auth_service",
        lambda request: FakeAuthService(),
    )
    monkeypatch.setattr(
        leetcode_dashboard,
        "get_feishu_user_credential_repository",
        unexpected_credential_access,
    )
    settings = Settings(
        debug_routes_enabled=False,
        feishu_app_id="cli_dashboard_test",
        feishu_app_secret="dashboard-secret",
        dashboard_public_base_url="",
    )
    client = TestClient(create_app(settings), follow_redirects=False)
    start = client.get(
        "/leetcode/dashboard/auth/start", params={"redirect": "/study/knowledge"}
    )
    state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]

    response = client.get(
        "/leetcode/dashboard/auth/callback",
        params={
            "code": "identity-only-code",
            "state": state,
            "calendar_owner_id": "feishu:ou_dashboard_user",
        },
    )

    assert response.status_code == 307
    assert response.headers["location"] == "/study/knowledge"
    session = DashboardTokenSigner("dashboard-secret").verify(
        response.cookies.get("offerpilot_dashboard_session"),
        purpose="dashboard_session",
    )
    assert session["open_id"] == "ou_dashboard_user"


def test_dashboard_oauth_callback_rejects_state_started_in_another_browser(
    monkeypatch,
) -> None:
    exchange_calls = []

    class FakeFeishuDashboardAuthService:
        async def exchange_code(self, code: str, redirect_uri: str) -> str:
            exchange_calls.append((code, redirect_uri))
            return "ou_attacker"

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
    app = create_app(settings)
    attacker = TestClient(app, follow_redirects=False)
    victim = TestClient(app, follow_redirects=False)
    attacker_start = attacker.get("/leetcode/dashboard/auth/start")
    victim.get("/leetcode/dashboard/auth/start")
    attacker_state = parse_qs(
        urlparse(attacker_start.headers["location"]).query
    )["state"][0]

    response = victim.get(
        "/leetcode/dashboard/auth/callback",
        params={"code": "attacker-code", "state": attacker_state},
    )

    assert response.status_code == 401
    assert "this browser" in response.json()["detail"]
    assert exchange_calls == []
    assert any(
        leetcode_dashboard._OAUTH_STATE_COOKIE in value and "Max-Age=0" in value
        for value in response.headers.get_list("set-cookie")
    )


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
