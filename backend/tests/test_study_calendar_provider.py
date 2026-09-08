import asyncio
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.routes import leetcode_dashboard
from app.core.config import Settings
from app.main import create_app
from app.models.study import StudyPriority, StudySession
from app.services.feishu_service import (
    FeishuCalendarBusyInterval,
    FeishuCalendarEventResult,
    FeishuCalendarService,
)
from app.services.feishu_user_credentials import (
    FeishuCredentialDecryptionError,
    FeishuTokenBundle,
    FeishuUserCredential,
    SQLiteFeishuUserCredentialRepository,
)
from app.services.leetcode_dashboard_auth import (
    DashboardTokenSigner,
    FEISHU_OAUTH_STATE_PURPOSE,
    FeishuDashboardAuthService,
    STUDY_CALENDAR_OWNER_TOKEN_PURPOSE,
)
from app.services.study_calendar_provider import (
    FeishuStudyCalendarProvider,
    StudyCalendarAuthorizationRequired,
    StudyCalendarCredentialRefreshError,
)


UTC = timezone.utc
NOW = datetime(2026, 9, 2, 8, 0, tzinfo=UTC)


def _credential(
    *,
    access_token: str = "access-secret",
    refresh_token: str = "refresh-secret",
    access_expires_at: datetime = NOW + timedelta(hours=1),
) -> FeishuUserCredential:
    return FeishuUserCredential(
        owner_id="feishu:ou_owner",
        open_id="ou_owner",
        access_token=access_token,
        refresh_token=refresh_token,
        access_expires_at=access_expires_at,
        refresh_expires_at=NOW + timedelta(days=30),
        scope="calendar:calendar:read calendar:calendar",
        updated_at=NOW,
    )


def _session() -> StudySession:
    return StudySession(
        id="study_session_1",
        owner_id="feishu:ou_owner",
        plan_id="study_plan_1",
        topic="系统设计",
        start_at=NOW + timedelta(days=1),
        end_at=NOW + timedelta(days=1, minutes=45),
        priority=StudyPriority.HIGH,
        source_refs=["gap:distributed-lock", "interview:1"],
        rationale="面试前优先补齐分布式锁知识点。",
    )


def test_sqlite_user_credentials_are_encrypted_and_owner_scoped(tmp_path) -> None:
    database_path = tmp_path / "credentials.db"
    repository = SQLiteFeishuUserCredentialRepository(
        str(database_path),
        "credential-encryption-secret",
    )

    saved = repository.save(_credential())
    reopened = SQLiteFeishuUserCredentialRepository(
        str(database_path),
        "credential-encryption-secret",
    )

    assert reopened.get("feishu:ou_owner") == saved
    assert reopened.get("feishu:ou_other") is None
    assert "access-secret" not in repr(saved)
    assert "refresh-secret" not in repr(saved)
    raw_database = database_path.read_bytes()
    assert b"access-secret" not in raw_database
    assert b"refresh-secret" not in raw_database

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT access_token_ciphertext, refresh_token_ciphertext "
            "FROM feishu_user_credentials WHERE owner_id = ?",
            ("feishu:ou_owner",),
        ).fetchone()
    assert row is not None
    assert row[0] != "access-secret"
    assert row[1] != "refresh-secret"


def test_sqlite_user_credentials_reject_wrong_encryption_secret(tmp_path) -> None:
    database_path = tmp_path / "credentials.db"
    SQLiteFeishuUserCredentialRepository(
        str(database_path),
        "correct-secret",
    ).save(_credential())

    wrong_repository = SQLiteFeishuUserCredentialRepository(
        str(database_path),
        "wrong-secret",
    )

    with pytest.raises(FeishuCredentialDecryptionError):
        wrong_repository.get("feishu:ou_owner")


def test_dashboard_oauth_exchanges_and_refreshes_full_token_bundle() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/authen/v1/user_info"):
            assert request.headers["authorization"] == "Bearer access-v1"
            return httpx.Response(
                200,
                json={"code": 0, "data": {"open_id": "ou_owner"}},
            )
        body = json.loads(request.content)
        if body["grant_type"] == "authorization_code":
            assert body["code"] == "oauth-code"
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "access_token": "access-v1",
                    "refresh_token": "refresh-v1",
                    "expires_in": 3600,
                    "refresh_token_expires_in": 7200,
                    "scope": "calendar:calendar:read calendar:calendar",
                    "token_type": "Bearer",
                },
            )
        assert body == {
            "grant_type": "refresh_token",
            "client_id": "cli_test",
            "client_secret": "app-secret",
            "refresh_token": "refresh-v1",
        }
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "access_token": "access-v2",
                    "refresh_token": "refresh-v2",
                    "expires_in": 1800,
                    "refresh_token_expires_in": 10800,
                    "scope": ["calendar:calendar:read", "calendar:calendar"],
                },
            },
        )

    service = FeishuDashboardAuthService(
        app_id="cli_test",
        app_secret="app-secret",
        api_base_url="https://open.feishu.test/open-apis",
        timeout_seconds=5,
        transport=httpx.MockTransport(handler),
    )

    credential = asyncio.run(
        service.exchange_code_bundle(
            "oauth-code",
            "https://offerpilot.test/leetcode/dashboard/auth/callback",
            now=NOW,
        )
    )
    refreshed = asyncio.run(
        service.refresh_access_token("refresh-v1", now=NOW + timedelta(hours=1))
    )

    assert credential.owner_id == "feishu:ou_owner"
    assert credential.open_id == "ou_owner"
    assert credential.access_token == "access-v1"
    assert credential.refresh_token == "refresh-v1"
    assert credential.access_expires_at == NOW + timedelta(hours=1)
    assert credential.refresh_expires_at == NOW + timedelta(hours=2)
    assert refreshed.access_token == "access-v2"
    assert refreshed.refresh_token == "refresh-v2"
    assert refreshed.access_expires_at == NOW + timedelta(hours=1, minutes=30)
    assert refreshed.refresh_expires_at == NOW + timedelta(hours=4)
    assert refreshed.scope == "calendar:calendar:read calendar:calendar"
    assert len(requests) == 3


def test_calendar_service_uses_user_token_for_freebusy_and_event(monkeypatch) -> None:
    calls = []
    service = FeishuCalendarService(
        app_id="cli_test",
        app_secret="app-secret",
        base_url="https://open.feishu.test/open-apis",
        timezone="Asia/Shanghai",
    )

    async def fake_post_json(path, payload, headers=None, params=None):
        calls.append(
            {"path": path, "payload": payload, "headers": headers, "params": params}
        )
        if path.endswith("/freebusy/list"):
            return {
                "code": 0,
                "data": {
                    "freebusy_list": [
                        {
                            "start_time": "2026-09-03T09:00:00+08:00",
                            "end_time": "2026-09-03T10:00:00+08:00",
                        },
                        {
                            "start_time": {"timestamp": "1788399000000"},
                            "end_time": {"timestamp": "1788400800000"},
                        },
                    ]
                },
            }
        return {"code": 0, "data": {"event": {"event_id": "evt_study_1"}}}

    monkeypatch.setattr(service, "_post_json", fake_post_json)
    start_at = datetime(2026, 9, 3, 0, 0, tzinfo=UTC)
    end_at = start_at + timedelta(days=1)

    busy = asyncio.run(
        service.list_user_freebusy("user-access", "ou_owner", start_at, end_at)
    )
    event = asyncio.run(
        service.create_user_study_event(
            user_access_token="user-access",
            topic="系统设计",
            start_at=start_at + timedelta(hours=2),
            end_at=start_at + timedelta(hours=3),
            description="复习分布式锁",
            operation_key="study_operation_1",
        )
    )

    assert len(busy) == 2
    assert busy[0].start_at.isoformat() == "2026-09-03T09:00:00+08:00"
    assert busy[1].start_at.tzinfo == UTC
    assert calls[0] == {
        "path": "/calendar/v4/freebusy/list",
        "payload": {
            "time_min": start_at.isoformat(),
            "time_max": end_at.isoformat(),
            "user_id": "ou_owner",
            "include_external_calendar": True,
        },
        "headers": {"Authorization": "Bearer user-access"},
        "params": {"user_id_type": "open_id"},
    }
    assert event.event_id == "evt_study_1"
    assert calls[1]["path"] == "/calendar/v4/calendars/primary/events"
    assert calls[1]["headers"] == {"Authorization": "Bearer user-access"}
    assert calls[1]["params"] == {"idempotency_key": "study_operation_1"}
    assert calls[1]["payload"]["summary"] == "OfferPilot 复习：系统设计"


def test_provider_exposes_authorization_url_when_owner_has_no_credential(tmp_path) -> None:
    repository = SQLiteFeishuUserCredentialRepository(
        str(tmp_path / "credentials.db"),
        "credential-secret",
    )
    calendar_calls = []

    class FakeCalendarService:
        async def list_user_freebusy(self, *args):
            calendar_calls.append(args)
            return []

    provider = FeishuStudyCalendarProvider(
        credential_repository=repository,
        auth_service=SimpleNamespace(),
        calendar_service=FakeCalendarService(),
        authorization_base_url="https://offerpilot.test",
        authorization_signing_secret="session-secret",
        clock=lambda: NOW,
    )

    with pytest.raises(StudyCalendarAuthorizationRequired) as exc_info:
        asyncio.run(
            provider.list_busy(
                "feishu:ou_missing",
                NOW,
                NOW + timedelta(days=1),
            )
        )

    auth_url = exc_info.value.authorization_url
    parsed = urlparse(auth_url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        "https://offerpilot.test/leetcode/dashboard/auth/start"
    )
    query = parse_qs(parsed.query)
    assert query["redirect"] == ["/leetcode/dashboard"]
    assert "feishu:ou_missing" not in auth_url
    owner_payload = DashboardTokenSigner("session-secret").verify(
        query["calendar_owner_token"][0],
        purpose=STUDY_CALENDAR_OWNER_TOKEN_PURPOSE,
    )
    assert owner_payload["owner_id"] == "feishu:ou_missing"
    assert calendar_calls == []


def test_provider_does_not_build_relative_authorization_url_without_public_base(
    tmp_path,
) -> None:
    repository = SQLiteFeishuUserCredentialRepository(
        str(tmp_path / "credentials.db"),
        "credential-secret",
    )

    class FakeCalendarService:
        async def list_user_freebusy(self, *args):
            raise AssertionError("calendar API must not be called without credentials")

    provider = FeishuStudyCalendarProvider(
        credential_repository=repository,
        auth_service=SimpleNamespace(),
        calendar_service=FakeCalendarService(),
        authorization_base_url="",
        clock=lambda: NOW,
    )

    assert provider.authorization_url("feishu:ou_missing") is None
    with pytest.raises(StudyCalendarAuthorizationRequired) as exc_info:
        asyncio.run(
            provider.list_busy(
                "feishu:ou_missing",
                NOW,
                NOW + timedelta(days=1),
            )
        )

    assert exc_info.value.authorization_url is None


def test_provider_refreshes_expired_credential_before_freebusy(tmp_path) -> None:
    repository = SQLiteFeishuUserCredentialRepository(
        str(tmp_path / "credentials.db"),
        "credential-secret",
    )
    repository.save(_credential(access_expires_at=NOW - timedelta(minutes=1)))
    refresh_calls = []
    calendar_calls = []

    class FakeAuthService:
        async def refresh_access_token(self, refresh_token, now=None):
            refresh_calls.append((refresh_token, now))
            return FeishuTokenBundle(
                access_token="access-refreshed",
                refresh_token="refresh-refreshed",
                access_expires_at=NOW + timedelta(hours=2),
                refresh_expires_at=NOW + timedelta(days=30),
                scope="calendar:calendar:read calendar:calendar",
            )

    class FakeCalendarService:
        async def list_user_freebusy(
            self, user_access_token, user_open_id, start_at, end_at
        ):
            calendar_calls.append(
                (user_access_token, user_open_id, start_at, end_at)
            )
            return [
                FeishuCalendarBusyInterval(
                    NOW + timedelta(hours=1),
                    NOW + timedelta(hours=2),
                )
            ]

    provider = FeishuStudyCalendarProvider(
        credential_repository=repository,
        auth_service=FakeAuthService(),
        calendar_service=FakeCalendarService(),
        clock=lambda: NOW,
    )

    busy = asyncio.run(
        provider.list_busy("feishu:ou_owner", NOW, NOW + timedelta(days=1))
    )

    assert refresh_calls == [("refresh-secret", NOW)]
    assert calendar_calls[0][0:2] == ("access-refreshed", "ou_owner")
    assert busy[0].start_at == NOW + timedelta(hours=1)
    persisted = repository.get("feishu:ou_owner")
    assert persisted is not None
    assert persisted.access_token == "access-refreshed"
    assert persisted.refresh_token == "refresh-refreshed"


def test_provider_classifies_refresh_transport_failure_before_calendar_write(tmp_path) -> None:
    repository = SQLiteFeishuUserCredentialRepository(
        str(tmp_path / "credentials.db"),
        "credential-secret",
    )
    repository.save(_credential(access_expires_at=NOW - timedelta(minutes=1)))
    calendar_calls = []

    class FailingAuthService:
        async def refresh_access_token(self, refresh_token, now=None):
            del refresh_token, now
            raise httpx.ConnectError("token endpoint unavailable")

    class FakeCalendarService:
        async def list_user_freebusy(self, *args):
            calendar_calls.append(args)
            return []

    provider = FeishuStudyCalendarProvider(
        credential_repository=repository,
        auth_service=FailingAuthService(),
        calendar_service=FakeCalendarService(),
        clock=lambda: NOW,
    )

    with pytest.raises(StudyCalendarCredentialRefreshError):
        asyncio.run(
            provider.list_busy(
                "feishu:ou_owner",
                NOW,
                NOW + timedelta(days=1),
            )
        )

    assert calendar_calls == []


def test_provider_creates_events_with_stable_operation_key(tmp_path) -> None:
    repository = SQLiteFeishuUserCredentialRepository(
        str(tmp_path / "credentials.db"),
        "credential-secret",
    )
    repository.save(_credential())
    calls = []

    class FakeCalendarService:
        async def create_user_study_event(self, **kwargs):
            calls.append(kwargs)
            return FeishuCalendarEventResult(
                event_id="evt_study_1",
                raw_response={"code": 0},
            )

    provider = FeishuStudyCalendarProvider(
        credential_repository=repository,
        auth_service=SimpleNamespace(),
        calendar_service=FakeCalendarService(),
        clock=lambda: NOW,
    )
    session = _session()

    first = asyncio.run(
        provider.create_study_event("feishu:ou_owner", session, "agent-operation")
    )
    second = asyncio.run(
        provider.create_study_event(
            "feishu:ou_owner",
            session,
            "different-agent-operation-after-replan",
        )
    )

    assert first.event_id == "evt_study_1"
    assert first.operation_key == second.operation_key
    assert first.operation_key.startswith("study_")
    assert calls[0]["operation_key"] == calls[1]["operation_key"]
    assert calls[0]["user_access_token"] == "access-secret"
    assert calls[0]["calendar_id"] == "primary"
    assert "gap:distributed-lock" in calls[0]["description"]


def test_provider_updates_and_deletes_existing_study_events(tmp_path) -> None:
    repository = SQLiteFeishuUserCredentialRepository(
        str(tmp_path / "credentials.db"),
        "credential-secret",
    )
    repository.save(_credential())
    calls = []

    class FakeCalendarService:
        async def update_user_study_event(self, **kwargs):
            calls.append(("update", kwargs))
            return FeishuCalendarEventResult(
                event_id=kwargs["event_id"],
                raw_response={"code": 0},
            )

        async def delete_user_calendar_event(self, **kwargs):
            calls.append(("delete", kwargs))
            return FeishuCalendarEventResult(
                event_id=kwargs["event_id"],
                raw_response={"code": 0},
            )

    provider = FeishuStudyCalendarProvider(
        credential_repository=repository,
        auth_service=SimpleNamespace(),
        calendar_service=FakeCalendarService(),
        clock=lambda: NOW,
    )
    session = _session()

    updated = asyncio.run(
        provider.update_study_event(
            "feishu:ou_owner", session, "evt_study_1", "update-operation"
        )
    )
    deleted = asyncio.run(
        provider.delete_study_event(
            "feishu:ou_owner", "evt_study_1", "delete-operation"
        )
    )

    assert updated.event_id == deleted.event_id == "evt_study_1"
    assert calls[0][0] == "update"
    assert calls[0][1]["user_access_token"] == "access-secret"
    assert calls[1] == (
        "delete",
        {
            "user_access_token": "access-secret",
            "event_id": "evt_study_1",
            "calendar_id": "primary",
        },
    )


def test_dashboard_oauth_callback_persists_calendar_credential(
    monkeypatch,
    tmp_path,
) -> None:
    credential = _credential()
    saved = []

    class FakeAuthService:
        async def exchange_code_bundle(self, code, redirect_uri):
            assert code == "oauth-code"
            assert redirect_uri == "http://testserver/leetcode/dashboard/auth/callback"
            return credential

    class FakeCredentialRepository:
        def save(self, value):
            saved.append(value)
            return value

    monkeypatch.setattr(
        leetcode_dashboard,
        "get_dashboard_auth_service",
        lambda request: FakeAuthService(),
    )
    monkeypatch.setattr(
        leetcode_dashboard,
        "get_feishu_user_credential_repository",
        lambda request: FakeCredentialRepository(),
    )
    settings = Settings(
        debug_routes_enabled=False,
        feishu_app_id="cli_test",
        feishu_app_secret="app-secret",
        dashboard_session_secret="session-secret",
        dashboard_public_base_url="",
        feishu_user_credentials_path=str(tmp_path / "credentials.db"),
        sqlite_path=str(tmp_path / "offerpilot.db"),
        agent_checkpoint_backend="memory",
        knowledge_markdown_sync_enabled=False,
        project_discovery_enabled=False,
    )
    oauth_nonce = "credential-callback-nonce"
    state = DashboardTokenSigner("session-secret").issue(
        purpose=FEISHU_OAUTH_STATE_PURPOSE,
        claims={
            "redirect": "/leetcode/dashboard",
            "nonce": oauth_nonce,
            "calendar_owner_id": credential.owner_id,
        },
        ttl_seconds=600,
    )

    with TestClient(create_app(settings), follow_redirects=False) as client:
        client.cookies.set(
            leetcode_dashboard._OAUTH_STATE_COOKIE,
            oauth_nonce,
            path=leetcode_dashboard._OAUTH_STATE_COOKIE_PATH,
        )
        response = client.get(
            "/leetcode/dashboard/auth/callback",
            params={
                "code": "oauth-code",
                "state": state,
                "calendar_owner_id": "api:caller_cannot_override_owner",
            },
        )

    assert response.status_code == 307
    assert response.headers["location"] == "/leetcode/dashboard"
    assert saved == [credential]
    assert "offerpilot_dashboard_session=" in response.headers["set-cookie"]
    assert "HttpOnly" in response.headers["set-cookie"]


def test_dashboard_oauth_callback_binds_calendar_credential_to_signed_api_owner(
    monkeypatch,
    tmp_path,
) -> None:
    credential = _credential()
    saved = []

    class FakeAuthService:
        async def exchange_code_bundle(self, code, redirect_uri):
            del code, redirect_uri
            return credential

    class FakeCredentialRepository:
        def save(self, value):
            saved.append(value)
            return value

    monkeypatch.setattr(
        leetcode_dashboard,
        "get_dashboard_auth_service",
        lambda request: FakeAuthService(),
    )
    monkeypatch.setattr(
        leetcode_dashboard,
        "get_feishu_user_credential_repository",
        lambda request: FakeCredentialRepository(),
    )
    settings = Settings(
        debug_routes_enabled=False,
        feishu_app_id="cli_test",
        feishu_app_secret="app-secret",
        dashboard_session_secret="session-secret",
        dashboard_public_base_url="",
        feishu_user_credentials_path=str(tmp_path / "credentials.db"),
        sqlite_path=str(tmp_path / "offerpilot.db"),
        agent_checkpoint_backend="memory",
        knowledge_markdown_sync_enabled=False,
        project_discovery_enabled=False,
    )
    oauth_nonce = "api-owner-callback-nonce"
    state = DashboardTokenSigner("session-secret").issue(
        purpose=FEISHU_OAUTH_STATE_PURPOSE,
        claims={
            "redirect": "/leetcode/dashboard",
            "calendar_owner_id": "api:local_user",
            "nonce": oauth_nonce,
        },
        ttl_seconds=600,
    )

    with TestClient(create_app(settings), follow_redirects=False) as client:
        client.cookies.set(
            leetcode_dashboard._OAUTH_STATE_COOKIE,
            oauth_nonce,
            path=leetcode_dashboard._OAUTH_STATE_COOKIE_PATH,
        )
        response = client.get(
            "/leetcode/dashboard/auth/callback",
            params={"code": "oauth-code", "state": state},
        )

    assert response.status_code == 307
    assert len(saved) == 1
    assert saved[0].owner_id == "api:local_user"
    assert saved[0].open_id == "ou_owner"
    assert saved[0].access_token == "access-secret"


def test_calendar_oauth_rejects_feishu_owner_account_mismatch(
    monkeypatch,
    tmp_path,
) -> None:
    saved = []

    class FakeAuthService:
        async def exchange_code_bundle(self, code, redirect_uri):
            del code, redirect_uri
            return _credential()

    class FakeCredentialRepository:
        def save(self, value):
            saved.append(value)
            return value

    monkeypatch.setattr(
        leetcode_dashboard,
        "get_dashboard_auth_service",
        lambda request: FakeAuthService(),
    )
    monkeypatch.setattr(
        leetcode_dashboard,
        "get_feishu_user_credential_repository",
        lambda request: FakeCredentialRepository(),
    )
    settings = Settings(
        debug_routes_enabled=False,
        feishu_app_id="cli_test",
        feishu_app_secret="app-secret",
        dashboard_session_secret="session-secret",
        dashboard_public_base_url="",
        feishu_user_credentials_path=str(tmp_path / "credentials.db"),
        sqlite_path=str(tmp_path / "offerpilot.db"),
        agent_checkpoint_backend="memory",
        knowledge_markdown_sync_enabled=False,
        project_discovery_enabled=False,
    )
    oauth_nonce = "mismatched-owner-callback-nonce"
    state = DashboardTokenSigner("session-secret").issue(
        purpose=FEISHU_OAUTH_STATE_PURPOSE,
        claims={
            "calendar_owner_id": "feishu:ou_someone_else",
            "nonce": oauth_nonce,
        },
        ttl_seconds=600,
    )

    with TestClient(create_app(settings), follow_redirects=False) as client:
        client.cookies.set(
            leetcode_dashboard._OAUTH_STATE_COOKIE,
            oauth_nonce,
            path=leetcode_dashboard._OAUTH_STATE_COOKIE_PATH,
        )
        response = client.get(
            "/leetcode/dashboard/auth/callback",
            params={"code": "oauth-code", "state": state},
        )

    assert response.status_code == 401
    assert saved == []
