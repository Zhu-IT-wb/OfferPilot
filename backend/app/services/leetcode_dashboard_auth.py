import base64
import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import httpx

from app.services.feishu_user_credentials import (
    FeishuTokenBundle,
    FeishuUserCredential,
)


class DashboardTokenError(ValueError):
    pass


class FeishuDashboardAuthError(RuntimeError):
    pass


FEISHU_OAUTH_STATE_PURPOSE = "feishu_oauth_state"
STUDY_CALENDAR_OWNER_TOKEN_PURPOSE = "study_calendar_oauth_owner"


class DashboardTokenSigner:
    def __init__(self, secret: str) -> None:
        if not secret:
            raise DashboardTokenError("Dashboard session secret is not configured.")
        self.secret = secret.encode("utf-8")

    def issue(
        self,
        purpose: str,
        claims: Dict[str, Any],
        ttl_seconds: int,
        now: Optional[int] = None,
    ) -> str:
        issued_at = int(time.time() if now is None else now)
        payload = {
            "purpose": purpose,
            "iat": issued_at,
            "exp": issued_at + ttl_seconds,
            **claims,
        }
        encoded = _urlsafe_encode(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
        signature = _urlsafe_encode(
            hmac.new(self.secret, encoded.encode("ascii"), hashlib.sha256).digest()
        )
        return f"{encoded}.{signature}"

    def verify(
        self,
        token: str,
        purpose: str,
        now: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            encoded, signature = token.split(".", 1)
        except ValueError as exc:
            raise DashboardTokenError("Dashboard token is malformed.") from exc
        expected = _urlsafe_encode(
            hmac.new(self.secret, encoded.encode("ascii"), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(signature, expected):
            raise DashboardTokenError("Dashboard token signature is invalid.")
        try:
            payload = json.loads(_urlsafe_decode(encoded).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DashboardTokenError("Dashboard token payload is invalid.") from exc
        current_time = int(time.time() if now is None else now)
        if payload.get("purpose") != purpose or int(payload.get("exp", 0)) <= current_time:
            raise DashboardTokenError("Dashboard token is expired or has the wrong purpose.")
        return payload


class FeishuDashboardAuthService:
    def __init__(
        self,
        app_id: str,
        app_secret: str,
        api_base_url: str,
        timeout_seconds: float,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.app_id = app_id
        self.app_secret = app_secret
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def exchange_code(self, code: str, redirect_uri: str) -> str:
        credential = await self.exchange_code_bundle(code, redirect_uri)
        return credential.open_id

    async def exchange_code_bundle(
        self,
        code: str,
        redirect_uri: str,
        now: Optional[datetime] = None,
    ) -> FeishuUserCredential:
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            user_token_response = await client.post(
                f"{self.api_base_url}/authen/v2/oauth/token",
                json={
                    "grant_type": "authorization_code",
                    "client_id": self.app_id,
                    "client_secret": self.app_secret,
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
            )
            user_token_payload = _feishu_json(user_token_response)
            token_bundle = _token_bundle(user_token_payload, now=now)
            user_info_response = await client.get(
                f"{self.api_base_url}/authen/v1/user_info",
                headers={"Authorization": f"Bearer {token_bundle.access_token}"},
            )
            user_info_payload = _feishu_json(user_info_response)
            user_info = user_info_payload.get("data")
            open_id = user_info.get("open_id") if isinstance(user_info, dict) else None
            if not isinstance(open_id, str) or not open_id:
                raise FeishuDashboardAuthError(
                    "Feishu user information does not contain open_id."
                )
            return FeishuUserCredential.from_token_bundle(
                owner_id=f"feishu:{open_id}",
                open_id=open_id,
                bundle=token_bundle,
                updated_at=_utc_now(now),
            )

    async def refresh_access_token(
        self,
        refresh_token: str,
        now: Optional[datetime] = None,
    ) -> FeishuTokenBundle:
        if not refresh_token:
            raise FeishuDashboardAuthError("Feishu refresh_token is missing.")
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            response = await client.post(
                f"{self.api_base_url}/authen/v2/oauth/token",
                json={
                    "grant_type": "refresh_token",
                    "client_id": self.app_id,
                    "client_secret": self.app_secret,
                    "refresh_token": refresh_token,
                },
            )
        return _token_bundle(
            _feishu_json(response),
            now=now,
            fallback_refresh_token=refresh_token,
        )


def _token_bundle(
    payload: Dict[str, Any],
    now: Optional[datetime] = None,
    fallback_refresh_token: Optional[str] = None,
) -> FeishuTokenBundle:
    token_payload = payload.get("data")
    if not isinstance(token_payload, dict) or not token_payload.get("access_token"):
        token_payload = payload
    access_token = token_payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise FeishuDashboardAuthError(
            "Feishu user_access_token response is incomplete."
        )
    refresh_token = token_payload.get("refresh_token") or fallback_refresh_token
    if refresh_token is not None and not isinstance(refresh_token, str):
        raise FeishuDashboardAuthError("Feishu refresh_token response is invalid.")
    current = _utc_now(now)
    access_ttl = _positive_seconds(token_payload.get("expires_in"), 3600)
    refresh_ttl = _positive_seconds(
        token_payload.get("refresh_token_expires_in")
        or token_payload.get("refresh_expires_in"),
        30 * 24 * 60 * 60,
    )
    raw_scope = token_payload.get("scope") or ""
    scope = (
        " ".join(str(item) for item in raw_scope)
        if isinstance(raw_scope, list)
        else str(raw_scope)
    )
    return FeishuTokenBundle(
        access_token=access_token,
        refresh_token=refresh_token,
        access_expires_at=current + timedelta(seconds=access_ttl),
        refresh_expires_at=(
            current + timedelta(seconds=refresh_ttl) if refresh_token else None
        ),
        scope=scope,
        token_type=str(token_payload.get("token_type") or "Bearer"),
    )


def _positive_seconds(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _utc_now(value: Optional[datetime] = None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("OAuth token clock must be timezone-aware.")
    return current.astimezone(timezone.utc)


def _feishu_json(response: httpx.Response) -> Dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise FeishuDashboardAuthError(
            "Feishu authentication returned invalid JSON."
        ) from exc
    if not isinstance(payload, dict):
        raise FeishuDashboardAuthError(
            "Feishu authentication returned an unexpected payload."
        )
    if response.status_code >= 400 or payload.get("code", 0) != 0:
        message = payload.get("msg") or f"HTTP {response.status_code}"
        raise FeishuDashboardAuthError(f"Feishu authentication failed: {message}")
    return payload


def _urlsafe_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _urlsafe_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)
