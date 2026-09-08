import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Optional, Sequence
from urllib.parse import urlencode

from app.core.config import Settings, settings as default_settings
from app.models.study import StudySession
from app.services.feishu_service import FeishuCalendarService
from app.services.feishu_user_credentials import (
    FeishuUserCredential,
    FeishuUserCredentialRepository,
    SQLiteFeishuUserCredentialRepository,
)
from app.services.leetcode_dashboard_auth import (
    DashboardTokenSigner,
    FeishuDashboardAuthService,
    STUDY_CALENDAR_OWNER_TOKEN_PURPOSE,
)
from app.services.study_scheduler import BusyInterval


class StudyCalendarAuthorizationRequired(RuntimeError):
    def __init__(self, authorization_url: Optional[str]) -> None:
        super().__init__("Feishu user calendar authorization is required.")
        self.authorization_url = authorization_url


class StudyCalendarCredentialRefreshError(RuntimeError):
    pass


@dataclass(frozen=True)
class FeishuStudyCalendarReceipt:
    event_id: str
    calendar_id: str = "primary"
    operation_key: str = ""


class FeishuStudyCalendarProvider:
    def __init__(
        self,
        credential_repository: FeishuUserCredentialRepository,
        auth_service: FeishuDashboardAuthService,
        calendar_service: FeishuCalendarService,
        authorization_base_url: str = "",
        authorization_signing_secret: str = "",
        authorization_ttl_seconds: int = 10 * 60,
        refresh_skew_seconds: int = 300,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.credential_repository = credential_repository
        self.auth_service = auth_service
        self.calendar_service = calendar_service
        self.authorization_base_url = (authorization_base_url or "").strip().rstrip("/")
        self.authorization_signing_secret = authorization_signing_secret or ""
        self.authorization_ttl_seconds = max(int(authorization_ttl_seconds), 1)
        self.refresh_skew = timedelta(seconds=max(refresh_skew_seconds, 0))
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._refresh_locks: Dict[str, asyncio.Lock] = {}
        self._refresh_locks_guard = asyncio.Lock()

    async def list_busy(
        self,
        owner_id: str,
        start_at: datetime,
        end_at: datetime,
    ) -> Sequence[BusyInterval]:
        credential = await self._valid_credential(owner_id)
        intervals = await self.calendar_service.list_user_freebusy(
            credential.access_token,
            credential.open_id,
            start_at,
            end_at,
        )
        return [BusyInterval(item.start_at, item.end_at) for item in intervals]

    async def create_study_event(
        self,
        owner_id: str,
        session: StudySession,
        operation_key: str,
    ) -> FeishuStudyCalendarReceipt:
        if session.owner_id != owner_id:
            raise ValueError("Study session owner does not match the calendar owner.")
        credential = await self._valid_credential(owner_id)
        stable_key = self.stable_operation_key(owner_id, session.id, operation_key)
        description_parts = [part for part in (session.rationale.strip(),) if part]
        if session.source_refs:
            description_parts.append("来源：" + "、".join(session.source_refs))
        result = await self.calendar_service.create_user_study_event(
            user_access_token=credential.access_token,
            topic=session.topic,
            start_at=session.start_at,
            end_at=session.end_at,
            description="\n".join(description_parts),
            operation_key=stable_key,
            calendar_id="primary",
        )
        if not result.event_id:
            raise RuntimeError("Feishu study calendar event did not return an event_id.")
        return FeishuStudyCalendarReceipt(
            event_id=result.event_id,
            calendar_id="primary",
            operation_key=stable_key,
        )

    async def update_study_event(
        self,
        owner_id: str,
        session: StudySession,
        event_id: str,
        operation_key: str,
    ) -> FeishuStudyCalendarReceipt:
        if session.owner_id != owner_id:
            raise ValueError("Study session owner does not match the calendar owner.")
        credential = await self._valid_credential(owner_id)
        description_parts = [part for part in (session.rationale.strip(),) if part]
        if session.source_refs:
            description_parts.append("来源：" + "、".join(session.source_refs))
        result = await self.calendar_service.update_user_study_event(
            user_access_token=credential.access_token,
            event_id=event_id,
            topic=session.topic,
            start_at=session.start_at,
            end_at=session.end_at,
            description="\n".join(description_parts),
            calendar_id="primary",
        )
        return FeishuStudyCalendarReceipt(
            event_id=result.event_id or event_id,
            calendar_id="primary",
            operation_key=operation_key,
        )

    async def delete_study_event(
        self,
        owner_id: str,
        event_id: str,
        operation_key: str,
    ) -> FeishuStudyCalendarReceipt:
        credential = await self._valid_credential(owner_id)
        result = await self.calendar_service.delete_user_calendar_event(
            user_access_token=credential.access_token,
            event_id=event_id,
            calendar_id="primary",
        )
        return FeishuStudyCalendarReceipt(
            event_id=result.event_id or event_id,
            calendar_id="primary",
            operation_key=operation_key,
        )

    def authorization_url(self, owner_id: str) -> Optional[str]:
        normalized_owner = (owner_id or "").strip()
        if (
            not self.authorization_base_url
            or not self.authorization_signing_secret
            or not normalized_owner
        ):
            return None
        owner_token = DashboardTokenSigner(self.authorization_signing_secret).issue(
            purpose=STUDY_CALENDAR_OWNER_TOKEN_PURPOSE,
            claims={"owner_id": normalized_owner},
            ttl_seconds=self.authorization_ttl_seconds,
        )
        path = "/leetcode/dashboard/auth/start"
        query = urlencode(
            {
                "redirect": "/leetcode/dashboard",
                "calendar_owner_token": owner_token,
            }
        )
        return f"{self.authorization_base_url}{path}?{query}"

    @staticmethod
    def stable_operation_key(
        owner_id: str,
        session_id: str,
        operation_key: str,
    ) -> str:
        # Agent tool-call ids may change after replanning; a session's create key must not.
        del operation_key
        digest = hashlib.sha256(
            f"study-calendar:v1|{owner_id}|{session_id}".encode("utf-8")
        ).hexdigest()
        return f"study_{digest[:40]}"

    async def _valid_credential(self, owner_id: str) -> FeishuUserCredential:
        normalized_owner = (owner_id or "").strip()
        try:
            credential = await asyncio.to_thread(
                self.credential_repository.get,
                normalized_owner,
            )
        except Exception as exc:
            raise StudyCalendarCredentialRefreshError(
                "Feishu user calendar credential could not be read."
            ) from exc
        if credential is None:
            raise StudyCalendarAuthorizationRequired(self.authorization_url(normalized_owner))
        now = self._now()
        if credential.access_expires_at.astimezone(timezone.utc) > now + self.refresh_skew:
            return credential

        lock = await self._refresh_lock(normalized_owner)
        async with lock:
            try:
                credential = await asyncio.to_thread(
                    self.credential_repository.get,
                    normalized_owner,
                )
            except Exception as exc:
                raise StudyCalendarCredentialRefreshError(
                    "Feishu user calendar credential could not be read."
                ) from exc
            if credential is None:
                raise StudyCalendarAuthorizationRequired(
                    self.authorization_url(normalized_owner)
                )
            now = self._now()
            if (
                credential.access_expires_at.astimezone(timezone.utc)
                > now + self.refresh_skew
            ):
                return credential
            if not credential.refresh_token or (
                credential.refresh_expires_at is not None
                and credential.refresh_expires_at.astimezone(timezone.utc) <= now
            ):
                raise StudyCalendarAuthorizationRequired(
                    self.authorization_url(normalized_owner)
                )
            try:
                bundle = await self.auth_service.refresh_access_token(
                    credential.refresh_token,
                    now=now,
                )
            except Exception as exc:
                raise StudyCalendarCredentialRefreshError(
                    "Feishu user calendar credential refresh failed."
                ) from exc
            refreshed = FeishuUserCredential.from_token_bundle(
                owner_id=credential.owner_id,
                open_id=credential.open_id,
                bundle=bundle,
                updated_at=now,
            )
            try:
                await asyncio.to_thread(self.credential_repository.save, refreshed)
            except Exception as exc:
                raise StudyCalendarCredentialRefreshError(
                    "Refreshed Feishu user calendar credential could not be saved."
                ) from exc
            return refreshed

    async def _refresh_lock(self, owner_id: str) -> asyncio.Lock:
        async with self._refresh_locks_guard:
            return self._refresh_locks.setdefault(owner_id, asyncio.Lock())

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Study calendar provider clock must be timezone-aware.")
        return value.astimezone(timezone.utc)


def build_default_study_calendar_provider(
    settings: Settings = default_settings,
) -> FeishuStudyCalendarProvider:
    encryption_secret = (
        settings.feishu_user_credentials_secret
        or settings.dashboard_session_secret
        or settings.feishu_app_secret
    )
    credential_repository = SQLiteFeishuUserCredentialRepository(
        settings.feishu_user_credentials_path,
        encryption_secret,
    )
    return FeishuStudyCalendarProvider(
        credential_repository=credential_repository,
        auth_service=FeishuDashboardAuthService(
            app_id=settings.feishu_app_id,
            app_secret=settings.feishu_app_secret,
            api_base_url=settings.feishu_api_base_url,
            timeout_seconds=settings.feishu_timeout_seconds,
        ),
        calendar_service=FeishuCalendarService(
            app_id=settings.feishu_app_id,
            app_secret=settings.feishu_app_secret,
            base_url=settings.feishu_api_base_url,
            timeout_seconds=settings.feishu_timeout_seconds,
            calendar_id=settings.feishu_calendar_id,
            timezone=settings.feishu_calendar_timezone,
            event_duration_minutes=settings.feishu_interview_event_duration_minutes,
            sync_enabled=settings.feishu_calendar_sync_enabled,
            auto_create_enabled=settings.feishu_calendar_auto_create_enabled,
        ),
        authorization_base_url=settings.dashboard_public_base_url,
        authorization_signing_secret=(
            settings.dashboard_session_secret or settings.feishu_app_secret
        ),
    )
