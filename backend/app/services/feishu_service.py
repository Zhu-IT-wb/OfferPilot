import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import httpx

from app.core.config import settings


class FeishuConfigurationError(RuntimeError):
    """Raised when Feishu app credentials are not configured."""


class FeishuRequestError(RuntimeError):
    """Raised when Feishu OpenAPI returns or causes an error."""


@dataclass(frozen=True)
class FeishuMessageResult:
    message_id: Optional[str]
    raw_response: Dict[str, Any]


@dataclass(frozen=True)
class FeishuCalendarEventResult:
    event_id: Optional[str]
    raw_response: Dict[str, Any]


@dataclass(frozen=True)
class FeishuCalendarResult:
    calendar_id: Optional[str]
    raw_response: Dict[str, Any]


@dataclass(frozen=True)
class FeishuCalendarAttendeeResult:
    attendee_ids: List[str]
    raw_response: Dict[str, Any]


class FeishuMessageService:
    def __init__(
        self,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.app_id = app_id if app_id is not None else settings.feishu_app_id
        self.app_secret = app_secret if app_secret is not None else settings.feishu_app_secret
        self.base_url = (base_url or settings.feishu_api_base_url).rstrip("/")
        self.timeout_seconds = timeout_seconds or settings.feishu_timeout_seconds
        self._tenant_access_token: Optional[str] = None
        self._tenant_access_token_expires_at = 0.0

    def is_configured(self) -> bool:
        return bool(self.app_id and self.app_secret)

    async def send_text_message(
        self,
        receive_id: str,
        text: str,
        receive_id_type: str = "open_id",
    ) -> FeishuMessageResult:
        token = await self.get_tenant_access_token()
        payload = {
            "receive_id": receive_id,
            "msg_type": "text",
            "content": json.dumps({"text": text}, ensure_ascii=False),
        }
        response_data = await self._post_json(
            path="/im/v1/messages",
            payload=payload,
            headers={"Authorization": f"Bearer {token}"},
            params={"receive_id_type": receive_id_type},
        )
        self._raise_for_feishu_code(response_data)

        data = response_data.get("data")
        message_id = data.get("message_id") if isinstance(data, dict) else None
        return FeishuMessageResult(
            message_id=message_id,
            raw_response=response_data,
        )

    async def get_tenant_access_token(self) -> str:
        if not self.is_configured():
            raise FeishuConfigurationError(
                "Feishu app credentials are not configured. Set FEISHU_APP_ID and FEISHU_APP_SECRET."
            )

        now = time.time()
        if self._tenant_access_token and now < self._tenant_access_token_expires_at:
            return self._tenant_access_token

        response_data = await self._post_json(
            path="/auth/v3/tenant_access_token/internal",
            payload={
                "app_id": self.app_id,
                "app_secret": self.app_secret,
            },
        )
        self._raise_for_feishu_code(response_data)

        token = response_data.get("tenant_access_token")
        if not isinstance(token, str) or not token:
            raise FeishuRequestError("Feishu token response does not contain tenant_access_token.")

        expire_seconds = response_data.get("expire", 7200)
        if not isinstance(expire_seconds, int):
            expire_seconds = 7200

        self._tenant_access_token = token
        self._tenant_access_token_expires_at = now + max(expire_seconds - 60, 60)
        return token

    def get_tenant_access_token_sync(self) -> str:
        if not self.is_configured():
            raise FeishuConfigurationError(
                "Feishu app credentials are not configured. Set FEISHU_APP_ID and FEISHU_APP_SECRET."
            )

        now = time.time()
        if self._tenant_access_token and now < self._tenant_access_token_expires_at:
            return self._tenant_access_token

        response_data = self._post_json_sync(
            path="/auth/v3/tenant_access_token/internal",
            payload={
                "app_id": self.app_id,
                "app_secret": self.app_secret,
            },
        )
        self._raise_for_feishu_code(response_data)

        token = response_data.get("tenant_access_token")
        if not isinstance(token, str) or not token:
            raise FeishuRequestError("Feishu token response does not contain tenant_access_token.")

        expire_seconds = response_data.get("expire", 7200)
        if not isinstance(expire_seconds, int):
            expire_seconds = 7200

        self._tenant_access_token = token
        self._tenant_access_token_expires_at = now + max(expire_seconds - 60, 60)
        return token

    async def _post_json(
        self,
        path: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers=request_headers,
                    params=params,
                )
                response.raise_for_status()
                response_data = response.json()
        except httpx.HTTPStatusError as exc:
            raise FeishuRequestError(
                f"Feishu OpenAPI returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise FeishuRequestError(f"Feishu OpenAPI request failed: {exc}") from exc
        except ValueError as exc:
            raise FeishuRequestError("Feishu OpenAPI response is not valid JSON.") from exc

        if not isinstance(response_data, dict):
            raise FeishuRequestError("Feishu OpenAPI response must be a JSON object.")

        return response_data

    def _post_json_sync(
        self,
        path: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    url,
                    json=payload,
                    headers=request_headers,
                    params=params,
                )
                response.raise_for_status()
                response_data = response.json()
        except httpx.HTTPStatusError as exc:
            raise FeishuRequestError(
                f"Feishu OpenAPI returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise FeishuRequestError(f"Feishu OpenAPI request failed: {exc}") from exc
        except ValueError as exc:
            raise FeishuRequestError("Feishu OpenAPI response is not valid JSON.") from exc

        if not isinstance(response_data, dict):
            raise FeishuRequestError("Feishu OpenAPI response must be a JSON object.")

        return response_data

    @staticmethod
    def _raise_for_feishu_code(response_data: Dict[str, Any]) -> None:
        code = response_data.get("code", 0)
        if code not in (0, None):
            message = response_data.get("msg") or response_data.get("message") or "Unknown Feishu OpenAPI error."
            raise FeishuRequestError(f"Feishu OpenAPI returned code {code}: {message}")


class FeishuCalendarService(FeishuMessageService):
    def __init__(
        self,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        calendar_id: Optional[str] = None,
        timezone: Optional[str] = None,
        event_duration_minutes: Optional[int] = None,
        sync_enabled: Optional[bool] = None,
        auto_create_enabled: Optional[bool] = None,
        managed_calendar_summary: Optional[str] = None,
        managed_calendar_description: Optional[str] = None,
        managed_calendar_permissions: Optional[str] = None,
    ) -> None:
        super().__init__(
            app_id=app_id,
            app_secret=app_secret,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        self.calendar_id = calendar_id if calendar_id is not None else settings.feishu_calendar_id
        self.timezone = timezone if timezone is not None else settings.feishu_calendar_timezone
        self.event_duration_minutes = (
            event_duration_minutes
            if event_duration_minutes is not None
            else settings.feishu_interview_event_duration_minutes
        )
        self.sync_enabled = (
            sync_enabled
            if sync_enabled is not None
            else settings.feishu_calendar_sync_enabled
        )
        self.auto_create_enabled = (
            auto_create_enabled
            if auto_create_enabled is not None
            else settings.feishu_calendar_auto_create_enabled
        )
        self.managed_calendar_summary = (
            managed_calendar_summary
            if managed_calendar_summary is not None
            else settings.feishu_offerpilot_calendar_summary
        )
        self.managed_calendar_description = (
            managed_calendar_description
            if managed_calendar_description is not None
            else settings.feishu_offerpilot_calendar_description
        )
        self.managed_calendar_permissions = (
            managed_calendar_permissions
            if managed_calendar_permissions is not None
            else settings.feishu_offerpilot_calendar_permissions
        )

    def is_calendar_sync_enabled(self) -> bool:
        return self.sync_enabled and self.is_configured() and bool(
            self.calendar_id or self.auto_create_enabled
        )

    def should_manage_offerpilot_calendar(self) -> bool:
        return self.auto_create_enabled and self.calendar_id.strip().lower() in {"", "primary"}

    def create_shared_calendar(
        self,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        permissions: Optional[str] = None,
    ) -> FeishuCalendarResult:
        if not self.sync_enabled:
            raise FeishuConfigurationError("Feishu calendar sync is not enabled.")

        token = self.get_tenant_access_token_sync()
        response_data = self._post_json_sync(
            path="/calendar/v4/calendars",
            payload={
                "summary": summary or self.managed_calendar_summary,
                "description": description or self.managed_calendar_description,
                "permissions": permissions or self.managed_calendar_permissions,
                "summary_alias": summary or self.managed_calendar_summary,
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        self._raise_for_feishu_code(response_data)

        return FeishuCalendarResult(
            calendar_id=_extract_calendar_id(response_data),
            raw_response=response_data,
        )

    def create_interview_event(
        self,
        company: str,
        role: Optional[str],
        round_name: str,
        start_time_text: Optional[str],
        start_at: Optional[Any] = None,
        reminder_minutes: int = 30,
        description: str = "",
        now: Optional[datetime] = None,
        calendar_id: Optional[str] = None,
    ) -> FeishuCalendarEventResult:
        if not self.is_calendar_sync_enabled():
            raise FeishuConfigurationError(
                "Feishu calendar sync is not enabled. Set FEISHU_CALENDAR_SYNC_ENABLED=true and FEISHU_CALENDAR_ID."
            )

        selected_calendar_id = calendar_id or self.calendar_id
        if not selected_calendar_id:
            raise FeishuConfigurationError("Feishu calendar id is missing.")

        event_start_at = _coerce_event_start_at(start_at, self.timezone)
        if event_start_at is None:
            event_start_at = parse_chinese_datetime(
                text=start_time_text,
                timezone=self.timezone,
                now=now,
            )
        if event_start_at is None:
            raise FeishuRequestError(f"Cannot parse interview start time: {start_time_text or start_at}")

        end_at = event_start_at + timedelta(minutes=max(self.event_duration_minutes, 1))
        summary = _build_interview_event_summary(company=company, role=role, round_name=round_name)
        payload = {
            "summary": summary,
            "description": description,
            "start_time": {
                "timestamp": str(int(event_start_at.timestamp())),
                "timezone": self.timezone,
            },
            "end_time": {
                "timestamp": str(int(end_at.timestamp())),
                "timezone": self.timezone,
            },
            "reminders": [
                {
                    "minutes": reminder_minutes,
                }
            ],
        }
        token = self.get_tenant_access_token_sync()
        response_data = self._post_json_sync(
            path=f"/calendar/v4/calendars/{selected_calendar_id}/events",
            payload=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        self._raise_for_feishu_code(response_data)

        return FeishuCalendarEventResult(
            event_id=_extract_calendar_event_id(response_data),
            raw_response=response_data,
        )

    def add_event_attendee(
        self,
        calendar_id: str,
        event_id: str,
        user_id: str,
        user_id_type: str = "open_id",
        need_notification: bool = True,
    ) -> FeishuCalendarAttendeeResult:
        if not self.is_calendar_sync_enabled():
            raise FeishuConfigurationError("Feishu calendar sync is not enabled.")
        if not calendar_id:
            raise FeishuConfigurationError("Feishu calendar id is missing.")
        if not event_id:
            raise FeishuConfigurationError("Feishu event id is missing.")
        if not user_id:
            raise FeishuConfigurationError("Feishu attendee user id is missing.")

        token = self.get_tenant_access_token_sync()
        response_data = self._post_json_sync(
            path=f"/calendar/v4/calendars/{calendar_id}/events/{event_id}/attendees",
            payload={
                "attendees": [
                    {
                        "type": "user",
                        "user_id": user_id,
                    }
                ],
                "need_notification": need_notification,
            },
            headers={"Authorization": f"Bearer {token}"},
            params={"user_id_type": user_id_type},
        )
        self._raise_for_feishu_code(response_data)

        return FeishuCalendarAttendeeResult(
            attendee_ids=_extract_attendee_ids(response_data),
            raw_response=response_data,
        )


def _coerce_event_start_at(start_at: Optional[Any], timezone: str) -> Optional[datetime]:
    if start_at is None:
        return None

    tz = ZoneInfo(timezone)
    if isinstance(start_at, datetime):
        return start_at.astimezone(tz) if start_at.tzinfo else start_at.replace(tzinfo=tz)

    if isinstance(start_at, str) and start_at.strip():
        try:
            parsed = datetime.fromisoformat(start_at.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.astimezone(tz) if parsed.tzinfo else parsed.replace(tzinfo=tz)

    return None


def parse_chinese_datetime(
    text: Optional[str],
    timezone: str = "Asia/Shanghai",
    now: Optional[datetime] = None,
) -> Optional[datetime]:
    if not text:
        return None

    tz = ZoneInfo(timezone)
    current = now.astimezone(tz) if now else datetime.now(tz)
    compact = text.replace(" ", "")

    target_date = current.date()
    if "大后天" in compact:
        target_date = (current + timedelta(days=3)).date()
    elif "后天" in compact:
        target_date = (current + timedelta(days=2)).date()
    elif "明天" in compact:
        target_date = (current + timedelta(days=1)).date()
    elif "今天" in compact or "今晚" in compact:
        target_date = current.date()
    else:
        week_date = _parse_weekday_date(compact, current)
        if week_date is not None:
            target_date = week_date

    hour, minute = _parse_clock_time(compact)
    if hour is None:
        return None

    if _contains_any(compact, ("下午", "晚上", "今晚")) and hour < 12:
        hour += 12
    elif "中午" in compact and hour < 11:
        hour += 12

    return datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        hour,
        minute,
        tzinfo=tz,
    )

def _parse_weekday_date(text: str, current: datetime) -> Optional[Any]:
    weekday_index = _extract_weekday_index(text)
    if weekday_index is None:
        return None

    days_ahead = weekday_index - current.weekday()
    if days_ahead <= 0 or text.startswith("下周"):
        days_ahead += 7
    return (current + timedelta(days=days_ahead)).date()


def _extract_weekday_index(text: str) -> Optional[int]:
    weekday_by_char = {
        "一": 0,
        "二": 1,
        "三": 2,
        "四": 3,
        "五": 4,
        "六": 5,
        "日": 6,
        "天": 6,
    }
    for prefix in ("下周", "周", "星期", "礼拜"):
        if prefix not in text:
            continue
        tail = text.split(prefix, 1)[1]
        if not tail:
            return None
        return weekday_by_char.get(tail[0])
    return None


def _parse_clock_time(text: str) -> tuple[Optional[int], int]:
    import re

    match = re.search(r"([零一二三四五六七八九十两\d]{1,3})点(?:(半)|([零一二三四五六七八九十\d]{1,2})分?)?", text)
    if not match:
        return None, 0

    hour = _parse_chinese_number(match.group(1))
    if hour is None or hour > 23:
        return None, 0

    if match.group(2):
        minute = 30
    elif match.group(3):
        parsed_minute = _parse_chinese_number(match.group(3))
        minute = parsed_minute if parsed_minute is not None else 0
    else:
        minute = 0

    if minute < 0 or minute > 59:
        minute = 0
    return hour, minute


def _parse_chinese_number(text: str) -> Optional[int]:
    if text.isdigit():
        return int(text)

    digit_by_char = {
        "零": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if text == "十":
        return 10
    if text.startswith("十"):
        tail = text[1:]
        return 10 + digit_by_char.get(tail, 0)
    if "十" in text:
        head, tail = text.split("十", 1)
        tens = digit_by_char.get(head)
        if tens is None:
            return None
        return tens * 10 + digit_by_char.get(tail, 0)
    if len(text) == 1:
        return digit_by_char.get(text)
    return None


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def _build_interview_event_summary(company: str, role: Optional[str], round_name: str) -> str:
    role_text = f" - {role}" if role else ""
    return f"OfferPilot 面试：{company}{role_text}（{round_name}）"


def _extract_calendar_event_id(response_data: Dict[str, Any]) -> Optional[str]:
    data = response_data.get("data")
    if isinstance(data, dict):
        event = data.get("event")
        if isinstance(event, dict) and isinstance(event.get("event_id"), str):
            return event["event_id"]
        if isinstance(data.get("event_id"), str):
            return data["event_id"]
    return None


def _extract_calendar_id(response_data: Dict[str, Any]) -> Optional[str]:
    data = response_data.get("data")
    if isinstance(data, dict):
        calendar = data.get("calendar")
        if isinstance(calendar, dict) and isinstance(calendar.get("calendar_id"), str):
            return calendar["calendar_id"]
        if isinstance(data.get("calendar_id"), str):
            return data["calendar_id"]
    return None


def _extract_attendee_ids(response_data: Dict[str, Any]) -> List[str]:
    data = response_data.get("data")
    if not isinstance(data, dict):
        return []

    attendees = data.get("attendees")
    if not isinstance(attendees, list):
        return []

    attendee_ids = []
    for attendee in attendees:
        if isinstance(attendee, dict) and isinstance(attendee.get("attendee_id"), str):
            attendee_ids.append(attendee["attendee_id"])
    return attendee_ids
