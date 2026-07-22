import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import httpx

from app.core.config import settings


# 表示当前模块抛出的业务异常。
class FeishuConfigurationError(RuntimeError):
    """Raised when Feishu app credentials are not configured."""


# 表示当前模块抛出的业务异常。
class FeishuRequestError(RuntimeError):
    """Raised when Feishu OpenAPI returns or causes an error."""


# 承载外部服务调用后的结构化结果。
@dataclass(frozen=True)
class FeishuMessageResult:
    message_id: Optional[str]
    raw_response: Dict[str, Any]


# 承载外部服务调用后的结构化结果。
@dataclass(frozen=True)
class FeishuCalendarEventResult:
    event_id: Optional[str]
    raw_response: Dict[str, Any]


# 承载外部服务调用后的结构化结果。
@dataclass(frozen=True)
class FeishuCalendarResult:
    calendar_id: Optional[str]
    raw_response: Dict[str, Any]


# 承载外部服务调用后的结构化结果。
@dataclass(frozen=True)
class FeishuCalendarAttendeeResult:
    attendee_ids: List[str]
    raw_response: Dict[str, Any]


# 承载外部服务调用后的结构化结果。
@dataclass(frozen=True)
class FeishuBitableAppResult:
    app_token: Optional[str]
    raw_response: Dict[str, Any]


# 承载外部服务调用后的结构化结果。
@dataclass(frozen=True)
class FeishuBitableTableResult:
    table_id: Optional[str]
    raw_response: Dict[str, Any]


# 承载外部服务调用后的结构化结果。
@dataclass(frozen=True)
class FeishuBitableRecordResult:
    record_id: Optional[str]
    raw_response: Dict[str, Any]
    fields: Optional[Dict[str, Any]] = None


# 承载外部服务调用后的结构化结果。
@dataclass(frozen=True)
class FeishuBitableRecordListResult:
    records: List[FeishuBitableRecordResult]
    has_more: bool
    page_token: Optional[str]
    raw_response: Dict[str, Any]


# 承载外部服务调用后的结构化结果。
@dataclass(frozen=True)
class FeishuBitableCollaboratorResult:
    member_id: Optional[str]
    raw_response: Dict[str, Any]


# 承载外部服务调用后的结构化结果。
@dataclass(frozen=True)
class FeishuFileSubscriptionResult:
    file_token: str
    raw_response: Dict[str, Any]


# 封装飞书消息发送和基础 OpenAPI 调用。
class FeishuMessageService:
    # 初始化当前组件所需的依赖和配置。
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

    # 判断飞书服务必要配置是否完整。
    def is_configured(self) -> bool:
        return bool(self.app_id and self.app_secret)

    # 调用飞书接口发送文本消息。
    async def send_text_message(
        self,
        receive_id: str,
        text: str,
        receive_id_type: str = "open_id",
        idempotency_key: Optional[str] = None,
    ) -> FeishuMessageResult:
        token = await self.get_tenant_access_token()
        payload = {
            "receive_id": receive_id,
            "msg_type": "text",
            "content": json.dumps({"text": text}, ensure_ascii=False),
        }
        if idempotency_key:
            payload["uuid"] = idempotency_key
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

    # 异步获取飞书 tenant_access_token。
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

    # 同步获取飞书 tenant_access_token。
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

    # 发送 POST 请求处理 json。
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

    # 发送 POST 请求处理 json sync。
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

    # 发送 PUT 请求处理 json sync。
    def _put_json_sync(
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
                response = client.put(
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

    # 发送 PATCH 请求处理 json sync。
    def _patch_json_sync(
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
                response = client.patch(
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

    # 发送 DELETE 请求处理 json sync；兼容 204 空响应。
    def _delete_json_sync(
        self,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        request_headers: Dict[str, str] = {}
        if headers:
            request_headers.update(headers)

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.delete(url, headers=request_headers, params=params)
                response.raise_for_status()
                if not response.content:
                    return {"code": 0}
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

    # 获取 json sync。
    def _get_json_sync(
        self,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        request_headers: Dict[str, str] = {}
        if headers:
            request_headers.update(headers)

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.get(
                    url,
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

    # 处理 raise_for_feishu_code 相关逻辑。
    @staticmethod
    def _raise_for_feishu_code(response_data: Dict[str, Any]) -> None:
        code = response_data.get("code", 0)
        if code not in (0, None):
            message = response_data.get("msg") or response_data.get("message") or "Unknown Feishu OpenAPI error."
            raise FeishuRequestError(f"Feishu OpenAPI returned code {code}: {message}")


# 封装飞书日历创建、日程创建和参与人同步。
class FeishuCalendarService(FeishuMessageService):
    # 初始化当前组件所需的依赖和配置。
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

    # 判断飞书日历同步是否开启。
    def is_calendar_sync_enabled(self) -> bool:
        return self.sync_enabled and self.is_configured() and bool(
            self.calendar_id or self.auto_create_enabled
        )

    # 判断是否由 OfferPilot 自动管理共享日历。
    def should_manage_offerpilot_calendar(self) -> bool:
        return self.auto_create_enabled and self.calendar_id.strip().lower() in {"", "primary"}

    # 创建 OfferPilot 专用共享日历。
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

    # 在飞书日历中创建面试日程。
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

        payload = _build_interview_event_payload(
            company=company,
            role=role,
            round_name=round_name,
            event_start_at=event_start_at,
            timezone=self.timezone,
            duration_minutes=self.event_duration_minutes,
            reminder_minutes=reminder_minutes,
            description=description,
        )
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

    # 更新飞书日历中已有的面试日程。
    def update_interview_event(
        self,
        calendar_id: str,
        event_id: str,
        company: str,
        role: Optional[str],
        round_name: str,
        start_time_text: Optional[str],
        start_at: Optional[Any] = None,
        reminder_minutes: int = 30,
        description: str = "",
        now: Optional[datetime] = None,
    ) -> FeishuCalendarEventResult:
        if not self.is_calendar_sync_enabled():
            raise FeishuConfigurationError("Feishu calendar sync is not enabled.")
        if not calendar_id:
            raise FeishuConfigurationError("Feishu calendar id is missing.")
        if not event_id:
            raise FeishuConfigurationError("Feishu event id is missing.")

        event_start_at = _coerce_event_start_at(start_at, self.timezone)
        if event_start_at is None:
            event_start_at = parse_chinese_datetime(
                text=start_time_text,
                timezone=self.timezone,
                now=now,
            )
        if event_start_at is None:
            raise FeishuRequestError(f"Cannot parse interview start time: {start_time_text or start_at}")

        payload = _build_interview_event_payload(
            company=company,
            role=role,
            round_name=round_name,
            event_start_at=event_start_at,
            timezone=self.timezone,
            duration_minutes=self.event_duration_minutes,
            reminder_minutes=reminder_minutes,
            description=description,
        )
        token = self.get_tenant_access_token_sync()
        response_data = self._patch_json_sync(
            path=f"/calendar/v4/calendars/{calendar_id}/events/{event_id}",
            payload=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        self._raise_for_feishu_code(response_data)
        return FeishuCalendarEventResult(
            event_id=_extract_calendar_event_id(response_data) or event_id,
            raw_response=response_data,
        )

    # 删除飞书日历中已有的面试日程。
    def delete_interview_event(
        self,
        calendar_id: str,
        event_id: str,
    ) -> FeishuCalendarEventResult:
        if not self.is_calendar_sync_enabled():
            raise FeishuConfigurationError("Feishu calendar sync is not enabled.")
        if not calendar_id:
            raise FeishuConfigurationError("Feishu calendar id is missing.")
        if not event_id:
            raise FeishuConfigurationError("Feishu event id is missing.")

        token = self.get_tenant_access_token_sync()
        response_data = self._delete_json_sync(
            path=f"/calendar/v4/calendars/{calendar_id}/events/{event_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        self._raise_for_feishu_code(response_data)
        return FeishuCalendarEventResult(event_id=event_id, raw_response=response_data)

    # 把用户加入飞书日程参与人。
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


# 构造创建和更新日程共用的请求体。
def _build_interview_event_payload(
    company: str,
    role: Optional[str],
    round_name: str,
    event_start_at: datetime,
    timezone: str,
    duration_minutes: int,
    reminder_minutes: int,
    description: str,
) -> Dict[str, Any]:
    end_at = event_start_at + timedelta(minutes=max(duration_minutes, 1))
    return {
        "summary": _build_interview_event_summary(
            company=company,
            role=role,
            round_name=round_name,
        ),
        "description": description,
        "start_time": {
            "timestamp": str(int(event_start_at.timestamp())),
            "timezone": timezone,
        },
        "end_time": {
            "timestamp": str(int(end_at.timestamp())),
            "timezone": timezone,
        },
        "reminders": [{"minutes": reminder_minutes}],
    }


# 将输入转换为 event start at。
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


# 把中文相对时间解析为具体日期时间。
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

# 解析 weekday date。
def _parse_weekday_date(text: str, current: datetime) -> Optional[Any]:
    weekday_index = _extract_weekday_index(text)
    if weekday_index is None:
        return None

    days_ahead = weekday_index - current.weekday()
    if days_ahead <= 0 or text.startswith("下周"):
        days_ahead += 7
    return (current + timedelta(days=days_ahead)).date()


# 从输入数据中提取 weekday index。
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


# 解析 clock time。
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


# 解析 chinese number。
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


# 处理 contains_any 相关逻辑。
def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


# 构造 interview event summary。
def _build_interview_event_summary(company: str, role: Optional[str], round_name: str) -> str:
    role_text = f" - {role}" if role else ""
    return f"OfferPilot 面试：{company}{role_text}（{round_name}）"


# 从输入数据中提取 calendar event id。
def _extract_calendar_event_id(response_data: Dict[str, Any]) -> Optional[str]:
    data = response_data.get("data")
    if isinstance(data, dict):
        event = data.get("event")
        if isinstance(event, dict) and isinstance(event.get("event_id"), str):
            return event["event_id"]
        if isinstance(data.get("event_id"), str):
            return data["event_id"]
    return None


# 从输入数据中提取 calendar id。
def _extract_calendar_id(response_data: Dict[str, Any]) -> Optional[str]:
    data = response_data.get("data")
    if isinstance(data, dict):
        calendar = data.get("calendar")
        if isinstance(calendar, dict) and isinstance(calendar.get("calendar_id"), str):
            return calendar["calendar_id"]
        if isinstance(data.get("calendar_id"), str):
            return data["calendar_id"]
    return None


# 从输入数据中提取 attendee ids。
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


_BITABLE_APPLICATION_STATUS_OPTIONS = [
    "待投递/待确认",
    "已投递",
    "笔试阶段",
    "笔试通过",
    "一面阶段",
    "一面通过",
    "二面阶段",
    "二面通过",
    "三面阶段",
    "三面通过",
    "HR 面阶段",
    "面试已安排",
    "Offer",
    "已拒绝",
    "暂无反馈",
    "已放弃",
]
_BITABLE_INTERVIEW_ROUND_OPTIONS = ["待确认", "笔试", "一面", "二面", "三面", "HR 面", "面试"]
_BITABLE_PRIORITY_OPTIONS = ["高", "中", "低"]
_BITABLE_SOURCE_OPTIONS = ["飞书助手", "官网", "内推", "Boss直聘", "牛客", "拉勾", "猎聘", "其他"]


# 构造 bitable select options。
def _build_bitable_select_options(names: List[str]) -> List[Dict[str, Any]]:
    return [
        {
            "name": name,
            "color": index % 54,
        }
        for index, name in enumerate(names)
    ]


# 封装飞书多维表格创建、记录读写和权限同步。
class FeishuBitableService(FeishuMessageService):
    TEXT_FIELD_TYPE = 1
    NUMBER_FIELD_TYPE = 2
    SINGLE_SELECT_FIELD_TYPE = 3
    MULTI_SELECT_FIELD_TYPE = 4
    DATETIME_FIELD_TYPE = 5

    APPLICATION_TABLE_FIELDS = [
        {"field_name": "投递记录", "type": TEXT_FIELD_TYPE},
        {"field_name": "OfferPilot记录ID", "type": TEXT_FIELD_TYPE},
        {"field_name": "公司", "type": TEXT_FIELD_TYPE},
        {"field_name": "岗位", "type": TEXT_FIELD_TYPE},
        {
            "field_name": "投递状态",
            "type": SINGLE_SELECT_FIELD_TYPE,
            "property": {"options": _build_bitable_select_options(_BITABLE_APPLICATION_STATUS_OPTIONS)},
        },
        {
            "field_name": "面试轮次",
            "type": SINGLE_SELECT_FIELD_TYPE,
            "property": {"options": _build_bitable_select_options(_BITABLE_INTERVIEW_ROUND_OPTIONS)},
        },
        {"field_name": "面试时间文本", "type": TEXT_FIELD_TYPE},
        {"field_name": "面试开始时间", "type": DATETIME_FIELD_TYPE},
        {"field_name": "提醒分钟", "type": NUMBER_FIELD_TYPE},
        {
            "field_name": "优先级",
            "type": SINGLE_SELECT_FIELD_TYPE,
            "property": {"options": _build_bitable_select_options(_BITABLE_PRIORITY_OPTIONS)},
        },
        {
            "field_name": "来源",
            "type": SINGLE_SELECT_FIELD_TYPE,
            "property": {"options": _build_bitable_select_options(_BITABLE_SOURCE_OPTIONS)},
        },
        {"field_name": "JD关键词", "type": MULTI_SELECT_FIELD_TYPE},
        {"field_name": "下一步", "type": TEXT_FIELD_TYPE},
        {"field_name": "备注", "type": TEXT_FIELD_TYPE},
        {"field_name": "日历事件ID", "type": TEXT_FIELD_TYPE},
        {"field_name": "最后同步时间", "type": DATETIME_FIELD_TYPE},
        {"field_name": "OfferPilot用户ID", "type": TEXT_FIELD_TYPE},
    ]

    # 初始化当前组件所需的依赖和配置。
    def __init__(
        self,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        app_token: Optional[str] = None,
        table_id: Optional[str] = None,
        sync_enabled: Optional[bool] = None,
        auto_create_enabled: Optional[bool] = None,
        bitable_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            app_id=app_id,
            app_secret=app_secret,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        self.app_token = app_token if app_token is not None else settings.feishu_bitable_app_token
        self.table_id = table_id if table_id is not None else settings.feishu_bitable_table_id
        self.sync_enabled = (
            sync_enabled
            if sync_enabled is not None
            else settings.feishu_bitable_sync_enabled
        )
        self.auto_create_enabled = (
            auto_create_enabled
            if auto_create_enabled is not None
            else settings.feishu_bitable_auto_create_enabled
        )
        self.bitable_name = (
            bitable_name
            if bitable_name is not None
            else settings.feishu_offerpilot_bitable_name
        )
        self.table_name = (
            table_name
            if table_name is not None
            else settings.feishu_offerpilot_bitable_table_name
        )

    # 判断飞书多维表格同步是否开启。
    def is_bitable_sync_enabled(self) -> bool:
        has_existing_table = bool(self.app_token and self.table_id)
        return self.sync_enabled and self.is_configured() and (
            has_existing_table or self.auto_create_enabled
        )

    # 判断是否由 OfferPilot 自动管理多维表格。
    def should_manage_offerpilot_bitable(self) -> bool:
        return self.auto_create_enabled and not (self.app_token and self.table_id)

    # 创建飞书多维表格应用。
    def create_app(self, name: Optional[str] = None) -> FeishuBitableAppResult:
        if not self.sync_enabled:
            raise FeishuConfigurationError("Feishu bitable sync is not enabled.")

        token = self.get_tenant_access_token_sync()
        response_data = self._post_json_sync(
            path="/bitable/v1/apps",
            payload={"name": name or self.bitable_name},
            headers={"Authorization": f"Bearer {token}"},
        )
        self._raise_for_feishu_code(response_data)

        return FeishuBitableAppResult(
            app_token=_extract_bitable_app_token(response_data),
            raw_response=response_data,
        )

    # 创建 OfferPilot 投递记录表。
    def create_application_table(
        self,
        app_token: str,
        table_name: Optional[str] = None,
    ) -> FeishuBitableTableResult:
        if not self.sync_enabled:
            raise FeishuConfigurationError("Feishu bitable sync is not enabled.")
        if not app_token:
            raise FeishuConfigurationError("Feishu bitable app token is missing.")

        token = self.get_tenant_access_token_sync()
        response_data = self._post_json_sync(
            path=f"/bitable/v1/apps/{app_token}/tables",
            payload={
                "table": {
                    "name": table_name or self.table_name,
                    "default_view_name": "全部投递",
                    "fields": self.APPLICATION_TABLE_FIELDS,
                }
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        self._raise_for_feishu_code(response_data)

        return FeishuBitableTableResult(
            table_id=_extract_bitable_table_id(response_data),
            raw_response=response_data,
        )

    # 将旧版投递表原地迁移为用户可读主标题，并移除内部状态展示列。
    def migrate_application_table_schema(
        self,
        app_token: str,
        table_id: str,
    ) -> Dict[str, int]:
        if not self.is_bitable_sync_enabled():
            raise FeishuConfigurationError("Feishu bitable sync is not enabled.")
        if not app_token or not table_id:
            raise FeishuConfigurationError("Feishu bitable app token or table id is missing.")

        fields = self.list_fields(app_token=app_token, table_id=table_id)
        fields_by_name = {
            field.get("field_name"): field
            for field in fields
            if isinstance(field.get("field_name"), str)
        }
        renamed_fields = 0
        created_fields = 0
        deleted_fields = 0
        updated_records = 0

        title_field = fields_by_name.get("投递记录")
        internal_id_field = fields_by_name.get("OfferPilot记录ID")
        if title_field is None and internal_id_field is not None:
            self.update_field(
                app_token=app_token,
                table_id=table_id,
                field_id=str(internal_id_field.get("field_id") or ""),
                field_name="投递记录",
                field_type=self.TEXT_FIELD_TYPE,
            )
            renamed_fields += 1
            title_field = {**internal_id_field, "field_name": "投递记录"}
            internal_id_field = None

        if internal_id_field is None:
            self.create_field(
                app_token=app_token,
                table_id=table_id,
                field_name="OfferPilot记录ID",
                field_type=self.TEXT_FIELD_TYPE,
            )
            created_fields += 1

        page_token: Optional[str] = None
        while True:
            page = self.list_records(
                app_token=app_token,
                table_id=table_id,
                page_size=500,
                page_token=page_token,
            )
            for record in page.records:
                record_fields = record.fields or {}
                previous_title = _bitable_field_text(record_fields.get("投递记录"))
                company = _bitable_field_text(record_fields.get("公司"))
                role = _bitable_field_text(record_fields.get("岗位"))
                readable_title = _build_bitable_application_title(company, role)
                updates: Dict[str, Any] = {"投递记录": readable_title}
                if (
                    previous_title.startswith("app_")
                    and _bitable_field_text(record_fields.get("投递状态")) == "待投递/待确认"
                ):
                    updates["投递状态"] = "已投递"
                existing_internal_id = _bitable_field_text(
                    record_fields.get("OfferPilot记录ID")
                )
                if not existing_internal_id and previous_title.startswith("app_"):
                    updates["OfferPilot记录ID"] = previous_title
                if record.record_id:
                    self.update_record(
                        app_token=app_token,
                        table_id=table_id,
                        record_id=record.record_id,
                        fields=updates,
                    )
                    updated_records += 1
            if not page.has_more or not page.page_token:
                break
            page_token = page.page_token

        status_value_field = fields_by_name.get("状态值")
        if status_value_field is not None:
            self.delete_field(
                app_token=app_token,
                table_id=table_id,
                field_id=str(status_value_field.get("field_id") or ""),
            )
            deleted_fields += 1

        return {
            "renamed_fields": renamed_fields,
            "created_fields": created_fields,
            "deleted_fields": deleted_fields,
            "updated_records": updated_records,
        }

    # 列出数据表字段。
    def list_fields(self, app_token: str, table_id: str) -> List[Dict[str, Any]]:
        token = self.get_tenant_access_token_sync()
        fields: List[Dict[str, Any]] = []
        page_token: Optional[str] = None
        while True:
            params = {"page_size": "100"}
            if page_token:
                params["page_token"] = page_token
            response_data = self._get_json_sync(
                path=f"/bitable/v1/apps/{app_token}/tables/{table_id}/fields",
                headers={"Authorization": f"Bearer {token}"},
                params=params,
            )
            self._raise_for_feishu_code(response_data)
            data = response_data.get("data")
            items = data.get("items") if isinstance(data, dict) else None
            fields.extend(item for item in items or [] if isinstance(item, dict))
            has_more = bool(data.get("has_more")) if isinstance(data, dict) else False
            next_page_token = data.get("page_token") if isinstance(data, dict) else None
            if not has_more or not isinstance(next_page_token, str) or not next_page_token:
                return fields
            page_token = next_page_token

    # 新增数据表字段。
    def create_field(
        self,
        app_token: str,
        table_id: str,
        field_name: str,
        field_type: int,
    ) -> Dict[str, Any]:
        token = self.get_tenant_access_token_sync()
        response_data = self._post_json_sync(
            path=f"/bitable/v1/apps/{app_token}/tables/{table_id}/fields",
            payload={"field_name": field_name, "type": field_type},
            headers={"Authorization": f"Bearer {token}"},
        )
        self._raise_for_feishu_code(response_data)
        return response_data

    # 更新数据表字段名称或类型。
    def update_field(
        self,
        app_token: str,
        table_id: str,
        field_id: str,
        field_name: str,
        field_type: int,
    ) -> Dict[str, Any]:
        if not field_id:
            raise FeishuRequestError("Feishu bitable field id is missing.")
        token = self.get_tenant_access_token_sync()
        response_data = self._put_json_sync(
            path=f"/bitable/v1/apps/{app_token}/tables/{table_id}/fields/{field_id}",
            payload={"field_name": field_name, "type": field_type},
            headers={"Authorization": f"Bearer {token}"},
        )
        self._raise_for_feishu_code(response_data)
        return response_data

    # 删除不再需要展示的数据表字段。
    def delete_field(
        self,
        app_token: str,
        table_id: str,
        field_id: str,
    ) -> Dict[str, Any]:
        if not field_id:
            raise FeishuRequestError("Feishu bitable field id is missing.")
        token = self.get_tenant_access_token_sync()
        response_data = self._delete_json_sync(
            path=f"/bitable/v1/apps/{app_token}/tables/{table_id}/fields/{field_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        self._raise_for_feishu_code(response_data)
        return response_data

    # 在飞书多维表格中创建记录。
    def create_record(
        self,
        app_token: str,
        table_id: str,
        fields: Dict[str, Any],
    ) -> FeishuBitableRecordResult:
        if not self.is_bitable_sync_enabled():
            raise FeishuConfigurationError("Feishu bitable sync is not enabled.")
        if not app_token:
            raise FeishuConfigurationError("Feishu bitable app token is missing.")
        if not table_id:
            raise FeishuConfigurationError("Feishu bitable table id is missing.")

        token = self.get_tenant_access_token_sync()
        response_data = self._post_json_sync(
            path=f"/bitable/v1/apps/{app_token}/tables/{table_id}/records",
            payload={"fields": fields},
            headers={"Authorization": f"Bearer {token}"},
        )
        self._raise_for_feishu_code(response_data)

        return FeishuBitableRecordResult(
            record_id=_extract_bitable_record_id(response_data),
            raw_response=response_data,
        )

    # 更新飞书多维表格中的记录。
    def update_record(
        self,
        app_token: str,
        table_id: str,
        record_id: str,
        fields: Dict[str, Any],
    ) -> FeishuBitableRecordResult:
        if not self.is_bitable_sync_enabled():
            raise FeishuConfigurationError("Feishu bitable sync is not enabled.")
        if not app_token:
            raise FeishuConfigurationError("Feishu bitable app token is missing.")
        if not table_id:
            raise FeishuConfigurationError("Feishu bitable table id is missing.")
        if not record_id:
            raise FeishuConfigurationError("Feishu bitable record id is missing.")

        token = self.get_tenant_access_token_sync()
        response_data = self._put_json_sync(
            path=f"/bitable/v1/apps/{app_token}/tables/{table_id}/records/{record_id}",
            payload={"fields": fields},
            headers={"Authorization": f"Bearer {token}"},
        )
        self._raise_for_feishu_code(response_data)

        return FeishuBitableRecordResult(
            record_id=_extract_bitable_record_id(response_data) or record_id,
            raw_response=response_data,
        )

    # 读取飞书多维表格中的单条记录。
    def get_record(
        self,
        app_token: str,
        table_id: str,
        record_id: str,
    ) -> FeishuBitableRecordResult:
        if not self.is_bitable_sync_enabled():
            raise FeishuConfigurationError("Feishu bitable sync is not enabled.")
        if not app_token:
            raise FeishuConfigurationError("Feishu bitable app token is missing.")
        if not table_id:
            raise FeishuConfigurationError("Feishu bitable table id is missing.")
        if not record_id:
            raise FeishuConfigurationError("Feishu bitable record id is missing.")

        token = self.get_tenant_access_token_sync()
        response_data = self._get_json_sync(
            path=f"/bitable/v1/apps/{app_token}/tables/{table_id}/records/{record_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        self._raise_for_feishu_code(response_data)

        return FeishuBitableRecordResult(
            record_id=_extract_bitable_record_id(response_data) or record_id,
            raw_response=response_data,
            fields=_extract_bitable_record_fields(response_data),
        )

    # 分页读取飞书多维表格记录。
    def list_records(
        self,
        app_token: str,
        table_id: str,
        page_size: int = 100,
        page_token: Optional[str] = None,
    ) -> FeishuBitableRecordListResult:
        if not self.is_bitable_sync_enabled():
            raise FeishuConfigurationError("Feishu bitable sync is not enabled.")
        if not app_token:
            raise FeishuConfigurationError("Feishu bitable app token is missing.")
        if not table_id:
            raise FeishuConfigurationError("Feishu bitable table id is missing.")

        token = self.get_tenant_access_token_sync()
        params = {"page_size": str(max(min(page_size, 500), 1))}
        if page_token:
            params["page_token"] = page_token
        response_data = self._get_json_sync(
            path=f"/bitable/v1/apps/{app_token}/tables/{table_id}/records",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
        )
        self._raise_for_feishu_code(response_data)

        return FeishuBitableRecordListResult(
            records=_extract_bitable_record_items(response_data),
            has_more=_extract_bitable_has_more(response_data),
            page_token=_extract_bitable_page_token(response_data),
            raw_response=response_data,
        )

    # 给用户授予多维表格协作者权限。
    def add_bitable_collaborator(
        self,
        app_token: str,
        member_id: str,
        member_id_type: str = "open_id",
        perm: str = "edit",
        need_notification: bool = True,
    ) -> FeishuBitableCollaboratorResult:
        if not self.is_bitable_sync_enabled():
            raise FeishuConfigurationError("Feishu bitable sync is not enabled.")
        if not app_token:
            raise FeishuConfigurationError("Feishu bitable app token is missing.")
        if not member_id:
            raise FeishuConfigurationError("Feishu permission member id is missing.")

        token = self.get_tenant_access_token_sync()
        response_data = self._post_json_sync(
            path=f"/drive/v1/permissions/{app_token}/members",
            payload={
                "member_type": _normalize_drive_permission_member_type(member_id_type),
                "member_id": member_id,
                "perm": perm,
            },
            headers={"Authorization": f"Bearer {token}"},
            params={
                "type": "bitable",
                "need_notification": str(need_notification).lower(),
            },
        )
        self._raise_for_feishu_code(response_data)

        return FeishuBitableCollaboratorResult(
            member_id=_extract_drive_permission_member_id(response_data) or member_id,
            raw_response=response_data,
        )

    # 订阅多维表格文件变更事件。
    def subscribe_bitable_file(self, app_token: str) -> FeishuFileSubscriptionResult:
        if not self.sync_enabled:
            raise FeishuConfigurationError("Feishu bitable sync is not enabled.")
        if not app_token:
            raise FeishuConfigurationError("Feishu bitable app token is missing.")

        token = self.get_tenant_access_token_sync()
        response_data = self._post_json_sync(
            path=f"/drive/v1/files/{app_token}/subscribe",
            payload={},
            headers={"Authorization": f"Bearer {token}"},
            params={"file_type": "bitable"},
        )
        self._raise_for_feishu_code(response_data)

        return FeishuFileSubscriptionResult(
            file_token=app_token,
            raw_response=response_data,
        )


# 从输入数据中提取 bitable app token。
def _extract_bitable_app_token(response_data: Dict[str, Any]) -> Optional[str]:
    data = response_data.get("data")
    if isinstance(data, dict):
        app = data.get("app")
        if isinstance(app, dict) and isinstance(app.get("app_token"), str):
            return app["app_token"]
        if isinstance(data.get("app_token"), str):
            return data["app_token"]
    return None


# 将多维表格文本字段响应转换为普通字符串。
def _bitable_field_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "".join(_bitable_field_text(item) for item in value).strip()
    if isinstance(value, dict):
        for key in ("text", "name", "value"):
            if key in value:
                return _bitable_field_text(value[key])
    return str(value).strip()


# 构造投递记录的用户可读主标题。
def _build_bitable_application_title(company: str, role: str) -> str:
    if company and role:
        return f"{company}｜{role}"
    return company or role or "未命名投递"


# 从输入数据中提取 bitable table id。
def _extract_bitable_table_id(response_data: Dict[str, Any]) -> Optional[str]:
    data = response_data.get("data")
    if isinstance(data, dict):
        table = data.get("table")
        if isinstance(table, dict) and isinstance(table.get("table_id"), str):
            return table["table_id"]
        if isinstance(data.get("table_id"), str):
            return data["table_id"]
    return None


# 从输入数据中提取 bitable record id。
def _extract_bitable_record_id(response_data: Dict[str, Any]) -> Optional[str]:
    data = response_data.get("data")
    if isinstance(data, dict):
        record = data.get("record")
        if isinstance(record, dict) and isinstance(record.get("record_id"), str):
            return record["record_id"]
        if isinstance(data.get("record_id"), str):
            return data["record_id"]
    return None


# 从输入数据中提取 bitable record fields。
def _extract_bitable_record_fields(response_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    data = response_data.get("data")
    if isinstance(data, dict):
        record = data.get("record")
        if isinstance(record, dict) and isinstance(record.get("fields"), dict):
            return record["fields"]
        if isinstance(data.get("fields"), dict):
            return data["fields"]
    return None


# 从输入数据中提取 bitable record items。
def _extract_bitable_record_items(response_data: Dict[str, Any]) -> List[FeishuBitableRecordResult]:
    data = response_data.get("data")
    if not isinstance(data, dict):
        return []

    items = data.get("items")
    if not isinstance(items, list):
        items = data.get("records")
    if not isinstance(items, list):
        return []

    records = []
    for item in items:
        if not isinstance(item, dict):
            continue
        record_id = item.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            continue
        fields = item.get("fields")
        records.append(
            FeishuBitableRecordResult(
                record_id=record_id,
                raw_response=item,
                fields=fields if isinstance(fields, dict) else None,
            )
        )
    return records


# 从输入数据中提取 bitable has more。
def _extract_bitable_has_more(response_data: Dict[str, Any]) -> bool:
    data = response_data.get("data")
    if not isinstance(data, dict):
        return False
    return bool(data.get("has_more"))


# 从输入数据中提取 bitable page token。
def _extract_bitable_page_token(response_data: Dict[str, Any]) -> Optional[str]:
    data = response_data.get("data")
    if isinstance(data, dict) and isinstance(data.get("page_token"), str):
        return data["page_token"]
    return None


# 标准化 drive permission member type。
def _normalize_drive_permission_member_type(member_id_type: str) -> str:
    normalized = (member_id_type or "open_id").strip().lower()
    mapping = {
        "open_id": "openid",
        "openid": "openid",
        "user_id": "userid",
        "userid": "userid",
        "union_id": "unionid",
        "unionid": "unionid",
    }
    return mapping.get(normalized, normalized)


# 从输入数据中提取 drive permission member id。
def _extract_drive_permission_member_id(response_data: Dict[str, Any]) -> Optional[str]:
    data = response_data.get("data")
    if isinstance(data, dict):
        member = data.get("member")
        if isinstance(member, dict) and isinstance(member.get("member_id"), str):
            return member["member_id"]
        if isinstance(data.get("member_id"), str):
            return data["member_id"]
    return None
