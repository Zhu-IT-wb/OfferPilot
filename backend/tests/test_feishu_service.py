import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.services.feishu_service import (
    FeishuCalendarService,
    FeishuConfigurationError,
    FeishuMessageService,
    FeishuRequestError,
    parse_chinese_datetime,
)


def test_feishu_service_requires_credentials() -> None:
    service = FeishuMessageService(app_id="", app_secret="")

    with pytest.raises(FeishuConfigurationError):
        asyncio.run(service.get_tenant_access_token())


def test_feishu_service_gets_and_caches_tenant_access_token(monkeypatch) -> None:
    calls = []
    service = FeishuMessageService(app_id="app_id", app_secret="app_secret")

    async def fake_post_json(path, payload, headers=None, params=None):
        calls.append(
            {
                "path": path,
                "payload": payload,
                "headers": headers,
                "params": params,
            }
        )
        return {
            "code": 0,
            "tenant_access_token": "tenant_token",
            "expire": 7200,
        }

    monkeypatch.setattr(service, "_post_json", fake_post_json)

    first_token = asyncio.run(service.get_tenant_access_token())
    second_token = asyncio.run(service.get_tenant_access_token())

    assert first_token == "tenant_token"
    assert second_token == "tenant_token"
    assert calls == [
        {
            "path": "/auth/v3/tenant_access_token/internal",
            "payload": {"app_id": "app_id", "app_secret": "app_secret"},
            "headers": None,
            "params": None,
        }
    ]


def test_feishu_service_sends_text_message(monkeypatch) -> None:
    calls = []
    service = FeishuMessageService(app_id="app_id", app_secret="app_secret")

    async def fake_post_json(path, payload, headers=None, params=None):
        calls.append(
            {
                "path": path,
                "payload": payload,
                "headers": headers,
                "params": params,
            }
        )
        if path == "/auth/v3/tenant_access_token/internal":
            return {
                "code": 0,
                "tenant_access_token": "tenant_token",
                "expire": 7200,
            }
        return {
            "code": 0,
            "data": {"message_id": "om_test"},
        }

    monkeypatch.setattr(service, "_post_json", fake_post_json)

    result = asyncio.run(
        service.send_text_message(
            receive_id="ou_test",
            text="今天的任务：1. LeetCode 206. 反转链表",
        )
    )

    assert result.message_id == "om_test"
    assert calls[1] == {
        "path": "/im/v1/messages",
        "payload": {
            "receive_id": "ou_test",
            "msg_type": "text",
            "content": '{"text": "今天的任务：1. LeetCode 206. 反转链表"}',
        },
        "headers": {"Authorization": "Bearer tenant_token"},
        "params": {"receive_id_type": "open_id"},
    }


def test_feishu_service_raises_for_feishu_error_code() -> None:
    with pytest.raises(FeishuRequestError):
        FeishuMessageService._raise_for_feishu_code(
            {
                "code": 999,
                "msg": "bad request",
            }
        )


def test_parse_chinese_datetime_for_relative_interview_time() -> None:
    now = datetime(2026, 6, 29, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    result = parse_chinese_datetime("明天早上八点", now=now)

    assert result is not None
    assert result.isoformat() == "2026-06-30T08:00:00+08:00"


def test_parse_chinese_datetime_for_afternoon_half_hour() -> None:
    now = datetime(2026, 6, 29, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    result = parse_chinese_datetime("后天下午两点半", now=now)

    assert result is not None
    assert result.isoformat() == "2026-07-01T14:30:00+08:00"


def test_feishu_calendar_service_creates_interview_event(monkeypatch) -> None:
    calls = []
    service = FeishuCalendarService(
        app_id="app_id",
        app_secret="app_secret",
        calendar_id="primary",
        sync_enabled=True,
        timezone="Asia/Shanghai",
        event_duration_minutes=60,
    )
    now = datetime(2026, 6, 29, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    def fake_post_json_sync(path, payload, headers=None, params=None):
        calls.append(
            {
                "path": path,
                "payload": payload,
                "headers": headers,
                "params": params,
            }
        )
        if path == "/auth/v3/tenant_access_token/internal":
            return {
                "code": 0,
                "tenant_access_token": "tenant_token",
                "expire": 7200,
            }
        return {
            "code": 0,
            "data": {"event": {"event_id": "evt_test_1"}},
        }

    monkeypatch.setattr(service, "_post_json_sync", fake_post_json_sync)

    result = service.create_interview_event(
        company="深信服",
        role="AI 应用开发",
        round_name="一面",
        start_time_text="明天早上八点",
        reminder_minutes=30,
        description="明天早上八点，深信服约我一面",
        now=now,
    )

    assert result.event_id == "evt_test_1"
    assert calls[1]["path"] == "/calendar/v4/calendars/primary/events"
    assert calls[1]["headers"] == {"Authorization": "Bearer tenant_token"}
    assert calls[1]["payload"]["summary"] == "OfferPilot 面试：深信服 - AI 应用开发（一面）"
    assert calls[1]["payload"]["start_time"]["timestamp"] == "1782777600"
    assert calls[1]["payload"]["end_time"]["timestamp"] == "1782781200"
    assert calls[1]["payload"]["reminders"] == [{"minutes": 30}]


def test_feishu_calendar_service_prefers_normalized_start_at(monkeypatch) -> None:
    calls = []
    service = FeishuCalendarService(
        app_id="app_id",
        app_secret="app_secret",
        calendar_id="primary",
        sync_enabled=True,
        timezone="Asia/Shanghai",
        event_duration_minutes=60,
    )

    def fake_post_json_sync(path, payload, headers=None, params=None):
        calls.append(
            {
                "path": path,
                "payload": payload,
                "headers": headers,
                "params": params,
            }
        )
        if path == "/auth/v3/tenant_access_token/internal":
            return {
                "code": 0,
                "tenant_access_token": "tenant_token",
                "expire": 7200,
            }
        return {
            "code": 0,
            "data": {"event": {"event_id": "evt_test_1"}},
        }

    monkeypatch.setattr(service, "_post_json_sync", fake_post_json_sync)

    result = service.create_interview_event(
        company="深信服",
        role="AI 应用开发",
        round_name="一面",
        start_time_text="明天下午三点",
        start_at="2026-06-30T15:00:00+08:00",
        reminder_minutes=30,
    )

    assert result.event_id == "evt_test_1"
    assert calls[1]["payload"]["start_time"]["timestamp"] == "1782802800"


def test_feishu_calendar_service_creates_shared_calendar(monkeypatch) -> None:
    calls = []
    service = FeishuCalendarService(
        app_id="app_id",
        app_secret="app_secret",
        calendar_id="primary",
        sync_enabled=True,
        auto_create_enabled=True,
        managed_calendar_summary="OfferPilot 秋招日历",
        managed_calendar_description="自动记录秋招安排",
        managed_calendar_permissions="private",
    )

    def fake_post_json_sync(path, payload, headers=None, params=None):
        calls.append(
            {
                "path": path,
                "payload": payload,
                "headers": headers,
                "params": params,
            }
        )
        if path == "/auth/v3/tenant_access_token/internal":
            return {
                "code": 0,
                "tenant_access_token": "tenant_token",
                "expire": 7200,
            }
        return {
            "code": 0,
            "data": {
                "calendar": {
                    "calendar_id": "feishu.cn_offerpilot@group.calendar.feishu.cn",
                }
            },
        }

    monkeypatch.setattr(service, "_post_json_sync", fake_post_json_sync)

    result = service.create_shared_calendar()

    assert result.calendar_id == "feishu.cn_offerpilot@group.calendar.feishu.cn"
    assert service.should_manage_offerpilot_calendar() is True
    assert calls[1] == {
        "path": "/calendar/v4/calendars",
        "payload": {
            "summary": "OfferPilot 秋招日历",
            "description": "自动记录秋招安排",
            "permissions": "private",
            "summary_alias": "OfferPilot 秋招日历",
        },
        "headers": {"Authorization": "Bearer tenant_token"},
        "params": None,
    }


def test_feishu_calendar_service_adds_event_attendee(monkeypatch) -> None:
    calls = []
    service = FeishuCalendarService(
        app_id="app_id",
        app_secret="app_secret",
        calendar_id="primary",
        sync_enabled=True,
    )

    def fake_post_json_sync(path, payload, headers=None, params=None):
        calls.append(
            {
                "path": path,
                "payload": payload,
                "headers": headers,
                "params": params,
            }
        )
        if path == "/auth/v3/tenant_access_token/internal":
            return {
                "code": 0,
                "tenant_access_token": "tenant_token",
                "expire": 7200,
            }
        return {
            "code": 0,
            "data": {
                "attendees": [
                    {
                        "type": "user",
                        "attendee_id": "user_attendee_1",
                    }
                ]
            },
        }

    monkeypatch.setattr(service, "_post_json_sync", fake_post_json_sync)

    result = service.add_event_attendee(
        calendar_id="feishu.cn_offerpilot@group.calendar.feishu.cn",
        event_id="evt_test_1",
        user_id="ou_test",
    )

    assert result.attendee_ids == ["user_attendee_1"]
    assert calls[1] == {
        "path": "/calendar/v4/calendars/feishu.cn_offerpilot@group.calendar.feishu.cn/events/evt_test_1/attendees",
        "payload": {
            "attendees": [
                {
                    "type": "user",
                    "user_id": "ou_test",
                }
            ],
            "need_notification": True,
        },
        "headers": {"Authorization": "Bearer tenant_token"},
        "params": {"user_id_type": "open_id"},
    }
