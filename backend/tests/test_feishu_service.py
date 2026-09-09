import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import pytest

from app.services.feishu_service import (
    FeishuBitableService,
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


def test_feishu_service_signs_page_url_and_caches_jssdk_ticket(monkeypatch) -> None:
    calls = []
    service = FeishuMessageService(app_id="app_id", app_secret="app_secret")

    async def fake_post_json(path, payload, headers=None, params=None):
        calls.append(path)
        if path == "/auth/v3/tenant_access_token/internal":
            return {"code": 0, "tenant_access_token": "tenant_token", "expire": 7200}
        return {"code": 0, "data": {"ticket": "ticket-test", "expire_in": 7200}}

    monkeypatch.setattr(service, "_post_json", fake_post_json)
    page_url = "https://offerpilot.example/study/knowledge?from=workplace"

    first = asyncio.run(service.get_jssdk_config(page_url))
    second = asyncio.run(service.get_jssdk_config(page_url))

    verify_string = (
        f"jsapi_ticket=ticket-test&noncestr={first.nonce_str}"
        f"&timestamp={first.timestamp}&url={page_url}"
    )
    assert first.app_id == "app_id"
    assert first.signature == hashlib.sha1(verify_string.encode()).hexdigest()
    assert second.signature
    assert calls == [
        "/auth/v3/tenant_access_token/internal",
        "/jssdk/ticket/get",
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
            idempotency_key="3d0e89f8-0180-52bd-b975-7fd07845af58",
        )
    )

    assert result.message_id == "om_test"
    assert calls[1] == {
        "path": "/im/v1/messages",
        "payload": {
            "receive_id": "ou_test",
            "msg_type": "text",
            "content": '{"text": "今天的任务：1. LeetCode 206. 反转链表"}',
            "uuid": "3d0e89f8-0180-52bd-b975-7fd07845af58",
        },
        "headers": {"Authorization": "Bearer tenant_token"},
        "params": {"receive_id_type": "open_id"},
    }


def test_feishu_service_sends_interactive_message(monkeypatch) -> None:
    calls = []
    service = FeishuMessageService(app_id="app_id", app_secret="app_secret")

    async def fake_post_json(path, payload, headers=None, params=None):
        calls.append({"path": path, "payload": payload, "headers": headers, "params": params})
        if path == "/auth/v3/tenant_access_token/internal":
            return {"code": 0, "tenant_access_token": "tenant_token", "expire": 7200}
        return {"code": 0, "data": {"message_id": "om_card"}}

    monkeypatch.setattr(service, "_post_json", fake_post_json)
    card = {
        "header": {"title": {"tag": "plain_text", "content": "今日 LeetCode"}},
        "elements": [],
    }

    result = asyncio.run(
        service.send_interactive_message(
            receive_id="ou_test",
            card=card,
            idempotency_key="3d0e89f8-0180-52bd-b975-7fd07845af58",
        )
    )

    assert result.message_id == "om_card"
    assert calls[1]["payload"]["msg_type"] == "interactive"
    assert json.loads(calls[1]["payload"]["content"]) == card
    assert calls[1]["payload"]["uuid"] == "3d0e89f8-0180-52bd-b975-7fd07845af58"


def test_feishu_service_replies_to_message_with_stable_uuid(monkeypatch) -> None:
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
            return {"code": 0, "tenant_access_token": "tenant_token", "expire": 7200}
        return {"code": 0, "data": {"message_id": "om_reply"}}

    monkeypatch.setattr(service, "_post_json", fake_post_json)

    result = asyncio.run(
        service.reply_text_message(
            message_id="om/source",
            text="任务已完成。",
            idempotency_key="3d0e89f8-0180-52bd-b975-7fd07845af58",
            reply_in_thread=True,
        )
    )

    assert result.message_id == "om_reply"
    assert calls[1] == {
        "path": "/im/v1/messages/om%2Fsource/reply",
        "payload": {
            "msg_type": "text",
            "content": json.dumps({"text": "任务已完成。"}, ensure_ascii=False),
            "uuid": "3d0e89f8-0180-52bd-b975-7fd07845af58",
            "reply_in_thread": True,
        },
        "headers": {"Authorization": "Bearer tenant_token"},
        "params": None,
    }


def test_feishu_service_replies_with_interactive_card(monkeypatch) -> None:
    calls = []
    service = FeishuMessageService(app_id="app_id", app_secret="app_secret")
    card = {"elements": [{"tag": "markdown", "content": "**复习计划**"}]}

    async def fake_post_json(path, payload, headers=None, params=None):
        calls.append({"path": path, "payload": payload})
        if path == "/auth/v3/tenant_access_token/internal":
            return {"code": 0, "tenant_access_token": "tenant_token", "expire": 7200}
        return {"code": 0, "data": {"message_id": "om_card_reply"}}

    monkeypatch.setattr(service, "_post_json", fake_post_json)

    result = asyncio.run(
        service.reply_interactive_message(
            message_id="om_source",
            card=card,
            idempotency_key="bca12f29-fb35-54cf-a0ef-930c5b1657f2",
        )
    )

    assert result.message_id == "om_card_reply"
    assert calls[1]["path"] == "/im/v1/messages/om_source/reply"
    assert calls[1]["payload"]["msg_type"] == "interactive"
    assert json.loads(calls[1]["payload"]["content"]) == card
    assert calls[1]["payload"]["uuid"] == "bca12f29-fb35-54cf-a0ef-930c5b1657f2"


def test_feishu_service_updates_interactive_message(monkeypatch) -> None:
    calls = []
    service = FeishuMessageService(app_id="app_id", app_secret="app_secret")
    card = {"elements": [{"tag": "markdown", "content": "处理完成"}]}

    async def fake_post_json(path, payload, headers=None, params=None):
        if path == "/auth/v3/tenant_access_token/internal":
            return {"code": 0, "tenant_access_token": "tenant_token", "expire": 7200}
        raise AssertionError(f"unexpected POST path: {path}")

    async def fake_patch_json(path, payload, headers=None, params=None):
        calls.append(
            {
                "path": path,
                "payload": payload,
                "headers": headers,
                "params": params,
            }
        )
        return {"code": 0, "data": {"message_id": "om_progress"}}

    monkeypatch.setattr(service, "_post_json", fake_post_json)
    monkeypatch.setattr(service, "_patch_json", fake_patch_json)

    result = asyncio.run(
        service.update_interactive_message(
            message_id="om/progress",
            card=card,
        )
    )

    assert result.message_id == "om_progress"
    assert calls == [
        {
            "path": "/im/v1/messages/om%2Fprogress",
            "payload": {"content": json.dumps(card, ensure_ascii=False)},
            "headers": {"Authorization": "Bearer tenant_token"},
            "params": None,
        }
    ]


def test_feishu_service_raises_for_feishu_error_code() -> None:
    with pytest.raises(FeishuRequestError) as raised:
        FeishuMessageService._raise_for_feishu_code(
            {
                "code": 999,
                "msg": "bad request",
            }
        )
    assert raised.value.code == 999
    assert str(raised.value) == "Feishu OpenAPI returned code 999: bad request"


def test_feishu_service_preserves_record_not_found_code() -> None:
    with pytest.raises(FeishuRequestError) as raised:
        FeishuMessageService._raise_for_feishu_code(
            {"code": 1254043, "msg": "RecordIdNotFound"}
        )

    assert raised.value.code == 1254043


def test_feishu_request_error_message_constructor_does_not_infer_code() -> None:
    error = FeishuRequestError("Feishu OpenAPI returned HTTP 404: RecordIdNotFound 1254043")

    assert error.code is None
    assert str(error) == "Feishu OpenAPI returned HTTP 404: RecordIdNotFound 1254043"


@pytest.mark.parametrize("method", ["get", "put"])
@pytest.mark.parametrize(
    ("status_code", "body", "expected_code"),
    [
        (400, b'{"code":1254043,"msg":"RecordIdNotFound"}', 1254043),
        (403, b'{"code":99991672,"msg":"permission denied"}', 99991672),
        (404, b'{"msg":"not found"}', None),
        (404, b'{"code":"1254043"}', None),
        (404, b'{"code":true}', None),
        (404, b"RecordIdNotFound 1254043", None),
    ],
)
def test_feishu_http_record_error_preserves_only_structured_numeric_code(
    monkeypatch, method, status_code, body, expected_code
) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(status_code, content=body))
    client_type = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client_type(transport=transport, **kwargs))
    service = FeishuMessageService(app_id="test_app", app_secret="test_secret")

    with pytest.raises(FeishuRequestError) as raised:
        if method == "get":
            service._get_json_sync("/record")
        else:
            service._put_json_sync("/record", payload={})

    assert raised.value.code == expected_code
    assert f"HTTP {status_code}" in str(raised.value)


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


def test_feishu_calendar_service_updates_interview_event(monkeypatch) -> None:
    calls = []
    service = FeishuCalendarService(
        app_id="app_id",
        app_secret="app_secret",
        calendar_id="primary",
        sync_enabled=True,
        timezone="Asia/Shanghai",
        event_duration_minutes=60,
    )
    service._tenant_access_token = "tenant_token"
    service._tenant_access_token_expires_at = float("inf")

    def fake_patch_json_sync(path, payload, headers=None, params=None):
        calls.append({"path": path, "payload": payload, "headers": headers, "params": params})
        return {"code": 0, "data": {"event": {"event_id": "evt_test_1"}}}

    monkeypatch.setattr(service, "_patch_json_sync", fake_patch_json_sync)

    result = service.update_interview_event(
        calendar_id="primary",
        event_id="evt_test_1",
        company="深信服",
        role="AI 应用开发",
        round_name="一面",
        start_time_text="后天下午四点",
        start_at="2026-07-18T16:00:00+08:00",
        reminder_minutes=30,
    )

    assert result.event_id == "evt_test_1"
    assert calls[0]["path"] == "/calendar/v4/calendars/primary/events/evt_test_1"
    assert calls[0]["headers"] == {"Authorization": "Bearer tenant_token"}
    assert calls[0]["payload"]["start_time"]["timestamp"] == "1784361600"


def test_feishu_calendar_service_deletes_interview_event(monkeypatch) -> None:
    calls = []
    service = FeishuCalendarService(
        app_id="app_id",
        app_secret="app_secret",
        calendar_id="primary",
        sync_enabled=True,
    )
    service._tenant_access_token = "tenant_token"
    service._tenant_access_token_expires_at = float("inf")

    def fake_delete_json_sync(path, headers=None, params=None):
        calls.append({"path": path, "headers": headers, "params": params})
        return {"code": 0}

    monkeypatch.setattr(service, "_delete_json_sync", fake_delete_json_sync)

    result = service.delete_interview_event(calendar_id="primary", event_id="evt_test_1")

    assert result.event_id == "evt_test_1"
    assert calls == [
        {
            "path": "/calendar/v4/calendars/primary/events/evt_test_1",
            "headers": {"Authorization": "Bearer tenant_token"},
            "params": None,
        }
    ]


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


def test_feishu_bitable_service_creates_app_table_record_and_updates_record(monkeypatch) -> None:
    calls = []
    service = FeishuBitableService(
        app_id="app_id",
        app_secret="app_secret",
        sync_enabled=True,
        auto_create_enabled=True,
        bitable_name="OfferPilot 秋招投递表",
        table_name="投递记录",
    )

    def fake_post_json_sync(path, payload, headers=None, params=None):
        calls.append(
            {
                "method": "POST",
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
        if path == "/bitable/v1/apps":
            return {
                "code": 0,
                "data": {"app": {"app_token": "bascn_test"}},
            }
        if path == "/bitable/v1/apps/bascn_test/tables":
            return {
                "code": 0,
                "data": {"table": {"table_id": "tbl_test"}},
            }
        return {
            "code": 0,
            "data": {"record": {"record_id": "rec_test"}},
        }

    def fake_put_json_sync(path, payload, headers=None, params=None):
        calls.append(
            {
                "method": "PUT",
                "path": path,
                "payload": payload,
                "headers": headers,
                "params": params,
            }
        )
        return {
            "code": 0,
            "data": {"record": {"record_id": "rec_test"}},
        }

    monkeypatch.setattr(service, "_post_json_sync", fake_post_json_sync)
    monkeypatch.setattr(service, "_put_json_sync", fake_put_json_sync)

    app_result = service.create_app()
    table_result = service.create_application_table(app_token=app_result.app_token)
    record_result = service.create_record(
        app_token=app_result.app_token,
        table_id=table_result.table_id,
        fields={"公司": "腾讯", "岗位": "AI 应用开发"},
    )
    updated_result = service.update_record(
        app_token=app_result.app_token,
        table_id=table_result.table_id,
        record_id=record_result.record_id,
        fields={"投递状态": "一面阶段"},
    )

    assert app_result.app_token == "bascn_test"
    assert table_result.table_id == "tbl_test"
    assert record_result.record_id == "rec_test"
    assert updated_result.record_id == "rec_test"
    assert calls[1] == {
        "method": "POST",
        "path": "/bitable/v1/apps",
        "payload": {"name": "OfferPilot 秋招投递表"},
        "headers": {"Authorization": "Bearer tenant_token"},
        "params": None,
    }
    assert calls[2]["path"] == "/bitable/v1/apps/bascn_test/tables"
    assert calls[2]["payload"]["table"]["name"] == "投递记录"
    assert calls[2]["payload"]["table"]["default_view_name"] == "全部投递"
    assert calls[2]["payload"]["table"]["fields"][0] == {
        "field_name": "投递记录",
        "type": 1,
    }
    assert calls[2]["payload"]["table"]["fields"][1] == {
        "field_name": "OfferPilot记录ID",
        "type": 1,
    }
    assert all(
        field["field_name"] != "状态值"
        for field in calls[2]["payload"]["table"]["fields"]
    )
    assert {"field_name": "工作地点", "type": 1} in calls[2]["payload"]["table"]["fields"]
    status_field = next(
        field
        for field in calls[2]["payload"]["table"]["fields"]
        if field["field_name"] == "投递状态"
    )
    assert status_field["field_name"] == "投递状态"
    assert status_field["type"] == 3
    assert [option["name"] for option in status_field["property"]["options"][:4]] == [
        "待投递/待确认",
        "已投递",
        "笔试阶段",
        "笔试通过",
    ]
    assert all("color" in option for option in status_field["property"]["options"])
    datetime_field = next(
        field
        for field in calls[2]["payload"]["table"]["fields"]
        if field["field_name"] == "面试开始时间"
    )
    assert datetime_field == {
        "field_name": "面试开始时间",
        "type": 5,
    }
    assert calls[3] == {
        "method": "POST",
        "path": "/bitable/v1/apps/bascn_test/tables/tbl_test/records",
        "payload": {"fields": {"公司": "腾讯", "岗位": "AI 应用开发"}},
        "headers": {"Authorization": "Bearer tenant_token"},
        "params": None,
    }
    assert calls[4] == {
        "method": "PUT",
        "path": "/bitable/v1/apps/bascn_test/tables/tbl_test/records/rec_test",
        "payload": {"fields": {"投递状态": "一面阶段"}},
        "headers": {"Authorization": "Bearer tenant_token"},
        "params": None,
    }


def test_feishu_bitable_service_migrates_legacy_application_table_in_place(monkeypatch) -> None:
    service = FeishuBitableService(app_id="app_id", app_secret="app_secret", sync_enabled=True)
    put_calls = []
    post_calls = []
    delete_calls = []
    monkeypatch.setattr(service, "get_tenant_access_token_sync", lambda: "tenant_token")

    def fake_get(path, headers=None, params=None):
        if path.endswith("/fields"):
            if params and params.get("page_token") == "fields_page_2":
                return {
                    "code": 0,
                    "data": {
                        "items": [
                            {
                                "field_id": "fld_status_value",
                                "field_name": "状态值",
                                "type": 1,
                            }
                        ],
                        "has_more": False,
                    },
                }
            return {
                "code": 0,
                "data": {
                    "items": [
                        {"field_id": "fld_primary", "field_name": "OfferPilot记录ID", "type": 1},
                        {"field_id": "fld_company", "field_name": "公司", "type": 1},
                        {"field_id": "fld_role", "field_name": "岗位", "type": 1},
                    ],
                    "has_more": True,
                    "page_token": "fields_page_2",
                },
            }
        return {
            "code": 0,
            "data": {
                "items": [
                    {
                        "record_id": "rec_1",
                        "fields": {
                            "投递记录": "app_20",
                            "公司": "腾讯",
                            "岗位": "Java 后端",
                            "投递状态": "待投递/待确认",
                        },
                    }
                ],
                "has_more": False,
            },
        }

    def fake_put(path, payload, headers=None, params=None):
        put_calls.append({"path": path, "payload": payload})
        return {"code": 0, "data": {"record": {"record_id": "rec_1"}}}

    def fake_post(path, payload, headers=None, params=None):
        post_calls.append({"path": path, "payload": payload})
        return {"code": 0, "data": {"field": {"field_id": "fld_internal"}}}

    def fake_delete(path, headers=None, params=None):
        delete_calls.append(path)
        return {"code": 0}

    monkeypatch.setattr(service, "_get_json_sync", fake_get)
    monkeypatch.setattr(service, "_put_json_sync", fake_put)
    monkeypatch.setattr(service, "_post_json_sync", fake_post)
    monkeypatch.setattr(service, "_delete_json_sync", fake_delete)

    result = service.migrate_application_table_schema("bascn_test", "tbl_test")

    assert result == {
        "renamed_fields": 1,
        "created_fields": 2,
        "deleted_fields": 1,
        "updated_records": 1,
    }
    assert put_calls[0] == {
        "path": "/bitable/v1/apps/bascn_test/tables/tbl_test/fields/fld_primary",
        "payload": {"field_name": "投递记录", "type": 1},
    }
    assert post_calls == [
        {
            "path": "/bitable/v1/apps/bascn_test/tables/tbl_test/fields",
            "payload": {"field_name": "OfferPilot记录ID", "type": 1},
        },
        {
            "path": "/bitable/v1/apps/bascn_test/tables/tbl_test/fields",
            "payload": {"field_name": "工作地点", "type": 1},
        },
    ]
    assert put_calls[1]["payload"] == {
        "fields": {
            "投递记录": "腾讯｜Java 后端",
            "OfferPilot记录ID": "app_20",
            "投递状态": "已投递",
        }
    }
    assert delete_calls == [
        "/bitable/v1/apps/bascn_test/tables/tbl_test/fields/fld_status_value"
    ]


def test_feishu_calendar_service_updates_and_deletes_user_study_event(monkeypatch) -> None:
    service = FeishuCalendarService(
        app_id="app_id",
        app_secret="app_secret",
        timezone="Asia/Shanghai",
    )
    patch_calls = []
    delete_calls = []

    def fake_patch(path, payload, headers=None, params=None):
        patch_calls.append((path, payload, headers, params))
        return {"code": 0, "data": {"event": {"event_id": "evt_study_1"}}}

    def fake_delete(path, headers=None, params=None):
        delete_calls.append((path, headers, params))
        return {"code": 0}

    monkeypatch.setattr(service, "_patch_json_sync", fake_patch)
    monkeypatch.setattr(service, "_delete_json_sync", fake_delete)
    start_at = datetime(2026, 9, 3, 19, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    updated = asyncio.run(
        service.update_user_study_event(
            user_access_token="user-access",
            event_id="evt_study_1",
            topic="系统设计",
            start_at=start_at,
            end_at=start_at + timedelta(hours=1),
            description="更新后的复习安排",
        )
    )
    deleted = asyncio.run(
        service.delete_user_calendar_event(
            user_access_token="user-access",
            event_id="evt_study_1",
        )
    )

    assert updated.event_id == deleted.event_id == "evt_study_1"
    assert patch_calls[0][0] == "/calendar/v4/calendars/primary/events/evt_study_1"
    assert patch_calls[0][2] == {"Authorization": "Bearer user-access"}
    assert patch_calls[0][1]["summary"] == "OfferPilot 复习：系统设计"
    assert delete_calls == [
        (
            "/calendar/v4/calendars/primary/events/evt_study_1",
            {"Authorization": "Bearer user-access"},
            None,
        )
    ]
def test_feishu_bitable_service_gets_record(monkeypatch) -> None:
    calls = []
    service = FeishuBitableService(
        app_id="app_id",
        app_secret="app_secret",
        sync_enabled=True,
        auto_create_enabled=True,
    )

    def fake_post_json_sync(path, payload, headers=None, params=None):
        calls.append(
            {
                "method": "POST",
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

    def fake_get_json_sync(path, headers=None, params=None):
        calls.append(
            {
                "method": "GET",
                "path": path,
                "headers": headers,
                "params": params,
            }
        )
        return {
            "code": 0,
            "data": {
                "record": {
                    "record_id": "rec_test",
                    "fields": {
                        "OfferPilot记录ID": "app_1",
                        "公司": "腾讯",
                        "岗位": "Java 后端实习",
                    },
                }
            },
        }

    monkeypatch.setattr(service, "_post_json_sync", fake_post_json_sync)
    monkeypatch.setattr(service, "_get_json_sync", fake_get_json_sync)

    result = service.get_record(
        app_token="bascn_test",
        table_id="tbl_test",
        record_id="rec_test",
    )

    assert result.record_id == "rec_test"
    assert result.fields == {
        "OfferPilot记录ID": "app_1",
        "公司": "腾讯",
        "岗位": "Java 后端实习",
    }
    assert calls[1] == {
        "method": "GET",
        "path": "/bitable/v1/apps/bascn_test/tables/tbl_test/records/rec_test",
        "headers": {"Authorization": "Bearer tenant_token"},
        "params": None,
    }


def test_feishu_bitable_service_adds_collaborator(monkeypatch) -> None:
    calls = []
    service = FeishuBitableService(
        app_id="app_id",
        app_secret="app_secret",
        sync_enabled=True,
        auto_create_enabled=True,
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
                "member": {
                    "member_id": "ou_test",
                }
            },
        }

    monkeypatch.setattr(service, "_post_json_sync", fake_post_json_sync)

    result = service.add_bitable_collaborator(
        app_token="bascn_offerpilot",
        member_id="ou_test",
        member_id_type="open_id",
    )

    assert result.member_id == "ou_test"
    assert calls[1] == {
        "path": "/drive/v1/permissions/bascn_offerpilot/members",
        "payload": {
            "member_type": "openid",
            "member_id": "ou_test",
            "perm": "edit",
        },
        "headers": {"Authorization": "Bearer tenant_token"},
        "params": {
            "type": "bitable",
            "need_notification": "true",
        },
    }
