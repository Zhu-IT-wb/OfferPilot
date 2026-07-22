import asyncio
import json
from datetime import datetime
from zoneinfo import ZoneInfo

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
    status_field = calls[2]["payload"]["table"]["fields"][4]
    assert status_field["field_name"] == "投递状态"
    assert status_field["type"] == 3
    assert [option["name"] for option in status_field["property"]["options"][:4]] == [
        "待投递/待确认",
        "已投递",
        "笔试阶段",
        "笔试通过",
    ]
    assert all("color" in option for option in status_field["property"]["options"])
    assert calls[2]["payload"]["table"]["fields"][7] == {
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
        "created_fields": 1,
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
        }
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
