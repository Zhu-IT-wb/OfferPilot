from datetime import datetime
from zoneinfo import ZoneInfo

from app.schemas.agent import AgentActionName
from app.models.application import ApplicationStatus
from app.services.feishu_service import (
    FeishuBitableAppResult,
    FeishuBitableCollaboratorResult,
    FeishuBitableRecordResult,
    FeishuBitableTableResult,
    FeishuCalendarAttendeeResult,
    FeishuCalendarEventResult,
    FeishuCalendarResult,
)
from app.schemas.tool import ToolResult
from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository
from app.tools import offerpilot_tools
from app.tools.offerpilot_tools import build_offerpilot_tool_registry, update_application
from app.tools.registry import ToolNotFoundError, ToolRegistry


def test_tool_registry_runs_registered_handler() -> None:
    registry = ToolRegistry()
    registry.register(
        "echo",
        lambda arguments: ToolResult(
            tool_name="echo",
            success=True,
            message=arguments["message"],
        ),
    )

    assert registry.has_tool("echo") is True
    result = registry.run("echo", {"message": "ok"})
    assert result.success is True
    assert result.message == "ok"


def test_tool_registry_raises_for_missing_tool() -> None:
    registry = ToolRegistry()

    try:
        registry.run("missing", {})
    except ToolNotFoundError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected ToolNotFoundError")


def test_offerpilot_tools_create_and_list_application(monkeypatch) -> None:
    monkeypatch.setattr(
        offerpilot_tools,
        "parse_chinese_datetime",
        lambda text: datetime(2026, 7, 1, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    repository = InMemoryOfferPilotRepository()
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)

    result = registry.run(
        AgentActionName.CREATE_APPLICATION.value,
        {
            "company": "深信服",
            "role": "开发实习",
            "interview_time": "明天下午三点",
            "round": "一面",
        },
    )

    assert result.success is True
    assert result.data["application"]["id"] == "app_1"
    assert result.data["application"]["status"] == "interview_1"
    assert result.data["interview_schedule"]["round"] == "一面"
    assert result.data["interview_schedule"]["start_time"] == "明天下午三点"
    assert result.data["interview_schedule"]["start_at"] == "2026-07-01T15:00:00+08:00"
    assert "公司：深信服" in result.message
    assert "状态：一面阶段" in result.message
    assert repository.applications[0].company == "深信服"


def test_offerpilot_tools_syncs_created_application_to_bitable() -> None:
    class FakeBitableService:
        app_token = ""
        table_id = ""

        def __init__(self):
            self.create_app_calls = 0
            self.create_table_calls = []
            self.created_records = []
            self.updated_records = []
            self.collaborator_calls = []

        def is_bitable_sync_enabled(self):
            return True

        def should_manage_offerpilot_bitable(self):
            return True

        def create_app(self):
            self.create_app_calls += 1
            return FeishuBitableAppResult(
                app_token="bascn_offerpilot",
                raw_response={"code": 0},
            )

        def create_application_table(self, app_token, table_name=None):
            self.create_table_calls.append({"app_token": app_token, "table_name": table_name})
            return FeishuBitableTableResult(
                table_id="tbl_applications",
                raw_response={"code": 0},
            )

        def create_record(self, app_token, table_id, fields):
            self.created_records.append(
                {
                    "app_token": app_token,
                    "table_id": table_id,
                    "fields": fields,
                }
            )
            return FeishuBitableRecordResult(
                record_id="rec_app_1",
                raw_response={"code": 0},
            )

        def update_record(self, app_token, table_id, record_id, fields):
            self.updated_records.append(
                {
                    "app_token": app_token,
                    "table_id": table_id,
                    "record_id": record_id,
                    "fields": fields,
                }
            )
            return FeishuBitableRecordResult(
                record_id=record_id,
                raw_response={"code": 0},
            )

        def add_bitable_collaborator(self, **kwargs):
            self.collaborator_calls.append(kwargs)
            return FeishuBitableCollaboratorResult(
                member_id=kwargs["member_id"],
                raw_response={"code": 0},
            )

    repository = InMemoryOfferPilotRepository()
    bitable_service = FakeBitableService()
    registry = build_offerpilot_tool_registry(
        repository,
        calendar_service=None,
        bitable_service=bitable_service,
    )

    result = registry.run(
        AgentActionName.CREATE_APPLICATION.value,
        {
            "company": "腾讯",
            "role": "AI 应用开发",
            "bitable_collaborator_user_id": "ou_test",
            "bitable_collaborator_user_id_type": "open_id",
        },
    )

    assert result.success is True
    assert bitable_service.create_app_calls == 1
    assert bitable_service.create_table_calls == [
        {"app_token": "bascn_offerpilot", "table_name": "投递记录"}
    ]
    assert bitable_service.created_records[0]["app_token"] == "bascn_offerpilot"
    assert bitable_service.created_records[0]["table_id"] == "tbl_applications"
    assert bitable_service.created_records[0]["fields"]["OfferPilot记录ID"] == "app_1"
    assert bitable_service.created_records[0]["fields"]["公司"] == "腾讯"
    assert bitable_service.created_records[0]["fields"]["岗位"] == "AI 应用开发"
    assert bitable_service.created_records[0]["fields"]["投递状态"] == "待投递/待确认"
    assert bitable_service.created_records[0]["fields"]["优先级"] == "高"
    assert bitable_service.created_records[0]["fields"]["来源"] == "飞书助手"
    assert bitable_service.created_records[0]["fields"]["下一步"] == "确认投递信息，补充投递渠道或 JD 关键词"
    assert repository.get_runtime_setting("feishu.offerpilot_bitable_app_token") == "bascn_offerpilot"
    assert repository.get_runtime_setting("feishu.offerpilot_bitable_table_id") == "tbl_applications"
    assert repository.get_runtime_setting("feishu.offerpilot_bitable_schema_version") == "v2"
    assert repository.get_runtime_setting("feishu.offerpilot_bitable_record_id.tbl_applications.app_1") == "rec_app_1"
    assert repository.get_runtime_setting("feishu.offerpilot_bitable_collaborator.bascn_offerpilot.ou_test") == "ou_test"
    assert bitable_service.collaborator_calls == [
        {
            "app_token": "bascn_offerpilot",
            "member_id": "ou_test",
            "member_id_type": "open_id",
            "perm": "edit",
            "need_notification": False,
        }
    ]
    assert result.data["bitable_sync"]["synced"] is True
    assert result.data["bitable_sync"]["collaborator_sync"]["synced"] is True
    assert result.data["bitable_sync"]["operation"] == "created"
    assert result.data["bitable_sync"]["bitable_auto_created"] is True
    assert result.data["bitable_sync"]["web_url"] == (
        "https://feishu.cn/base/bascn_offerpilot?table=tbl_applications"
    )
    assert "已自动创建 OfferPilot 秋招投递多维表格" in result.message
    assert "已同步到飞书多维表格" in result.message
    assert "已授予你飞书多维表格编辑权限" in result.message
    assert "打开多维表格：https://feishu.cn/base/bascn_offerpilot?table=tbl_applications" in result.message


def test_offerpilot_tools_updates_existing_bitable_record_for_application_progress() -> None:
    class FakeBitableService:
        app_token = ""
        table_id = ""

        def __init__(self):
            self.create_app_calls = 0
            self.create_table_calls = 0
            self.created_records = []
            self.updated_records = []
            self.collaborator_calls = []

        def is_bitable_sync_enabled(self):
            return True

        def should_manage_offerpilot_bitable(self):
            return True

        def create_app(self):
            self.create_app_calls += 1
            return FeishuBitableAppResult(
                app_token="bascn_offerpilot",
                raw_response={"code": 0},
            )

        def create_application_table(self, app_token, table_name=None):
            self.create_table_calls += 1
            return FeishuBitableTableResult(
                table_id="tbl_applications",
                raw_response={"code": 0},
            )

        def create_record(self, app_token, table_id, fields):
            self.created_records.append(fields)
            return FeishuBitableRecordResult(
                record_id="rec_app_1",
                raw_response={"code": 0},
            )

        def update_record(self, app_token, table_id, record_id, fields):
            self.updated_records.append(
                {
                    "app_token": app_token,
                    "table_id": table_id,
                    "record_id": record_id,
                    "fields": fields,
                }
            )
            return FeishuBitableRecordResult(
                record_id=record_id,
                raw_response={"code": 0},
            )

        def add_bitable_collaborator(self, **kwargs):
            self.collaborator_calls.append(kwargs)
            return FeishuBitableCollaboratorResult(
                member_id=kwargs["member_id"],
                raw_response={"code": 0},
            )

    repository = InMemoryOfferPilotRepository()
    bitable_service = FakeBitableService()
    registry = build_offerpilot_tool_registry(
        repository,
        calendar_service=None,
        bitable_service=bitable_service,
    )

    create_result = registry.run(
        AgentActionName.CREATE_APPLICATION.value,
        {
            "company": "腾讯",
            "role": "AI 应用开发",
            "bitable_collaborator_user_id": "ou_test",
            "bitable_collaborator_user_id_type": "open_id",
        },
    )
    update_result = registry.run(
        AgentActionName.UPDATE_APPLICATION.value,
        {
            "company": "腾讯",
            "round": "一面",
            "interview_time": "后天下午三点",
            "start_at": "2026-07-03T15:00:00+08:00",
            "update_type": "schedule_interview",
            "status": "interview_1",
            "bitable_collaborator_user_id": "ou_test",
            "bitable_collaborator_user_id_type": "open_id",
        },
    )

    assert create_result.success is True
    assert update_result.success is True
    assert bitable_service.create_app_calls == 1
    assert bitable_service.create_table_calls == 1
    assert len(bitable_service.collaborator_calls) == 1
    assert len(bitable_service.created_records) == 1
    assert bitable_service.updated_records == [
        {
            "app_token": "bascn_offerpilot",
            "table_id": "tbl_applications",
            "record_id": "rec_app_1",
            "fields": {
                "OfferPilot记录ID": "app_1",
                "公司": "腾讯",
                "岗位": "AI 应用开发",
                "投递状态": "一面阶段",
                "状态值": "interview_1",
                "优先级": "高",
                "来源": "飞书助手",
                "下一步": "准备 一面，时间：后天下午三点",
                "最后同步时间": bitable_service.updated_records[0]["fields"]["最后同步时间"],
                "面试轮次": "一面",
                "面试时间文本": "后天下午三点",
                "面试开始时间": 1783062000000,
                "提醒分钟": 30,
            },
        }
    ]
    assert update_result.data["bitable_sync"]["operation"] == "updated"
    assert update_result.data["bitable_sync"]["bitable_auto_created"] is False
    assert update_result.data["bitable_sync"]["web_url"] == (
        "https://feishu.cn/base/bascn_offerpilot?table=tbl_applications"
    )
    assert "已同步到飞书多维表格" in update_result.message


def test_offerpilot_tools_summarizes_bitable_collaborator_permission_error() -> None:
    raw_error = (
        'Feishu OpenAPI returned HTTP 400: {"code":99991672,'
        '"msg":"Access denied. One of the following scopes is required: '
        '[drive:drive, drive:file, docs:doc, sheets:spreadsheet]."}'
    )

    summary = offerpilot_tools._summarize_bitable_collaborator_error(raw_error)

    assert "缺少云文档协作者授权权限" in summary
    assert "drive:drive 或 drive:file" in summary


def test_offerpilot_tools_create_application_with_generic_interview_creates_schedule(monkeypatch) -> None:
    monkeypatch.setattr(
        offerpilot_tools,
        "parse_chinese_datetime",
        lambda text: datetime(2026, 7, 1, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    repository = InMemoryOfferPilotRepository()
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)

    result = registry.run(
        AgentActionName.CREATE_APPLICATION.value,
        {
            "company": "深信服",
            "role": "AI 应用开发",
            "interview_time": "明天下午三点",
            "raw_message": "我投递了深信服 AI 应用开发，明天下午三点有个面试",
        },
    )

    assert result.success is True
    assert result.data["application"]["status"] == "interview_scheduled"
    assert result.data["application"]["round"] == "面试"
    assert result.data["interview_schedule"]["round"] == "面试"
    assert result.data["interview_schedule"]["start_time"] == "明天下午三点"
    assert result.data["interview_schedule"]["start_at"] == "2026-07-01T15:00:00+08:00"
    assert "已记录面试安排" in result.message


def test_offerpilot_tools_query_company_application_status() -> None:
    repository = InMemoryOfferPilotRepository()
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)
    repository.create_application(
        company="深信服",
        role="AI 应用开发",
        interview_time="今晚",
        round_name="二面",
    )

    result = registry.run(
        AgentActionName.QUERY_APPLICATION.value,
        {"query_type": "company_status", "company": "深信服"},
    )

    assert result.success is True
    assert result.data["applications"][0]["status"] == "interview_2"
    assert "深信服 当前进度" in result.message
    assert "二面阶段" in result.message
    assert "今晚" in result.message


def test_offerpilot_tools_withdraws_all_company_applications() -> None:
    repository = InMemoryOfferPilotRepository()
    first = repository.create_application(company="vivo", role="AI 应用开发")
    second = repository.create_application(company="vivo", role="Java 后端")
    repository.create_application(company="小米", role="AI 应用开发")

    result = update_application(
        repository,
        {
            "company": "vivo",
            "update_type": "withdraw",
            "status": "withdrawn",
            "apply_to_all": True,
        },
        calendar_service=None,
        bitable_service=None,
    )

    vivo_applications = repository.list_applications(company="vivo")
    assert result.success is True
    assert "已更新 2 条 vivo 投递记录" in result.message
    assert [application.status for application in vivo_applications] == [
        ApplicationStatus.WITHDRAWN,
        ApplicationStatus.WITHDRAWN,
    ]
    assert result.data["applications"][0]["id"] == first.id
    assert result.data["applications"][1]["id"] == second.id


def test_offerpilot_tools_query_upcoming_interviews() -> None:
    repository = InMemoryOfferPilotRepository()
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)
    repository.create_application(
        company="深信服",
        role="AI 应用开发",
        interview_time="今晚",
        round_name="二面",
    )
    repository.create_application(company="美团", role="Java 后端")

    result = registry.run(
        AgentActionName.QUERY_APPLICATION.value,
        {"query_type": "upcoming_interviews"},
    )

    assert result.success is True
    assert len(result.data["applications"]) == 1
    assert "近期笔试/面试安排" in result.message
    assert "深信服" in result.message
    assert "美团" not in result.message


def test_offerpilot_tools_update_application_and_create_interview_schedule(monkeypatch) -> None:
    monkeypatch.setattr(
        offerpilot_tools,
        "parse_chinese_datetime",
        lambda text: datetime(2026, 6, 30, 8, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    repository = InMemoryOfferPilotRepository()
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)
    repository.create_application(company="深信服", role="AI 应用开发")

    result = registry.run(
        AgentActionName.UPDATE_APPLICATION.value,
        {
            "company": "深信服",
            "round": "一面",
            "interview_time": "明天早上八点",
            "update_type": "schedule_interview",
            "status": "interview_1",
            "raw_message": "明天早上八点，深信服约我一面",
        },
    )

    assert result.success is True
    assert result.data["application"]["status"] == "interview_1"
    assert result.data["application"]["round"] == "一面"
    assert result.data["interview_schedule"]["company"] == "深信服"
    assert result.data["interview_schedule"]["start_time"] == "明天早上八点"
    assert result.data["interview_schedule"]["start_at"] == "2026-06-30T08:00:00+08:00"
    assert result.data["interview_schedule"]["reminder_minutes"] == 30
    assert "默认提前 30 分钟提醒" in result.message


def test_offerpilot_tools_syncs_interview_schedule_to_calendar() -> None:
    class FakeCalendarService:
        def is_calendar_sync_enabled(self):
            return True

        def create_interview_event(self, **kwargs):
            assert kwargs["company"] == "深信服"
            assert kwargs["role"] == "AI 应用开发"
            assert kwargs["round_name"] == "一面"
            assert kwargs["start_time_text"] == "明天早上八点"
            assert kwargs["start_at"] == "2026-06-30T08:00:00+08:00"
            assert kwargs["reminder_minutes"] == 30
            return FeishuCalendarEventResult(
                event_id="evt_test_1",
                raw_response={"code": 0},
            )

    repository = InMemoryOfferPilotRepository()
    registry = build_offerpilot_tool_registry(
        repository,
        calendar_service=FakeCalendarService(),
        bitable_service=None,
    )
    repository.create_application(company="深信服", role="AI 应用开发")

    result = registry.run(
        AgentActionName.UPDATE_APPLICATION.value,
        {
            "company": "深信服",
            "round": "一面",
            "interview_time": "明天早上八点",
            "start_at": "2026-06-30T08:00:00+08:00",
            "update_type": "schedule_interview",
            "status": "interview_1",
        },
    )

    assert result.success is True
    assert result.data["calendar_sync"]["synced"] is True
    assert result.data["calendar_sync"]["status"] == "synced"
    assert result.data["calendar_sync"]["calendar_event_id"] == "evt_test_1"
    assert result.data["interview_schedule"]["calendar_event_id"] == "evt_test_1"
    assert repository.interview_schedules[0].calendar_event_id == "evt_test_1"
    assert "已同步到飞书日历" in result.message


def test_offerpilot_tools_auto_creates_managed_calendar_once() -> None:
    class FakeCalendarService:
        calendar_id = "primary"

        def __init__(self):
            self.create_calendar_calls = 0
            self.created_events = []

        def is_calendar_sync_enabled(self):
            return True

        def should_manage_offerpilot_calendar(self):
            return True

        def create_shared_calendar(self):
            self.create_calendar_calls += 1
            return FeishuCalendarResult(
                calendar_id="feishu.cn_offerpilot@group.calendar.feishu.cn",
                raw_response={"code": 0},
            )

        def create_interview_event(self, **kwargs):
            self.created_events.append(kwargs)
            return FeishuCalendarEventResult(
                event_id=f"evt_test_{len(self.created_events)}",
                raw_response={"code": 0},
            )

    repository = InMemoryOfferPilotRepository()
    calendar_service = FakeCalendarService()
    registry = build_offerpilot_tool_registry(
        repository,
        calendar_service=calendar_service,
        bitable_service=None,
    )
    repository.create_application(company="深信服", role="AI 应用开发")

    first_result = registry.run(
        AgentActionName.UPDATE_APPLICATION.value,
        {
            "company": "深信服",
            "round": "一面",
            "interview_time": "明天早上八点",
            "start_at": "2026-06-30T08:00:00+08:00",
            "update_type": "schedule_interview",
            "status": "interview_1",
        },
    )
    second_result = registry.run(
        AgentActionName.UPDATE_APPLICATION.value,
        {
            "company": "深信服",
            "round": "二面",
            "interview_time": "后天下午三点",
            "start_at": "2026-07-02T15:00:00+08:00",
            "update_type": "schedule_interview",
            "status": "interview_2",
        },
    )

    assert first_result.success is True
    assert second_result.success is True
    assert calendar_service.create_calendar_calls == 1
    assert repository.get_runtime_setting("feishu.offerpilot_calendar_id") == (
        "feishu.cn_offerpilot@group.calendar.feishu.cn"
    )
    assert calendar_service.created_events[0]["calendar_id"] == (
        "feishu.cn_offerpilot@group.calendar.feishu.cn"
    )
    assert calendar_service.created_events[1]["calendar_id"] == (
        "feishu.cn_offerpilot@group.calendar.feishu.cn"
    )
    assert first_result.data["calendar_sync"]["calendar_auto_created"] is True
    assert second_result.data["calendar_sync"]["calendar_auto_created"] is False
    assert "已自动创建 OfferPilot 秋招日历" in first_result.message


def test_offerpilot_tools_adds_feishu_user_as_event_attendee() -> None:
    class FakeCalendarService:
        calendar_id = "primary"

        def __init__(self):
            self.add_attendee_calls = []

        def is_calendar_sync_enabled(self):
            return True

        def should_manage_offerpilot_calendar(self):
            return True

        def create_shared_calendar(self):
            return FeishuCalendarResult(
                calendar_id="feishu.cn_offerpilot@group.calendar.feishu.cn",
                raw_response={"code": 0},
            )

        def create_interview_event(self, **kwargs):
            return FeishuCalendarEventResult(
                event_id="evt_test_1",
                raw_response={"code": 0},
            )

        def add_event_attendee(self, **kwargs):
            self.add_attendee_calls.append(kwargs)
            return FeishuCalendarAttendeeResult(
                attendee_ids=["user_attendee_1"],
                raw_response={"code": 0},
            )

    repository = InMemoryOfferPilotRepository()
    calendar_service = FakeCalendarService()
    registry = build_offerpilot_tool_registry(
        repository,
        calendar_service=calendar_service,
        bitable_service=None,
    )
    repository.create_application(company="深信服", role="AI 应用开发")

    result = registry.run(
        AgentActionName.UPDATE_APPLICATION.value,
        {
            "company": "深信服",
            "round": "一面",
            "interview_time": "明天早上八点",
            "start_at": "2026-06-30T08:00:00+08:00",
            "update_type": "schedule_interview",
            "status": "interview_1",
            "attendee_user_id": "ou_test",
            "attendee_user_id_type": "open_id",
        },
    )

    assert result.success is True
    assert calendar_service.add_attendee_calls == [
        {
            "calendar_id": "feishu.cn_offerpilot@group.calendar.feishu.cn",
            "event_id": "evt_test_1",
            "user_id": "ou_test",
            "user_id_type": "open_id",
            "need_notification": True,
        }
    ]
    assert result.data["calendar_sync"]["attendee_sync"] == {
        "synced": True,
        "status": "synced",
        "attendee_ids": ["user_attendee_1"],
    }
    assert "已将你加入日程参与人" in result.message


def test_offerpilot_tools_query_upcoming_interviews_uses_schedule_records() -> None:
    repository = InMemoryOfferPilotRepository()
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)
    application = repository.create_application(company="深信服", role="AI 应用开发")
    repository.create_interview_schedule(
        application_id=application.id,
        company="深信服",
        role="AI 应用开发",
        round_name="一面",
        start_time="明天早上八点",
    )

    result = registry.run(
        AgentActionName.QUERY_APPLICATION.value,
        {"query_type": "upcoming_interviews"},
    )

    assert result.success is True
    assert result.data["interview_schedules"][0]["start_time"] == "明天早上八点"
    assert "提前 30 分钟提醒" in result.message


def test_offerpilot_tools_update_application_passed_round() -> None:
    repository = InMemoryOfferPilotRepository()
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)
    repository.create_application(company="深信服", role="AI 应用开发", round_name="一面")

    result = registry.run(
        AgentActionName.UPDATE_APPLICATION.value,
        {
            "company": "深信服",
            "round": "一面",
            "update_type": "pass_round",
            "status": "interview_1_passed",
        },
    )

    assert result.success is True
    assert result.data["application"]["status"] == "interview_1_passed"
    assert result.data["interview_schedule"] is None
    assert "一面通过" in result.message


def test_offerpilot_tools_complete_seed_task() -> None:
    repository = InMemoryOfferPilotRepository()
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)

    result = registry.run(
        AgentActionName.COMPLETE_TASK.value,
        {"task_title": "反转链表"},
    )

    assert result.success is True
    assert result.data["task"]["id"] == "task_1"
    assert result.data["task"]["status"] == "passed"


def test_update_application_requires_application_id_when_company_has_multiple_matches() -> None:
    repository = InMemoryOfferPilotRepository()
    first = repository.create_application(company="美团", role="Java 后端")
    second = repository.create_application(company="美团", role="AI 应用开发")

    result = update_application(
        repository,
        {"company": "美团", "status": "rejected", "update_type": "reject"},
        calendar_service=None,
        bitable_service=None,
    )

    assert result.success is False
    assert result.data["requires_selection"] is True
    assert result.data["selection_slot"] == "application_id"
    assert [candidate["application_id"] for candidate in result.data["candidates"]] == [first.id, second.id]
    assert all(application.status != ApplicationStatus.REJECTED for application in repository.applications)


def test_reschedule_interview_updates_existing_schedule_by_stable_ids() -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="美团", role="Java 后端", round_name="一面")
    schedule = repository.create_interview_schedule(
        application_id=application.id,
        company=application.company,
        role=application.role,
        round_name="一面",
        start_time="明天下午三点",
        start_at="2026-07-17T15:00:00+08:00",
    )
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)

    result = registry.run(
        "reschedule_interview",
        {
            "application_id": application.id,
            "schedule_id": schedule.id,
            "interview_time": "后天下午四点",
            "start_at": "2026-07-18T16:00:00+08:00",
        },
    )

    assert result.success is True
    assert result.data["application"]["id"] == application.id
    assert result.data["interview_schedule"]["id"] == schedule.id
    assert result.data["interview_schedule"]["start_time"] == "后天下午四点"
    assert len(repository.interview_schedules) == 1


def test_cancel_interview_updates_schedule_status_by_schedule_id() -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="美团", role="Java 后端", round_name="一面")
    schedule = repository.create_interview_schedule(
        application_id=application.id,
        company=application.company,
        role=application.role,
        round_name="一面",
        start_time="明天下午三点",
    )
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)

    result = registry.run(
        "cancel_interview",
        {"application_id": application.id, "schedule_id": schedule.id},
    )

    assert result.success is True
    assert result.data["interview_schedule"]["id"] == schedule.id
    assert result.data["interview_schedule"]["status"] == "cancelled"
    assert result.data["application"]["status"] == "submitted"
    assert result.data["application"]["interview_time"] is None
    assert result.data["application"]["round"] is None


def test_update_application_uses_stable_id_even_when_descriptive_slots_are_stale() -> None:
    repository = InMemoryOfferPilotRepository()
    target = repository.create_application(company="美团", role="Java 后端")
    repository.create_application(company="字节跳动", role="AI 应用开发")

    result = update_application(
        repository,
        {
            "application_id": target.id,
            "company": "错误公司",
            "role": "错误岗位",
            "status": "rejected",
            "update_type": "reject",
        },
        calendar_service=None,
        bitable_service=None,
    )

    assert result.success is True
    assert result.data["application"]["id"] == target.id
    assert result.data["application"]["status"] == "rejected"


def test_reschedule_interview_rejects_unparseable_time_without_preserving_stale_start_at() -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="美团", role="Java 后端")
    schedule = repository.create_interview_schedule(
        application_id=application.id,
        company="美团",
        role="Java 后端",
        round_name="一面",
        start_time="明天下午三点",
        start_at="2026-07-17T15:00:00+08:00",
    )
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)

    result = registry.run(
        "reschedule_interview",
        {"schedule_id": schedule.id, "interview_time": "有空的时候"},
    )

    assert result.success is False
    assert repository.interview_schedules[0].start_time == "明天下午三点"
    assert repository.interview_schedules[0].start_at == "2026-07-17T15:00:00+08:00"


def test_reschedule_interview_rejects_invalid_explicit_start_at() -> None:
    repository = InMemoryOfferPilotRepository()
    schedule = repository.create_interview_schedule(
        company="美团",
        role="Java 后端",
        round_name="一面",
        start_time="明天下午三点",
        start_at="2026-07-17T15:00:00+08:00",
    )
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)

    result = registry.run(
        "reschedule_interview",
        {
            "schedule_id": schedule.id,
            "interview_time": "后天下午四点",
            "start_at": "not-an-iso-datetime",
        },
    )

    assert result.success is False
    assert repository.interview_schedules[0].start_time == "明天下午三点"
    assert repository.interview_schedules[0].start_at == "2026-07-17T15:00:00+08:00"


def test_reschedule_interview_rejects_explicit_start_at_that_contradicts_user_time() -> None:
    repository = InMemoryOfferPilotRepository()
    schedule = repository.create_interview_schedule(
        company="美团",
        role="Java 后端",
        round_name="一面",
        start_time="明天下午三点",
        start_at="2026-07-17T15:00:00+08:00",
    )
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)

    result = registry.run(
        "reschedule_interview",
        {
            "schedule_id": schedule.id,
            "interview_time": "后天下午四点",
            "start_at": "2026-08-18T16:00:00+08:00",
        },
    )

    assert result.success is False
    assert repository.interview_schedules[0].start_time == "明天下午三点"
    assert repository.interview_schedules[0].start_at == "2026-07-17T15:00:00+08:00"


def test_cancel_interview_reconciles_application_to_earliest_remaining_schedule() -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="美团", role="Java 后端", round_name="一面")
    cancelled = repository.create_interview_schedule(
        application_id=application.id,
        company="美团",
        role="Java 后端",
        round_name="一面",
        start_time="明天下午三点",
        start_at="2026-07-17T15:00:00+08:00",
    )
    repository.create_interview_schedule(
        application_id=application.id,
        company="美团",
        role="Java 后端",
        round_name="三面",
        start_time="下周三下午三点",
        start_at="2026-07-22T15:00:00+08:00",
    )
    earliest = repository.create_interview_schedule(
        application_id=application.id,
        company="美团",
        role="Java 后端",
        round_name="二面",
        start_time="下周一下午三点",
        start_at="2026-07-20T15:00:00+08:00",
    )
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)

    result = registry.run("cancel_interview", {"schedule_id": cancelled.id})

    assert result.success is True
    assert result.data["application"]["round"] == earliest.round
    assert result.data["application"]["interview_time"] == earliest.start_time
    assert result.data["application"]["status"] == "interview_2"


def test_reschedule_interview_returns_schedule_candidates_without_mutating() -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="美团", role="Java 后端")
    first = repository.create_interview_schedule(
        application_id=application.id,
        company="美团",
        role="Java 后端",
        round_name="一面",
        start_time="明天下午三点",
    )
    second = repository.create_interview_schedule(
        application_id=application.id,
        company="美团",
        role="Java 后端",
        round_name="二面",
        start_time="后天下午三点",
    )
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)

    result = registry.run(
        "reschedule_interview",
        {"application_id": application.id, "interview_time": "下周一下午四点"},
    )

    assert result.success is False
    assert result.data["requires_selection"] is True
    assert result.data["selection_slot"] == "schedule_id"
    assert [candidate["schedule_id"] for candidate in result.data["candidates"]] == [first.id, second.id]
    assert first.start_time == "明天下午三点"
    assert second.start_time == "后天下午三点"
