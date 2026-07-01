from datetime import datetime
from zoneinfo import ZoneInfo

from app.schemas.agent import AgentActionName
from app.services.feishu_service import (
    FeishuCalendarAttendeeResult,
    FeishuCalendarEventResult,
    FeishuCalendarResult,
)
from app.schemas.tool import ToolResult
from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository
from app.tools import offerpilot_tools
from app.tools.offerpilot_tools import build_offerpilot_tool_registry
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
    registry = build_offerpilot_tool_registry(repository)

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


def test_offerpilot_tools_create_application_with_generic_interview_creates_schedule(monkeypatch) -> None:
    monkeypatch.setattr(
        offerpilot_tools,
        "parse_chinese_datetime",
        lambda text: datetime(2026, 7, 1, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    repository = InMemoryOfferPilotRepository()
    registry = build_offerpilot_tool_registry(repository)

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
    registry = build_offerpilot_tool_registry(repository)
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


def test_offerpilot_tools_query_upcoming_interviews() -> None:
    repository = InMemoryOfferPilotRepository()
    registry = build_offerpilot_tool_registry(repository)
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
    registry = build_offerpilot_tool_registry(repository)
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
    registry = build_offerpilot_tool_registry(repository)
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
    registry = build_offerpilot_tool_registry(repository)
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
    registry = build_offerpilot_tool_registry(repository)

    result = registry.run(
        AgentActionName.COMPLETE_TASK.value,
        {"task_title": "反转链表"},
    )

    assert result.success is True
    assert result.data["task"]["id"] == "task_1"
    assert result.data["task"]["status"] == "passed"
