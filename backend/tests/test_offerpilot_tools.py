from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Event
from zoneinfo import ZoneInfo

import pytest

from app.core.config import Settings
from app.tools.tool_names import AgentActionName
from app.models.application import ApplicationStatus
from app.services.feishu_service import (
    FeishuBitableAppResult,
    FeishuBitableCollaboratorResult,
    FeishuBitableRecordResult,
    FeishuBitableTableResult,
    FeishuCalendarAttendeeResult,
    FeishuCalendarEventResult,
    FeishuCalendarResult,
    FeishuRequestError,
)
from app.schemas.tool import ToolResult
from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository
from app.tools import offerpilot_tools
from app.tools.offerpilot_tools import build_offerpilot_tool_registry, update_application
from app.tools.registry import ToolNotFoundError, ToolRegistry
from app.services.bitable_sync_service import BitableRecordSyncResult


class MissingRecordBitableService:
    app_token = "bascn_missing_record_test"
    table_id = "tbl_missing_record_test"

    def __init__(self, error=None):
        self.error = error
        self.updated_records = []
        self.created_records = []

    def is_bitable_sync_enabled(self):
        return True

    def should_manage_offerpilot_bitable(self):
        return False

    def update_record(self, **kwargs):
        self.updated_records.append(kwargs)
        if self.error:
            raise self.error
        return FeishuBitableRecordResult(record_id=kwargs["record_id"], raw_response={"code": 0})

    def create_record(self, **kwargs):
        self.created_records.append(kwargs)
        return FeishuBitableRecordResult(
            record_id=f"rec_created_{len(self.created_records)}",
            raw_response={"code": 0},
        )


def _remember_missing_record_test_mapping(repository, service, application):
    offerpilot_tools.remember_bitable_table_owner(
        repository, service.app_token, service.table_id, application.owner_id
    )
    offerpilot_tools.remember_bitable_record_mapping(
        repository,
        service.app_token,
        service.table_id,
        "rec_removed",
        application.id,
        application.owner_id,
    )


def test_bitable_missing_record_deletes_local_application_without_retry_or_recreation() -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="Example", role="Engineer")
    service = MissingRecordBitableService(FeishuRequestError("RecordIdNotFound", code=1254043))
    _remember_missing_record_test_mapping(repository, service, application)
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=service)

    result = registry.run("update_application", {"application_id": application.id, "status": "offer"})

    assert result.success is False
    assert result.data["operation_status"] == "deleted"
    assert result.data["retryable"] is False
    assert result.data["application"] is None
    assert result.data["bitable_sync"]["synced"] is False
    assert result.data["bitable_sync"]["status"] == "deleted"
    assert result.data["bitable_sync"]["retry_safe"] is False
    assert result.data["bitable_sync"]["attempts"] == 1
    assert "已在飞书多维表格中删除" in result.message
    assert "已更新投递记录" not in result.message
    assert repository.is_application_deleted(application.id, application.owner_id)
    assert repository.list_applications() == []
    assert len(service.updated_records) == 1
    assert service.created_records == []


@pytest.mark.parametrize(
    "error",
    [
        FeishuRequestError("Feishu OpenAPI returned HTTP 404: not found"),
        FeishuRequestError("permission denied", code=99991672),
        FeishuRequestError("network timeout"),
        FeishuRequestError("RecordIdNotFound 1254043"),
        FeishuRequestError("TableIdNotFound", code=1254004),
    ],
)
def test_bitable_other_errors_do_not_delete_application_or_create_record(error) -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="Example", role="Engineer")
    service = MissingRecordBitableService(error)
    _remember_missing_record_test_mapping(repository, service, application)

    result = offerpilot_tools._sync_application_to_bitable(
        repository, application, None, None, service
    )

    assert result["status"] == "failed"
    assert result["attempts"] == 3
    assert repository.is_application_deleted(application.id, application.owner_id) is False
    assert repository.list_applications() == [application]
    assert len(service.updated_records) == 3
    assert service.created_records == []


def test_bitable_deleted_stale_application_cannot_replay_success_or_recreate_record() -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="Example", role="Engineer")
    service = MissingRecordBitableService()
    repository.delete_application(application.id, application.owner_id)
    offerpilot_tools._store_sync_result(
        repository, "previous-sync", {"synced": True, "status": "synced", "record_id": "rec_removed"}
    )

    once = offerpilot_tools._sync_application_to_bitable_once(
        repository, application, None, None, service
    )
    replay = offerpilot_tools._sync_application_to_bitable(
        repository, application, None, None, service, idempotency_key="previous-sync"
    )

    assert once["status"] == "deleted"
    assert replay["status"] == "deleted"
    assert replay["synced"] is False
    assert replay["attempts"] == 0
    assert service.updated_records == []
    assert service.created_records == []
    assert repository.list_runtime_settings("feishu.offerpilot_bitable") == {}


def test_bitable_new_application_after_remote_deletion_creates_distinct_record() -> None:
    repository = InMemoryOfferPilotRepository()
    previous = repository.create_application(company="Example", role="Engineer")
    service = MissingRecordBitableService(FeishuRequestError("RecordIdNotFound", code=1254043))
    _remember_missing_record_test_mapping(repository, service, previous)
    deleted = offerpilot_tools._sync_application_to_bitable(repository, previous, None, None, service)
    service.error = None
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=service)

    created = registry.run("create_application", {"company": "Example", "role": "Engineer"})

    assert deleted["status"] == "deleted"
    assert created.success is True
    assert created.data["application"]["id"] != previous.id
    assert created.data["bitable_sync"]["record_id"] != "rec_removed"
    assert len(service.created_records) == 1
    assert len(service.updated_records) == 1
    assert len(repository.list_applications()) == 1


@pytest.mark.parametrize(
    "deletion_status",
    ["owner_conflict", "unmapped_deleted", "ignored_replaced_record", "ignored_foreign_table", "missing_application"],
)
def test_bitable_missing_record_requires_confirmed_application_deletion(monkeypatch, deletion_status) -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="Example", role="Engineer")
    service = MissingRecordBitableService(FeishuRequestError("RecordIdNotFound", code=1254043))
    _remember_missing_record_test_mapping(repository, service, application)
    monkeypatch.setattr(
        offerpilot_tools,
        "sync_deleted_bitable_record_to_repository",
        lambda **kwargs: BitableRecordSyncResult(synced=deletion_status != "owner_conflict", status=deletion_status),
    )

    result = offerpilot_tools._sync_application_to_bitable(repository, application, None, None, service)

    assert result["status"] == "failed"
    assert result["retry_safe"] is False
    assert result["attempts"] == 1
    assert result["deletion_sync_status"] == deletion_status
    assert repository.is_application_deleted(application.id, application.owner_id) is False
    assert len(service.updated_records) == 1
    assert service.created_records == []


def test_bitable_missing_record_with_foreign_mapping_does_not_delete_either_owner() -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="Example", role="Engineer")
    foreign_application = repository.create_application(
        company="Other", role="Engineer", owner_id="other_owner"
    )
    service = MissingRecordBitableService(FeishuRequestError("RecordIdNotFound", code=1254043))
    _remember_missing_record_test_mapping(repository, service, application)
    offerpilot_tools.remember_bitable_record_mapping(
        repository,
        service.app_token,
        service.table_id,
        "rec_removed",
        foreign_application.id,
        foreign_application.owner_id,
    )

    result = offerpilot_tools._sync_application_to_bitable(repository, application, None, None, service)

    assert result["status"] == "failed"
    assert result["deletion_sync_status"] == "owner_conflict"
    assert result["attempts"] == 1
    assert repository.is_application_deleted(application.id, application.owner_id) is False
    assert repository.is_application_deleted(foreign_application.id, foreign_application.owner_id) is False
    assert service.created_records == []


def test_bitable_missing_record_accepts_already_deleted_application() -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="Example", role="Engineer")

    class ConcurrentDeletionBitableService(MissingRecordBitableService):
        def update_record(self, **kwargs):
            repository.delete_application(application.id, application.owner_id)
            return super().update_record(**kwargs)

    service = ConcurrentDeletionBitableService(FeishuRequestError("RecordIdNotFound", code=1254043))
    _remember_missing_record_test_mapping(repository, service, application)

    result = offerpilot_tools._sync_application_to_bitable(repository, application, None, None, service)

    assert result["status"] == "deleted"
    assert result["synced"] is False
    assert result["attempts"] == 1
    assert service.created_records == []


def test_bitable_deletion_during_table_setup_cannot_recreate_remote_record(monkeypatch) -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="Example", role="Engineer")
    service = MissingRecordBitableService()
    _remember_missing_record_test_mapping(repository, service, application)
    ensure_bitable = offerpilot_tools._ensure_bitable_for_sync

    def delete_during_setup(*args, **kwargs):
        context = ensure_bitable(*args, **kwargs)
        deletion = offerpilot_tools.sync_deleted_bitable_record_to_repository(
            repository, "rec_removed", service.app_token, service.table_id, application.owner_id
        )
        assert deletion.status == "deleted"
        return context

    monkeypatch.setattr(offerpilot_tools, "_ensure_bitable_for_sync", delete_during_setup)

    result = offerpilot_tools._sync_application_to_bitable(repository, application, None, None, service)

    assert result["status"] == "deleted"
    assert result["synced"] is False
    assert service.updated_records == []
    assert service.created_records == []


def test_bitable_record_write_and_remote_deletion_share_table_lock() -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="Example", role="Engineer")
    update_started = Event()
    release_update = Event()
    deletion_started = Event()

    class BlockingBitableService(MissingRecordBitableService):
        def update_record(self, **kwargs):
            update_started.set()
            assert release_update.wait(timeout=5)
            return super().update_record(**kwargs)

    service = BlockingBitableService()
    _remember_missing_record_test_mapping(repository, service, application)

    def delete_remote_record():
        deletion_started.set()
        return offerpilot_tools.sync_deleted_bitable_record_to_repository(
            repository, "rec_removed", service.app_token, service.table_id, application.owner_id
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        writer = executor.submit(
            offerpilot_tools._sync_application_to_bitable, repository, application, None, None, service
        )
        try:
            assert update_started.wait(timeout=5)
            deletion = executor.submit(delete_remote_record)
            assert deletion_started.wait(timeout=5)
            assert deletion.done() is False
            assert repository.is_application_deleted(application.id, application.owner_id) is False
        finally:
            release_update.set()
        writer.result(timeout=5)
        assert deletion.result(timeout=5).status == "deleted"

    stale_write = offerpilot_tools._sync_application_to_bitable(repository, application, None, None, service)

    assert stale_write["status"] == "deleted"
    assert repository.get_runtime_setting(
        offerpilot_tools._bitable_record_setting_key(service.table_id, application.id)
    ) == ""
    assert len(service.updated_records) == 1
    assert service.created_records == []


@pytest.mark.parametrize(
    "sync_function",
    [
        offerpilot_tools._sync_interview_schedule_to_calendar,
        offerpilot_tools._sync_rescheduled_interview_to_calendar,
        offerpilot_tools._sync_cancelled_interview_to_calendar,
    ],
)
@pytest.mark.parametrize("event_id", [None, "event_retained"])
def test_deleted_application_stale_schedule_cannot_sync_calendar_or_replay_success(sync_function, event_id) -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="Example", role="Engineer", owner_id="calendar_owner")
    schedule = repository.create_interview_schedule(
        application_id=application.id,
        company=application.company,
        role=application.role,
        round_name="一面",
        start_time="明天下午三点",
        start_at="2026-09-10T15:00:00+08:00",
        owner_id=application.owner_id,
    )
    repository.update_interview_schedule_calendar_event(schedule.id, event_id, owner_id=application.owner_id)
    repository.delete_application(application.id, application.owner_id)
    offerpilot_tools._store_sync_result(
        repository, "previous-calendar-sync", {"synced": True, "status": "synced", "calendar_event_id": event_id}
    )

    class UnexpectedCalendarService:
        def is_calendar_sync_enabled(self):
            raise AssertionError("Deleted applications must stop before accessing calendar service")

    result = sync_function(
        repository=repository,
        schedule=schedule,
        calendar_service=UnexpectedCalendarService(),
        owner_id=application.owner_id,
        idempotency_key="previous-calendar-sync",
    )

    assert result["status"] == "deleted"
    assert result["synced"] is False
    assert result["retry_safe"] is False
    assert result["attempts"] == 0
    assert "停止关联面试的日历同步" in result["message"]
    assert offerpilot_tools._format_calendar_sync_lines(result) == [result["message"]]
    assert schedule.calendar_event_id == event_id


def test_bulk_update_reports_remote_deleted_application_without_success() -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="Example", role="Engineer")
    service = MissingRecordBitableService(FeishuRequestError("RecordIdNotFound", code=1254043))
    _remember_missing_record_test_mapping(repository, service, application)
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=service)

    result = registry.run("update_application", {"company": "Example", "apply_to_all": True, "status": "offer"})

    assert result.success is False
    assert result.data["applications"] == []
    assert result.data["deleted_application_ids"] == [application.id]
    assert "1 条记录已在飞书多维表格中删除" in result.message
    assert "已更新 1 条" not in result.message


@pytest.mark.parametrize("tool_name", ["reschedule_interview", "cancel_interview"])
def test_schedule_update_reports_remote_deleted_application_without_success(monkeypatch, tool_name) -> None:
    monkeypatch.setattr(
        offerpilot_tools, "_normalize_interview_start_at", lambda value: "2026-09-11T16:00:00+08:00"
    )
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="Example", role="Engineer")
    schedule = repository.create_interview_schedule(
        application_id=application.id,
        company=application.company,
        role=application.role,
        round_name="一面",
        start_time="2026-09-10 15:00",
        start_at="2026-09-10T15:00:00+08:00",
    )
    service = MissingRecordBitableService(FeishuRequestError("RecordIdNotFound", code=1254043))
    _remember_missing_record_test_mapping(repository, service, application)
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=service)

    result = registry.run(
        tool_name,
        {"schedule_id": schedule.id, "interview_time": "2026-09-11 16:00"},
    )

    assert result.success is False
    assert result.data["application"] is None
    assert result.data["operation_status"] == "deleted"
    assert result.data["retryable"] is False
    assert "已在飞书多维表格中删除" in result.message
    assert len(service.updated_records) == 1
    assert service.created_records == []


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


def test_application_bitable_link_reports_disabled_and_not_ready_states() -> None:
    repository = InMemoryOfferPilotRepository()

    disabled = offerpilot_tools.get_application_bitable_link(
        repository,
        app_settings=Settings(feishu_bitable_sync_enabled=False),
    )
    not_ready = offerpilot_tools.get_application_bitable_link(
        repository,
        app_settings=Settings(feishu_bitable_sync_enabled=True),
    )

    assert disabled.success is False
    assert disabled.data["error"]["code"] == "bitable_sync_disabled"
    assert "尚未启用" in disabled.message
    assert not_ready.success is False
    assert not_ready.data["error"]["code"] == "bitable_not_ready"
    assert "尚未创建" in not_ready.message


def test_application_bitable_link_requires_recorded_nonlocal_access() -> None:
    repository = InMemoryOfferPilotRepository()
    repository.set_runtime_setting(
        "feishu.offerpilot_bitable_app_token", "bascn_offerpilot"
    )
    repository.set_runtime_setting(
        "feishu.offerpilot_bitable_table_id", "tbl_applications"
    )

    result = offerpilot_tools.get_application_bitable_link(
        repository,
        {"owner_id": "feishu:ou_unknown"},
        app_settings=Settings(feishu_bitable_sync_enabled=True),
    )

    assert result.success is False
    assert result.data["error"]["code"] == "bitable_access_unverified"
    assert "尚未确认你有访问权限" in result.message


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
            "base_location": "深圳",
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
    assert "工作地点：深圳" in result.message
    assert "状态：一面阶段" in result.message
    assert repository.applications[0].company == "深信服"


def test_offerpilot_tools_updates_and_queries_application_base_location() -> None:
    repository = InMemoryOfferPilotRepository()
    repository.create_application(
        company="字节",
        role="Agent 开发",
        base_location="杭州",
    )
    registry = build_offerpilot_tool_registry(
        repository,
        calendar_service=None,
        bitable_service=None,
    )

    update_result = registry.run(
        AgentActionName.UPDATE_APPLICATION.value,
        {
            "company": "字节",
            "base_location": "上海",
            "update_type": "location_update",
        },
    )
    query_result = registry.run(
        AgentActionName.QUERY_APPLICATION.value,
        {"company": "字节", "query_type": "company_status"},
    )

    assert update_result.success is True
    assert update_result.data["application"]["base_location"] == "上海"
    assert "工作地点：上海" in update_result.message
    assert "工作地点：上海" in query_result.message


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
            "base_location": "深圳",
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
    assert bitable_service.created_records[0]["fields"]["投递记录"] == "腾讯｜AI 应用开发"
    assert bitable_service.created_records[0]["fields"]["OfferPilot记录ID"] == "app_1"
    assert bitable_service.created_records[0]["fields"]["公司"] == "腾讯"
    assert bitable_service.created_records[0]["fields"]["岗位"] == "AI 应用开发"
    assert bitable_service.created_records[0]["fields"]["工作地点"] == "深圳"
    assert bitable_service.created_records[0]["fields"]["投递状态"] == "已投递"
    assert "状态值" not in bitable_service.created_records[0]["fields"]
    assert bitable_service.created_records[0]["fields"]["优先级"] == "高"
    assert bitable_service.created_records[0]["fields"]["来源"] == "飞书助手"
    assert bitable_service.created_records[0]["fields"]["下一步"] == "等待反馈，超过 7 天可跟进"
    assert repository.get_runtime_setting("feishu.offerpilot_bitable_app_token") == "bascn_offerpilot"
    assert repository.get_runtime_setting("feishu.offerpilot_bitable_table_id") == "tbl_applications"
    assert repository.get_runtime_setting("feishu.offerpilot_bitable_schema_version") == "v4"
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
                "投递记录": "腾讯｜AI 应用开发",
                "OfferPilot记录ID": "app_1",
                "公司": "腾讯",
                "岗位": "AI 应用开发",
                "工作地点": "",
                "投递状态": "一面阶段",
                "优先级": "高",
                "来源": "飞书助手",
                "下一步": "准备 一面，时间：后天下午三点",
                "最后同步时间": bitable_service.updated_records[0]["fields"]["最后同步时间"],
                "面试轮次": "一面",
                "面试时间文本": "后天下午三点",
                "面试开始时间": 1783062000000,
                "提醒分钟": 30,
                "日历事件ID": None,
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


def test_offerpilot_tools_creates_a_separate_managed_calendar_per_owner() -> None:
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
                calendar_id=f"calendar_{self.create_calendar_calls}",
                raw_response={"code": 0},
            )

        def create_interview_event(self, **kwargs):
            self.created_events.append(kwargs)
            return FeishuCalendarEventResult(
                event_id=f"event_{len(self.created_events)}",
                raw_response={"code": 0},
            )

    repository = InMemoryOfferPilotRepository()
    calendar_service = FakeCalendarService()
    registry = build_offerpilot_tool_registry(
        repository,
        calendar_service=calendar_service,
        bitable_service=None,
    )
    for owner_id, company in (("feishu:ou_a", "公司A"), ("feishu:ou_b", "公司B")):
        repository.create_application(
            company=company,
            role="后端开发",
            owner_id=owner_id,
        )
        result = registry.run(
            AgentActionName.UPDATE_APPLICATION.value,
            {
                "company": company,
                "round": "一面",
                "interview_time": "明天早上八点",
                "start_at": "2026-06-30T08:00:00+08:00",
                "update_type": "schedule_interview",
                "status": "interview_1",
                "owner_id": owner_id,
            },
        )
        assert result.success is True

    assert calendar_service.create_calendar_calls == 2
    assert [event["calendar_id"] for event in calendar_service.created_events] == [
        "calendar_1",
        "calendar_2",
    ]
    owner_settings = repository.list_runtime_settings(
        "feishu.offerpilot_calendar_id.owner."
    )
    assert len(owner_settings) == 2
    assert set(owner_settings.values()) == {"calendar_1", "calendar_2"}


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
        {"task_title": "HashMap"},
    )

    assert result.success is True
    assert result.data["task"]["id"] == "task_1"
    assert result.data["task"]["status"] == "passed"


def test_task_update_requires_selection_before_mutating_multiple_candidates() -> None:
    repository = InMemoryOfferPilotRepository()
    registry = build_offerpilot_tool_registry(
        repository,
        calendar_service=None,
        bitable_service=None,
    )
    before = [task.status.value for task in repository.list_today_tasks()]

    result = registry.run(AgentActionName.COMPLETE_TASK.value, {})

    assert result.success is False
    assert result.data["requires_selection"] is True
    assert result.data["selection_slot"] == "task_title"
    assert [candidate["task_id"] for candidate in result.data["candidates"]] == [
        "task_1",
        "task_2",
    ]
    assert [task.status.value for task in repository.list_today_tasks()] == before


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


def test_reschedule_interview_updates_existing_schedule_by_stable_ids(monkeypatch) -> None:
    monkeypatch.setattr(
        offerpilot_tools,
        "_normalize_interview_start_at",
        lambda value: "2026-07-18T16:00:00+08:00",
    )
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


def test_reschedule_interview_rejects_invalid_explicit_start_at(monkeypatch) -> None:
    monkeypatch.setattr(
        offerpilot_tools,
        "_normalize_interview_start_at",
        lambda value: "2026-07-18T16:00:00+08:00",
    )
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


def test_reschedule_interview_rejects_explicit_start_at_that_contradicts_user_time(monkeypatch) -> None:
    monkeypatch.setattr(
        offerpilot_tools,
        "_normalize_interview_start_at",
        lambda value: "2026-07-18T16:00:00+08:00",
    )
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


def test_schedule_interview_replay_reuses_schedule_and_external_sync() -> None:
    class FakeCalendarService:
        calendar_id = "primary"

        def __init__(self):
            self.create_calls = 0

        def is_calendar_sync_enabled(self):
            return True

        def should_manage_offerpilot_calendar(self):
            return False

        def create_interview_event(self, **kwargs):
            self.create_calls += 1
            return FeishuCalendarEventResult(event_id="evt_once", raw_response={"code": 0})

    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="美团", role="Java 后端")
    calendar_service = FakeCalendarService()
    registry = build_offerpilot_tool_registry(repository, calendar_service, bitable_service=None)
    arguments = {
        "application_id": application.id,
        "company": "美团",
        "round": "一面",
        "interview_time": "明天下午三点",
        "start_at": "2026-07-18T15:00:00+08:00",
        "update_type": "schedule_interview",
        "idempotency_key": "schedule-meituan-round-1",
    }

    first = registry.run("update_application", arguments)
    second = registry.run("update_application", arguments)

    assert first.success is True
    assert second.success is True
    assert len(repository.interview_schedules) == 1
    assert calendar_service.create_calls == 1
    assert second.data["calendar_sync"]["idempotent_replay"] is True


def test_schedule_interview_can_retry_after_user_reconciles_unknown_calendar_result() -> None:
    class FakeCalendarService:
        calendar_id = "primary"

        def __init__(self):
            self.create_calls = 0

        def is_calendar_sync_enabled(self):
            return True

        def should_manage_offerpilot_calendar(self):
            return False

        def create_interview_event(self, **kwargs):
            self.create_calls += 1
            if self.create_calls == 1:
                raise FeishuRequestError("response timeout; remote result unknown")
            return FeishuCalendarEventResult(event_id="evt_after_reconcile", raw_response={"code": 0})

    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="美团", role="Java 后端")
    calendar_service = FakeCalendarService()
    registry = build_offerpilot_tool_registry(repository, calendar_service, bitable_service=None)
    arguments = {
        "application_id": application.id,
        "company": "美团",
        "round": "一面",
        "interview_time": "明天下午三点",
        "start_at": "2026-07-18T15:00:00+08:00",
        "update_type": "schedule_interview",
        "idempotency_key": "schedule-after-reconcile",
    }

    first = registry.run("update_application", arguments)
    blocked_replay = registry.run("update_application", arguments)
    recovered = registry.run(
        "update_application",
        {**arguments, "retry_after_reconciliation": True},
    )

    assert first.data["operation_status"] == "reconciliation_required"
    assert blocked_replay.data["calendar_sync"]["idempotent_replay"] is True
    assert calendar_service.create_calls == 2
    assert recovered.data["calendar_sync"]["calendar_event_id"] == "evt_after_reconcile"
    assert len(repository.interview_schedules) == 1


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


def test_cancel_interview_syncs_remaining_schedule_to_bitable() -> None:
    class FakeBitableService:
        app_token = "bascn_offerpilot"
        table_id = "tbl_applications"

        def __init__(self):
            self.fields = None

        def is_bitable_sync_enabled(self):
            return True

        def should_manage_offerpilot_bitable(self):
            return False

        def update_record(self, **kwargs):
            self.fields = kwargs["fields"]
            return FeishuBitableRecordResult(record_id=kwargs["record_id"], raw_response={"code": 0})

    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="美团", role="Java 后端", round_name="一面")
    cancelled = repository.create_interview_schedule(
        application_id=application.id,
        company="美团",
        role="Java 后端",
        round_name="一面",
        start_time="明天下午三点",
        start_at="2026-07-18T15:00:00+08:00",
    )
    remaining = repository.create_interview_schedule(
        application_id=application.id,
        company="美团",
        role="Java 后端",
        round_name="二面",
        start_time="下周一下午三点",
        start_at="2026-07-20T15:00:00+08:00",
    )
    repository.set_runtime_setting(
        "feishu.offerpilot_bitable_record_id.tbl_applications.app_1",
        "rec_app_1",
    )
    bitable_service = FakeBitableService()
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=bitable_service)

    result = registry.run("cancel_interview", {"schedule_id": cancelled.id})

    assert result.success is True
    assert result.data["operation_status"] == "completed_with_skips"
    assert bitable_service.fields["面试轮次"] == remaining.round
    assert bitable_service.fields["面试时间文本"] == remaining.start_time
    assert bitable_service.fields["面试开始时间"] is not None
    assert "日历事件ID" in bitable_service.fields
    assert bitable_service.fields["日历事件ID"] is None


def test_reschedule_sync_retries_failed_step_and_replays_successful_step_idempotently() -> None:
    class FakeCalendarService:
        calendar_id = "primary"

        def __init__(self):
            self.update_calls = 0

        def is_calendar_sync_enabled(self):
            return True

        def should_manage_offerpilot_calendar(self):
            return False

        def update_interview_event(self, **kwargs):
            self.update_calls += 1
            return FeishuCalendarEventResult(event_id=kwargs["event_id"], raw_response={"code": 0})

    class FakeBitableService:
        app_token = "bascn_offerpilot"
        table_id = "tbl_applications"

        def __init__(self):
            self.update_calls = 0
            self.should_fail = True

        def is_bitable_sync_enabled(self):
            return True

        def should_manage_offerpilot_bitable(self):
            return False

        def update_record(self, **kwargs):
            self.update_calls += 1
            if self.should_fail:
                raise FeishuRequestError("temporary bitable failure")
            return FeishuBitableRecordResult(record_id=kwargs["record_id"], raw_response={"code": 0})

    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="美团", role="Java 后端", round_name="一面")
    schedule = repository.create_interview_schedule(
        application_id=application.id,
        company="美团",
        role="Java 后端",
        round_name="一面",
        start_time="明天下午三点",
        start_at="2026-07-18T15:00:00+08:00",
    )
    repository.update_interview_schedule_calendar_event(schedule.id, "evt_test_1")
    repository.set_runtime_setting(
        "feishu.offerpilot_bitable_record_id.tbl_applications.app_1",
        "rec_app_1",
    )
    calendar_service = FakeCalendarService()
    bitable_service = FakeBitableService()
    registry = build_offerpilot_tool_registry(repository, calendar_service, bitable_service)
    arguments = {
        "schedule_id": schedule.id,
        "interview_time": "后天下午四点",
        "idempotency_key": "scenario-1-reschedule",
    }

    first = registry.run("reschedule_interview", arguments)
    bitable_service.should_fail = False
    second = registry.run("reschedule_interview", arguments)

    assert first.success is True
    assert first.data["operation_status"] == "partial_success"
    assert first.data["retryable"] is True
    assert first.data["calendar_sync"]["attempts"] == 1
    assert first.data["bitable_sync"]["attempts"] == 3
    assert second.data["operation_status"] == "completed"
    assert second.data["calendar_sync"]["idempotent_replay"] is True
    assert calendar_service.update_calls == 1
    assert bitable_service.update_calls == 4


def test_reschedule_rejects_reusing_idempotency_key_for_different_time() -> None:
    repository = InMemoryOfferPilotRepository()
    schedule = repository.create_interview_schedule(
        company="美团",
        role="Java 后端",
        round_name="一面",
        start_time="明天下午三点",
        start_at="2026-07-18T15:00:00+08:00",
    )
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)

    first = registry.run(
        "reschedule_interview",
        {
            "schedule_id": schedule.id,
            "interview_time": "后天下午四点",
            "idempotency_key": "same-request-key",
        },
    )
    second = registry.run(
        "reschedule_interview",
        {
            "schedule_id": schedule.id,
            "interview_time": "下周一下午五点",
            "idempotency_key": "same-request-key",
        },
    )

    assert first.success is True
    assert second.success is False
    assert second.data["idempotency_conflict"] is True
    assert repository.interview_schedules[0].start_time == "后天下午四点"


def test_cancel_sync_deletes_calendar_event_and_clears_bitable_interview_fields() -> None:
    class FakeCalendarService:
        calendar_id = "primary"

        def __init__(self):
            self.deleted = []

        def is_calendar_sync_enabled(self):
            return True

        def should_manage_offerpilot_calendar(self):
            return False

        def delete_interview_event(self, **kwargs):
            self.deleted.append(kwargs)
            return FeishuCalendarEventResult(event_id=kwargs["event_id"], raw_response={"code": 0})

    class FakeBitableService:
        app_token = "bascn_offerpilot"
        table_id = "tbl_applications"

        def __init__(self):
            self.fields = None

        def is_bitable_sync_enabled(self):
            return True

        def should_manage_offerpilot_bitable(self):
            return False

        def update_record(self, **kwargs):
            self.fields = kwargs["fields"]
            return FeishuBitableRecordResult(record_id=kwargs["record_id"], raw_response={"code": 0})

    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(
        company="美团",
        role="Java 后端",
        round_name="一面",
        interview_time="明天下午三点",
    )
    schedule = repository.create_interview_schedule(
        application_id=application.id,
        company="美团",
        role="Java 后端",
        round_name="一面",
        start_time="明天下午三点",
        start_at="2026-07-18T15:00:00+08:00",
    )
    repository.update_interview_schedule_calendar_event(schedule.id, "evt_test_1")
    repository.set_runtime_setting(
        "feishu.offerpilot_bitable_record_id.tbl_applications.app_1",
        "rec_app_1",
    )
    calendar_service = FakeCalendarService()
    bitable_service = FakeBitableService()
    registry = build_offerpilot_tool_registry(repository, calendar_service, bitable_service)

    result = registry.run(
        "cancel_interview",
        {"schedule_id": schedule.id, "idempotency_key": "scenario-1-cancel"},
    )

    assert result.success is True
    assert result.data["operation_status"] == "completed"
    assert calendar_service.deleted == [{"calendar_id": "primary", "event_id": "evt_test_1"}]
    assert bitable_service.fields["面试轮次"] is None
    assert bitable_service.fields["面试时间文本"] is None
    assert bitable_service.fields["面试开始时间"] is None
    assert bitable_service.fields["日历事件ID"] is None


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
