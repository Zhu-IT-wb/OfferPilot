import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository
from app.repositories.sqlite_offerpilot_repository import SQLiteOfferPilotRepository
from app.services.bitable_sync_service import (
    remember_bitable_record_mapping,
    remember_bitable_table_owner,
    sync_bitable_record_fields_to_repository,
    sync_bitable_record_to_repository,
    sync_deleted_bitable_record_to_repository,
)
from app.services.feishu_service import FeishuBitableRecordResult, FeishuRequestError


class FakeBitableRecordService:
    app_token = ""
    table_id = ""

    def __init__(self, fields):
        self.fields = fields
        self.get_record_calls = []

    def is_bitable_sync_enabled(self):
        return True

    def get_record(self, app_token, table_id, record_id):
        self.get_record_calls.append(
            {
                "app_token": app_token,
                "table_id": table_id,
                "record_id": record_id,
            }
        )
        return FeishuBitableRecordResult(
            record_id=record_id,
            raw_response={"code": 0},
            fields=self.fields,
        )


class WritableBitableRecordService(FakeBitableRecordService):
    def __init__(self, fields):
        super().__init__(fields)
        self.update_record_calls = []

    def update_record(self, app_token, table_id, record_id, fields):
        self.update_record_calls.append(
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


def test_bitable_record_sync_updates_existing_application_by_offerpilot_id() -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="腾讯", role="Java 后端实习")
    repository.set_runtime_setting("feishu.offerpilot_bitable_app_token", "bascn_offerpilot")
    repository.set_runtime_setting("feishu.offerpilot_bitable_table_id", "tbl_applications")
    repository.set_runtime_setting(
        "feishu.offerpilot_bitable_record_id.tbl_applications.app_1",
        "rec_app_1",
    )
    service = FakeBitableRecordService(
        {
            "OfferPilot记录ID": application.id,
            "公司": "腾讯",
            "岗位": "AI 应用开发实习",
            "工作地点": "上海",
            "投递状态": "一面阶段",
            "面试轮次": "一面",
            "面试时间文本": "明天下午三点",
            "JD关键词": ["Java", "Agent"],
        }
    )

    result = sync_bitable_record_to_repository(
        repository=repository,
        bitable_service=service,
        record_id="rec_app_1",
        app_token="bascn_offerpilot",
        table_id="tbl_applications",
    )

    assert result.synced is True
    assert result.status == "updated"
    assert result.application_id == application.id
    assert repository.applications[0].role == "AI 应用开发实习"
    assert repository.applications[0].base_location == "上海"
    assert repository.applications[0].status.value == "interview_1"
    assert repository.applications[0].round == "一面"
    assert repository.applications[0].interview_time == "明天下午三点"
    assert repository.applications[0].jd_keywords == ["Java", "Agent"]
    assert service.get_record_calls == [
        {
            "app_token": "bascn_offerpilot",
            "table_id": "tbl_applications",
            "record_id": "rec_app_1",
        }
    ]


def test_bitable_record_sync_creates_application_for_manual_row() -> None:
    repository = InMemoryOfferPilotRepository()
    repository.set_runtime_setting("feishu.offerpilot_bitable_app_token", "bascn_offerpilot")
    repository.set_runtime_setting("feishu.offerpilot_bitable_table_id", "tbl_applications")
    service = FakeBitableRecordService(
        {
            "公司": "小红书",
            "岗位": "Java 后端实习",
            "工作地点": "北京/上海",
            "投递状态": "已投递",
        }
    )

    result = sync_bitable_record_to_repository(
        repository=repository,
        bitable_service=service,
        record_id="rec_manual_1",
        table_id="tbl_applications",
    )

    assert result.synced is True
    assert result.status == "created"
    assert result.created is True
    assert repository.applications[0].company == "小红书"
    assert repository.applications[0].role == "Java 后端实习"
    assert repository.applications[0].base_location == "北京/上海"
    assert repository.applications[0].status.value == "submitted"
    assert (
        repository.get_runtime_setting("feishu.offerpilot_bitable_record_id.tbl_applications.app_1")
        == "rec_manual_1"
    )


def test_bitable_record_sync_can_clear_application_base_location() -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(
        company="腾讯",
        role="Agent 开发",
        base_location="深圳",
    )
    service = FakeBitableRecordService(
        {
            "OfferPilot记录ID": application.id,
            "工作地点": None,
        }
    )

    result = sync_bitable_record_to_repository(
        repository=repository,
        bitable_service=service,
        record_id="rec_app_1",
        app_token="bascn_offerpilot",
        table_id="tbl_applications",
    )

    assert result.synced is True
    assert repository.applications[0].base_location is None
    assert result.updated_fields == ["工作地点"]


def test_bitable_record_sync_uses_fallback_owner_and_persists_identity() -> None:
    repository = InMemoryOfferPilotRepository()
    repository.set_runtime_setting("feishu.offerpilot_bitable_app_token", "bascn_offerpilot")
    repository.set_runtime_setting("feishu.offerpilot_bitable_table_id", "tbl_applications")
    service = WritableBitableRecordService(
        {
            "公司": "美团",
            "岗位": "AI 全栈工程师",
            "工作地点": "深圳",
            "投递状态": "已投递",
        }
    )

    result = sync_bitable_record_to_repository(
        repository=repository,
        bitable_service=service,
        record_id="rec_manual_owner",
        app_token="bascn_offerpilot",
        table_id="tbl_applications",
        fallback_owner_id="feishu:ou_test",
        allow_default_local_owner=False,
    )

    assert result.synced is True
    assert result.status == "created"
    assert result.owner_id == "feishu:ou_test"
    assert result.metadata_updated is True
    applications = repository.list_applications(owner_id="feishu:ou_test")
    assert len(applications) == 1
    assert applications[0].company == "美团"
    assert repository.list_applications(owner_id="local_user") == []
    assert service.update_record_calls == [
        {
            "app_token": "bascn_offerpilot",
            "table_id": "tbl_applications",
            "record_id": "rec_manual_owner",
            "fields": {
                "OfferPilot记录ID": applications[0].id,
                "OfferPilot用户ID": "feishu:ou_test",
            },
        }
    ]
    assert (
        repository.get_runtime_setting(
            f"feishu.offerpilot_bitable_record_id.tbl_applications.{applications[0].id}"
        )
        == "rec_manual_owner"
    )
    reverse_mapping = repository.get_runtime_setting(
        "feishu.offerpilot_bitable_application_id."
        "bascn_offerpilot.tbl_applications.rec_manual_owner"
    )
    assert json.loads(reverse_mapping or "{}") == {
        "application_id": applications[0].id,
        "owner_id": "feishu:ou_test",
    }
    assert (
        repository.get_runtime_setting(
            "feishu.offerpilot_bitable_owner.bascn_offerpilot.tbl_applications"
        )
        == "feishu:ou_test"
    )


def test_bitable_record_sync_rejects_explicit_row_owner_over_verified_fallback() -> None:
    repository = InMemoryOfferPilotRepository()
    service = WritableBitableRecordService(
        {
            "OfferPilot用户ID": "feishu:ou_explicit",
            "公司": "字节跳动",
            "岗位": "Agent 工程师",
        }
    )

    result = sync_bitable_record_to_repository(
        repository=repository,
        bitable_service=service,
        record_id="rec_explicit_owner",
        app_token="bascn_offerpilot",
        table_id="tbl_applications",
        fallback_owner_id="feishu:ou_fallback",
        allow_default_local_owner=False,
    )

    assert result.synced is False
    assert result.status == "owner_conflict"
    assert result.owner_id == "feishu:ou_fallback"
    assert repository.list_applications(owner_id="feishu:ou_explicit") == []
    assert repository.list_applications(owner_id="feishu:ou_fallback") == []
    assert service.update_record_calls == []


def test_bitable_record_sync_reassigns_legacy_local_mapping() -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="美团", role="AI 全栈工程师")
    repository.set_runtime_setting(
        f"feishu.offerpilot_bitable_record_id.tbl_applications.{application.id}",
        "rec_legacy_local",
    )
    service = WritableBitableRecordService(
        {
            "公司": "美团",
            "岗位": "AI 全栈工程师",
        }
    )

    result = sync_bitable_record_to_repository(
        repository=repository,
        bitable_service=service,
        record_id="rec_legacy_local",
        app_token="bascn_offerpilot",
        table_id="tbl_applications",
        fallback_owner_id="feishu:ou_test",
        allow_default_local_owner=False,
    )

    assert result.synced is True
    assert result.status == "updated"
    assert result.application_id == application.id
    assert repository.list_applications(owner_id="local_user") == []
    migrated = repository.list_applications(owner_id="feishu:ou_test")
    assert [item.id for item in migrated] == [application.id]
    assert service.update_record_calls[0]["fields"] == {
        "OfferPilot记录ID": application.id,
        "OfferPilot用户ID": "feishu:ou_test",
    }


def test_bitable_record_sync_does_not_duplicate_concurrent_manual_record() -> None:
    worker_count = 8
    repository = InMemoryOfferPilotRepository()

    class ConcurrentBitableRecordService(WritableBitableRecordService):
        def __init__(self):
            super().__init__({"公司": "小米", "岗位": "大模型应用工程师"})
            self.get_record_barrier = Barrier(worker_count)

        def get_record(self, app_token, table_id, record_id):
            self.get_record_barrier.wait()
            return super().get_record(app_token, table_id, record_id)

    service = ConcurrentBitableRecordService()

    def sync_once(_index):
        return sync_bitable_record_to_repository(
            repository=repository,
            bitable_service=service,
            record_id="rec_concurrent_manual",
            app_token="bascn_offerpilot",
            table_id="tbl_applications",
            fallback_owner_id="feishu:ou_test",
            allow_default_local_owner=False,
        )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(sync_once, range(worker_count)))

    applications = repository.list_applications(owner_id="feishu:ou_test")
    assert len(applications) == 1
    assert {result.application_id for result in results} == {applications[0].id}
    assert sum(result.created for result in results) == 1
    assert all(result.synced for result in results)
    reverse_mapping = repository.get_runtime_setting(
        "feishu.offerpilot_bitable_application_id."
        "bascn_offerpilot.tbl_applications.rec_concurrent_manual"
    )
    assert json.loads(reverse_mapping or "{}") == {
        "application_id": applications[0].id,
        "owner_id": "feishu:ou_test",
    }


DELETE_APP_TOKEN = "base_deleted"
DELETE_TABLE = "tbl_deleted"
DELETE_OWNER = "feishu:ou_delete_owner"


def _mapped_application(repository, record_id="rec_deleted"):
    application = repository.create_application("Example", "Engineer", owner_id=DELETE_OWNER)
    remember_bitable_table_owner(repository, DELETE_APP_TOKEN, DELETE_TABLE, DELETE_OWNER)
    remember_bitable_record_mapping(
        repository, DELETE_APP_TOKEN, DELETE_TABLE, record_id, application.id, DELETE_OWNER
    )
    return application


class FailedRecordRead:
    def __init__(self, error):
        self.error = error

    def get_record(self, **kwargs):
        raise self.error


def _sync_failed_read(repository, error, **overrides):
    arguments = dict(
        repository=repository,
        bitable_service=FailedRecordRead(error),
        record_id="rec_deleted",
        app_token=DELETE_APP_TOKEN,
        table_id=DELETE_TABLE,
        fallback_owner_id=DELETE_OWNER,
        allow_default_local_owner=False,
    )
    arguments.update(overrides)
    return sync_bitable_record_to_repository(**arguments)


def test_deleted_remote_record_is_hidden_and_late_edit_cannot_recreate_it():
    repository = InMemoryOfferPilotRepository()
    application = _mapped_application(repository)
    repository.create_interview_schedule(
        "Example", "First", application_id=application.id, owner_id=DELETE_OWNER
    )
    error = FeishuRequestError("RecordIdNotFound", code=1254043)

    assert _sync_failed_read(repository, error).status == "deleted"
    assert _sync_failed_read(repository, error).status == "already_deleted"
    assert repository.list_applications(owner_id=DELETE_OWNER) == []
    assert repository.list_interview_schedules(owner_id=DELETE_OWNER) == []
    assert repository.get_runtime_setting(
        f"feishu.offerpilot_bitable_record_id.{DELETE_TABLE}.{application.id}"
    ) == ""

    late_result = sync_bitable_record_fields_to_repository(
        repository, "rec_deleted", {"公司": "Example", "岗位": "Engineer"},
        DELETE_APP_TOKEN, DELETE_TABLE, fallback_owner_id=DELETE_OWNER,
    )
    assert late_result.status == "already_deleted"
    assert repository.list_applications(owner_id=DELETE_OWNER) == []
    new_application = repository.create_application("New", "Engineer", owner_id=DELETE_OWNER)
    assert new_application.id != application.id
    assert [item.id for item in repository.list_applications(owner_id=DELETE_OWNER)] == [new_application.id]


@pytest.mark.parametrize("error", [
    FeishuRequestError("Forbidden", code=1254302),
    FeishuRequestError("TableNotFound", code=1254041),
    FeishuRequestError("Network timeout"),
    FeishuRequestError("HTTP 404"),
    FeishuRequestError("Feishu OpenAPI returned code 1254043: unstructured message"),
])
def test_failed_remote_read_is_not_treated_as_deletion(error):
    repository = InMemoryOfferPilotRepository()
    application = _mapped_application(repository)
    result = _sync_failed_read(repository, error)
    assert result.status == "failed"
    assert not repository.is_application_deleted(application.id, DELETE_OWNER)
    assert repository.get_runtime_setting(
        f"feishu.offerpilot_bitable_record_id.{DELETE_TABLE}.{application.id}"
    ) == "rec_deleted"


@pytest.mark.parametrize("overrides", [
    {"fallback_owner_id": "feishu:ou_someone_else"},
    {"app_token": "base_foreign"},
    {"table_id": "tbl_foreign"},
])
def test_deleted_record_does_not_cross_owner_or_resource_boundaries(overrides):
    repository = InMemoryOfferPilotRepository()
    application = _mapped_application(repository)
    _sync_failed_read(repository, FeishuRequestError("RecordIdNotFound", code=1254043), **overrides)
    assert not repository.is_application_deleted(application.id, DELETE_OWNER)


def test_old_deleted_record_cannot_delete_application_mapped_to_replacement():
    repository = InMemoryOfferPilotRepository()
    application = _mapped_application(repository)
    remember_bitable_record_mapping(
        repository, DELETE_APP_TOKEN, DELETE_TABLE, "rec_replacement", application.id, DELETE_OWNER
    )
    result = _sync_failed_read(repository, FeishuRequestError("RecordIdNotFound", code=1254043))
    assert result.status == "ignored_replaced_record"
    assert not repository.is_application_deleted(application.id, DELETE_OWNER)
    assert repository.get_runtime_setting(
        f"feishu.offerpilot_bitable_record_id.{DELETE_TABLE}.{application.id}"
    ) == "rec_replacement"


def test_unmapped_deleted_record_blocks_delayed_first_import():
    repository = InMemoryOfferPilotRepository()
    remember_bitable_table_owner(repository, DELETE_APP_TOKEN, DELETE_TABLE, DELETE_OWNER)
    result = sync_deleted_bitable_record_to_repository(
        repository, "rec_deleted", DELETE_APP_TOKEN, DELETE_TABLE, DELETE_OWNER
    )
    assert result.status == "unmapped_deleted"
    late_result = sync_bitable_record_fields_to_repository(
        repository, "rec_deleted", {"公司": "Example", "岗位": "Engineer"},
        DELETE_APP_TOKEN, DELETE_TABLE, fallback_owner_id=DELETE_OWNER,
    )
    assert late_result.status == "already_deleted"
    assert repository.list_applications(owner_id=DELETE_OWNER) == []


def test_deleted_legacy_mapping_is_retained_as_receipt_after_restart(tmp_path):
    path = str(tmp_path / "applications.db")
    repository = SQLiteOfferPilotRepository(path)
    application = repository.create_application("Example", "Engineer", owner_id=DELETE_OWNER)
    remember_bitable_table_owner(repository, DELETE_APP_TOKEN, DELETE_TABLE, DELETE_OWNER)
    repository.set_runtime_setting(
        f"feishu.offerpilot_bitable_record_id.{DELETE_TABLE}.{application.id}", "rec_deleted"
    )
    assert _sync_failed_read(repository, FeishuRequestError("RecordIdNotFound", code=1254043)).status == "deleted"

    reopened = SQLiteOfferPilotRepository(path)
    assert reopened.list_applications(owner_id=DELETE_OWNER) == []
    assert _sync_failed_read(reopened, FeishuRequestError("RecordIdNotFound", code=1254043)).status == "already_deleted"
    assert reopened.create_application("New", "Engineer", owner_id=DELETE_OWNER).id != application.id


def test_reverse_mapping_cannot_assert_a_different_application_owner():
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application("Other", "Engineer", owner_id="feishu:ou_other")
    remember_bitable_table_owner(repository, DELETE_APP_TOKEN, DELETE_TABLE, DELETE_OWNER)
    remember_bitable_record_mapping(
        repository, DELETE_APP_TOKEN, DELETE_TABLE, "rec_deleted", application.id, DELETE_OWNER
    )
    result = _sync_failed_read(repository, FeishuRequestError("RecordIdNotFound", code=1254043))
    assert result.status == "owner_conflict"
    assert not repository.is_application_deleted(application.id, "feishu:ou_other")


def test_legacy_deletion_interrupted_after_soft_delete_does_not_resurrect(tmp_path, monkeypatch):
    path = str(tmp_path / "interrupted.db")
    repository = SQLiteOfferPilotRepository(path)
    application = repository.create_application("Original", "Engineer", owner_id=DELETE_OWNER)
    remember_bitable_table_owner(repository, DELETE_APP_TOKEN, DELETE_TABLE, DELETE_OWNER)
    repository.set_runtime_setting(
        f"feishu.offerpilot_bitable_record_id.{DELETE_TABLE}.{application.id}", "rec_deleted"
    )
    original_delete = repository.delete_application

    def interrupt_after_delete(*args, **kwargs):
        original_delete(*args, **kwargs)
        raise RuntimeError("Simulated process interruption")

    monkeypatch.setattr(repository, "delete_application", interrupt_after_delete)
    with pytest.raises(RuntimeError, match="Simulated"):
        sync_deleted_bitable_record_to_repository(
            repository, "rec_deleted", DELETE_APP_TOKEN, DELETE_TABLE, DELETE_OWNER
        )

    reopened = SQLiteOfferPilotRepository(path)
    result = sync_bitable_record_fields_to_repository(
        reopened, "rec_deleted", {"公司": "Original", "岗位": "Engineer"},
        DELETE_APP_TOKEN, DELETE_TABLE, fallback_owner_id=DELETE_OWNER,
    )
    assert result.status == "already_deleted"
    assert reopened.list_applications(owner_id=DELETE_OWNER) == []
    replay = _sync_failed_read(reopened, FeishuRequestError("RecordIdNotFound", code=1254043))
    assert replay.status == "already_deleted"
    assert reopened.get_runtime_setting(
        f"feishu.offerpilot_bitable_record_id.{DELETE_TABLE}.{application.id}"
    ) == ""

