import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository
from app.services.bitable_sync_service import sync_bitable_record_to_repository
from app.services.feishu_service import FeishuBitableRecordResult


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

