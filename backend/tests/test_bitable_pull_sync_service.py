import app.services.bitable_pull_sync_service as pull_sync_module
from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository
from app.services.bitable_pull_sync_service import BitablePullSyncService
from app.services.bitable_sync_service import remember_bitable_table_owner
from app.services.feishu_service import (
    FeishuBitableRecordListResult,
    FeishuBitableRecordResult,
)


APP_TOKEN = "bascn_offerpilot"
TABLE_ID = "tbl_applications"
OWNER_ID = "feishu:ou_owner"


class FakeBitablePullService:
    app_token = APP_TOKEN
    table_id = TABLE_ID

    def __init__(self, pages):
        self.pages = pages
        self.list_record_calls = []
        self.update_record_calls = []

    def is_bitable_sync_enabled(self):
        return True

    def list_records(self, app_token, table_id, page_size, page_token=None):
        self.list_record_calls.append(
            {
                "app_token": app_token,
                "table_id": table_id,
                "page_size": page_size,
                "page_token": page_token,
            }
        )
        records, has_more, next_page_token = self.pages[page_token]
        return FeishuBitableRecordListResult(
            records=records,
            has_more=has_more,
            page_token=next_page_token,
            raw_response={"code": 0},
        )

    def update_record(self, app_token, table_id, record_id, fields):
        self.update_record_calls.append(
            {
                "app_token": app_token,
                "table_id": table_id,
                "record_id": record_id,
                "fields": dict(fields),
            }
        )
        record = self._find_record(record_id)
        record.fields.update(fields)
        return FeishuBitableRecordResult(
            record_id=record_id,
            raw_response={"code": 0},
            fields=dict(record.fields),
        )

    def _find_record(self, record_id):
        for records, _has_more, _page_token in self.pages.values():
            for record in records:
                if record.record_id == record_id:
                    return record
        raise AssertionError(f"Unknown fake Bitable record: {record_id}")


def _record(record_id, company, role, **extra_fields):
    return FeishuBitableRecordResult(
        record_id=record_id,
        raw_response={"record_id": record_id},
        fields={"公司": company, "岗位": role, **extra_fields},
    )


def _configured_repository():
    repository = InMemoryOfferPilotRepository()
    repository.set_runtime_setting("feishu.offerpilot_bitable_app_token", APP_TOKEN)
    repository.set_runtime_setting("feishu.offerpilot_bitable_table_id", TABLE_ID)
    return repository


def test_pull_sync_recovers_manual_row_with_table_owner_and_is_idempotent() -> None:
    repository = _configured_repository()
    remember_bitable_table_owner(repository, APP_TOKEN, TABLE_ID, OWNER_ID)
    manual_record = _record("rec_manual_1", "小红书", "Java 后端实习")
    bitable_service = FakeBitablePullService(
        {None: ([manual_record], False, None)}
    )
    service = BitablePullSyncService(
        repository=repository,
        bitable_service=bitable_service,
    )

    first_summary = service.sync_once()

    assert first_summary.synced_count == 1
    assert first_summary.failed_count == 0
    applications = repository.list_applications(owner_id=OWNER_ID)
    assert len(applications) == 1
    assert applications[0].company == "小红书"
    assert repository.list_applications(owner_id="local_user") == []
    assert bitable_service.update_record_calls == [
        {
            "app_token": APP_TOKEN,
            "table_id": TABLE_ID,
            "record_id": "rec_manual_1",
            "fields": {
                "OfferPilot记录ID": applications[0].id,
                "OfferPilot用户ID": OWNER_ID,
            },
        }
    ]

    second_summary = service.sync_once()

    assert second_summary.synced_count == 1
    assert second_summary.failed_count == 0
    assert len(repository.list_applications(owner_id=OWNER_ID)) == 1
    assert len(bitable_service.update_record_calls) == 1


def test_pull_sync_reads_all_pages() -> None:
    repository = _configured_repository()
    remember_bitable_table_owner(repository, APP_TOKEN, TABLE_ID, OWNER_ID)
    bitable_service = FakeBitablePullService(
        {
            None: ([_record("rec_page_1", "腾讯", "Agent 工程师")], True, "page_2"),
            "page_2": ([_record("rec_page_2", "美团", "后端工程师")], False, None),
        }
    )
    service = BitablePullSyncService(
        repository=repository,
        bitable_service=bitable_service,
        page_size=1,
    )

    summary = service.sync_once()

    assert summary.synced_count == 2
    assert summary.failed_count == 0
    assert len(repository.list_applications(owner_id=OWNER_ID)) == 2
    assert [call["page_token"] for call in bitable_service.list_record_calls] == [
        None,
        "page_2",
    ]
    assert all(call["page_size"] == 1 for call in bitable_service.list_record_calls)


def test_pull_sync_rejects_manual_row_without_an_owner(monkeypatch) -> None:
    repository = _configured_repository()
    bitable_service = FakeBitablePullService(
        {None: ([_record("rec_orphan", "字节跳动", "平台工程师")], False, None)}
    )
    observed_statuses = []
    original_sync = pull_sync_module.sync_bitable_record_fields_to_repository

    def capture_status(**kwargs):
        result = original_sync(**kwargs)
        observed_statuses.append(result.status)
        return result

    monkeypatch.setattr(
        pull_sync_module,
        "sync_bitable_record_fields_to_repository",
        capture_status,
    )
    service = BitablePullSyncService(
        repository=repository,
        bitable_service=bitable_service,
    )

    summary = service.sync_once()

    assert summary.synced_count == 0
    assert summary.failed_count == 1
    assert observed_statuses == ["missing_owner"]
    assert repository.applications == []
    assert repository.list_applications(owner_id="local_user") == []
    assert bitable_service.update_record_calls == []
