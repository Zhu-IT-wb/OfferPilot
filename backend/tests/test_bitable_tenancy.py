from app.core.config import Settings
from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository
from app.repositories.sqlite_offerpilot_repository import SQLiteOfferPilotRepository
from app.services.bitable_pull_sync_service import BitablePullSyncService
from app.services.bitable_sync_service import (
    remember_bitable_record_mapping,
    sync_bitable_record_to_repository,
)
from app.services.bitable_tenancy import (
    get_owner_bitable_resource,
    list_owner_bitable_resources,
    owner_bitable_setting_key,
    save_owner_bitable_resource,
)
from app.services.feishu_service import (
    FeishuBitableAppResult,
    FeishuBitableRecordListResult,
    FeishuBitableRecordResult,
    FeishuBitableTableResult,
)
from app.tools.offerpilot_tools import (
    build_offerpilot_tool_registry,
    get_application_bitable_link,
)
from app.tools.tool_names import AgentActionName


class PerOwnerBitableService:
    app_token = ""
    table_id = ""

    def __init__(self) -> None:
        self.created_apps = 0
        self.created_tables = []
        self.created_records = []

    def is_bitable_sync_enabled(self):
        return True

    def should_manage_offerpilot_bitable(self):
        return True

    def create_app(self):
        self.created_apps += 1
        return FeishuBitableAppResult(
            app_token=f"bascn_owner_{self.created_apps}",
            raw_response={"code": 0},
        )

    def create_application_table(self, app_token, table_name=None):
        table_id = f"tbl_owner_{len(self.created_tables) + 1}"
        self.created_tables.append((app_token, table_id, table_name))
        return FeishuBitableTableResult(table_id=table_id, raw_response={"code": 0})

    def create_record(self, app_token, table_id, fields):
        self.created_records.append((app_token, table_id, dict(fields)))
        return FeishuBitableRecordResult(
            record_id=f"rec_{len(self.created_records)}",
            raw_response={"code": 0},
        )


def test_application_bitable_resources_and_links_are_isolated_by_owner() -> None:
    repository = InMemoryOfferPilotRepository()
    bitable_service = PerOwnerBitableService()
    registry = build_offerpilot_tool_registry(
        repository,
        calendar_service=None,
        bitable_service=bitable_service,
    )

    for owner_id, company in (
        ("feishu:ou_alice", "腾讯"),
        ("feishu:ou_bob", "美团"),
    ):
        result = registry.run(
            AgentActionName.CREATE_APPLICATION.value,
            {
                "owner_id": owner_id,
                "company": company,
                "role": "Agent 工程师",
            },
        )
        assert result.success is True
        assert result.data["bitable_sync"]["synced"] is True

    alice = get_owner_bitable_resource(repository, "feishu:ou_alice")
    bob = get_owner_bitable_resource(repository, "feishu:ou_bob")
    assert alice is not None
    assert bob is not None
    assert (alice.app_token, alice.table_id) != (bob.app_token, bob.table_id)
    assert repository.get_runtime_setting("feishu.offerpilot_bitable_app_token") is None
    assert repository.get_runtime_setting("feishu.offerpilot_bitable_table_id") is None
    assert repository.get_runtime_setting(
        owner_bitable_setting_key("feishu:ou_alice", "app_token")
    ) == alice.app_token
    assert {item.owner_id for item in list_owner_bitable_resources(repository)} == {
        "feishu:ou_alice",
        "feishu:ou_bob",
    }

    settings = Settings(feishu_bitable_sync_enabled=True)
    alice_link = get_application_bitable_link(
        repository,
        {"owner_id": "feishu:ou_alice"},
        app_settings=settings,
    )
    bob_link = get_application_bitable_link(
        repository,
        {"owner_id": "feishu:ou_bob"},
        app_settings=settings,
    )
    assert alice_link.success is True
    assert bob_link.success is True
    assert alice.app_token in alice_link.data["web_url"]
    assert bob.app_token in bob_link.data["web_url"]
    assert bob.app_token not in alice_link.data["web_url"]


def test_registered_table_owner_controls_event_writeback() -> None:
    repository = InMemoryOfferPilotRepository()
    save_owner_bitable_resource(
        repository,
        owner_id="feishu:ou_alice",
        app_token="bascn_alice",
        table_id="tbl_alice",
    )

    class RecordService:
        app_token = ""
        table_id = ""

        def __init__(self):
            self.get_record_calls = 0

        def get_record(self, app_token, table_id, record_id):
            self.get_record_calls += 1
            return FeishuBitableRecordResult(
                record_id=record_id,
                raw_response={"code": 0},
                fields={"公司": "字节跳动", "岗位": "后端工程师"},
            )

    service = RecordService()
    conflict = sync_bitable_record_to_repository(
        repository=repository,
        bitable_service=service,
        app_token="bascn_alice",
        table_id="tbl_alice",
        record_id="rec_1",
        fallback_owner_id="feishu:ou_bob",
        allow_default_local_owner=False,
    )
    assert conflict.synced is False
    assert conflict.status == "owner_conflict"
    assert service.get_record_calls == 0
    assert repository.applications == []

    synced = sync_bitable_record_to_repository(
        repository=repository,
        bitable_service=service,
        app_token="bascn_alice",
        table_id="tbl_alice",
        record_id="rec_1",
        allow_default_local_owner=False,
    )
    assert synced.synced is True
    assert synced.owner_id == "feishu:ou_alice"
    assert len(repository.list_applications(owner_id="feishu:ou_alice")) == 1
    assert repository.list_applications(owner_id="feishu:ou_bob") == []


def test_legacy_local_user_bitable_configuration_remains_readable() -> None:
    repository = InMemoryOfferPilotRepository()
    repository.set_runtime_setting(
        "feishu.offerpilot_bitable_app_token",
        "bascn_legacy",
    )
    repository.set_runtime_setting(
        "feishu.offerpilot_bitable_table_id",
        "tbl_legacy",
    )

    local_resource = get_owner_bitable_resource(repository, "local_user")
    other_resource = get_owner_bitable_resource(repository, "feishu:ou_other")

    assert local_resource is not None
    assert local_resource.legacy is True
    assert local_resource.app_token == "bascn_legacy"
    assert other_resource is None


def test_mapped_record_owner_cannot_be_changed_by_hidden_table_field() -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(
        company="腾讯",
        role="Agent 工程师",
        owner_id="feishu:ou_alice",
    )
    save_owner_bitable_resource(
        repository,
        owner_id="feishu:ou_alice",
        app_token="bascn_alice",
        table_id="tbl_alice",
    )
    remember_bitable_record_mapping(
        repository,
        app_token="bascn_alice",
        table_id="tbl_alice",
        record_id="rec_alice",
        application_id=application.id,
        owner_id="feishu:ou_alice",
    )

    class TamperedRecordService:
        app_token = ""
        table_id = ""

        def get_record(self, app_token, table_id, record_id):
            return FeishuBitableRecordResult(
                record_id=record_id,
                raw_response={"code": 0},
                fields={
                    "OfferPilot记录ID": application.id,
                    "OfferPilot用户ID": "feishu:ou_bob",
                    "公司": "腾讯",
                    "岗位": "被篡改的岗位",
                },
            )

    result = sync_bitable_record_to_repository(
        repository=repository,
        bitable_service=TamperedRecordService(),
        app_token="bascn_alice",
        table_id="tbl_alice",
        record_id="rec_alice",
        allow_default_local_owner=False,
    )

    assert result.synced is False
    assert result.status == "owner_conflict"
    assert result.owner_id == "feishu:ou_alice"
    assert repository.list_applications(owner_id="feishu:ou_bob") == []
    assert repository.list_applications(owner_id="feishu:ou_alice")[0].role == "Agent 工程师"


def test_pull_sync_iterates_every_registered_owner_table() -> None:
    repository = InMemoryOfferPilotRepository()
    for owner_id, app_token, table_id in (
        ("feishu:ou_alice", "bascn_alice", "tbl_alice"),
        ("feishu:ou_bob", "bascn_bob", "tbl_bob"),
    ):
        save_owner_bitable_resource(
            repository,
            owner_id=owner_id,
            app_token=app_token,
            table_id=table_id,
        )

    class MultiTableService:
        app_token = ""
        table_id = ""

        def __init__(self):
            self.calls = []
            self.records = {
                ("bascn_alice", "tbl_alice"): FeishuBitableRecordResult(
                    record_id="rec_alice",
                    raw_response={"code": 0},
                    fields={"公司": "腾讯", "岗位": "后端工程师"},
                ),
                ("bascn_bob", "tbl_bob"): FeishuBitableRecordResult(
                    record_id="rec_bob",
                    raw_response={"code": 0},
                    fields={"公司": "美团", "岗位": "平台工程师"},
                ),
            }

        def is_bitable_sync_enabled(self):
            return True

        def list_records(self, app_token, table_id, page_size, page_token=None):
            self.calls.append((app_token, table_id, page_token))
            return FeishuBitableRecordListResult(
                records=[self.records[(app_token, table_id)]],
                has_more=False,
                page_token=None,
                raw_response={"code": 0},
            )

        def update_record(self, app_token, table_id, record_id, fields):
            self.records[(app_token, table_id)].fields.update(fields)
            return FeishuBitableRecordResult(
                record_id=record_id,
                raw_response={"code": 0},
            )

    bitable_service = MultiTableService()
    summary = BitablePullSyncService(
        repository=repository,
        bitable_service=bitable_service,
    ).sync_once()

    assert summary.synced_count == 2
    assert summary.failed_count == 0
    assert set(bitable_service.calls) == {
        ("bascn_alice", "tbl_alice", None),
        ("bascn_bob", "tbl_bob", None),
    }
    assert [
        item.company for item in repository.list_applications(owner_id="feishu:ou_alice")
    ] == ["腾讯"]
    assert [
        item.company for item in repository.list_applications(owner_id="feishu:ou_bob")
    ] == ["美团"]


def test_owner_resource_mapping_survives_sqlite_repository_restart(tmp_path) -> None:
    database_path = tmp_path / "offerpilot.db"
    repository = SQLiteOfferPilotRepository(str(database_path))
    save_owner_bitable_resource(
        repository,
        owner_id="feishu:ou_alice",
        app_token="bascn_alice",
        table_id="tbl_alice",
        schema_version="v4",
    )

    reopened = SQLiteOfferPilotRepository(str(database_path))
    resource = get_owner_bitable_resource(reopened, "feishu:ou_alice")

    assert resource is not None
    assert resource.app_token == "bascn_alice"
    assert resource.table_id == "tbl_alice"
    assert resource.schema_version == "v4"
    assert [item.owner_id for item in list_owner_bitable_resources(reopened)] == [
        "feishu:ou_alice"
    ]
