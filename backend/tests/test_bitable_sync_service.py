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
    assert repository.applications[0].status.value == "submitted"
    assert (
        repository.get_runtime_setting("feishu.offerpilot_bitable_record_id.tbl_applications.app_1")
        == "rec_manual_1"
    )

