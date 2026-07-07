from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.models.application import (
    APPLICATION_STATUS_LABELS,
    Application,
    ApplicationStatus,
    application_status_from_round,
)
from app.repositories.offerpilot_repository import OfferPilotRepository
from app.services.feishu_service import (
    FeishuBitableService,
    FeishuConfigurationError,
    FeishuRequestError,
)


_OFFERPILOT_BITABLE_APP_TOKEN_SETTING = "feishu.offerpilot_bitable_app_token"
_OFFERPILOT_BITABLE_TABLE_ID_SETTING = "feishu.offerpilot_bitable_table_id"
_OFFERPILOT_BITABLE_RECORD_ID_PREFIX = "feishu.offerpilot_bitable_record_id."
_DEFAULT_OWNER_ID = "local_user"


# 承载外部服务调用后的结构化结果。
@dataclass(frozen=True)
class BitableRecordSyncResult:
    synced: bool
    status: str
    app_token: Optional[str] = None
    table_id: Optional[str] = None
    record_id: Optional[str] = None
    application_id: Optional[str] = None
    created: bool = False
    updated_fields: List[str] = field(default_factory=list)
    error: Optional[str] = None


# 将飞书多维表格记录同步回本地数据仓库。
def sync_bitable_record_to_repository(
    repository: OfferPilotRepository,
    bitable_service: FeishuBitableService,
    record_id: str,
    app_token: Optional[str] = None,
    table_id: Optional[str] = None,
) -> BitableRecordSyncResult:
    resolved_app_token = _resolve_app_token(repository, bitable_service, app_token)
    resolved_table_id = _resolve_table_id(repository, bitable_service, table_id)
    if not resolved_app_token:
        return BitableRecordSyncResult(
            synced=False,
            status="missing_app_token",
            table_id=resolved_table_id,
            record_id=record_id,
        )
    if not resolved_table_id:
        return BitableRecordSyncResult(
            synced=False,
            status="missing_table_id",
            app_token=resolved_app_token,
            record_id=record_id,
        )

    stored_table_id = repository.get_runtime_setting(_OFFERPILOT_BITABLE_TABLE_ID_SETTING)
    if table_id and stored_table_id and table_id != stored_table_id:
        return BitableRecordSyncResult(
            synced=False,
            status="ignored_foreign_table",
            app_token=resolved_app_token,
            table_id=resolved_table_id,
            record_id=record_id,
        )

    try:
        record = bitable_service.get_record(
            app_token=resolved_app_token,
            table_id=resolved_table_id,
            record_id=record_id,
        )
    except (FeishuConfigurationError, FeishuRequestError) as exc:
        return BitableRecordSyncResult(
            synced=False,
            status="failed",
            app_token=resolved_app_token,
            table_id=resolved_table_id,
            record_id=record_id,
            error=_summarize_error(str(exc)),
        )

    fields = record.fields or {}
    return sync_bitable_record_fields_to_repository(
        repository=repository,
        record_id=record.record_id or record_id,
        fields=fields,
        app_token=resolved_app_token,
        table_id=resolved_table_id,
    )


# 将多维表格字段变更同步回本地投递记录。
def sync_bitable_record_fields_to_repository(
    repository: OfferPilotRepository,
    record_id: str,
    fields: Dict[str, Any],
    app_token: str,
    table_id: str,
) -> BitableRecordSyncResult:
    owner_id = _owner_id_from_bitable_fields(fields)
    application_id = (
        _field_to_text(fields.get("OfferPilot记录ID")).strip()
        or _find_application_id_by_record_id(repository, table_id, record_id, owner_id=owner_id)
    )
    values = _application_values_from_bitable_fields(fields)

    if application_id:
        application = repository.update_application_by_id(
            application_id=application_id,
            owner_id=owner_id,
            **values,
        )
        if application is None:
            return _create_application_from_fields(
                repository=repository,
                table_id=table_id,
                record_id=record_id,
                fields=fields,
                values=values,
                app_token=app_token,
                owner_id=owner_id,
            )

        repository.set_runtime_setting(
            _bitable_record_setting_key(table_id, application.id),
            record_id,
        )
        return BitableRecordSyncResult(
            synced=True,
            status="updated",
            app_token=app_token,
            table_id=table_id,
            record_id=record_id,
            application_id=application.id,
            updated_fields=_updated_field_names(values),
        )

    return _create_application_from_fields(
        repository=repository,
        table_id=table_id,
        record_id=record_id,
        fields=fields,
        values=values,
        app_token=app_token,
        owner_id=owner_id,
    )


# 创建 application from fields。
def _create_application_from_fields(
    repository: OfferPilotRepository,
    table_id: str,
    record_id: str,
    fields: Dict[str, Any],
    values: Dict[str, Any],
    app_token: str,
    owner_id: str,
) -> BitableRecordSyncResult:
    company = values.get("company") or _field_to_text(fields.get("公司")).strip()
    role = values.get("role") or _field_to_text(fields.get("岗位")).strip()
    if not company or not role:
        return BitableRecordSyncResult(
            synced=False,
            status="missing_required_fields",
            app_token=app_token,
            table_id=table_id,
            record_id=record_id,
        )

    application = repository.create_application(
        company=company,
        role=role,
        interview_time=values.get("interview_time"),
        round_name=values.get("round_name"),
        jd_keywords=values.get("jd_keywords") or [],
        owner_id=owner_id,
    )
    status = values.get("status")
    if status is not None:
        repository.update_application_by_id(application.id, status=status, owner_id=owner_id)
        application.status = status

    repository.set_runtime_setting(
        _bitable_record_setting_key(table_id, application.id),
        record_id,
    )
    return BitableRecordSyncResult(
        synced=True,
        status="created",
        app_token=app_token,
        table_id=table_id,
        record_id=record_id,
        application_id=application.id,
        created=True,
        updated_fields=_updated_field_names(values),
    )


# 处理 application_values_from_bitable_fields 相关逻辑。
def _application_values_from_bitable_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    if "公司" in fields:
        company = _field_to_text(fields.get("公司")).strip()
        if company:
            values["company"] = company
    if "岗位" in fields:
        role = _field_to_text(fields.get("岗位")).strip()
        if role:
            values["role"] = role

    status = _status_from_fields(fields)
    if status is not None:
        values["status"] = status

    if "面试时间文本" in fields:
        values["interview_time"] = _field_to_text(fields.get("面试时间文本")).strip()
    if "面试轮次" in fields:
        round_name = _field_to_text(fields.get("面试轮次")).strip()
        values["round_name"] = round_name
        if status is None and round_name:
            values["status"] = application_status_from_round(round_name)
    if "JD关键词" in fields:
        values["jd_keywords"] = _field_to_text_list(fields.get("JD关键词"))
    return values


# 从多维表格字段中提取数据归属人。
def _owner_id_from_bitable_fields(fields: Dict[str, Any]) -> str:
    owner_id = _field_to_text(fields.get("OfferPilot用户ID")).strip()
    return owner_id or _DEFAULT_OWNER_ID


# 处理 status_from_fields 相关逻辑。
def _status_from_fields(fields: Dict[str, Any]) -> Optional[ApplicationStatus]:
    label = _field_to_text(fields.get("投递状态")).strip()
    label_mapping = {value: key for key, value in APPLICATION_STATUS_LABELS.items()}
    if label in label_mapping:
        return label_mapping[label]

    raw_status = _field_to_text(fields.get("状态值")).strip()
    if raw_status:
        try:
            return ApplicationStatus(raw_status)
        except ValueError:
            return None
    return None


# 处理 field_to_text 相关逻辑。
def _field_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "".join(_field_to_text(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "name", "title", "value"):
            nested = value.get(key)
            if nested is not None:
                return _field_to_text(nested)
    return str(value)


# 处理 field_to_text_list 相关逻辑。
def _field_to_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [_field_to_text(item).strip() for item in value]
        return [item for item in items if item]
    text = _field_to_text(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.replace("，", ",").split(",") if item.strip()]


# 解析并确定 app token。
def _resolve_app_token(
    repository: OfferPilotRepository,
    bitable_service: FeishuBitableService,
    app_token: Optional[str],
) -> Optional[str]:
    return (
        app_token
        or repository.get_runtime_setting(_OFFERPILOT_BITABLE_APP_TOKEN_SETTING)
        or getattr(bitable_service, "app_token", None)
    )


# 解析并确定 table id。
def _resolve_table_id(
    repository: OfferPilotRepository,
    bitable_service: FeishuBitableService,
    table_id: Optional[str],
) -> Optional[str]:
    return (
        table_id
        or repository.get_runtime_setting(_OFFERPILOT_BITABLE_TABLE_ID_SETTING)
        or getattr(bitable_service, "table_id", None)
    )


# 查找 application id by record id。
def _find_application_id_by_record_id(
    repository: OfferPilotRepository,
    table_id: str,
    record_id: str,
    owner_id: str = _DEFAULT_OWNER_ID,
) -> Optional[str]:
    for application in repository.list_applications(owner_id=owner_id):
        stored_record_id = repository.get_runtime_setting(
            _bitable_record_setting_key(table_id, application.id)
        )
        if stored_record_id == record_id:
            return application.id
    return None


# 处理 bitable_record_setting_key 相关逻辑。
def _bitable_record_setting_key(table_id: str, application_id: str) -> str:
    return f"{_OFFERPILOT_BITABLE_RECORD_ID_PREFIX}{table_id}.{application_id}"


# 处理 updated_field_names 相关逻辑。
def _updated_field_names(values: Dict[str, Any]) -> List[str]:
    field_names = {
        "company": "公司",
        "role": "岗位",
        "status": "投递状态",
        "interview_time": "面试时间文本",
        "round_name": "面试轮次",
        "jd_keywords": "JD关键词",
    }
    return [field_names[key] for key in values if key in field_names]


# 压缩并说明 error。
def _summarize_error(error: str) -> str:
    return error.splitlines()[0].strip()[:200] if error else "unknown"
