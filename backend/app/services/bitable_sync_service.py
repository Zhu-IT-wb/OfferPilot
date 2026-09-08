import json
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Any, Dict, List, Optional

from app.models.application import (
    APPLICATION_STATUS_LABELS,
    Application,
    ApplicationStatus,
    application_status_from_round,
)
from app.repositories.offerpilot_repository import OfferPilotRepository
from app.services.bitable_tenancy import (
    bitable_resource_owner_setting_key,
    get_bitable_resource_owner,
    list_owner_bitable_resources,
)
from app.services.feishu_service import (
    FeishuBitableService,
    FeishuConfigurationError,
    FeishuRequestError,
)


_OFFERPILOT_BITABLE_APP_TOKEN_SETTING = "feishu.offerpilot_bitable_app_token"
_OFFERPILOT_BITABLE_TABLE_ID_SETTING = "feishu.offerpilot_bitable_table_id"
_OFFERPILOT_BITABLE_RECORD_ID_PREFIX = "feishu.offerpilot_bitable_record_id."
_OFFERPILOT_BITABLE_APPLICATION_ID_PREFIX = "feishu.offerpilot_bitable_application_id."
_DEFAULT_OWNER_ID = "local_user"
_RECORD_LOCKS_GUARD = Lock()
_RECORD_LOCKS: Dict[tuple[str, str, str], Lock] = {}


# 承载外部服务调用后的结构化结果。
@dataclass(frozen=True)
class BitableRecordSyncResult:
    synced: bool
    status: str
    app_token: Optional[str] = None
    table_id: Optional[str] = None
    record_id: Optional[str] = None
    application_id: Optional[str] = None
    owner_id: Optional[str] = None
    created: bool = False
    updated_fields: List[str] = field(default_factory=list)
    metadata_updated: bool = False
    metadata_error: Optional[str] = None
    error: Optional[str] = None


# 将飞书多维表格记录同步回本地数据仓库。
def sync_bitable_record_to_repository(
    repository: OfferPilotRepository,
    bitable_service: FeishuBitableService,
    record_id: str,
    app_token: Optional[str] = None,
    table_id: Optional[str] = None,
    fallback_owner_id: Optional[str] = None,
    allow_default_local_owner: bool = True,
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

    stored_app_token = repository.get_runtime_setting(_OFFERPILOT_BITABLE_APP_TOKEN_SETTING)
    stored_table_id = repository.get_runtime_setting(_OFFERPILOT_BITABLE_TABLE_ID_SETTING)
    registered_owner_id = get_bitable_resource_owner(
        repository,
        resolved_app_token,
        resolved_table_id,
        include_legacy_local=False,
    )
    known_resources = list_owner_bitable_resources(repository)
    is_legacy_resource = bool(
        stored_app_token
        and stored_table_id
        and (resolved_app_token, resolved_table_id)
        == (stored_app_token, stored_table_id)
    )
    if (
        app_token
        and table_id
        and not registered_owner_id
        and not is_legacy_resource
        and (
            known_resources
            or (
                stored_app_token
                and stored_table_id
                and (app_token, table_id) != (stored_app_token, stored_table_id)
            )
        )
    ):
        return BitableRecordSyncResult(
            synced=False,
            status="ignored_foreign_table",
            app_token=resolved_app_token,
            table_id=resolved_table_id,
            record_id=record_id,
        )

    normalized_fallback_owner = (fallback_owner_id or "").strip() or None
    if (
        registered_owner_id
        and normalized_fallback_owner
        and normalized_fallback_owner != registered_owner_id
    ):
        return BitableRecordSyncResult(
            synced=False,
            status="owner_conflict",
            app_token=resolved_app_token,
            table_id=resolved_table_id,
            record_id=record_id,
            owner_id=registered_owner_id,
        )
    if registered_owner_id:
        fallback_owner_id = registered_owner_id

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
        fallback_owner_id=fallback_owner_id,
        allow_default_local_owner=allow_default_local_owner,
        bitable_service=bitable_service,
    )


# 将多维表格字段变更同步回本地投递记录。
def sync_bitable_record_fields_to_repository(
    repository: OfferPilotRepository,
    record_id: str,
    fields: Dict[str, Any],
    app_token: str,
    table_id: str,
    fallback_owner_id: Optional[str] = None,
    allow_default_local_owner: bool = True,
    bitable_service: Optional[FeishuBitableService] = None,
) -> BitableRecordSyncResult:
    record_lock = _record_sync_lock(app_token, table_id, record_id)
    with record_lock:
        return _sync_bitable_record_fields_locked(
            repository=repository,
            record_id=record_id,
            fields=fields,
            app_token=app_token,
            table_id=table_id,
            fallback_owner_id=fallback_owner_id,
            allow_default_local_owner=allow_default_local_owner,
            bitable_service=bitable_service,
        )


def _sync_bitable_record_fields_locked(
    repository: OfferPilotRepository,
    record_id: str,
    fields: Dict[str, Any],
    app_token: str,
    table_id: str,
    fallback_owner_id: Optional[str],
    allow_default_local_owner: bool,
    bitable_service: Optional[FeishuBitableService],
) -> BitableRecordSyncResult:
    field_application_id = _field_to_text(fields.get("OfferPilot记录ID")).strip() or None
    explicit_owner_id = extract_bitable_owner_id(fields)
    mapped_application_id, mapped_owner_id = _load_record_mapping(
        repository=repository,
        app_token=app_token,
        table_id=table_id,
        record_id=record_id,
    )
    mapping_is_authoritative = bool(mapped_application_id and mapped_owner_id)
    if (
        mapping_is_authoritative
        and field_application_id
        and field_application_id != mapped_application_id
    ):
        return BitableRecordSyncResult(
            synced=False,
            status="owner_conflict",
            app_token=app_token,
            table_id=table_id,
            record_id=record_id,
            application_id=mapped_application_id,
            owner_id=mapped_owner_id,
        )
    if field_application_id and not mapped_application_id:
        mapped_application_id = field_application_id
        mapped_owner_id = _find_application_owner_id(repository, field_application_id)
        mapping_is_authoritative = bool(mapped_owner_id)
    legacy_mapping_used = False
    if not mapped_application_id:
        legacy_mapping = _find_legacy_record_mapping(
            repository,
            table_id,
            record_id,
        )
        if legacy_mapping:
            mapped_application_id, mapped_owner_id = legacy_mapping
            legacy_mapping_used = True

    table_owner_id = get_bitable_table_owner(repository, app_token, table_id)
    authoritative_owner_id = (
        mapped_owner_id if mapping_is_authoritative else table_owner_id or fallback_owner_id
    )
    for asserted_owner_id in (
        explicit_owner_id,
        fallback_owner_id,
        table_owner_id,
    ):
        normalized_asserted_owner = (asserted_owner_id or "").strip()
        if (
            authoritative_owner_id
            and normalized_asserted_owner
            and normalized_asserted_owner != authoritative_owner_id
        ):
            return BitableRecordSyncResult(
                synced=False,
                status="owner_conflict",
                app_token=app_token,
                table_id=table_id,
                record_id=record_id,
                application_id=field_application_id or mapped_application_id,
                owner_id=authoritative_owner_id,
            )
    preferred_nonlocal_owner = next(
        (
            candidate
            for candidate in (explicit_owner_id, fallback_owner_id, table_owner_id)
            if candidate and candidate != _DEFAULT_OWNER_ID
        ),
        None,
    )
    owner_for_resolution = mapped_owner_id
    if mapped_owner_id == _DEFAULT_OWNER_ID and preferred_nonlocal_owner:
        owner_for_resolution = None
    owner_id = authoritative_owner_id or _owner_id_from_bitable_fields(
        fields,
        mapped_owner_id=owner_for_resolution,
        fallback_owner_id=fallback_owner_id,
        table_owner_id=table_owner_id,
        allow_default_local_owner=allow_default_local_owner,
    )
    if not owner_id:
        return BitableRecordSyncResult(
            synced=False,
            status="missing_owner",
            app_token=app_token,
            table_id=table_id,
            record_id=record_id,
        )

    remember_bitable_table_owner(repository, app_token, table_id, owner_id)
    application_id = (
        field_application_id
        or mapped_application_id
    )
    if (
        application_id
        and mapped_owner_id
        and mapped_owner_id != owner_id
    ):
        may_migrate_legacy_local_mapping = (
            legacy_mapping_used
            and mapped_owner_id == _DEFAULT_OWNER_ID
            and owner_id != _DEFAULT_OWNER_ID
        )
        if not may_migrate_legacy_local_mapping or not _reassign_application_owner(
            repository,
            application_id=application_id,
            from_owner_id=mapped_owner_id,
            to_owner_id=owner_id,
        ):
            return BitableRecordSyncResult(
                synced=False,
                status="owner_conflict",
                app_token=app_token,
                table_id=table_id,
                record_id=record_id,
                application_id=application_id,
                owner_id=owner_id,
            )
    values = _application_values_from_bitable_fields(fields)

    if application_id:
        application = repository.update_application_by_id(
            application_id=application_id,
            owner_id=owner_id,
            **values,
        )
        if application is None:
            result = _create_application_from_fields(
                repository=repository,
                table_id=table_id,
                record_id=record_id,
                fields=fields,
                values=values,
                app_token=app_token,
                owner_id=owner_id,
            )
        else:
            remember_bitable_record_mapping(
                repository=repository,
                app_token=app_token,
                table_id=table_id,
                record_id=record_id,
                application_id=application.id,
                owner_id=owner_id,
            )
            result = BitableRecordSyncResult(
                synced=True,
                status="updated",
                app_token=app_token,
                table_id=table_id,
                record_id=record_id,
                application_id=application.id,
                owner_id=owner_id,
                updated_fields=_updated_field_names(values),
            )
    else:
        result = _create_application_from_fields(
            repository=repository,
            table_id=table_id,
            record_id=record_id,
            fields=fields,
            values=values,
            app_token=app_token,
            owner_id=owner_id,
        )

    return _write_missing_bitable_metadata(
        result=result,
        fields=fields,
        bitable_service=bitable_service,
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
        base_location=values.get("base_location"),
        interview_time=values.get("interview_time"),
        round_name=values.get("round_name"),
        jd_keywords=values.get("jd_keywords") or [],
        owner_id=owner_id,
    )
    status = values.get("status")
    if status is not None:
        repository.update_application_by_id(application.id, status=status, owner_id=owner_id)
        application.status = status

    remember_bitable_record_mapping(
        repository=repository,
        app_token=app_token,
        table_id=table_id,
        record_id=record_id,
        application_id=application.id,
        owner_id=owner_id,
    )
    return BitableRecordSyncResult(
        synced=True,
        status="created",
        app_token=app_token,
        table_id=table_id,
        record_id=record_id,
        application_id=application.id,
        owner_id=owner_id,
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
    if "工作地点" in fields:
        values["base_location"] = _field_to_text(fields.get("工作地点")).strip()

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
def _owner_id_from_bitable_fields(
    fields: Dict[str, Any],
    *,
    mapped_owner_id: Optional[str] = None,
    fallback_owner_id: Optional[str] = None,
    table_owner_id: Optional[str] = None,
    allow_default_local_owner: bool = True,
) -> Optional[str]:
    candidates = (
        extract_bitable_owner_id(fields) or "",
        (mapped_owner_id or "").strip(),
        (fallback_owner_id or "").strip(),
        (table_owner_id or "").strip(),
    )
    for owner_id in candidates:
        if owner_id:
            return owner_id
    return _DEFAULT_OWNER_ID if allow_default_local_owner else None


def extract_bitable_owner_id(fields: Dict[str, Any]) -> Optional[str]:
    owner_id = _field_to_text(fields.get("OfferPilot用户ID")).strip()
    return owner_id or None


def get_bitable_table_owner(
    repository: OfferPilotRepository,
    app_token: str,
    table_id: str,
) -> Optional[str]:
    owner_id = get_bitable_resource_owner(
        repository,
        app_token,
        table_id,
        include_legacy_local=False,
    )
    return owner_id.strip() if owner_id and owner_id.strip() else None


def remember_bitable_table_owner(
    repository: OfferPilotRepository,
    app_token: str,
    table_id: str,
    owner_id: Optional[str],
) -> Optional[str]:
    normalized_owner_id = (owner_id or "").strip()
    if not normalized_owner_id or normalized_owner_id == _DEFAULT_OWNER_ID:
        return get_bitable_table_owner(repository, app_token, table_id)

    setting_key = _bitable_table_owner_setting_key(app_token, table_id)
    existing_owner_id = repository.get_runtime_setting(setting_key)
    if existing_owner_id and existing_owner_id.strip():
        return existing_owner_id.strip()
    repository.set_runtime_setting(setting_key, normalized_owner_id)
    return normalized_owner_id


def remember_bitable_record_mapping(
    repository: OfferPilotRepository,
    app_token: str,
    table_id: str,
    record_id: str,
    application_id: str,
    owner_id: str,
) -> None:
    repository.set_runtime_setting(
        _bitable_record_setting_key(table_id, application_id),
        record_id,
    )
    repository.set_runtime_setting(
        _bitable_application_setting_key(app_token, table_id, record_id),
        json.dumps(
            {
                "application_id": application_id,
                "owner_id": owner_id,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


def _load_record_mapping(
    repository: OfferPilotRepository,
    app_token: str,
    table_id: str,
    record_id: str,
) -> tuple[Optional[str], Optional[str]]:
    raw_mapping = repository.get_runtime_setting(
        _bitable_application_setting_key(app_token, table_id, record_id)
    )
    if not raw_mapping:
        return None, None
    try:
        mapping = json.loads(raw_mapping)
    except (TypeError, json.JSONDecodeError):
        return None, None
    if not isinstance(mapping, dict):
        return None, None
    application_id = mapping.get("application_id")
    owner_id = mapping.get("owner_id")
    return (
        (
            application_id.strip()
            if isinstance(application_id, str) and application_id.strip()
            else None
        ),
        owner_id.strip() if isinstance(owner_id, str) and owner_id.strip() else None,
    )


def _write_missing_bitable_metadata(
    result: BitableRecordSyncResult,
    fields: Dict[str, Any],
    bitable_service: Optional[FeishuBitableService],
) -> BitableRecordSyncResult:
    if not result.synced or not result.application_id or not result.owner_id:
        return result
    update_record = getattr(bitable_service, "update_record", None)
    if not callable(update_record):
        return result

    metadata_fields: Dict[str, Any] = {}
    if not _field_to_text(fields.get("OfferPilot记录ID")).strip():
        metadata_fields["OfferPilot记录ID"] = result.application_id
    if (
        result.owner_id != _DEFAULT_OWNER_ID
        and not _field_to_text(fields.get("OfferPilot用户ID")).strip()
    ):
        metadata_fields["OfferPilot用户ID"] = result.owner_id
    if not metadata_fields:
        return result

    try:
        update_record(
            app_token=result.app_token or "",
            table_id=result.table_id or "",
            record_id=result.record_id or "",
            fields=metadata_fields,
        )
    except (FeishuConfigurationError, FeishuRequestError) as exc:
        return replace(result, metadata_error=_summarize_error(str(exc)))
    return replace(result, metadata_updated=True)


def _reassign_application_owner(
    repository: OfferPilotRepository,
    application_id: str,
    from_owner_id: str,
    to_owner_id: str,
) -> bool:
    reassign_owner = getattr(repository, "reassign_application_owner", None)
    if not callable(reassign_owner):
        return False
    return bool(
        reassign_owner(
            application_id=application_id,
            from_owner_id=from_owner_id,
            to_owner_id=to_owner_id,
        )
    )


def _find_application_owner_id(
    repository: OfferPilotRepository,
    application_id: str,
) -> Optional[str]:
    find_owner = getattr(repository, "find_application_owner_id", None)
    if not callable(find_owner):
        return None
    owner_id = find_owner(application_id)
    return owner_id.strip() if isinstance(owner_id, str) and owner_id.strip() else None


def _find_legacy_record_mapping(
    repository: OfferPilotRepository,
    table_id: str,
    record_id: str,
) -> Optional[tuple[str, str]]:
    find_mapping = getattr(repository, "find_application_by_bitable_record_id", None)
    if not callable(find_mapping):
        return None
    mapping = find_mapping(table_id, record_id)
    if not isinstance(mapping, tuple) or len(mapping) != 2:
        return None
    application_id, owner_id = mapping
    if not isinstance(application_id, str) or not application_id.strip():
        return None
    if not isinstance(owner_id, str) or not owner_id.strip():
        return None
    return application_id.strip(), owner_id.strip()


def _record_sync_lock(app_token: str, table_id: str, record_id: str) -> Lock:
    key = (app_token, table_id, record_id)
    with _RECORD_LOCKS_GUARD:
        lock = _RECORD_LOCKS.get(key)
        if lock is None:
            lock = Lock()
            _RECORD_LOCKS[key] = lock
        return lock


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


# 处理 bitable_record_setting_key 相关逻辑。
def _bitable_record_setting_key(table_id: str, application_id: str) -> str:
    return f"{_OFFERPILOT_BITABLE_RECORD_ID_PREFIX}{table_id}.{application_id}"


def _bitable_application_setting_key(
    app_token: str,
    table_id: str,
    record_id: str,
) -> str:
    return f"{_OFFERPILOT_BITABLE_APPLICATION_ID_PREFIX}{app_token}.{table_id}.{record_id}"


def _bitable_table_owner_setting_key(app_token: str, table_id: str) -> str:
    return bitable_resource_owner_setting_key(app_token, table_id)


# 处理 updated_field_names 相关逻辑。
def _updated_field_names(values: Dict[str, Any]) -> List[str]:
    field_names = {
        "company": "公司",
        "role": "岗位",
        "base_location": "工作地点",
        "status": "投递状态",
        "interview_time": "面试时间文本",
        "round_name": "面试轮次",
        "jd_keywords": "JD关键词",
    }
    return [field_names[key] for key in values if key in field_names]


# 压缩并说明 error。
def _summarize_error(error: str) -> str:
    return error.splitlines()[0].strip()[:200] if error else "unknown"
