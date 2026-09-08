import hashlib
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional

from app.repositories.offerpilot_repository import OfferPilotRepository


LEGACY_BITABLE_APP_TOKEN_SETTING = "feishu.offerpilot_bitable_app_token"
LEGACY_BITABLE_TABLE_ID_SETTING = "feishu.offerpilot_bitable_table_id"
LEGACY_BITABLE_SCHEMA_VERSION_SETTING = "feishu.offerpilot_bitable_schema_version"
BITABLE_RESOURCE_PREFIX = "feishu.offerpilot_bitable_resource.v1."
BITABLE_OWNER_PREFIX = "feishu.offerpilot_bitable_owner."
DEFAULT_OWNER_ID = "local_user"

_RESOURCE_SETTINGS_LOCK = Lock()


@dataclass(frozen=True)
class OwnerBitableResource:
    owner_id: str
    app_token: str
    table_id: str
    schema_version: Optional[str] = None
    legacy: bool = False


class BitableResourceConflictError(ValueError):
    pass


def normalize_bitable_owner_id(owner_id: Optional[str]) -> str:
    return (owner_id or "").strip() or DEFAULT_OWNER_ID


def owner_bitable_setting_key(owner_id: str, field: str) -> str:
    normalized_owner = normalize_bitable_owner_id(owner_id)
    owner_scope = hashlib.sha256(normalized_owner.encode("utf-8")).hexdigest()
    return f"{BITABLE_RESOURCE_PREFIX}{owner_scope}.{field}"


def get_owner_bitable_resource(
    repository: OfferPilotRepository,
    owner_id: str,
    *,
    legacy_app_token: Optional[str] = None,
    legacy_table_id: Optional[str] = None,
) -> Optional[OwnerBitableResource]:
    normalized_owner = normalize_bitable_owner_id(owner_id)
    app_token = _setting(repository, normalized_owner, "app_token")
    table_id = _setting(repository, normalized_owner, "table_id")
    if app_token and table_id:
        stored_owner = _setting(repository, normalized_owner, "owner_id")
        if stored_owner and stored_owner != normalized_owner:
            return None
        registered_owner = get_bitable_resource_owner(
            repository,
            app_token,
            table_id,
            include_legacy_local=False,
        )
        if registered_owner and registered_owner != normalized_owner:
            return None
        return OwnerBitableResource(
            owner_id=normalized_owner,
            app_token=app_token,
            table_id=table_id,
            schema_version=_setting(repository, normalized_owner, "schema_version"),
        )

    fallback_app_token = (
        (legacy_app_token or "").strip()
        or (repository.get_runtime_setting(LEGACY_BITABLE_APP_TOKEN_SETTING) or "").strip()
    )
    fallback_table_id = (
        (legacy_table_id or "").strip()
        or (repository.get_runtime_setting(LEGACY_BITABLE_TABLE_ID_SETTING) or "").strip()
    )
    if not fallback_app_token or not fallback_table_id:
        return None

    registered_owner = get_bitable_resource_owner(
        repository,
        fallback_app_token,
        fallback_table_id,
        include_legacy_local=False,
    )
    if registered_owner and registered_owner != normalized_owner:
        return None
    if normalized_owner != DEFAULT_OWNER_ID and registered_owner != normalized_owner:
        return None
    return OwnerBitableResource(
        owner_id=normalized_owner,
        app_token=fallback_app_token,
        table_id=fallback_table_id,
        schema_version=(
            repository.get_runtime_setting(LEGACY_BITABLE_SCHEMA_VERSION_SETTING)
            if normalized_owner == DEFAULT_OWNER_ID
            else None
        ),
        legacy=True,
    )


def save_owner_bitable_resource(
    repository: OfferPilotRepository,
    *,
    owner_id: str,
    app_token: str,
    table_id: str,
    schema_version: Optional[str] = None,
    mirror_legacy: Optional[bool] = None,
) -> OwnerBitableResource:
    normalized_owner = normalize_bitable_owner_id(owner_id)
    normalized_app_token = (app_token or "").strip()
    normalized_table_id = (table_id or "").strip()
    if not normalized_app_token or not normalized_table_id:
        raise ValueError("Bitable app_token and table_id are required.")

    with _RESOURCE_SETTINGS_LOCK:
        current_owner = get_bitable_resource_owner(
            repository,
            normalized_app_token,
            normalized_table_id,
            include_legacy_local=False,
        )
        if current_owner and current_owner != normalized_owner:
            raise BitableResourceConflictError(
                "The Bitable resource is already assigned to another owner."
            )

        existing_app_token = _setting(repository, normalized_owner, "app_token")
        existing_table_id = _setting(repository, normalized_owner, "table_id")
        if (
            existing_app_token
            and existing_table_id
            and (existing_app_token, existing_table_id)
            != (normalized_app_token, normalized_table_id)
        ):
            raise BitableResourceConflictError(
                "The owner already has a different Bitable resource."
            )

        repository.set_runtime_setting(
            owner_bitable_setting_key(normalized_owner, "owner_id"),
            normalized_owner,
        )
        repository.set_runtime_setting(
            owner_bitable_setting_key(normalized_owner, "app_token"),
            normalized_app_token,
        )
        repository.set_runtime_setting(
            owner_bitable_setting_key(normalized_owner, "table_id"),
            normalized_table_id,
        )
        if schema_version:
            repository.set_runtime_setting(
                owner_bitable_setting_key(normalized_owner, "schema_version"),
                schema_version,
            )
        repository.set_runtime_setting(
            bitable_resource_owner_setting_key(normalized_app_token, normalized_table_id),
            normalized_owner,
        )

        should_mirror_legacy = (
            normalized_owner == DEFAULT_OWNER_ID
            if mirror_legacy is None
            else mirror_legacy
        )
        if should_mirror_legacy:
            repository.set_runtime_setting(
                LEGACY_BITABLE_APP_TOKEN_SETTING,
                normalized_app_token,
            )
            repository.set_runtime_setting(
                LEGACY_BITABLE_TABLE_ID_SETTING,
                normalized_table_id,
            )
            if schema_version:
                repository.set_runtime_setting(
                    LEGACY_BITABLE_SCHEMA_VERSION_SETTING,
                    schema_version,
                )

    return OwnerBitableResource(
        owner_id=normalized_owner,
        app_token=normalized_app_token,
        table_id=normalized_table_id,
        schema_version=schema_version,
    )


def set_owner_bitable_schema_version(
    repository: OfferPilotRepository,
    owner_id: str,
    schema_version: str,
) -> None:
    normalized_owner = normalize_bitable_owner_id(owner_id)
    repository.set_runtime_setting(
        owner_bitable_setting_key(normalized_owner, "schema_version"),
        schema_version,
    )
    if normalized_owner == DEFAULT_OWNER_ID:
        repository.set_runtime_setting(
            LEGACY_BITABLE_SCHEMA_VERSION_SETTING,
            schema_version,
        )


def get_bitable_resource_owner(
    repository: OfferPilotRepository,
    app_token: str,
    table_id: str,
    *,
    include_legacy_local: bool = True,
) -> Optional[str]:
    normalized_app_token = (app_token or "").strip()
    normalized_table_id = (table_id or "").strip()
    if not normalized_app_token or not normalized_table_id:
        return None
    owner_id = repository.get_runtime_setting(
        bitable_resource_owner_setting_key(normalized_app_token, normalized_table_id)
    )
    if owner_id and owner_id.strip():
        return normalize_bitable_owner_id(owner_id)

    if include_legacy_local:
        legacy_app_token = (
            repository.get_runtime_setting(LEGACY_BITABLE_APP_TOKEN_SETTING) or ""
        ).strip()
        legacy_table_id = (
            repository.get_runtime_setting(LEGACY_BITABLE_TABLE_ID_SETTING) or ""
        ).strip()
        if (legacy_app_token, legacy_table_id) == (
            normalized_app_token,
            normalized_table_id,
        ):
            return DEFAULT_OWNER_ID
    return None


def list_owner_bitable_resources(
    repository: OfferPilotRepository,
    *,
    legacy_app_token: Optional[str] = None,
    legacy_table_id: Optional[str] = None,
) -> list[OwnerBitableResource]:
    list_settings = getattr(repository, "list_runtime_settings", None)
    settings_by_key: Dict[str, str] = (
        list_settings(BITABLE_RESOURCE_PREFIX) if callable(list_settings) else {}
    )
    grouped: Dict[str, Dict[str, str]] = {}
    for key, value in settings_by_key.items():
        suffix = key[len(BITABLE_RESOURCE_PREFIX) :]
        owner_scope, separator, field = suffix.partition(".")
        if not separator or not owner_scope or not field:
            continue
        grouped.setdefault(owner_scope, {})[field] = value

    resources: list[OwnerBitableResource] = []
    seen: set[tuple[str, str]] = set()
    for values in grouped.values():
        owner_id = (values.get("owner_id") or "").strip()
        app_token = (values.get("app_token") or "").strip()
        table_id = (values.get("table_id") or "").strip()
        if not owner_id or not app_token or not table_id:
            continue
        registered_owner = get_bitable_resource_owner(
            repository,
            app_token,
            table_id,
            include_legacy_local=False,
        )
        if registered_owner and registered_owner != owner_id:
            continue
        resource_key = (app_token, table_id)
        if resource_key in seen:
            continue
        seen.add(resource_key)
        resources.append(
            OwnerBitableResource(
                owner_id=normalize_bitable_owner_id(owner_id),
                app_token=app_token,
                table_id=table_id,
                schema_version=(values.get("schema_version") or "").strip() or None,
            )
        )

    fallback_app_token = (
        (legacy_app_token or "").strip()
        or (repository.get_runtime_setting(LEGACY_BITABLE_APP_TOKEN_SETTING) or "").strip()
    )
    fallback_table_id = (
        (legacy_table_id or "").strip()
        or (repository.get_runtime_setting(LEGACY_BITABLE_TABLE_ID_SETTING) or "").strip()
    )
    if (
        fallback_app_token
        and fallback_table_id
        and (fallback_app_token, fallback_table_id) not in seen
    ):
        resources.append(
            OwnerBitableResource(
                owner_id=(
                    get_bitable_resource_owner(
                        repository,
                        fallback_app_token,
                        fallback_table_id,
                        include_legacy_local=True,
                    )
                    or DEFAULT_OWNER_ID
                ),
                app_token=fallback_app_token,
                table_id=fallback_table_id,
                schema_version=(
                    repository.get_runtime_setting(LEGACY_BITABLE_SCHEMA_VERSION_SETTING)
                    or ""
                ).strip()
                or None,
                legacy=True,
            )
        )
    return sorted(resources, key=lambda item: (item.owner_id, item.app_token, item.table_id))


def bitable_resource_owner_setting_key(app_token: str, table_id: str) -> str:
    return f"{BITABLE_OWNER_PREFIX}{app_token}.{table_id}"


def _setting(
    repository: OfferPilotRepository,
    owner_id: str,
    field: str,
) -> Optional[str]:
    value = repository.get_runtime_setting(owner_bitable_setting_key(owner_id, field))
    return value.strip() if value and value.strip() else None
