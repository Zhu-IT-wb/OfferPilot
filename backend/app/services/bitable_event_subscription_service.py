from dataclasses import dataclass
from typing import Optional

from app.core.config import settings
from app.repositories.offerpilot_repository import OfferPilotRepository
from app.services.feishu_service import (
    FeishuBitableService,
    FeishuConfigurationError,
    FeishuRequestError,
)


_OFFERPILOT_BITABLE_APP_TOKEN_SETTING = "feishu.offerpilot_bitable_app_token"
_OFFERPILOT_BITABLE_EVENT_SUBSCRIPTION_PREFIX = "feishu.offerpilot_bitable_event_subscription."


# 承载外部服务调用后的结构化结果。
@dataclass(frozen=True)
class BitableEventSubscriptionResult:
    subscribed: bool
    status: str
    app_token: Optional[str] = None
    error: Optional[str] = None


# 确保指定多维表格文件已订阅变更事件。
def ensure_bitable_event_subscription(
    repository: OfferPilotRepository,
    bitable_service: Optional[FeishuBitableService] = None,
    force: bool = False,
    app_token: Optional[str] = None,
) -> BitableEventSubscriptionResult:
    service = bitable_service or FeishuBitableService()
    if not settings.feishu_bitable_sync_enabled:
        return BitableEventSubscriptionResult(False, "disabled")
    if hasattr(service, "is_configured") and not service.is_configured():
        return BitableEventSubscriptionResult(False, "missing_credentials")

    resolved_app_token = (
        (app_token or "").strip()
        or repository.get_runtime_setting(_OFFERPILOT_BITABLE_APP_TOKEN_SETTING)
        or getattr(service, "app_token", None)
    )
    if not resolved_app_token:
        return BitableEventSubscriptionResult(False, "missing_app_token")
    if not hasattr(service, "subscribe_bitable_file"):
        return BitableEventSubscriptionResult(
            False,
            "not_supported",
            app_token=resolved_app_token,
        )

    setting_key = _bitable_event_subscription_setting_key(resolved_app_token)
    current_status = repository.get_runtime_setting(setting_key)
    if not force and current_status == "subscribed":
        return BitableEventSubscriptionResult(
            True,
            "cached",
            app_token=resolved_app_token,
        )

    try:
        service.subscribe_bitable_file(app_token=resolved_app_token)
    except (FeishuConfigurationError, FeishuRequestError) as exc:
        error = _summarize_error(str(exc))
        repository.set_runtime_setting(setting_key, f"failed: {error}")
        return BitableEventSubscriptionResult(
            False,
            "failed",
            app_token=resolved_app_token,
            error=error,
        )

    repository.set_runtime_setting(setting_key, "subscribed")
    return BitableEventSubscriptionResult(
        True,
        "subscribed",
        app_token=resolved_app_token,
    )


# 处理 bitable_event_subscription_setting_key 相关逻辑。
def _bitable_event_subscription_setting_key(app_token: str) -> str:
    return f"{_OFFERPILOT_BITABLE_EVENT_SUBSCRIPTION_PREFIX}{app_token}"


# 压缩并说明 error。
def _summarize_error(error: str) -> str:
    return error.splitlines()[0].strip()[:200] if error else "unknown"
