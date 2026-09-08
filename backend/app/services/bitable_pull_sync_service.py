import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from app.core.config import settings
from app.repositories.offerpilot_repository import OfferPilotRepository
from app.services.bitable_sync_service import (
    get_bitable_table_owner,
    sync_bitable_record_fields_to_repository,
)
from app.services.bitable_tenancy import (
    OwnerBitableResource,
    list_owner_bitable_resources,
)
from app.services.feishu_service import (
    FeishuBitableService,
    FeishuConfigurationError,
    FeishuRequestError,
)


logger = logging.getLogger(__name__)
_OFFERPILOT_BITABLE_APP_TOKEN_SETTING = "feishu.offerpilot_bitable_app_token"
_OFFERPILOT_BITABLE_TABLE_ID_SETTING = "feishu.offerpilot_bitable_table_id"


# 定义 BitablePullSyncSummary 相关的数据结构或领域对象。
@dataclass(frozen=True)
class BitablePullSyncSummary:
    synced_count: int
    failed_count: int
    skipped: bool = False
    reason: Optional[str] = None


# 定时从飞书多维表格拉取变更并同步到本地。
class BitablePullSyncService:
    # 初始化当前组件所需的依赖和配置。
    def __init__(
        self,
        repository: OfferPilotRepository,
        bitable_service: Optional[FeishuBitableService] = None,
        interval_seconds: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> None:
        self.repository = repository
        self.bitable_service = bitable_service or FeishuBitableService()
        self.interval_seconds = max(
            interval_seconds
            if interval_seconds is not None
            else settings.feishu_bitable_pull_sync_interval_seconds,
            5,
        )
        self.page_size = max(
            min(
                page_size
                if page_size is not None
                else settings.feishu_bitable_pull_sync_page_size,
                500,
            ),
            1,
        )
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    # 判断 enabled 是否成立。
    def is_enabled(self) -> bool:
        if not settings.feishu_bitable_pull_sync_enabled:
            return False
        if self.bitable_service.is_bitable_sync_enabled():
            return True
        if not self._resolve_resources() or not bool(
            getattr(self.bitable_service, "sync_enabled", False)
        ):
            return False
        is_configured = getattr(self.bitable_service, "is_configured", None)
        return not callable(is_configured) or bool(is_configured())

    # 启动后台任务。
    def start(self) -> None:
        if not self.is_enabled():
            logger.info("Feishu bitable pull sync is disabled.")
            return
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())
        logger.info("Feishu bitable pull sync started: interval_seconds=%s", self.interval_seconds)

    # 停止后台任务。
    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            await self._task
            self._task = None

    # 处理 run 相关逻辑。
    async def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                summary = await asyncio.to_thread(self.sync_once)
                if summary.skipped:
                    logger.info("Feishu bitable pull sync skipped: reason=%s", summary.reason)
                else:
                    logger.info(
                        "Feishu bitable pull sync finished: synced=%s failed=%s",
                        summary.synced_count,
                        summary.failed_count,
                    )
            except Exception as exc:
                logger.warning("Feishu bitable pull sync failed: %s", exc)

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.interval_seconds,
                )
            except asyncio.TimeoutError:
                continue

    # 执行一次多维表格全量拉取同步。
    def sync_once(self) -> BitablePullSyncSummary:
        resources = self._resolve_resources()
        if not resources:
            return BitablePullSyncSummary(0, 0, skipped=True, reason="missing_app_token")

        synced_count = 0
        failed_count = 0
        first_error: Optional[str] = None
        for resource in resources:
            summary = self._sync_resource(resource)
            synced_count += summary.synced_count
            failed_count += summary.failed_count
            if summary.reason and first_error is None:
                first_error = summary.reason
        return BitablePullSyncSummary(
            synced_count=synced_count,
            failed_count=failed_count,
            reason=first_error,
        )

    def _sync_resource(self, resource: OwnerBitableResource) -> BitablePullSyncSummary:
        synced_count = 0
        failed_count = 0
        page_token: Optional[str] = None
        while True:
            try:
                page = self.bitable_service.list_records(
                    app_token=resource.app_token,
                    table_id=resource.table_id,
                    page_size=self.page_size,
                    page_token=page_token,
                )
            except (FeishuConfigurationError, FeishuRequestError) as exc:
                logger.warning("Feishu bitable record list failed: %s", exc)
                return BitablePullSyncSummary(
                    synced_count,
                    failed_count + 1,
                    skipped=False,
                    reason=str(exc),
                )

            fallback_owner_id = get_bitable_table_owner(
                self.repository,
                resource.app_token,
                resource.table_id,
            )

            for record in page.records:
                result = sync_bitable_record_fields_to_repository(
                    repository=self.repository,
                    record_id=record.record_id or "",
                    fields=record.fields or {},
                    app_token=resource.app_token,
                    table_id=resource.table_id,
                    fallback_owner_id=fallback_owner_id,
                    allow_default_local_owner=False,
                    bitable_service=self.bitable_service,
                )
                if result.synced:
                    synced_count += 1
                else:
                    failed_count += 1

            if not page.has_more:
                break
            page_token = page.page_token
            if not page_token:
                break

        return BitablePullSyncSummary(
            synced_count=synced_count,
            failed_count=failed_count,
        )

    def _resolve_resources(self) -> list[OwnerBitableResource]:
        return list_owner_bitable_resources(
            self.repository,
            legacy_app_token=getattr(self.bitable_service, "app_token", None),
            legacy_table_id=getattr(self.bitable_service, "table_id", None),
        )

    # 解析并确定 app token。
    def _resolve_app_token(self) -> Optional[str]:
        return (
            self.repository.get_runtime_setting(_OFFERPILOT_BITABLE_APP_TOKEN_SETTING)
            or getattr(self.bitable_service, "app_token", None)
        )

    # 解析并确定 table id。
    def _resolve_table_id(self) -> Optional[str]:
        return (
            self.repository.get_runtime_setting(_OFFERPILOT_BITABLE_TABLE_ID_SETTING)
            or getattr(self.bitable_service, "table_id", None)
        )

