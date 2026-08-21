import logging

from fastapi import FastAPI

from app.api.routes.agent import router as agent_router
from app.api.routes.debug import router as debug_router
from app.api.routes.feishu import router as feishu_router
from app.api.routes.health import router as health_router
from app.api.routes.leetcode_dashboard import (
    api_router as leetcode_dashboard_api_router,
    router as leetcode_dashboard_router,
)
from app.api.routes.knowledge_dashboard import (
    api_router as knowledge_dashboard_api_router,
    router as knowledge_dashboard_router,
)
from app.api.routes.project_training_dashboard import (
    api_router as project_training_api_router,
    router as project_training_router,
    training_api_router,
)
from app.api.routes.project_discovery_dashboard import (
    router as project_discovery_api_router,
)
from app.core.config import Settings, settings
from app.services.bitable_event_subscription_service import ensure_bitable_event_subscription
from app.services.bitable_pull_sync_service import BitablePullSyncService
from app.services.leetcode_push_service import LeetCodePushService
from app.services.knowledge_dependencies import (
    get_default_knowledge_repository,
    sync_default_knowledge_repository,
)
from app.services.knowledge_corpus import KnowledgeCorpusSyncError
from app.services.knowledge_push_service import KnowledgePushService
from app.services.project_discovery_dependencies import (
    build_project_discovery_services,
)
from app.services.project_training_dependencies import (
    get_default_project_training_repository,
)
from app.mcp.client import close_default_mcp_client
from app.tools.offerpilot_tools import (
    get_default_leetcode_repository,
    get_default_offerpilot_repository,
)

logger = logging.getLogger(__name__)


# 创建飞书多维表格应用。
def create_app(app_settings: Settings = settings) -> FastAPI:
    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        description="OfferPilot backend service for job search preparation workflows.",
    )
    app.state.settings = app_settings
    app.state.project_discovery_services = build_project_discovery_services(
        app_settings, get_default_project_training_repository()
    )
    app.include_router(agent_router, prefix=app_settings.api_prefix, tags=["agent"])
    app.include_router(feishu_router, prefix=app_settings.api_prefix, tags=["feishu"])
    if app_settings.debug_routes_enabled:
        app.include_router(debug_router, prefix=app_settings.api_prefix, tags=["debug"])
    app.include_router(health_router, prefix=app_settings.api_prefix, tags=["health"])
    app.include_router(
        leetcode_dashboard_api_router,
        prefix=app_settings.api_prefix,
        tags=["leetcode-dashboard"],
    )
    app.include_router(leetcode_dashboard_router, tags=["leetcode-dashboard"])
    app.include_router(
        knowledge_dashboard_api_router,
        prefix=app_settings.api_prefix,
        tags=["knowledge-dashboard"],
    )
    app.include_router(knowledge_dashboard_router, tags=["knowledge-dashboard"])
    app.include_router(
        project_training_api_router,
        prefix=app_settings.api_prefix,
        tags=["project-training"],
    )
    app.include_router(project_training_router, tags=["project-training"])
    app.include_router(
        training_api_router,
        prefix=app_settings.api_prefix,
        tags=["project-training"],
    )
    app.include_router(
        project_discovery_api_router,
        prefix=app_settings.api_prefix,
        tags=["project-discovery"],
    )

    # 在应用启动时启动多维表格定时拉取同步任务。
    @app.on_event("startup")
    async def start_bitable_pull_sync() -> None:
        try:
            sync_default_knowledge_repository(app_settings)
        except KnowledgeCorpusSyncError as exc:
            logger.warning(
                "Knowledge Markdown startup sync was rejected; retaining the "
                "last-known-good corpus: %s",
                exc,
            )
        repository = get_default_offerpilot_repository()
        event_subscription = ensure_bitable_event_subscription(repository=repository, force=True)
        if event_subscription.subscribed:
            logger.info(
                "Feishu bitable event subscription ready: status=%s app_token_present=%s",
                event_subscription.status,
                bool(event_subscription.app_token),
            )
        else:
            logger.warning(
                "Feishu bitable event subscription unavailable: status=%s app_token_present=%s error=%s",
                event_subscription.status,
                bool(event_subscription.app_token),
                event_subscription.error,
            )
        service = BitablePullSyncService(
            repository=repository,
        )
        app.state.bitable_pull_sync_service = service
        service.start()

        leetcode_push_service = LeetCodePushService(
            repository=get_default_leetcode_repository(),
            dashboard_url=(
                f"{app_settings.dashboard_public_base_url.rstrip('/')}/leetcode/dashboard"
                if app_settings.dashboard_public_base_url
                else None
            ),
        )
        app.state.leetcode_push_service = leetcode_push_service
        leetcode_push_service.start()

        knowledge_push_service = KnowledgePushService(
            repository=get_default_knowledge_repository(),
            dashboard_url=(
                f"{app_settings.dashboard_public_base_url.rstrip('/')}/study/knowledge"
                if app_settings.dashboard_public_base_url
                else None
            ),
        )
        app.state.knowledge_push_service = knowledge_push_service
        knowledge_push_service.start()

        if app_settings.project_discovery_enabled:
            app.state.project_discovery_services.runner.start()

    # 在应用关闭时停止多维表格定时拉取同步任务。
    @app.on_event("shutdown")
    async def stop_bitable_pull_sync() -> None:
        await close_default_mcp_client()
        service = getattr(app.state, "bitable_pull_sync_service", None)
        if service is not None:
            await service.stop()
        leetcode_push_service = getattr(app.state, "leetcode_push_service", None)
        if leetcode_push_service is not None:
            await leetcode_push_service.stop()
        knowledge_push_service = getattr(app.state, "knowledge_push_service", None)
        if knowledge_push_service is not None:
            await knowledge_push_service.stop()
        discovery_services = getattr(app.state, "project_discovery_services", None)
        if discovery_services is not None:
            await discovery_services.runner.stop()

    return app


app = create_app()
