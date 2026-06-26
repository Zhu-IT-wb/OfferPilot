from fastapi import FastAPI

from app.api.routes.agent import router as agent_router
from app.api.routes.debug import router as debug_router
from app.api.routes.feishu import router as feishu_router
from app.api.routes.health import router as health_router
from app.core.config import Settings, settings


def create_app(app_settings: Settings = settings) -> FastAPI:
    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        description="OfferPilot backend service for job search preparation workflows.",
    )
    app.include_router(agent_router, prefix=app_settings.api_prefix, tags=["agent"])
    app.include_router(feishu_router, prefix=app_settings.api_prefix, tags=["feishu"])
    if app_settings.debug_routes_enabled:
        app.include_router(debug_router, prefix=app_settings.api_prefix, tags=["debug"])
    app.include_router(health_router, prefix=app_settings.api_prefix, tags=["health"])
    return app


app = create_app()
