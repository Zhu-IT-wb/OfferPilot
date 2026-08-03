from typing import Optional

from fastapi import HTTPException, Request, status

from app.services.leetcode_dashboard_auth import DashboardTokenError, DashboardTokenSigner


def dashboard_open_id(request: Request) -> Optional[str]:
    token = request.cookies.get("offerpilot_dashboard_session")
    if not token:
        return None
    app_settings = request.app.state.settings
    secret = app_settings.dashboard_session_secret or app_settings.feishu_app_secret
    try:
        payload = DashboardTokenSigner(secret).verify(
            token,
            purpose="dashboard_session",
        )
    except DashboardTokenError:
        return None
    open_id = payload.get("open_id")
    return open_id if isinstance(open_id, str) and open_id else None


def require_dashboard_open_id(request: Request) -> str:
    open_id = dashboard_open_id(request)
    if open_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Dashboard session is missing or expired.",
        )
    return open_id
