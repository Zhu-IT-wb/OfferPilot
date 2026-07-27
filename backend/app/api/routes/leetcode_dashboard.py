from datetime import datetime
from pathlib import Path
import secrets
from typing import Dict, Optional
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from app.models.leetcode import LeetCodePracticeResult
from app.repositories.leetcode_repository import LeetCodeRepository
from app.services.leetcode_card_service import LeetCodeCardService
from app.services.leetcode_dashboard_auth import (
    DashboardTokenError,
    DashboardTokenSigner,
    FeishuDashboardAuthError,
    FeishuDashboardAuthService,
)
from app.services.leetcode_recommendation import LeetCodeRecommendationWorkflow
from app.tools.offerpilot_tools import get_default_leetcode_repository


router = APIRouter(prefix="/leetcode/dashboard")
api_router = APIRouter(prefix="/leetcode/dashboard")
_DASHBOARD_HTML_PATH = Path(__file__).resolve().parents[2] / "web" / "leetcode_dashboard.html"


class LeetCodeDashboardResultRequest(BaseModel):
    assignment_id: str
    result: LeetCodePracticeResult


@router.get("")
async def show_leetcode_dashboard(request: Request):
    if _dashboard_open_id(request) is None:
        return RedirectResponse(url="/leetcode/dashboard/auth/start")
    return HTMLResponse(_DASHBOARD_HTML_PATH.read_text(encoding="utf-8"))


@api_router.get("/today")
async def get_leetcode_dashboard_today(request: Request):
    open_id = _require_dashboard_open_id(request)
    owner_id = f"feishu:{open_id}"
    repository = get_default_leetcode_repository()
    return _build_dashboard_payload(repository=repository, owner_id=owner_id)


@api_router.post("/results")
async def record_leetcode_dashboard_result(
    request: Request,
    payload: LeetCodeDashboardResultRequest,
):
    open_id = _require_dashboard_open_id(request)
    owner_id = f"feishu:{open_id}"
    repository = get_default_leetcode_repository()
    tool_result = LeetCodeCardService(repository).record_result(
        owner_id=owner_id,
        assignment_id=payload.assignment_id,
        result=payload.result.value,
    )
    if not tool_result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=tool_result.message,
        )
    return {
        "message": tool_result.message,
        "dashboard": _build_dashboard_payload(
            repository=repository,
            owner_id=owner_id,
        ),
    }


def _build_dashboard_payload(
    repository: LeetCodeRepository,
    owner_id: str,
) -> Dict[str, object]:
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    recommendations = LeetCodeRecommendationWorkflow(repository).get_today(
        owner_id=owner_id,
        today=today,
    )
    items = []
    for recommendation in recommendations:
        assignment = recommendation.assignment
        problem = recommendation.problem
        progress = repository.get_progress(owner_id, problem.id)
        items.append(
            {
                "assignment_id": assignment.id,
                "assignment_type": assignment.assignment_type.value,
                "recommendation_reason": assignment.recommendation_reason,
                "status": assignment.status.value,
                "result": assignment.result.value if assignment.result else None,
                "postpone_count": assignment.postpone_count,
                "problem": {
                    "frontend_id": problem.frontend_id,
                    "title": problem.title_zh,
                    "difficulty": problem.difficulty.value,
                    "category": problem.category,
                    "url": problem.url,
                },
                "progress": (
                    {
                        "mastery_status": progress.mastery_status.value,
                        "independent_streak": progress.independent_streak,
                        "next_review_on": (
                            progress.next_review_on.isoformat()
                            if progress.next_review_on
                            else None
                        ),
                    }
                    if progress
                    else None
                ),
            }
        )
    return {
        "date": today.isoformat(),
        "summary": {
            "handled": sum(1 for item in recommendations if item.assignment.result),
            "total": len(recommendations),
        },
        "items": items,
    }


@router.get("/auth/start")
async def start_leetcode_dashboard_auth(
    request: Request,
    redirect: str = Query(default="/leetcode/dashboard"),
):
    app_settings = request.app.state.settings
    session_secret = app_settings.dashboard_session_secret or app_settings.feishu_app_secret
    if not app_settings.feishu_app_id or not session_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feishu dashboard authentication is not configured.",
        )
    callback_url = _callback_url(request)
    safe_redirect = _safe_dashboard_redirect(redirect)
    state_token = DashboardTokenSigner(session_secret).issue(
        purpose="feishu_oauth_state",
        claims={
            "redirect": safe_redirect,
            "nonce": secrets.token_urlsafe(16),
        },
        ttl_seconds=10 * 60,
    )
    query = urlencode(
        {
            "client_id": app_settings.feishu_app_id,
            "redirect_uri": callback_url,
            "state": state_token,
            "scope": app_settings.dashboard_oauth_scope,
        }
    )
    return RedirectResponse(url=f"{app_settings.feishu_authorize_url}?{query}")


@router.get("/auth/callback", name="finish_leetcode_dashboard_auth")
async def finish_leetcode_dashboard_auth(
    request: Request,
    code: str = Query(...),
    state: str = Query(...),
):
    app_settings = request.app.state.settings
    session_secret = app_settings.dashboard_session_secret or app_settings.feishu_app_secret
    try:
        state_payload = DashboardTokenSigner(session_secret).verify(
            state,
            purpose="feishu_oauth_state",
        )
        open_id = await get_dashboard_auth_service(request).exchange_code(
            code,
            _callback_url(request),
        )
    except (DashboardTokenError, FeishuDashboardAuthError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc
    redirect_path = _safe_dashboard_redirect(state_payload.get("redirect"))
    session_token = DashboardTokenSigner(session_secret).issue(
        purpose="dashboard_session",
        claims={"open_id": open_id},
        ttl_seconds=app_settings.dashboard_session_ttl_seconds,
    )
    response = RedirectResponse(url=redirect_path)
    response.set_cookie(
        key="offerpilot_dashboard_session",
        value=session_token,
        max_age=app_settings.dashboard_session_ttl_seconds,
        httponly=True,
        secure=_is_secure_request(request),
        samesite="lax",
        path="/",
    )
    return response


def get_dashboard_auth_service(request: Request) -> FeishuDashboardAuthService:
    app_settings = request.app.state.settings
    return FeishuDashboardAuthService(
        app_id=app_settings.feishu_app_id,
        app_secret=app_settings.feishu_app_secret,
        api_base_url=app_settings.feishu_api_base_url,
        timeout_seconds=app_settings.feishu_timeout_seconds,
    )


def _callback_url(request: Request) -> str:
    app_settings = request.app.state.settings
    if app_settings.dashboard_public_base_url:
        return (
            app_settings.dashboard_public_base_url.rstrip("/")
            + "/leetcode/dashboard/auth/callback"
        )
    return str(request.base_url).rstrip("/") + "/leetcode/dashboard/auth/callback"


def _is_secure_request(request: Request) -> bool:
    app_settings = request.app.state.settings
    if app_settings.dashboard_public_base_url:
        return app_settings.dashboard_public_base_url.startswith("https://")
    return request.url.scheme == "https"


def _dashboard_open_id(request: Request) -> Optional[str]:
    token = request.cookies.get("offerpilot_dashboard_session")
    if not token:
        return None
    app_settings = request.app.state.settings
    session_secret = app_settings.dashboard_session_secret or app_settings.feishu_app_secret
    try:
        payload = DashboardTokenSigner(session_secret).verify(
            token,
            purpose="dashboard_session",
        )
    except DashboardTokenError:
        return None
    open_id = payload.get("open_id")
    return open_id if isinstance(open_id, str) and open_id else None


def _require_dashboard_open_id(request: Request) -> str:
    open_id = _dashboard_open_id(request)
    if open_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Dashboard session is missing or expired.",
        )
    return open_id


def _safe_dashboard_redirect(value: object) -> str:
    allowed = {"/leetcode/dashboard", "/study/knowledge"}
    return value if isinstance(value, str) and value in allowed else "/leetcode/dashboard"
