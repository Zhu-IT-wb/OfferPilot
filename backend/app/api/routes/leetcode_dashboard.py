import asyncio
from dataclasses import replace
from datetime import datetime
import hmac
from pathlib import Path
import secrets
from typing import Dict, Optional
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from app.models.leetcode import LeetCodePracticeResult
from app.repositories.leetcode_repository import LeetCodeRepository
from app.services.leetcode_card_service import LeetCodeCardService
from app.services.leetcode_dashboard_auth import (
    DashboardTokenError,
    DashboardTokenSigner,
    FEISHU_OAUTH_STATE_PURPOSE,
    FeishuDashboardAuthError,
    FeishuDashboardAuthService,
    STUDY_CALENDAR_OWNER_TOKEN_PURPOSE,
)
from app.services.feishu_user_credentials import (
    FeishuUserCredentialRepository,
    SQLiteFeishuUserCredentialRepository,
)
from app.services.dashboard_session import (
    dashboard_open_id,
    require_dashboard_open_id,
)
from app.services.leetcode_recommendation import LeetCodeRecommendationWorkflow
from app.tools.offerpilot_tools import get_default_leetcode_repository


router = APIRouter(prefix="/leetcode/dashboard")
api_router = APIRouter(prefix="/leetcode/dashboard")
_DASHBOARD_HTML_PATH = Path(__file__).resolve().parents[2] / "web" / "leetcode_dashboard.html"
_OAUTH_STATE_COOKIE = "offerpilot_dashboard_oauth_state"
_OAUTH_STATE_COOKIE_PATH = "/leetcode/dashboard/auth/callback"
_OAUTH_STATE_TTL_SECONDS = 10 * 60
_REQUIRED_USER_CALENDAR_SCOPES = (
    "auth:user.id:read",
    "offline_access",
    "calendar:calendar:read",
    "calendar:calendar",
)


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
    calendar_owner_token: Optional[str] = Query(default=None),
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
    oauth_nonce = secrets.token_urlsafe(32)
    state_claims = {
        "redirect": safe_redirect,
        "nonce": oauth_nonce,
    }
    if calendar_owner_token:
        try:
            owner_payload = DashboardTokenSigner(session_secret).verify(
                calendar_owner_token,
                purpose=STUDY_CALENDAR_OWNER_TOKEN_PURPOSE,
            )
            state_claims["calendar_owner_id"] = _calendar_owner_id(owner_payload)
        except DashboardTokenError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(exc),
            ) from exc
    state_token = DashboardTokenSigner(session_secret).issue(
        purpose=FEISHU_OAUTH_STATE_PURPOSE,
        claims=state_claims,
        ttl_seconds=_OAUTH_STATE_TTL_SECONDS,
    )
    query = urlencode(
        {
            "client_id": app_settings.feishu_app_id,
            "redirect_uri": callback_url,
            "state": state_token,
            "scope": _oauth_scope(
                app_settings.feishu_user_calendar_oauth_scope
                if calendar_owner_token
                else app_settings.dashboard_oauth_scope,
                calendar_authorization=bool(calendar_owner_token),
            ),
        }
    )
    response = RedirectResponse(url=f"{app_settings.feishu_authorize_url}?{query}")
    response.set_cookie(
        key=_OAUTH_STATE_COOKIE,
        value=oauth_nonce,
        max_age=_OAUTH_STATE_TTL_SECONDS,
        httponly=True,
        secure=_is_secure_request(request),
        samesite="lax",
        path=_OAUTH_STATE_COOKIE_PATH,
    )
    return response


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
            purpose=FEISHU_OAUTH_STATE_PURPOSE,
        )
        _validate_oauth_browser_nonce(
            state_payload,
            request.cookies.get(_OAUTH_STATE_COOKIE),
        )
        calendar_owner_id = _optional_calendar_owner_id(state_payload)
        auth_service = get_dashboard_auth_service(request)
        if calendar_owner_id:
            exchange_bundle = getattr(auth_service, "exchange_code_bundle", None)
            if not callable(exchange_bundle):
                raise FeishuDashboardAuthError(
                    "Feishu calendar authorization requires a credential token bundle."
                )
            credential = await exchange_bundle(code, _callback_url(request))
            _validate_calendar_owner_identity(calendar_owner_id, credential.open_id)
            credential = replace(credential, owner_id=calendar_owner_id)
            await asyncio.to_thread(
                get_feishu_user_credential_repository(request).save,
                credential,
            )
            open_id = credential.open_id
        else:
            open_id = await auth_service.exchange_code(code, _callback_url(request))
    except (DashboardTokenError, FeishuDashboardAuthError) as exc:
        response = JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": str(exc)},
        )
        _clear_oauth_state_cookie(response, request)
        return response
    redirect_path = _safe_dashboard_redirect(state_payload.get("redirect"))
    session_token = DashboardTokenSigner(session_secret).issue(
        purpose="dashboard_session",
        claims={"open_id": open_id},
        ttl_seconds=app_settings.dashboard_session_ttl_seconds,
    )
    response = RedirectResponse(url=redirect_path)
    _clear_oauth_state_cookie(response, request)
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


def get_feishu_user_credential_repository(
    request: Request,
) -> FeishuUserCredentialRepository:
    app_settings = request.app.state.settings
    encryption_secret = (
        app_settings.feishu_user_credentials_secret
        or app_settings.dashboard_session_secret
        or app_settings.feishu_app_secret
    )
    return SQLiteFeishuUserCredentialRepository(
        app_settings.feishu_user_credentials_path,
        encryption_secret,
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


def _validate_oauth_browser_nonce(
    state_payload: Dict[str, object],
    browser_nonce: Optional[str],
) -> None:
    state_nonce = state_payload.get("nonce")
    expected = state_nonce if isinstance(state_nonce, str) else ""
    received = browser_nonce if isinstance(browser_nonce, str) else ""
    if not expected or not received or not hmac.compare_digest(
        expected.encode("utf-8"),
        received.encode("utf-8"),
    ):
        raise DashboardTokenError(
            "Dashboard authorization was not started in this browser or has expired."
        )


def _clear_oauth_state_cookie(response, request: Request) -> None:
    response.delete_cookie(
        key=_OAUTH_STATE_COOKIE,
        path=_OAUTH_STATE_COOKIE_PATH,
        secure=_is_secure_request(request),
        httponly=True,
        samesite="lax",
    )


def _dashboard_open_id(request: Request) -> Optional[str]:
    return dashboard_open_id(request)


def _require_dashboard_open_id(request: Request) -> str:
    return require_dashboard_open_id(request)


def _safe_dashboard_redirect(value: object) -> str:
    allowed = {"/leetcode/dashboard", "/study/knowledge", "/study/projects"}
    if isinstance(value, str) and value in allowed:
        return value
    if isinstance(value, str) and value.startswith(
        "/study/projects/training?session_id=project_session_"
    ):
        session_id = value.split("?session_id=", 1)[1]
        if session_id.replace("_", "").isalnum() and len(session_id) <= 160:
            return value
    if isinstance(value, str) and value.startswith(
        "/study/projects/discovery?job_id=discovery_"
    ):
        job_id = value.split("?job_id=", 1)[1]
        if job_id.replace("_", "").isalnum() and len(job_id) <= 160:
            return value
    return "/leetcode/dashboard"


def _oauth_scope(configured_scope: str, *, calendar_authorization: bool = False) -> str:
    scopes = str(configured_scope or "").split()
    if calendar_authorization:
        scopes.extend(_REQUIRED_USER_CALENDAR_SCOPES)
    else:
        # Old installations bundled calendar access into the dashboard setting.
        scopes = [
            scope for scope in scopes
            if scope != "offline_access" and not scope.startswith("calendar:")
        ]
        scopes.append("auth:user.id:read")
    return " ".join(dict.fromkeys(scope for scope in scopes if scope))


def _calendar_owner_id(payload: Dict[str, object]) -> str:
    owner_id = payload.get("owner_id")
    if not isinstance(owner_id, str):
        raise DashboardTokenError("Calendar authorization owner is missing.")
    normalized = owner_id.strip()
    if (
        not normalized
        or normalized != owner_id
        or len(normalized) > 256
        or any(character in normalized for character in ("\r", "\n", "\0"))
    ):
        raise DashboardTokenError("Calendar authorization owner is invalid.")
    return normalized


def _optional_calendar_owner_id(payload: Dict[str, object]) -> Optional[str]:
    if "calendar_owner_id" not in payload:
        return None
    return _calendar_owner_id({"owner_id": payload.get("calendar_owner_id")})


def _validate_calendar_owner_identity(owner_id: str, open_id: str) -> None:
    if owner_id.startswith("feishu:") and owner_id != f"feishu:{open_id}":
        raise FeishuDashboardAuthError(
            "The authorized Feishu account does not match the requested calendar owner."
        )
