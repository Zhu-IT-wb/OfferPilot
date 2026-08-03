from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from app.services.dashboard_session import (
    dashboard_open_id,
    require_dashboard_open_id,
)
from app.repositories.project_training_repository import ProjectTrainingRepository
from app.agents.interview_agent import InterviewAgentError
from app.models.project_training import (
    ProjectTrainingAnswerSource,
    ProjectTrainingDifficulty,
)
from app.repositories.project_training_repository import (
    ProjectTrainingSubmissionInProgressError,
)
from app.services.project_training_dependencies import (
    get_default_project_training_repository,
)
from app.services.speech_http import transcribe_audio_request
from app.services.speech_transcription import create_speech_transcription_service
from app.services.feishu_service import (
    FeishuConfigurationError,
    FeishuMessageService,
    FeishuRequestError,
)


router = APIRouter(prefix="/study/projects")
api_router = APIRouter(prefix="/study/projects")
training_api_router = APIRouter(prefix="/study/project-training")
_HTML_PATH = Path(__file__).resolve().parents[2] / "web" / "project_profiles.html"
_TRAINING_HTML_PATH = Path(__file__).resolve().parents[2] / "web" / "project_training.html"
_WEB_ROOT = _HTML_PATH.parent
_ASSETS = {
    "project_training.css",
    "project_profiles.js",
    "project_training.js",
    "voice_recorder.js",
    "web_common.js",
}


class ProjectProfileRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    target_role: str = Field(default="", max_length=120)
    background: str = Field(default="", max_length=3000)
    responsibilities: List[str] = Field(default_factory=list, max_length=30)
    tech_stack: List[str] = Field(default_factory=list, max_length=50)
    architecture: str = Field(default="", max_length=5000)
    key_decisions: List[str] = Field(default_factory=list, max_length=30)
    technical_challenges: List[str] = Field(default_factory=list, max_length=30)
    metrics: List[str] = Field(default_factory=list, max_length=30)
    outcomes: List[str] = Field(default_factory=list, max_length=30)
    resume_description: str = Field(default="", max_length=3000)
    supplemental_text: str = Field(default="", max_length=20000)


class ProjectTrainingSessionRequest(BaseModel):
    project_id: str = Field(..., min_length=1, max_length=160)
    creation_id: str = Field(..., min_length=8, max_length=160)
    target_role: str = Field(default="", max_length=120)
    goal: str = Field(default="", max_length=500)
    difficulty: ProjectTrainingDifficulty = ProjectTrainingDifficulty.MEDIUM
    max_turns: int = Field(default=6, ge=3, le=10)
    focus_topics: List[str] = Field(default_factory=list, max_length=5)


class ProjectTrainingAnswerRequest(BaseModel):
    turn_id: str = Field(..., min_length=1, max_length=160)
    submission_id: str = Field(..., min_length=8, max_length=160)
    answer_text: str = Field(..., min_length=1, max_length=8000)
    answer_source: ProjectTrainingAnswerSource = ProjectTrainingAnswerSource.TEXT


def get_project_training_workflow():
    from app.services.project_training import ProjectTrainingWorkflow

    return ProjectTrainingWorkflow(get_default_project_training_repository())


@router.get("")
async def show_projects(request: Request):
    if dashboard_open_id(request) is None:
        return RedirectResponse(
            url="/leetcode/dashboard/auth/start?redirect=/study/projects"
        )
    return HTMLResponse(_HTML_PATH.read_text(encoding="utf-8"))


@router.get("/training")
async def show_project_training(request: Request):
    if dashboard_open_id(request) is None:
        redirect = request.url.path
        if request.url.query:
            redirect += f"?{request.url.query}"
        return RedirectResponse(
            url=f"/leetcode/dashboard/auth/start?redirect={redirect}"
        )
    return HTMLResponse(_TRAINING_HTML_PATH.read_text(encoding="utf-8"))


@router.get("/assets/{asset_name}")
async def project_training_asset(asset_name: str):
    if asset_name not in _ASSETS:
        raise HTTPException(status_code=404, detail="Asset was not found.")
    return FileResponse(_WEB_ROOT / asset_name)


@api_router.get("")
async def list_projects(request: Request) -> Dict[str, Any]:
    owner_id = _owner_id(request)
    projects = get_default_project_training_repository().list_projects(
        owner_id, include_archived=True
    )
    return {"projects": [item.to_dict() for item in projects]}


@api_router.post("", status_code=status.HTTP_201_CREATED)
async def create_project(
    request: Request, payload: ProjectProfileRequest
) -> Dict[str, Any]:
    project = get_default_project_training_repository().create_project(
        _owner_id(request), _model_dict(payload)
    )
    return {"project": project.to_dict()}


@api_router.get("/{project_id}")
async def get_project(request: Request, project_id: str) -> Dict[str, Any]:
    owner_id = _owner_id(request)
    repository = get_default_project_training_repository()
    project = repository.get_project(owner_id, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project was not found.")
    return {
        "project": project.to_dict(),
        "versions": repository.list_project_versions(owner_id, project_id),
    }


@api_router.put("/{project_id}")
async def update_project(
    request: Request, project_id: str, payload: ProjectProfileRequest
) -> Dict[str, Any]:
    project = get_default_project_training_repository().update_project(
        _owner_id(request), project_id, _model_dict(payload)
    )
    if project is None:
        raise HTTPException(status_code=404, detail="Project was not found.")
    return {"project": project.to_dict()}


@api_router.post("/{project_id}/archive")
async def archive_project(request: Request, project_id: str) -> Dict[str, Any]:
    project = get_default_project_training_repository().archive_project(
        _owner_id(request), project_id
    )
    if project is None:
        raise HTTPException(status_code=404, detail="Project was not found.")
    return {"project": project.to_dict()}


@training_api_router.post("/sessions", status_code=status.HTTP_201_CREATED)
async def create_training_session(
    request: Request, response: Response, payload: ProjectTrainingSessionRequest
) -> Dict[str, Any]:
    owner_id = _owner_id(request)
    repository = get_default_project_training_repository()
    if repository.get_project(owner_id, payload.project_id) is None:
        raise HTTPException(status_code=404, detail="Project was not found.")
    try:
        outcome = get_project_training_workflow().start(
            owner_id=owner_id,
            project_id=payload.project_id,
            creation_id=payload.creation_id,
            target_role=payload.target_role,
            goal=payload.goal,
            difficulty=payload.difficulty,
            max_turns=payload.max_turns,
            focus_topics=payload.focus_topics,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    body = _training_outcome_payload(outcome, owner_id)
    if outcome.resumed:
        response.status_code = status.HTTP_200_OK
    return body


@training_api_router.get("/sessions/{session_id}")
async def get_training_session(request: Request, session_id: str) -> Dict[str, Any]:
    return _resume_training(request, session_id)


@training_api_router.get("/sessions/{session_id}/current")
async def get_current_training_turn(
    request: Request, session_id: str
) -> Dict[str, Any]:
    return _resume_training(request, session_id)


@training_api_router.get("/sessions")
async def list_training_sessions(request: Request) -> Dict[str, Any]:
    sessions = get_default_project_training_repository().list_sessions(
        _owner_id(request)
    )
    return {"sessions": [item.to_dict() for item in sessions]}


@training_api_router.post("/transcriptions")
async def transcribe_project_answer(
    request: Request, turn_id: str
) -> Dict[str, str]:
    owner_id = _owner_id(request)
    repository = get_default_project_training_repository()
    turn = repository.get_turn(owner_id, turn_id)
    if turn is None:
        raise HTTPException(status_code=404, detail="项目训练题目不存在。")
    session = repository.get_session(owner_id, turn.session_id)
    if session is None or session.current_turn_id != turn.id:
        raise HTTPException(status_code=400, detail="只能转写当前项目训练题目。")
    project = repository.get_project(
        owner_id, session.project_id, version=session.project_version
    )
    if project is None:
        raise HTTPException(status_code=404, detail="项目历史版本不存在。")
    context = "；".join(
        [
            f"项目：{project.name}",
            f"岗位：{session.target_role}",
            f"题目：{turn.question_text}",
            "技术术语：" + "、".join(project.tech_stack[:20]),
        ]
    )[:300]
    result = await transcribe_audio_request(
        request,
        service=get_speech_transcription_service(request),
        context=context,
    )
    return {
        "transcript": result.transcript,
        "provider": result.provider,
        "model": result.model,
    }


def get_speech_transcription_service(request: Request):
    return create_speech_transcription_service(request.app.state.settings)


@training_api_router.get("/jsapi-config")
async def get_project_jsapi_config(request: Request, url: str) -> Dict[str, Any]:
    _owner_id(request)
    page_url = url.split("#", 1)[0]
    parsed = urlsplit(page_url)
    if parsed.scheme not in {"http", "https"} or parsed.path != "/study/projects/training":
        raise HTTPException(status_code=400, detail="只能为项目训练页面申请录音权限。")
    settings = request.app.state.settings
    if not settings.dashboard_public_base_url:
        raise HTTPException(status_code=503, detail="尚未配置 OfferPilot 网页应用公网地址。")
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin != settings.dashboard_public_base_url.rstrip("/"):
        raise HTTPException(status_code=400, detail="网页地址与 OfferPilot 配置不匹配。")
    try:
        config = await _feishu_jssdk_service(request).get_jssdk_config(page_url)
    except (FeishuConfigurationError, FeishuRequestError) as exc:
        raise HTTPException(
            status_code=503,
            detail="飞书移动录音暂时不可用，请使用浏览器录音或文字回答。",
        ) from exc
    return {
        "app_id": config.app_id,
        "timestamp": config.timestamp,
        "nonce_str": config.nonce_str,
        "signature": config.signature,
    }


def _feishu_jssdk_service(request: Request) -> FeishuMessageService:
    existing = getattr(request.app.state, "project_training_feishu_jssdk_service", None)
    if existing is not None:
        return existing
    settings = request.app.state.settings
    service = FeishuMessageService(
        app_id=settings.feishu_app_id,
        app_secret=settings.feishu_app_secret,
        base_url=settings.feishu_api_base_url,
        timeout_seconds=settings.feishu_timeout_seconds,
    )
    request.app.state.project_training_feishu_jssdk_service = service
    return service


@training_api_router.post("/sessions/{session_id}/answers")
async def submit_training_answer(
    request: Request, session_id: str, payload: ProjectTrainingAnswerRequest
) -> Dict[str, Any]:
    owner_id = _owner_id(request)
    repository = get_default_project_training_repository()
    session = repository.get_session(owner_id, session_id)
    turn = repository.get_turn(owner_id, payload.turn_id)
    if session is None or turn is None or turn.session_id != session_id:
        raise HTTPException(status_code=404, detail="Project training was not found.")
    try:
        outcome = await get_project_training_workflow().submit_answer(
            owner_id=owner_id,
            session_id=session_id,
            turn_id=payload.turn_id,
            answer_text=payload.answer_text,
            answer_source=payload.answer_source,
            submission_id=payload.submission_id,
            submitted_at=_now(),
        )
    except ProjectTrainingSubmissionInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InterviewAgentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    response_payload = {
        "session": outcome.session.to_dict(),
        "completed_turn": outcome.completed_turn.to_dict(),
        "answer": outcome.answer.to_dict(),
        "evaluation": outcome.evaluation,
        "current_turn": (
            outcome.current_turn.to_dict() if outcome.current_turn is not None else None
        ),
        "progress": outcome.progress.to_dict(),
    }
    response_payload.update(
        _session_display_context(
            owner_id, outcome.session, outcome.current_turn
        )
    )
    return response_payload


@training_api_router.post("/sessions/{session_id}/finish")
async def finish_training(request: Request, session_id: str) -> Dict[str, Any]:
    owner_id = _owner_id(request)
    if get_default_project_training_repository().get_session(owner_id, session_id) is None:
        raise HTTPException(status_code=404, detail="Project training was not found.")
    try:
        summary = get_project_training_workflow().finish(owner_id, session_id)
    except ProjectTrainingSubmissionInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"summary": summary.to_dict()}


@training_api_router.post("/sessions/{session_id}/abandon")
async def abandon_training(request: Request, session_id: str) -> Dict[str, Any]:
    owner_id = _owner_id(request)
    if get_default_project_training_repository().get_session(owner_id, session_id) is None:
        raise HTTPException(status_code=404, detail="Project training was not found.")
    try:
        session = get_project_training_workflow().abandon(
            owner_id, session_id
        )
    except ProjectTrainingSubmissionInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"session": session.to_dict()}


@training_api_router.get("/sessions/{session_id}/summary")
async def get_training_summary(request: Request, session_id: str) -> Dict[str, Any]:
    try:
        summary = get_project_training_workflow().summary(_owner_id(request), session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"summary": summary.to_dict()}


def _resume_training(request: Request, session_id: str) -> Dict[str, Any]:
    owner_id = _owner_id(request)
    try:
        outcome = get_project_training_workflow().resume(owner_id, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _training_outcome_payload(outcome, owner_id)


def _training_outcome_payload(outcome, owner_id: str) -> Dict[str, Any]:
    payload = {
        "session": outcome.session.to_dict(),
        "current_turn": (
            outcome.current_turn.to_dict() if outcome.current_turn is not None else None
        ),
        "resumed": outcome.resumed,
    }
    payload.update(
        _session_display_context(owner_id, outcome.session, outcome.current_turn)
    )
    return payload


def _session_display_context(owner_id, session, current_turn) -> Dict[str, Any]:
    project = get_default_project_training_repository().get_project(
        owner_id, session.project_id, version=session.project_version
    )
    remaining_turns = (
        max(0, session.max_turns - current_turn.sequence + 1)
        if current_turn is not None
        else 0
    )
    return {
        "project": project.to_dict() if project is not None else None,
        "remaining_turns": remaining_turns,
    }


def _owner_id(request: Request) -> str:
    return f"feishu:{require_dashboard_open_id(request)}"


def _model_dict(model: BaseModel) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _now():
    from datetime import datetime

    return datetime.now().astimezone()
