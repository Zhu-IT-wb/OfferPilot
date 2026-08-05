from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.models.project_discovery import ProjectDiscoveryStatus
from app.services.dashboard_session import require_dashboard_open_id
from app.services.project_discovery_runner import ProjectDiscoveryQueueFull


router = APIRouter(prefix="/study/project-discovery")


class StartDiscoveryRequest(BaseModel):
    repository_url: str = Field(..., min_length=1, max_length=500)
    request_id: str = Field(..., min_length=8, max_length=160)
    project_id: Optional[str] = Field(default=None, max_length=160)


class AnswerDiscoveryRequest(BaseModel):
    question_id: str = Field(..., min_length=1, max_length=160)
    answer_text: str = Field(..., min_length=1, max_length=5000)
    submission_id: str = Field(..., min_length=8, max_length=160)


class ConfirmDiscoveryRequest(BaseModel):
    confirmation_id: str = Field(..., min_length=8, max_length=160)


def _owner_id(request):
    return f"feishu:{require_dashboard_open_id(request)}"


def _workflow(request):
    return request.app.state.project_discovery_services.workflow


def _runner(request):
    return request.app.state.project_discovery_services.runner


def _enabled(request):
    if not request.app.state.settings.project_discovery_enabled:
        raise HTTPException(status_code=503, detail="项目代码分析功能暂未启用。")


def _payload(workflow, job):
    evidence = workflow.repository.list_evidence(job.owner_id, job.id)
    return {"job": job.to_dict(evidence=evidence)}


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def start_discovery(request: Request, payload: StartDiscoveryRequest):
    _enabled(request)
    workflow = _workflow(request)
    try:
        job = workflow.start(
            _owner_id(request), payload.repository_url, payload.request_id,
            payload.project_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if job.status == ProjectDiscoveryStatus.QUEUED:
        try:
            _runner(request).enqueue(job.id)
        except ProjectDiscoveryQueueFull as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
    return _payload(workflow, job)


@router.get("/jobs")
async def list_discovery_jobs(request: Request):
    _enabled(request)
    workflow = _workflow(request)
    return {"jobs": [item.to_dict() for item in workflow.repository.list(_owner_id(request))]}


@router.get("/jobs/{job_id}")
async def get_discovery_job(request: Request, job_id: str):
    _enabled(request)
    workflow = _workflow(request)
    job = workflow.resume(_owner_id(request), job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="项目分析任务不存在。")
    return _payload(workflow, job)


@router.post("/jobs/{job_id}/answers")
async def answer_discovery_job(request: Request, job_id: str,
                               payload: AnswerDiscoveryRequest):
    _enabled(request)
    workflow = _workflow(request)
    try:
        job = workflow.answer(
            _owner_id(request), job_id, payload.question_id,
            payload.answer_text, payload.submission_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="项目分析任务不存在。")
    return _payload(workflow, job)


@router.post("/jobs/{job_id}/confirm")
async def confirm_discovery_job(request: Request, job_id: str,
                                payload: ConfirmDiscoveryRequest):
    _enabled(request)
    workflow = _workflow(request)
    try:
        project = workflow.confirm(
            _owner_id(request), job_id, payload.confirmation_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if project is None:
        raise HTTPException(status_code=404, detail="项目分析任务不存在。")
    return {"project": project.to_dict()}


@router.post("/jobs/{job_id}/retry", status_code=status.HTTP_202_ACCEPTED)
async def retry_discovery_job(request: Request, job_id: str):
    _enabled(request)
    workflow = _workflow(request)
    try:
        job = workflow.retry(_owner_id(request), job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="项目分析任务不存在。")
    try:
        _runner(request).enqueue(job.id)
    except ProjectDiscoveryQueueFull as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return _payload(workflow, job)


@router.post("/jobs/{job_id}/cancel")
async def cancel_discovery_job(request: Request, job_id: str):
    _enabled(request)
    workflow = _workflow(request)
    job = workflow.cancel(_owner_id(request), job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="项目分析任务不存在。")
    _runner(request).cancel(job.id)
    return _payload(workflow, job)
