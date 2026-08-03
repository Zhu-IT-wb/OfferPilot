import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from app.services.dashboard_session import (
    dashboard_open_id,
    require_dashboard_open_id,
)
from app.models.interview_knowledge import (
    KnowledgeAnswerSource,
    KnowledgeAssignmentType,
    KnowledgeMasteryStatus,
    KnowledgeSubscription,
)
from app.services.knowledge_dependencies import get_default_knowledge_repository
from app.services.knowledge_evaluation import (
    KnowledgeEvaluationError,
    KnowledgeEvaluationService,
)
from app.services.feishu_service import (
    FeishuConfigurationError,
    FeishuMessageService,
    FeishuRequestError,
)
from app.services.knowledge_practice import (
    KnowledgePracticeWorkflow,
    KnowledgeSubmissionInProgressError,
)
from app.services.knowledge_recommendation import (
    KnowledgeRecommendation,
    KnowledgeRecommendationWorkflow,
)
from app.services.speech_transcription import (
    NoSpeechDetectedError,
    SpeechTranscriptionConfigurationError,
    SpeechTranscriptionError,
    create_speech_transcription_service,
)
from app.services.speech_http import (
    AUDIO_FORMATS,
    MAX_AUDIO_BYTES,
    MAX_AUDIO_DURATION_MS,
    audio_duration_ms,
    read_limited_audio,
    transcribe_audio_request,
)


router = APIRouter(prefix="/study/knowledge")
api_router = APIRouter(prefix="/study/knowledge")
_HTML_PATH = Path(__file__).resolve().parents[2] / "web" / "knowledge_dashboard.html"
_MAX_AUDIO_BYTES = MAX_AUDIO_BYTES
_MAX_AUDIO_DURATION_MS = MAX_AUDIO_DURATION_MS
_AUDIO_FORMATS = AUDIO_FORMATS


class KnowledgeAnswerRequest(BaseModel):
    assignment_id: str = Field(..., min_length=1, max_length=160)
    submission_id: str = Field(..., min_length=8, max_length=160)
    answer_text: str = Field(default="", max_length=6000)
    answer_source: KnowledgeAnswerSource = KnowledgeAnswerSource.TEXT


class KnowledgePracticeRequest(BaseModel):
    question_id: str = Field(default="", max_length=160)
    module_id: str = Field(default="", max_length=100)
    chapter_id: str = Field(default="", max_length=100)
    weak_only: bool = False


@router.get("")
async def show_knowledge_dashboard(request: Request):
    if dashboard_open_id(request) is None:
        return RedirectResponse(
            url="/leetcode/dashboard/auth/start?redirect=/study/knowledge"
        )
    return HTMLResponse(_HTML_PATH.read_text(encoding="utf-8"))


@api_router.get("/today")
async def get_knowledge_today(request: Request) -> Dict[str, Any]:
    open_id = require_dashboard_open_id(request)
    owner_id = f"feishu:{open_id}"
    repository = get_default_knowledge_repository()
    repository.save_subscription(
        KnowledgeSubscription(owner_id=owner_id, feishu_open_id=open_id)
    )
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    recommendations = KnowledgeRecommendationWorkflow(repository).get_today(
        owner_id, today
    )
    return _dashboard_payload(repository, owner_id, recommendations, today.isoformat())


@api_router.get("/materials")
async def get_knowledge_materials(
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    module_id: str = Query(default="", max_length=100),
    chapter_id: str = Query(default="", max_length=100),
) -> Dict[str, Any]:
    require_dashboard_open_id(request)
    repository = get_default_knowledge_repository()
    questions = [
        question
        for question in repository.list_questions()
        if (not module_id or question.module_id == module_id)
        and (not chapter_id or question.chapter_id == chapter_id)
    ]
    page = questions[offset : offset + limit]
    return {
        "total": len(questions),
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(page) < len(questions),
        "questions": [
            {
                "id": question.id,
                "module_id": question.module_id,
                "module_title": question.module_title,
                "chapter_id": question.chapter_id,
                "chapter_title": question.chapter_title,
                "prompt": question.prompt,
                "short_reference_answer": question.short_reference_answer,
                "full_reference_answer": question.full_reference_answer,
                "source": {
                    "title": question.source_title,
                    "url": question.source_url,
                    "chapter": question.source_chapter,
                    "page_start": question.source_page_start,
                    "page_end": question.source_page_end,
                    "file": question.source_file,
                    "heading": question.source_heading,
                },
            }
            for question in page
        ]
    }


@api_router.get("/catalog/questions")
async def get_knowledge_catalog_questions(
    request: Request,
    chapter_id: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(default=100, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    owner_id = f"feishu:{require_dashboard_open_id(request)}"
    repository = get_default_knowledge_repository()
    questions = [
        question
        for question in repository.list_questions()
        if question.chapter_id == chapter_id
    ]
    progress_by_id = {
        item.question_id: item for item in repository.list_progress(owner_id)
    }
    page = questions[offset : offset + limit]
    return {
        "total": len(questions),
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(page) < len(questions),
        "questions": [
            _catalog_question_payload(question, progress_by_id.get(question.id))
            for question in page
        ],
    }


@api_router.post("/practice")
async def start_knowledge_practice(
    request: Request, payload: KnowledgePracticeRequest
) -> Dict[str, Any]:
    owner_id = f"feishu:{require_dashboard_open_id(request)}"
    repository = get_default_knowledge_repository()
    questions = repository.list_questions()
    if payload.question_id:
        questions = [item for item in questions if item.id == payload.question_id]
    elif payload.chapter_id:
        questions = [item for item in questions if item.chapter_id == payload.chapter_id]
    elif payload.module_id:
        questions = [item for item in questions if item.module_id == payload.module_id]
    elif payload.weak_only:
        weak_ids = {
            item.question_id
            for item in repository.list_progress(owner_id)
            if item.mastery_status != KnowledgeMasteryStatus.MASTERED
            and (item.last_score is None or item.last_score < 80)
        }
        questions = [item for item in questions if item.id in weak_ids]
    else:
        raise HTTPException(status_code=400, detail="A practice selector is required.")
    if not questions:
        raise HTTPException(status_code=404, detail="No matching knowledge questions.")
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    recommendations = []
    for question in questions:
        assignment = repository.create_assignment(
            owner_id=owner_id,
            question_id=question.id,
            assigned_on=today,
            assignment_type=KnowledgeAssignmentType.MANUAL,
            recommendation_reason="从知识导航主动开始练习。",
        )
        recommendations.append(
            KnowledgeRecommendation(assignment=assignment, question=question)
        )
    response = _dashboard_payload(repository, owner_id, recommendations, today.isoformat())
    response["focus_question_id"] = questions[0].id
    return response


@api_router.post("/answers")
async def submit_knowledge_answer(
    request: Request, payload: KnowledgeAnswerRequest
) -> Dict[str, Any]:
    owner_id = f"feishu:{require_dashboard_open_id(request)}"
    repository = get_default_knowledge_repository()
    workflow = KnowledgePracticeWorkflow(
        repository,
        get_knowledge_evaluation_service(),
    )
    try:
        outcome = await workflow.submit_answer(
            owner_id=owner_id,
            assignment_id=payload.assignment_id,
            answer_text=payload.answer_text,
            answer_source=payload.answer_source,
            submitted_at=datetime.now(ZoneInfo("Asia/Shanghai")),
            submission_id=payload.submission_id,
        )
    except KnowledgeSubmissionInProgressError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except KnowledgeEvaluationError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    question = repository.get_question(outcome.assignment.question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="Knowledge question was not found.")
    evaluation_payload = outcome.evaluation.to_dict()
    point_labels = {point.id: point.label for point in question.rubric_points}
    evaluation_payload["matched_required_labels"] = [
        point_labels[item] for item in outcome.evaluation.matched_required_point_ids
    ]
    evaluation_payload["matched_bonus_labels"] = [
        point_labels[item] for item in outcome.evaluation.matched_bonus_point_ids
    ]
    return {
        "attempt": {"id": outcome.attempt.id, "score": outcome.attempt.score},
        "evaluation": evaluation_payload,
        "reference_answer": {
            "short": question.short_reference_answer,
            "full": question.full_reference_answer,
        },
        "source": {
            "title": question.source_title,
            "url": question.source_url,
            "chapter": question.source_chapter,
            "page_start": question.source_page_start,
            "page_end": question.source_page_end,
        },
        "progress": _progress_payload(outcome.progress),
    }


@api_router.post("/transcriptions")
async def transcribe_knowledge_answer(
    request: Request,
    question_id: str = Query(..., min_length=1, max_length=160),
) -> Dict[str, str]:
    require_dashboard_open_id(request)
    repository = get_default_knowledge_repository()
    question = repository.get_question(question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="Knowledge question was not found.")

    result = await transcribe_audio_request(
        request,
        service=get_speech_transcription_service(request),
        context=_speech_context(question),
    )
    return {
        "transcript": result.transcript,
        "provider": result.provider,
        "model": result.model,
    }


@api_router.get("/jsapi-config")
async def get_knowledge_jsapi_config(
    request: Request,
    url: str = Query(..., min_length=1, max_length=2048),
) -> Dict[str, Any]:
    require_dashboard_open_id(request)
    page_url = url.split("#", 1)[0]
    _validate_knowledge_page_url(request, page_url)
    try:
        config = await get_feishu_jssdk_service(request).get_jssdk_config(page_url)
    except (FeishuConfigurationError, FeishuRequestError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="飞书移动录音暂时不可用，请使用浏览器录音或文字回答。",
        ) from exc
    return {
        "app_id": config.app_id,
        "timestamp": config.timestamp,
        "nonce_str": config.nonce_str,
        "signature": config.signature,
    }


def get_speech_transcription_service(request: Request):
    return create_speech_transcription_service(request.app.state.settings)


def get_feishu_jssdk_service(request: Request) -> FeishuMessageService:
    existing = getattr(request.app.state, "knowledge_feishu_jssdk_service", None)
    if existing is not None:
        return existing
    settings = request.app.state.settings
    service = FeishuMessageService(
        app_id=settings.feishu_app_id,
        app_secret=settings.feishu_app_secret,
        base_url=settings.feishu_api_base_url,
        timeout_seconds=settings.feishu_timeout_seconds,
    )
    request.app.state.knowledge_feishu_jssdk_service = service
    return service


def _validate_knowledge_page_url(request: Request, page_url: str) -> None:
    parsed = urlsplit(page_url)
    if parsed.scheme not in {"http", "https"} or parsed.path != "/study/knowledge":
        raise HTTPException(
            status_code=400,
            detail="只能为八股学习页面申请飞书录音权限。",
        )
    settings = request.app.state.settings
    if not settings.dashboard_public_base_url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="尚未配置 OfferPilot 网页应用公网地址。",
        )
    allowed_origin = settings.dashboard_public_base_url.rstrip("/")
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin != allowed_origin:
        raise HTTPException(status_code=400, detail="网页地址与 OfferPilot 配置不匹配。")


def _audio_duration_ms(request: Request) -> int:
    return audio_duration_ms(request)


async def _read_limited_audio(request: Request) -> bytes:
    return await read_limited_audio(request)


def _speech_context(question) -> str:
    parts = [
        f"模块：{question.module_title}",
        f"章节：{question.chapter_title}",
        f"题目：{question.prompt}",
    ]
    keywords = _safe_speech_keywords(question)
    if keywords:
        parts.append("技术术语：" + "、".join(keywords))
    return "；".join(parts)[:300]


def _safe_speech_keywords(question) -> List[str]:
    public_text = " ".join(
        [question.module_title, question.chapter_title, question.prompt]
    )
    normalized_public_text = public_text.casefold()
    candidates = [
        item.strip()
        for item in question.keywords
        if item.strip() and item.strip().casefold() in normalized_public_text
    ]
    candidates.extend(
        re.findall(r"[A-Za-z][A-Za-z0-9+#._/-]{1,31}", public_text)
    )
    result = []
    seen = set()
    for candidate in candidates:
        normalized = candidate.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(candidate)
        if len(result) == 20:
            break
    return result


def get_knowledge_evaluation_service() -> KnowledgeEvaluationService:
    return KnowledgeEvaluationService()


def _dashboard_payload(repository, owner_id, recommendations, date_value: str):
    items = []
    for recommendation in recommendations:
        assignment = recommendation.assignment
        question = recommendation.question
        progress = repository.get_progress(owner_id, question.id)
        items.append(
            {
                "assignment_id": assignment.id,
                "assignment_type": assignment.assignment_type.value,
                "recommendation_reason": assignment.recommendation_reason,
                "status": assignment.status.value,
                "question": {
                    "id": question.id,
                    "module_id": question.module_id,
                    "module_title": question.module_title,
                    "chapter_id": question.chapter_id,
                    "chapter_title": question.chapter_title,
                    "prompt": question.prompt,
                    "difficulty": question.difficulty.value,
                    "frequency": question.frequency,
                    "hint": question.hint,
                    "source_title": question.source_title,
                    "source_url": question.source_url,
                    "source_chapter": question.source_chapter,
                },
                "progress": _progress_payload(progress) if progress else None,
            }
        )
    return {
        "date": date_value,
        "summary": {
            "handled": sum(1 for item in recommendations if item.assignment.attempt_id),
            "total": len(recommendations),
        },
        "catalog": _catalog_payload(repository),
        "items": items,
    }


def _catalog_payload(repository) -> List[Dict[str, Any]]:
    modules: Dict[str, Dict[str, Any]] = {}
    for question in repository.list_questions():
        module = modules.setdefault(
            question.module_id,
            {
                "id": question.module_id,
                "title": question.module_title,
                "order": question.module_order,
                "chapters": {},
            },
        )
        chapter = module["chapters"].setdefault(
            question.chapter_id,
            {
                "id": question.chapter_id,
                "title": question.chapter_title,
                "order": question.chapter_order,
                "question_count": 0,
            },
        )
        chapter["question_count"] += 1
    result = []
    for module in sorted(modules.values(), key=lambda item: item["order"]):
        module["chapters"] = sorted(
            module["chapters"].values(), key=lambda item: item["order"]
        )
        result.append(module)
    return result


def _catalog_question_payload(question, progress) -> Dict[str, Any]:
    return {
        "id": question.id,
        "prompt": question.prompt,
        "status": progress.mastery_status.value if progress else "unseen",
        "score": progress.last_score if progress else None,
        "source_url": question.source_url,
    }


def _progress_payload(progress) -> Dict[str, Any]:
    return {
        "mastery_status": progress.mastery_status.value,
        "mastery_score": progress.mastery_score,
        "attempt_count": progress.attempt_count,
        "last_score": progress.last_score,
        "next_review_on": (
            progress.next_review_on.isoformat() if progress.next_review_on else None
        ),
        "last_detected_gaps": list(progress.last_detected_gaps),
    }
