from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from app.api.routes.leetcode_dashboard import (
    _dashboard_open_id,
    _require_dashboard_open_id,
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
from app.services.knowledge_practice import (
    KnowledgePracticeWorkflow,
    KnowledgeSubmissionInProgressError,
)
from app.services.knowledge_recommendation import (
    KnowledgeRecommendation,
    KnowledgeRecommendationWorkflow,
)


router = APIRouter(prefix="/study/knowledge")
api_router = APIRouter(prefix="/study/knowledge")
_HTML_PATH = Path(__file__).resolve().parents[2] / "web" / "knowledge_dashboard.html"


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
    if _dashboard_open_id(request) is None:
        return RedirectResponse(
            url="/leetcode/dashboard/auth/start?redirect=/study/knowledge"
        )
    return HTMLResponse(_HTML_PATH.read_text(encoding="utf-8"))


@api_router.get("/today")
async def get_knowledge_today(request: Request) -> Dict[str, Any]:
    open_id = _require_dashboard_open_id(request)
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
async def get_knowledge_materials(request: Request) -> Dict[str, Any]:
    _require_dashboard_open_id(request)
    repository = get_default_knowledge_repository()
    return {
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
                },
            }
            for question in repository.list_questions()
        ]
    }


@api_router.post("/practice")
async def start_knowledge_practice(
    request: Request, payload: KnowledgePracticeRequest
) -> Dict[str, Any]:
    owner_id = f"feishu:{_require_dashboard_open_id(request)}"
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
    owner_id = f"feishu:{_require_dashboard_open_id(request)}"
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
        "catalog": _catalog_payload(repository, owner_id),
        "items": items,
    }


def _catalog_payload(repository, owner_id: str) -> List[Dict[str, Any]]:
    modules: Dict[str, Dict[str, Any]] = {}
    progress_by_id = {item.question_id: item for item in repository.list_progress(owner_id)}
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
                "questions": [],
            },
        )
        progress = progress_by_id.get(question.id)
        chapter["questions"].append(
            {
                "id": question.id,
                "prompt": question.prompt,
                "status": progress.mastery_status.value if progress else "unseen",
                "score": progress.last_score if progress else None,
                "source_url": question.source_url,
            }
        )
    result = []
    for module in sorted(modules.values(), key=lambda item: item["order"]):
        module["chapters"] = sorted(
            module["chapters"].values(), key=lambda item: item["order"]
        )
        result.append(module)
    return result


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
