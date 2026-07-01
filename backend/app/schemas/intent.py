from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, Field


class IntentName(str, Enum):
    GET_TODAY_TASKS = "get_today_tasks"
    COMPLETE_TASK = "complete_task"
    POSTPONE_TASK = "postpone_task"
    ADD_APPLICATION = "add_application"
    QUERY_APPLICATION = "query_application"
    UPDATE_APPLICATION = "update_application"
    ADD_INTERVIEW_REVIEW = "add_interview_review"
    START_MOCK_INTERVIEW = "start_mock_interview"
    ANSWER_QUESTION = "answer_question"
    ASK_HELP = "ask_help"
    SUMMARIZE_WEEK = "summarize_week"
    UNKNOWN = "unknown"


class DebugIntentRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class IntentClassification(BaseModel):
    intent: IntentName
    confidence: float = Field(..., ge=0.0, le=1.0)
    slots: Dict[str, Any] = Field(default_factory=dict)
