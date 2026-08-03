from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.intent import IntentName
from app.schemas.tool import ToolResult


# 枚举 AgentActionName 的可选值。
class AgentActionName(str, Enum):
    LIST_TODAY_TASKS = "list_today_tasks"
    ENABLE_LEETCODE_PLAN = "enable_leetcode_plan"
    DISABLE_LEETCODE_PLAN = "disable_leetcode_plan"
    GET_TODAY_LEETCODE = "get_today_leetcode"
    RECORD_LEETCODE_RESULT = "record_leetcode_result"
    CREATE_APPLICATION = "create_application"
    QUERY_APPLICATION = "query_application"
    UPDATE_APPLICATION = "update_application"
    RESCHEDULE_INTERVIEW = "reschedule_interview"
    CANCEL_INTERVIEW = "cancel_interview"
    COMPLETE_TASK = "complete_task"
    POSTPONE_TASK = "postpone_task"
    CREATE_INTERVIEW_REVIEW = "create_interview_review"
    START_MOCK_INTERVIEW = "start_mock_interview"
    START_PROJECT_TRAINING = "start_project_training"
    RESUME_PROJECT_TRAINING = "resume_project_training"
    GET_PROJECT_TRAINING_SUMMARY = "get_project_training_summary"
    RECORD_ANSWER = "record_answer"
    ANSWER_HELP = "answer_help"
    SUMMARIZE_WEEK = "summarize_week"
    ASK_CLARIFICATION = "ask_clarification"
    NO_OP = "no_op"


# 定义接口请求体的数据结构。
class AgentMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    confirmed: bool = False
    user_id: str = Field(default="local_user", min_length=1, max_length=128)
    source: str = Field(default="api", min_length=1, max_length=64)


# 定义接口请求体的数据结构。
class DebugAgentRequest(AgentMessageRequest):
    pass


# 定义接口响应体的数据结构。
class AgentResponse(BaseModel):
    intent: IntentName
    confidence: float = Field(..., ge=0.0, le=1.0)
    action: AgentActionName
    reply: str
    need_confirmation: bool = False
    slots: Dict[str, Any] = Field(default_factory=dict)
    missing_slots: List[str] = Field(default_factory=list)
    tool_result: Optional[ToolResult] = None
