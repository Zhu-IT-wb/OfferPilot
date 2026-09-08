"""Stable names for tools exposed by OfferPilot's domain registry."""

from enum import Enum


class AgentActionName(str, Enum):
    """Registry keys kept as an enum for legacy tool-adapter compatibility."""

    LIST_TODAY_TASKS = "list_today_tasks"
    ENABLE_LEETCODE_PLAN = "enable_leetcode_plan"
    DISABLE_LEETCODE_PLAN = "disable_leetcode_plan"
    GET_TODAY_LEETCODE = "get_today_leetcode"
    RECORD_LEETCODE_RESULT = "record_leetcode_result"
    CREATE_APPLICATION = "create_application"
    GET_APPLICATION_BITABLE_LINK = "get_application_bitable_link"
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
    SEARCH_CAREER_KNOWLEDGE = "search_career_knowledge"
    SEARCH_PROJECT_EVIDENCE = "search_project_evidence"
