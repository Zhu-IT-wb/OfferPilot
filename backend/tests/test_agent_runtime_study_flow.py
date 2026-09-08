import asyncio
import json
from collections import Counter
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Sequence
from zoneinfo import ZoneInfo

from langgraph.checkpoint.memory import InMemorySaver

from app.agents.runtime import AgentRuntime
from app.core.config import settings as default_settings
from app.models.interview_knowledge import KnowledgeMasteryStatus, KnowledgeProgress
from app.models.study import StudySessionSyncStatus
from app.models.tool_calling import ModelToolCall, ModelTurn, TokenUsage
from app.repositories.interview_knowledge_repository import (
    InMemoryInterviewKnowledgeRepository,
)
from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository
from app.tools.offerpilot_tools import query_application
from app.tools.registry import ToolRegistry
from app.tools.runtime_tools import build_runtime_tool_registry
from app.tools.study_tools import CalendarSyncReceipt


class FakeCalendarProvider:
    def __init__(self) -> None:
        self.busy_reads: List[tuple[str, datetime, datetime]] = []
        self.created: List[tuple[str, str, str]] = []

    async def list_busy(self, owner_id, start_at, end_at):
        self.busy_reads.append((owner_id, start_at, end_at))
        return []

    async def create_study_event(self, owner_id, session, operation_key):
        self.created.append((owner_id, session.id, operation_key))
        return CalendarSyncReceipt(
            event_id=f"event_{session.id}",
            calendar_id="primary",
        )

    def authorization_url(self, owner_id):
        return None


class StudyFlowModel:
    def __init__(
        self,
        *,
        range_start: date,
        range_end: date,
        availability_start: datetime,
        availability_end: datetime,
        interview_at: datetime,
        interview_id: str,
    ) -> None:
        self.range_start = range_start
        self.range_end = range_end
        self.availability_start = availability_start
        self.availability_end = availability_end
        self.interview_at = interview_at
        self.interview_id = interview_id
        self.calls: List[List[Dict[str, Any]]] = []
        self.plan_id: str | None = None

    async def complete(self, messages, tools, options) -> ModelTurn:
        history = [dict(message) for message in messages]
        self.calls.append(history)
        stage = len(self.calls) - 1

        if stage == 0:
            return _tool_turn(
                _call("find_interviews", "list_interviews"),
                _call("find_gaps", "list_learning_gaps", limit=5),
            )

        if stage == 1:
            interviews = _tool_payload(history, "find_interviews")
            gaps = _tool_payload(history, "find_gaps")
            assert interviews["status"] == "success"
            assert interviews["data"]["interview_schedules"][0]["id"] == self.interview_id
            assert gaps["status"] == "success"
            assert gaps["data"]["learning_gaps"][0]["question_id"] == "gap_jvm"
            return _tool_turn(
                _call("read_preferences", "get_study_preferences")
            )

        if stage == 2:
            supplied = _tool_payload(history, "read_preferences")
            assert supplied["status"] == "success"
            assert supplied["data"]["human_input"]["timezone"] == "Asia/Shanghai"
            return _tool_turn(
                _call(
                    "save_preferences",
                    "save_study_preferences",
                    timezone="Asia/Shanghai",
                    weekday_windows=[{"start_time": "09:00", "end_time": "18:00"}],
                    weekend_windows=[{"start_time": "09:00", "end_time": "18:00"}],
                    daily_max_minutes=120,
                    session_minutes=45,
                )
            )

        if stage == 3:
            assert _tool_payload(history, "save_preferences")["status"] == "success"
            return _tool_turn(
                _call(
                    "read_availability",
                    "get_calendar_availability",
                    range_start=self.availability_start.isoformat(),
                    range_end=self.availability_end.isoformat(),
                )
            )

        if stage == 4:
            availability = _tool_payload(history, "read_availability")
            assert availability["status"] == "success"
            assert availability["data"]["provider"] == "local_and_feishu"
            return _tool_turn(
                _call(
                    "create_plan",
                    "create_study_plan",
                    goal="准备本周 JVM 面试",
                    range_start=self.range_start.isoformat(),
                    range_end=self.range_end.isoformat(),
                    topics=[
                        {
                            "topic": "JVM GC 与可达性分析",
                            "duration_minutes": 90,
                            "priority": 3,
                            "deadline": self.interview_at.isoformat(),
                            "source_refs": ["gap:gap_jvm"],
                            "rationale": "最近掌握度较低，且面试在本周。",
                        }
                    ],
                    source_interview_ids=[self.interview_id],
                )
            )

        if stage == 5:
            created = _tool_payload(history, "create_plan")
            assert created["status"] == "success"
            assert len(created["data"]["sessions"]) == 2
            self.plan_id = created["data"]["plan"]["id"]
            return _tool_turn(
                _call(
                    "sync_plan",
                    "sync_study_plan_to_calendar",
                    plan_id=self.plan_id,
                )
            )

        if stage == 6:
            synced = _tool_payload(history, "sync_plan")
            assert synced["status"] == "success"
            assert len(synced["data"]["synced"]) == 2
            return _answer_turn("已结合本周面试和 JVM 薄弱点生成复习计划，并同步到日历。")

        raise AssertionError(f"Unexpected model stage: {stage}")


def test_runtime_completes_study_plan_flow_across_two_native_interrupts() -> None:
    async def scenario():
        zone = ZoneInfo("Asia/Shanghai")
        first_day = datetime.now(zone).date() + timedelta(days=2)
        last_day = first_day + timedelta(days=2)
        availability_start = datetime.combine(first_day, time.min, zone)
        availability_end = datetime.combine(last_day + timedelta(days=1), time.min, zone)
        interview_at = datetime.combine(last_day, time(hour=16), zone)
        owner_id = "api:user_study_flow"

        repository = InMemoryOfferPilotRepository()
        interview = repository.create_interview_schedule(
            company="Example Tech",
            round_name="技术二面",
            role="Java 后端",
            start_at=interview_at.isoformat(),
            owner_id=owner_id,
        )
        knowledge = InMemoryInterviewKnowledgeRepository()
        knowledge.save_progress(
            KnowledgeProgress(
                owner_id=owner_id,
                question_id="gap_jvm",
                mastery_status=KnowledgeMasteryStatus.LEARNING,
                mastery_score=25,
                last_score=40,
                last_detected_gaps=["可达性分析与 GC Roots"],
            )
        )
        calendar = FakeCalendarProvider()

        legacy = ToolRegistry()
        legacy.register(
            "query_application",
            lambda arguments: query_application(repository, arguments),
            description="查询投递记录和近期面试。",
            optional_slots=["query_type", "company"],
        )
        registry = build_runtime_tool_registry(
            legacy,
            offerpilot_repository=repository,
            knowledge_repository=knowledge,
            calendar_provider=calendar,
        )
        model = StudyFlowModel(
            range_start=first_day,
            range_end=last_day,
            availability_start=availability_start,
            availability_end=availability_end,
            interview_at=interview_at,
            interview_id=interview.id,
        )
        runtime = AgentRuntime(
            model=model,
            tool_registry=registry,
            checkpointer=InMemorySaver(),
            settings=replace(
                default_settings,
                agent_max_model_turns=12,
                agent_max_tool_calls=20,
                agent_no_progress_limit=3,
                agent_max_verifier_passes=0,
                context_max_input_tokens=40000,
            ),
        )

        missing_preferences = await runtime.start(
            "查询我本周的面试，结合薄弱知识点生成复习计划，并同步到飞书日历。",
            user_id="user_study_flow",
            request_id="req_study_start",
        )
        assert missing_preferences.status.value == "waiting_for_input"
        assert missing_preferences.interaction.type.value == "preference_form"
        assert missing_preferences.interaction.tool_name == "get_study_preferences"
        assert calendar.created == []

        preferences = {
            "timezone": "Asia/Shanghai",
            "weekday_windows": [{"start_time": "09:00", "end_time": "18:00"}],
            "weekend_windows": [{"start_time": "09:00", "end_time": "18:00"}],
            "daily_max_minutes": 120,
            "session_minutes": 45,
        }
        calendar_approval = await runtime.resume(
            missing_preferences.thread_id,
            missing_preferences.interaction.id,
            "answer",
            value=preferences,
            user_id="user_study_flow",
            request_id="req_study_preferences",
        )
        assert calendar_approval.status.value == "waiting_for_input"
        assert calendar_approval.interaction.type.value == "approval"
        assert calendar_approval.interaction.tool_name == "sync_study_plan_to_calendar"
        assert calendar.created == []

        completed = await runtime.resume(
            calendar_approval.thread_id,
            calendar_approval.interaction.id,
            "approve",
            user_id="user_study_flow",
            request_id="req_study_calendar_approval",
        )
        created_before_duplicate = list(calendar.created)
        model_calls_before_duplicate = len(model.calls)
        duplicate = await runtime.resume(
            calendar_approval.thread_id,
            calendar_approval.interaction.id,
            "approve",
            user_id="user_study_flow",
            request_id="req_study_calendar_approval",
        )

        snapshot = await runtime.graph.aget_state(
            {"configurable": {"thread_id": completed.thread_id}}
        )
        return (
            repository,
            calendar,
            model,
            completed,
            duplicate,
            created_before_duplicate,
            model_calls_before_duplicate,
            list(snapshot.values.get("messages") or []),
        )

    (
        repository,
        calendar,
        model,
        completed,
        duplicate,
        created_before_duplicate,
        model_calls_before_duplicate,
        messages,
    ) = asyncio.run(scenario())

    assert completed.status.value == "completed"
    assert completed.interaction is None
    assert completed.reply == "已结合本周面试和 JVM 薄弱点生成复习计划，并同步到日历。"
    assert completed.usage.model_calls == 7
    assert completed.usage.tool_calls == 7
    assert [item.name for item in completed.tool_executions] == [
        "list_interviews",
        "list_learning_gaps",
        "get_study_preferences",
        "save_study_preferences",
        "get_calendar_availability",
        "create_study_plan",
        "sync_study_plan_to_calendar",
    ]
    assert all(item.status.value == "success" for item in completed.tool_executions)

    sessions = repository.list_study_sessions(owner_id="api:user_study_flow")
    assert len(sessions) == 2
    assert all(item.sync_status == StudySessionSyncStatus.SYNCED for item in sessions)
    assert {item.calendar_event_id for item in sessions} == {
        f"event_{item.id}" for item in sessions
    }
    # The write path refreshes availability instead of trusting the model's
    # earlier observation, so a changed calendar cannot create stale slots.
    assert len(calendar.busy_reads) == 2
    assert len(calendar.created) == 2
    assert len({item[1] for item in calendar.created}) == 2

    assistant_call_ids = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            assistant_call_ids.append(str(call["id"]))
        assert "reasoning_content" not in message
    tool_message_ids = [
        str(message["tool_call_id"])
        for message in messages
        if message.get("role") == "tool"
    ]
    assert Counter(tool_message_ids) == Counter(assistant_call_ids)
    assert all(count == 1 for count in Counter(tool_message_ids).values())

    assert duplicate == completed
    assert calendar.created == created_before_duplicate
    assert len(model.calls) == model_calls_before_duplicate


def _call(call_id: str, name: str, **arguments: Any) -> ModelToolCall:
    return ModelToolCall(id=call_id, name=name, arguments=arguments)


def _tool_turn(*calls: ModelToolCall) -> ModelTurn:
    return ModelTurn(
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in calls
            ],
            "reasoning_content": "hidden",
        },
        tool_calls=list(calls),
        content="",
        reasoning_content="hidden",
        finish_reason="tool_calls",
        usage=TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
        provider="fake",
        model="study-flow",
    )


def _answer_turn(content: str) -> ModelTurn:
    return ModelTurn(
        assistant_message={
            "role": "assistant",
            "content": content,
            "reasoning_content": "hidden",
        },
        tool_calls=[],
        content=content,
        reasoning_content="hidden",
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
        provider="fake",
        model="study-flow",
    )


def _tool_payload(messages: Sequence[Dict[str, Any]], call_id: str) -> Dict[str, Any]:
    for message in reversed(messages):
        if message.get("role") == "tool" and message.get("tool_call_id") == call_id:
            return json.loads(str(message["content"]))
    raise AssertionError(f"ToolMessage not found for {call_id}")
