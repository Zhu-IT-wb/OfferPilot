import asyncio
import json

from app.agents.conversation import (
    ConversationEvent,
    PendingAgentAction,
    RecentAgentContext,
)
from app.agents.orchestrator import AgentOrchestrator
from app.agents.planner import AgentPlanner, AgentPlannerContext
from app.repositories.sqlite_conversation_repository import SQLiteConversationStore
from app.schemas.agent import AgentActionName, AgentResponse
from app.schemas.agent_plan import AgentPlan
from app.schemas.intent import IntentName
from app.schemas.tool import ToolResult
from app.services.llm_service import LLMResult
from app.tools.registry import ToolRegistry


def _query_response() -> AgentResponse:
    return AgentResponse(
        intent=IntentName.QUERY_APPLICATION,
        confidence=0.94,
        action=AgentActionName.QUERY_APPLICATION,
        reply="Found two upcoming interviews.",
        slots={"query_type": "upcoming_interviews"},
        tool_result=ToolResult(
            tool_name="query_application",
            success=True,
            message="Query completed.",
            data={
                "interview_schedules": [
                    {"company": "Alpha", "interview_time": "2026-08-22T10:00:00"},
                    {"company": "Beta", "interview_time": "2026-08-24T15:00:00"},
                ]
            },
        ),
    )


def test_sqlite_conversation_state_and_events_survive_restart(tmp_path) -> None:
    database_path = tmp_path / "offerpilot.db"
    conversation_id = "feishu:tenant-a:chat-a:user-a"
    response = _query_response()

    first_store = SQLiteConversationStore(str(database_path))
    first_store.set_pending_action(
        conversation_id,
        PendingAgentAction.from_response(response, "show my interviews"),
    )
    first_store.set_recent_context(
        conversation_id,
        RecentAgentContext.from_response(response, "show my interviews"),
    )
    first_store.append_events(
        conversation_id,
        [
            ConversationEvent(
                event_type="user_message",
                role="user",
                payload={"text": "show my interviews"},
                turn_id="turn-1",
            ),
            ConversationEvent(
                event_type="tool_result",
                role="tool",
                payload={
                    "tool_name": "query_application",
                    "success": True,
                    "data": response.tool_result.data,
                },
                turn_id="turn-1",
            ),
        ],
    )

    restarted_store = SQLiteConversationStore(str(database_path))

    pending = restarted_store.get_pending_action(conversation_id)
    recent = restarted_store.get_recent_context(conversation_id)
    history = restarted_store.get_recent_events(conversation_id)
    assert pending is not None
    assert pending.original_message == "show my interviews"
    assert recent is not None
    assert recent.tool_result is not None
    assert recent.tool_result["data"]["interview_schedules"][0]["company"] == "Alpha"
    assert [event.sequence for event in history] == [1, 2]
    assert history[1].payload["data"]["interview_schedules"][1]["company"] == "Beta"


def test_sqlite_conversation_histories_are_isolated(tmp_path) -> None:
    store = SQLiteConversationStore(str(tmp_path / "offerpilot.db"))
    store.append_events(
        "feishu:chat-a:user-a",
        [
            ConversationEvent(
                event_type="user_message",
                role="user",
                payload={"text": "message-a"},
                turn_id="turn-a",
            )
        ],
    )
    store.append_events(
        "feishu:chat-b:user-a",
        [
            ConversationEvent(
                event_type="user_message",
                role="user",
                payload={"text": "message-b"},
                turn_id="turn-b",
            )
        ],
    )

    assert store.get_recent_events("feishu:chat-a:user-a")[0].payload["text"] == "message-a"
    assert store.get_recent_events("feishu:chat-b:user-a")[0].payload["text"] == "message-b"


def test_orchestrator_loads_history_after_store_restart(tmp_path) -> None:
    database_path = tmp_path / "offerpilot.db"
    conversation_id = "feishu:tenant-a:chat-a:user-a"
    first_store = SQLiteConversationStore(str(database_path))
    first_store.append_events(
        conversation_id,
        [
            ConversationEvent(
                event_type="tool_result",
                role="tool",
                payload={
                    "tool_name": "query_application",
                    "success": True,
                    "data": {"applications": [{"company": "Alpha", "status": "interview"}]},
                },
                turn_id="event-old",
            )
        ],
    )

    class CapturingPlanner:
        def __init__(self) -> None:
            self.context = None

        async def plan_message(self, message, context, tool_specs):
            self.context = context
            return AgentPlan(
                intent=IntentName.ASK_HELP,
                confidence=0.9,
                action=AgentActionName.ANSWER_HELP,
                reply="Alpha is already in the interview stage.",
                reason="answer from persisted history",
            )

    planner = CapturingPlanner()
    restarted_store = SQLiteConversationStore(str(database_path))
    orchestrator = AgentOrchestrator(
        planner=planner,
        tool_registry=ToolRegistry(),
        conversation_store=restarted_store,
    )

    response = asyncio.run(
        orchestrator.handle_message(
            "what about that one?",
            user_id="user-a",
            source="feishu",
            conversation_scope="tenant-a:chat-a",
            external_event_id="event-new",
        )
    )

    assert response.action == AgentActionName.ANSWER_HELP
    assert planner.context is not None
    assert planner.context.conversation_id == conversation_id
    assert planner.context.conversation_history[0].payload["data"]["applications"][0]["company"] == "Alpha"
    persisted = SQLiteConversationStore(str(database_path)).get_recent_events(conversation_id)
    assert [event.event_type for event in persisted] == [
        "tool_result",
        "user_message",
        "assistant_message",
    ]
    assert persisted[-1].turn_id == "event-new"


def test_llm_planner_prompt_contains_persisted_tool_results(tmp_path) -> None:
    database_path = tmp_path / "offerpilot.db"
    conversation_id = "api:session-a:user-a"
    store = SQLiteConversationStore(str(database_path))
    store.append_events(
        conversation_id,
        [
            ConversationEvent(
                event_type="tool_result",
                role="tool",
                payload={
                    "tool_name": "query_application",
                    "success": True,
                    "data": {"applications": [{"company": "Alpha"}]},
                },
                turn_id="turn-1",
            )
        ],
    )

    class CapturingLLMService:
        def __init__(self) -> None:
            self.prompt = ""

        async def generate_text(self, **kwargs):
            self.prompt = kwargs["prompt"]
            return LLMResult(
                provider="fake",
                model="fake",
                content=json.dumps(
                    {
                        "intent": "ask_help",
                        "confidence": 0.9,
                        "action": "answer_help",
                        "reply": "Alpha was mentioned earlier.",
                        "need_confirmation": False,
                        "slots": {},
                        "missing_slots": [],
                        "steps": [],
                        "reason": "use conversation history",
                    }
                ),
                raw_response={},
            )

    llm_service = CapturingLLMService()
    planner = AgentPlanner(
        llm_service=llm_service,
        llm_planner_enabled=True,
    )
    plan = asyncio.run(
        planner.plan_message(
            message="what about that one?",
            context=AgentPlannerContext(
                conversation_id=conversation_id,
                user_id="user-a",
                source="api",
                conversation_history=store.get_recent_events(conversation_id),
            ),
            tool_specs=[],
        )
    )

    prompt = json.loads(llm_service.prompt)
    history = prompt["conversation"]["history"]
    assert plan is not None and plan.action == AgentActionName.ANSWER_HELP
    assert history[0]["event_type"] == "tool_result"
    assert history[0]["payload"]["data"]["applications"][0]["company"] == "Alpha"
