import asyncio
import json
import sqlite3

from app.agents.conversation import (
    ConversationEvent,
    ConversationSummary,
    InMemoryConversationStore,
)
from app.agents.orchestrator import AgentOrchestrator
from app.agents.planner import AgentPlanner, AgentPlannerContext
from app.repositories.sqlite_conversation_repository import SQLiteConversationStore
from app.schemas.agent import AgentActionName
from app.schemas.agent_plan import AgentPlan
from app.schemas.intent import IntentName
from app.services.context_compaction_service import (
    ContextCompactionService,
    estimate_value_tokens,
)
from app.services.llm_service import LLMResult
from app.tools.registry import ToolRegistry


class InvalidSummaryLLM:
    async def generate_text(self, **kwargs):
        return LLMResult(
            provider="fake",
            model="fake",
            content="not-json",
            raw_response={},
        )


def _append_turn(
    store,
    conversation_id: str,
    turn_id: str,
    user_text: str,
    tool_name: str = "",
) -> None:
    events = [
        ConversationEvent(
            event_type="user_message",
            role="user",
            payload={"text": user_text},
            turn_id=turn_id,
        )
    ]
    if tool_name:
        events.extend(
            [
                ConversationEvent(
                    event_type="tool_call",
                    role="assistant",
                    payload={"tool_name": tool_name, "arguments": {}},
                    turn_id=turn_id,
                ),
                ConversationEvent(
                    event_type="tool_result",
                    role="tool",
                    payload={
                        "tool_name": tool_name,
                        "success": True,
                        "message": "Found Alpha interview.",
                        "data": {
                            "interview_schedules": [
                                {
                                    "company": "Alpha",
                                    "round": "first",
                                    "interview_time": "2026-08-25T10:00:00",
                                }
                            ]
                        },
                    },
                    turn_id=turn_id,
                ),
            ]
        )
    events.append(
        ConversationEvent(
            event_type="assistant_message",
            role="assistant",
            payload={
                "action": "query_application" if tool_name else "answer_help",
                "reply": f"reply for {turn_id} " + ("x" * 120),
                "need_confirmation": False,
                "missing_slots": [],
                "slots": {},
            },
            turn_id=turn_id,
        )
    )
    store.append_events(conversation_id, events)


def _compactor(store, **kwargs) -> ContextCompactionService:
    return ContextCompactionService(
        conversation_store=store,
        llm_service=InvalidSummaryLLM(),
        enabled=True,
        trigger_tokens=kwargs.get("trigger_tokens", 40),
        max_context_tokens=kwargs.get("max_context_tokens", 4000),
        keep_recent_turns=kwargs.get("keep_recent_turns", 1),
        summary_max_tokens=kwargs.get("summary_max_tokens", 500),
    )


def test_compaction_keeps_raw_events_and_preserves_recent_complete_turn() -> None:
    store = InMemoryConversationStore()
    conversation_id = "feishu:chat:user"
    _append_turn(store, conversation_id, "turn-1", "My goal is an Agent engineer role.")
    _append_turn(
        store,
        conversation_id,
        "turn-2",
        "show interviews",
        tool_name="query_application",
    )
    _append_turn(store, conversation_id, "turn-3", "which one should I prepare first?")

    prepared = asyncio.run(_compactor(store).prepare_context(conversation_id))

    assert prepared.compacted is True
    assert prepared.summary is not None
    assert prepared.summary.summary_upto_sequence == 6
    assert prepared.summary.compaction_count == 1
    assert prepared.summary.upcoming_interviews[0]["company"] == "Alpha"
    assert prepared.summary.tool_evidence[0]["source_sequences"] == [5]
    assert {event.turn_id for event in prepared.events} == {"turn-3"}
    assert [event.event_type for event in prepared.events] == [
        "user_message",
        "assistant_message",
    ]
    assert len(store.get_recent_events(conversation_id, limit=100)) == 8


def test_repeated_compaction_merges_existing_summary_without_losing_evidence() -> None:
    store = InMemoryConversationStore()
    conversation_id = "api:session:user"
    _append_turn(
        store,
        conversation_id,
        "turn-1",
        "show interviews",
        tool_name="query_application",
    )
    _append_turn(store, conversation_id, "turn-2", "prepare Alpha first")
    first = asyncio.run(_compactor(store).prepare_context(conversation_id))
    assert first.summary is not None

    _append_turn(store, conversation_id, "turn-3", "My goal is an AI application role.")
    _append_turn(store, conversation_id, "turn-4", "continue the plan")
    second = asyncio.run(_compactor(store).prepare_context(conversation_id))

    assert second.summary is not None
    assert second.summary.compaction_count == 2
    assert second.summary.tool_evidence[0]["tool_name"] == "query_application"
    assert second.summary.upcoming_interviews[0]["company"] == "Alpha"
    assert second.summary.summary_upto_sequence > first.summary.summary_upto_sequence
    assert {event.turn_id for event in second.events} == {"turn-4"}


def test_sqlite_compaction_summary_survives_restart(tmp_path) -> None:
    database_path = tmp_path / "offerpilot.db"
    conversation_id = "feishu:tenant:chat:user"
    first_store = SQLiteConversationStore(str(database_path))
    _append_turn(first_store, conversation_id, "turn-1", "I want an Agent role.")
    _append_turn(first_store, conversation_id, "turn-2", "continue")

    first = asyncio.run(_compactor(first_store).prepare_context(conversation_id))
    restarted_store = SQLiteConversationStore(str(database_path))
    restarted = asyncio.run(
        ContextCompactionService(
            restarted_store,
            enabled=False,
        ).prepare_context(conversation_id)
    )

    assert first.summary is not None
    assert restarted.summary is not None
    assert restarted.summary.to_dict() == first.summary.to_dict()
    assert {event.turn_id for event in restarted.events} == {"turn-2"}


def test_sqlite_store_migrates_existing_conversation_table(tmp_path) -> None:
    database_path = tmp_path / "legacy.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE agent_conversations (
                conversation_id TEXT PRIMARY KEY,
                pending_action TEXT,
                recent_context TEXT,
                summary TEXT NOT NULL DEFAULT '',
                last_sequence INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    SQLiteConversationStore(str(database_path))

    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(agent_conversations)"
            ).fetchall()
        }
    assert {
        "summary_upto_sequence",
        "compaction_count",
        "summary_token_count",
        "last_compacted_at",
    }.issubset(columns)


def test_context_budget_fallback_keeps_newest_turn_as_an_atomic_unit() -> None:
    store = InMemoryConversationStore()
    conversation_id = "api:budget:user"
    _append_turn(store, conversation_id, "turn-1", "old " + ("x" * 500))
    _append_turn(
        store,
        conversation_id,
        "turn-2",
        "new " + ("y" * 500),
        tool_name="query_application",
    )

    prepared = asyncio.run(
        ContextCompactionService(
            store,
            enabled=False,
            trigger_tokens=20,
            max_context_tokens=20,
        ).prepare_context(conversation_id)
    )

    assert {event.turn_id for event in prepared.events} == {"turn-2"}
    assert [event.event_type for event in prepared.events] == [
        "user_message",
        "tool_call",
        "tool_result",
        "assistant_message",
    ]


def test_valid_model_summary_is_merged_with_deterministic_tool_evidence() -> None:
    class ValidSummaryLLM:
        async def generate_text(self, **kwargs):
            return LLMResult(
                provider="fake",
                model="fake",
                content=json.dumps(
                    {
                        "decisions": ["Prepare Alpha before Beta."],
                        "open_loops": ["Confirm the preparation schedule."],
                    }
                ),
                raw_response={},
            )

    store = InMemoryConversationStore()
    conversation_id = "api:model-summary:user"
    _append_turn(
        store,
        conversation_id,
        "turn-1",
        "show interviews",
        tool_name="query_application",
    )
    _append_turn(store, conversation_id, "turn-2", "continue")
    service = ContextCompactionService(
        store,
        llm_service=ValidSummaryLLM(),
        trigger_tokens=40,
        keep_recent_turns=1,
        summary_max_tokens=500,
    )

    prepared = asyncio.run(service.prepare_context(conversation_id))

    assert prepared.summary is not None
    assert "Prepare Alpha before Beta." in prepared.summary.decisions
    assert prepared.summary.tool_evidence[0]["tool_name"] == "query_application"


def test_orchestrator_passes_summary_and_recent_complete_turn_to_planner() -> None:
    class CapturingPlanner:
        def __init__(self) -> None:
            self.context = None

        async def plan_message(self, message, context, tool_specs):
            self.context = context
            return AgentPlan(
                intent=IntentName.ASK_HELP,
                confidence=0.9,
                action=AgentActionName.ANSWER_HELP,
                reply="Continue with Alpha preparation.",
                reason="answer from compacted context",
            )

    store = InMemoryConversationStore()
    conversation_id = "api:session:user"
    _append_turn(
        store,
        conversation_id,
        "turn-1",
        "show interviews",
        tool_name="query_application",
    )
    _append_turn(store, conversation_id, "turn-2", "prepare Alpha first")
    planner = CapturingPlanner()
    orchestrator = AgentOrchestrator(
        planner=planner,
        tool_registry=ToolRegistry(),
        conversation_store=store,
        context_compaction_service=_compactor(store),
    )

    response = asyncio.run(
        orchestrator.handle_message(
            "continue",
            user_id="user",
            source="api",
            conversation_scope="session",
            external_event_id="turn-3",
        )
    )

    assert response.action == AgentActionName.ANSWER_HELP
    assert planner.context is not None
    assert planner.context.conversation_summary is not None
    assert planner.context.conversation_summary.tool_evidence[0]["tool_name"] == (
        "query_application"
    )
    assert {event.turn_id for event in planner.context.conversation_history} == {
        "turn-2"
    }


def test_sqlite_summary_compare_and_swap_rejects_stale_compaction(tmp_path) -> None:
    store = SQLiteConversationStore(str(tmp_path / "offerpilot.db"))
    conversation_id = "api:cas:user"
    _append_turn(store, conversation_id, "turn-1", "first turn")
    summary = ConversationSummary.from_dict(
        {
            "decisions": ["Keep the first result."],
            "summary_upto_sequence": 2,
            "compaction_count": 1,
        }
    )
    stale_summary = ConversationSummary.from_dict(
        {
            "decisions": ["Overwrite with stale result."],
            "summary_upto_sequence": 2,
            "compaction_count": 1,
        }
    )

    assert store.save_conversation_summary(conversation_id, summary, 0) is True
    assert store.save_conversation_summary(conversation_id, stale_summary, 0) is False
    persisted = store.get_conversation_summary(conversation_id)
    assert persisted is not None
    assert persisted.decisions == ["Keep the first result."]


def test_compaction_cas_loser_reloads_summary_written_by_winner() -> None:
    class ConcurrentWinnerStore(InMemoryConversationStore):
        def save_conversation_summary(
            self,
            conversation_id,
            summary,
            expected_upto_sequence,
        ):
            saved = super().save_conversation_summary(
                conversation_id,
                summary,
                expected_upto_sequence,
            )
            assert saved is True
            return False

    store = ConcurrentWinnerStore()
    conversation_id = "api:concurrent:user"
    _append_turn(
        store,
        conversation_id,
        "turn-1",
        "show interviews",
        tool_name="query_application",
    )
    _append_turn(store, conversation_id, "turn-2", "continue")

    prepared = asyncio.run(_compactor(store).prepare_context(conversation_id))

    assert prepared.compacted is False
    assert prepared.summary is not None
    assert prepared.summary.tool_evidence[0]["tool_name"] == "query_application"
    assert {event.turn_id for event in prepared.events} == {"turn-2"}


def test_planner_prompt_contains_structured_conversation_summary() -> None:
    summary = ConversationSummary.from_dict(
        {
            "user_goals": ["Find an Agent engineering role."],
            "decisions": ["Prepare Alpha first."],
            "summary_upto_sequence": 12,
            "compaction_count": 1,
        }
    )
    planner = AgentPlanner(llm_planner_enabled=False)

    prompt = json.loads(
        planner._build_prompt(
            message="continue",
            context=AgentPlannerContext(
                conversation_id="api:session:user",
                user_id="user",
                source="api",
                conversation_summary=summary,
            ),
            tool_specs=[],
        )
    )

    assert prompt["conversation"]["summary"]["summary_upto_sequence"] == 12
    assert prompt["conversation"]["summary"]["decisions"] == [
        "Prepare Alpha first."
    ]


def test_token_estimator_is_conservative_for_chinese_text() -> None:
    chinese_tokens = estimate_value_tokens({"text": "上下文压缩需要保留关键事实"})
    ascii_tokens = estimate_value_tokens({"text": "context compression keeps key facts"})

    assert chinese_tokens >= 12
    assert ascii_tokens >= 5
