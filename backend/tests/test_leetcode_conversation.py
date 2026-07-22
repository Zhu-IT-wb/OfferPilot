import asyncio
from datetime import date

from app.agents.conversation import InMemoryConversationStore
from app.agents.orchestrator import AgentOrchestrator
from app.models.leetcode import LeetCodeDifficulty, LeetCodeProblem
from app.repositories.leetcode_repository import InMemoryLeetCodeRepository
from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository
from app.tools.offerpilot_tools import build_offerpilot_tool_registry


def _problem(index: int) -> LeetCodeProblem:
    return LeetCodeProblem(
        id=f"leetcode_{index}",
        frontend_id=str(index),
        title_zh=f"Problem {index}",
        title_en=f"Problem {index}",
        slug=f"problem-{index}",
        difficulty=LeetCodeDifficulty.EASY,
        topics=["数组"],
        category="数组",
        category_order=0,
        problem_order=index,
        url=f"https://leetcode.cn/problems/problem-{index}/",
    )


def _orchestrator(monkeypatch) -> AgentOrchestrator:
    monkeypatch.setattr("app.tools.leetcode_tools.today_in_shanghai", lambda: date(2026, 7, 22))
    leetcode_repository = InMemoryLeetCodeRepository(
        problems=[_problem(index) for index in range(1, 4)]
    )
    return AgentOrchestrator(
        tool_registry=build_offerpilot_tool_registry(
            InMemoryOfferPilotRepository(),
            calendar_service=None,
            bitable_service=None,
            leetcode_repository=leetcode_repository,
        ),
        conversation_store=InMemoryConversationStore(),
    )


def test_ambiguous_problem_can_be_selected_in_the_next_message(monkeypatch) -> None:
    orchestrator = _orchestrator(monkeypatch)
    asyncio.run(orchestrator.handle_message("今天刷什么", user_id="ou_1", source="feishu"))

    ambiguous = asyncio.run(
        orchestrator.handle_message(
            "LeetCode Problem 看题解完成", user_id="ou_1", source="feishu"
        )
    )
    selected = asyncio.run(
        orchestrator.handle_message("第2题", user_id="ou_1", source="feishu")
    )

    assert ambiguous.missing_slots == ["problem_index"]
    assert selected.tool_result is not None and selected.tool_result.success is True
    assert selected.tool_result.data["problem"]["frontend_id"] == "2"
    assert selected.tool_result.data["assignment"]["result"] == "with_solution"


def test_ambiguous_result_can_be_supplied_in_the_next_message(monkeypatch) -> None:
    orchestrator = _orchestrator(monkeypatch)
    asyncio.run(orchestrator.handle_message("今天刷什么", user_id="ou_1", source="feishu"))

    ambiguous = asyncio.run(
        orchestrator.handle_message("第1题做完了", user_id="ou_1", source="feishu")
    )
    completed = asyncio.run(
        orchestrator.handle_message("看题解完成", user_id="ou_1", source="feishu")
    )

    assert ambiguous.missing_slots == ["result"]
    assert completed.tool_result is not None and completed.tool_result.success is True
    assert completed.tool_result.data["problem"]["frontend_id"] == "1"
    assert completed.tool_result.data["assignment"]["result"] == "with_solution"
