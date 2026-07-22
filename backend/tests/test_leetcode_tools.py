from datetime import date

from app.repositories.leetcode_repository import InMemoryLeetCodeRepository
from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository
from app.services.leetcode_catalog import load_hot100_snapshot
from app.tools.offerpilot_tools import build_offerpilot_tool_registry


def test_leetcode_tools_enable_list_and_record_explicit_feedback(monkeypatch) -> None:
    monkeypatch.setattr("app.tools.leetcode_tools.today_in_shanghai", lambda: date(2026, 7, 22))
    leetcode_repository = InMemoryLeetCodeRepository(problems=load_hot100_snapshot().problems)
    registry = build_offerpilot_tool_registry(
        InMemoryOfferPilotRepository(),
        calendar_service=None,
        bitable_service=None,
        leetcode_repository=leetcode_repository,
    )
    actor = {
        "owner_id": "feishu:ou_1",
        "attendee_user_id": "ou_1",
    }

    enabled = registry.run("enable_leetcode_plan", actor)
    listed = registry.run("get_today_leetcode", {"owner_id": "feishu:ou_1"})
    feedback = registry.run(
        "record_leetcode_result",
        {
            "owner_id": "feishu:ou_1",
            "problem_index": 1,
            "result": "with_solution",
        },
    )
    disabled = registry.run("disable_leetcode_plan", {"owner_id": "feishu:ou_1"})

    assert enabled.success is True
    assert len(enabled.data["recommendations"]) == 3
    assert "https://leetcode.cn/problems/" in enabled.message
    assert leetcode_repository.get_subscription("feishu:ou_1").feishu_open_id == "ou_1"
    assert listed.data["recommendations"] == enabled.data["recommendations"]
    assert feedback.success is True
    assert feedback.data["assignment"]["result"] == "with_solution"
    assert feedback.data["progress"]["next_review_on"] == "2026-07-23"
    assert disabled.success is True
    assert leetcode_repository.get_subscription("feishu:ou_1").enabled is False
    assert leetcode_repository.get_progress(
        "feishu:ou_1", feedback.data["problem"]["id"]
    ) is not None


def test_leetcode_feedback_requires_a_problem_when_multiple_are_pending(monkeypatch) -> None:
    monkeypatch.setattr("app.tools.leetcode_tools.today_in_shanghai", lambda: date(2026, 7, 22))
    leetcode_repository = InMemoryLeetCodeRepository(problems=load_hot100_snapshot().problems)
    registry = build_offerpilot_tool_registry(
        InMemoryOfferPilotRepository(),
        calendar_service=None,
        bitable_service=None,
        leetcode_repository=leetcode_repository,
    )
    registry.run("get_today_leetcode", {"owner_id": "feishu:ou_1"})

    result = registry.run(
        "record_leetcode_result",
        {"owner_id": "feishu:ou_1", "result": "independent"},
    )

    assert result.success is False
    assert result.data["missing_slots"] == ["problem_index"]
    assert "第几题" in result.message


def test_today_tasks_combines_leetcode_and_other_job_search_tasks(monkeypatch) -> None:
    monkeypatch.setattr("app.tools.leetcode_tools.today_in_shanghai", lambda: date(2026, 7, 22))
    leetcode_repository = InMemoryLeetCodeRepository(problems=load_hot100_snapshot().problems)
    registry = build_offerpilot_tool_registry(
        InMemoryOfferPilotRepository(),
        calendar_service=None,
        bitable_service=None,
        leetcode_repository=leetcode_repository,
    )

    result = registry.run("list_today_tasks", {"owner_id": "feishu:ou_1"})

    assert result.success is True
    assert len(result.data["leetcode_recommendations"]) == 3
    assert len(result.data["tasks"]) == 2
    assert "今日 LeetCode" in result.message
    assert "其他秋招任务" in result.message
    assert "LeetCode 206. 反转链表" not in result.message
