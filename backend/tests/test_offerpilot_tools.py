from app.schemas.agent import AgentActionName
from app.schemas.tool import ToolResult
from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository
from app.tools.offerpilot_tools import build_offerpilot_tool_registry
from app.tools.registry import ToolNotFoundError, ToolRegistry


def test_tool_registry_runs_registered_handler() -> None:
    registry = ToolRegistry()
    registry.register(
        "echo",
        lambda arguments: ToolResult(
            tool_name="echo",
            success=True,
            message=arguments["message"],
        ),
    )

    assert registry.has_tool("echo") is True
    result = registry.run("echo", {"message": "ok"})
    assert result.success is True
    assert result.message == "ok"


def test_tool_registry_raises_for_missing_tool() -> None:
    registry = ToolRegistry()

    try:
        registry.run("missing", {})
    except ToolNotFoundError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected ToolNotFoundError")


def test_offerpilot_tools_create_and_list_application() -> None:
    repository = InMemoryOfferPilotRepository()
    registry = build_offerpilot_tool_registry(repository)

    result = registry.run(
        AgentActionName.CREATE_APPLICATION.value,
        {
            "company": "深信服",
            "role": "开发实习",
            "interview_time": "明天下午三点",
            "round": "一面",
        },
    )

    assert result.success is True
    assert result.data["application"]["id"] == "app_1"
    assert result.data["application"]["status"] == "interview_1"
    assert repository.applications[0].company == "深信服"


def test_offerpilot_tools_complete_seed_task() -> None:
    repository = InMemoryOfferPilotRepository()
    registry = build_offerpilot_tool_registry(repository)

    result = registry.run(
        AgentActionName.COMPLETE_TASK.value,
        {"task_title": "反转链表"},
    )

    assert result.success is True
    assert result.data["task"]["id"] == "task_1"
    assert result.data["task"]["status"] == "passed"
