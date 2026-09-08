from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from app.agents.runtime import AgentRuntime, build_default_agent_runtime
from app.core.config import Settings
from app.repositories.sqlite_offerpilot_repository import SQLiteOfferPilotRepository
from app.services.agent_runtime_store import SQLiteAgentRuntimeStore
from app.services.tool_calling_model import OpenAIToolCallingModel
from app.tools.agent_tool import ToolApproval, ToolEffect


def test_default_runtime_is_composed_from_the_passed_settings(tmp_path) -> None:
    database_path = tmp_path / "custom-offerpilot.db"
    app_settings = Settings(
        storage_backend="sqlite",
        sqlite_path=str(database_path),
        agent_checkpoint_backend="sqlite",
        rag_enabled=False,
        knowledge_markdown_sync_enabled=False,
        feishu_calendar_sync_enabled=False,
        feishu_bitable_sync_enabled=False,
        llm_api_key="custom-key",
        llm_base_url="https://custom.example/v1",
        llm_provider="custom-provider",
        llm_model="custom-model",
        llm_timeout_seconds=17,
    )

    runtime = build_default_agent_runtime(InMemorySaver(), app_settings)

    assert isinstance(runtime, AgentRuntime)
    assert isinstance(runtime.runtime_store, SQLiteAgentRuntimeStore)
    assert runtime.runtime_store.database_path == database_path
    assert isinstance(runtime.offerpilot_repository, SQLiteOfferPilotRepository)
    assert runtime.offerpilot_repository.db_path == database_path
    assert runtime.model.api_key == "custom-key"
    assert runtime.model.base_url == "https://custom.example/v1"
    assert runtime.model.provider == "custom-provider"
    assert runtime.model.default_model == "custom-model"
    assert runtime.model.timeout_seconds == 17
    assert runtime._owned_mcp_client is None

    tool_names = {item.name for item in runtime.tool_registry.definitions()}
    assert "search_interview_knowledge" not in tool_names
    assert "search_project_evidence" not in tool_names
    assert "read_evidence" not in tool_names
    create_application = runtime.tool_registry.get_definition("create_application")
    assert create_application is not None
    assert create_application.effect == ToolEffect.LOCAL_WRITE
    assert create_application.approval == ToolApproval.IF_IMPLICIT
    assert Path(runtime.project_training_repository.database_path) == database_path


def test_default_runtime_selects_openai_model_adapter(tmp_path) -> None:
    app_settings = Settings(
        storage_backend="sqlite",
        sqlite_path=str(tmp_path / "openai-runtime.db"),
        agent_checkpoint_backend="memory",
        rag_enabled=False,
        knowledge_markdown_sync_enabled=False,
        feishu_calendar_sync_enabled=False,
        feishu_bitable_sync_enabled=False,
        llm_api_key="openai-key",
        llm_base_url="https://api.openai.com/v1",
        llm_provider="openai",
        llm_model="gpt-4.1-mini",
    )

    runtime = build_default_agent_runtime(InMemorySaver(), app_settings)

    assert isinstance(runtime.model, OpenAIToolCallingModel)
    assert runtime.model.provider == "openai"
    assert runtime.model.base_url == "https://api.openai.com/v1"
    assert runtime.model.default_model == "gpt-4.1-mini"
