from dataclasses import dataclass
from functools import partial

from app.core.config import Settings
from app.project_analysis.repository_analysis_agent import (
    RepositoryAnalysisAgent,
)
from app.agents.tool_calling_agent import (
    ToolCallingAgentLimits,
)
from app.project_analysis.repository_analysis_service import (
    RepositoryAnalysisService,
)
from app.project_analysis.repository_analysis_result import (
    RepositoryAnalysisResultParser,
)
from app.project_analysis.repository_workspace import (
    GitHubRepositoryWorkspace,
)
from app.repositories.project_discovery_repository import (
    InMemoryProjectDiscoveryRepository,
    SQLiteProjectDiscoveryRepository,
)
from app.services.project_discovery_runner import ProjectDiscoveryRunner
from app.services.project_import import ProjectImportWorkflow
from app.services.tool_calling_model import (
    build_tool_calling_model,
)


@dataclass(frozen=True)
class ProjectDiscoveryServices:
    workflow: ProjectImportWorkflow
    runner: ProjectDiscoveryRunner


def build_project_discovery_services(app_settings: Settings, project_repository):
    """组装持久化任务、手写代码分析 Agent 和后台 Runner。"""

    repository = (
        SQLiteProjectDiscoveryRepository(app_settings.sqlite_path)
        if app_settings.storage_backend.strip().lower() == "sqlite"
        else InMemoryProjectDiscoveryRepository()
    )
    model = build_tool_calling_model(
        api_key=app_settings.llm_api_key,
        base_url=app_settings.llm_base_url,
        provider=app_settings.llm_provider,
        default_model=app_settings.llm_model,
        timeout_seconds=(
            app_settings
            .project_analysis_llm_timeout_seconds
        ),
    )
    agent = RepositoryAnalysisAgent(
        model=model,
        limits=ToolCallingAgentLimits(
            max_model_turns=(
                app_settings
                .project_analysis_max_model_turns
            ),
            max_tool_calls=(
                app_settings
                .project_analysis_max_tool_calls
            ),
            max_total_tokens=(
                app_settings
                .project_analysis_max_total_tokens
            ),
            max_request_chars=(
                app_settings
                .project_analysis_max_request_chars
            ),
        ),
        max_findings=(
            app_settings
            .project_analysis_max_findings
        ),
        max_quote_chars=(
            app_settings
            .project_analysis_max_quote_chars
        ),
    )
    workspace_factory = partial(
        GitHubRepositoryWorkspace,
        clone_timeout_seconds=(
            app_settings
            .project_discovery_clone_timeout_seconds
        ),
        max_workspace_mb=(
            app_settings
            .project_discovery_max_git_metadata_mb
        ),
    )
    analysis_service = RepositoryAnalysisService(
        agent=agent,
        parser=RepositoryAnalysisResultParser(
            max_findings=(
                app_settings
                .project_analysis_max_findings
            ),
            max_quote_chars=(
                app_settings
                .project_analysis_max_quote_chars
            ),
        ),
        workspace_factory=workspace_factory,
    )
    workflow = ProjectImportWorkflow(
        repository=repository,
        project_repository=project_repository,
        analysis_service=analysis_service,
    )
    return ProjectDiscoveryServices(
        workflow=workflow,
        runner=ProjectDiscoveryRunner(
            workflow, app_settings.project_discovery_max_concurrency
        ),
    )
