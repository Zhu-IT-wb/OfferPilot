from dataclasses import dataclass

from app.agents.project_discovery_agent import ProjectDiscoveryGraph
from app.core.config import Settings
from app.repositories.project_discovery_repository import (
    InMemoryProjectDiscoveryRepository,
    SQLiteProjectDiscoveryRepository,
)
from app.services.github_repository_source import GitHubRepositorySource
from app.services.llm_service import LLMService
from app.services.project_discovery_runner import ProjectDiscoveryRunner
from app.services.project_import import ProjectImportWorkflow


@dataclass(frozen=True)
class ProjectDiscoveryServices:
    workflow: ProjectImportWorkflow
    runner: ProjectDiscoveryRunner


def build_project_discovery_services(app_settings: Settings, project_repository):
    repository = (
        SQLiteProjectDiscoveryRepository(app_settings.sqlite_path)
        if app_settings.storage_backend.strip().lower() == "sqlite"
        else InMemoryProjectDiscoveryRepository()
    )
    source = GitHubRepositorySource(
        clone_timeout_seconds=app_settings.project_discovery_clone_timeout_seconds,
        max_git_metadata_mb=app_settings.project_discovery_max_git_metadata_mb,
    )
    llm = LLMService(
        api_key=app_settings.llm_api_key, base_url=app_settings.llm_base_url,
        provider=app_settings.llm_provider, default_model=app_settings.llm_model,
        timeout_seconds=app_settings.llm_timeout_seconds,
    )
    workflow = ProjectImportWorkflow(
        repository, project_repository, ProjectDiscoveryGraph(source, llm)
    )
    return ProjectDiscoveryServices(
        workflow=workflow,
        runner=ProjectDiscoveryRunner(
            workflow, app_settings.project_discovery_max_concurrency
        ),
    )
