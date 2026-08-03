from app.core.config import Settings, settings
from app.repositories.project_training_repository import (
    InMemoryProjectTrainingRepository,
    ProjectTrainingRepository,
)
from app.repositories.sqlite_project_training_repository import (
    SQLiteProjectTrainingRepository,
)


def build_project_training_repository(
    app_settings: Settings = settings,
) -> ProjectTrainingRepository:
    if app_settings.storage_backend.strip().lower() == "sqlite":
        return SQLiteProjectTrainingRepository(app_settings.sqlite_path)
    return InMemoryProjectTrainingRepository()


_default_repository = build_project_training_repository()


def get_default_project_training_repository() -> ProjectTrainingRepository:
    return _default_repository
