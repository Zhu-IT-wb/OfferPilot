from app.agents.conversation import ConversationStore, InMemoryConversationStore
from app.core.config import settings
from app.repositories.sqlite_conversation_repository import SQLiteConversationStore


def build_default_conversation_store() -> ConversationStore:
    if settings.storage_backend.strip().lower() == "sqlite":
        return SQLiteConversationStore(settings.sqlite_path)
    return InMemoryConversationStore()


_default_conversation_store = build_default_conversation_store()


def get_default_conversation_store() -> ConversationStore:
    return _default_conversation_store
