import json
import sqlite3
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.agents.conversation import (
    ConversationEvent,
    PendingAgentAction,
    RecentAgentContext,
)


class SQLiteConversationStore:
    def __init__(self, database_path: str) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def get_pending_action(self, conversation_id: str) -> Optional[PendingAgentAction]:
        value = self._get_state_value(conversation_id, "pending_action")
        return PendingAgentAction.from_dict(value) if value is not None else None

    def set_pending_action(self, conversation_id: str, pending: PendingAgentAction) -> None:
        self._set_state_value(conversation_id, "pending_action", pending.to_dict())

    def clear_pending_action(self, conversation_id: str) -> None:
        self._clear_state_value(conversation_id, "pending_action")

    def get_recent_context(self, conversation_id: str) -> Optional[RecentAgentContext]:
        value = self._get_state_value(conversation_id, "recent_context")
        return RecentAgentContext.from_dict(value) if value is not None else None

    def set_recent_context(
        self,
        conversation_id: str,
        recent_context: RecentAgentContext,
    ) -> None:
        self._set_state_value(conversation_id, "recent_context", recent_context.to_dict())

    def append_events(
        self,
        conversation_id: str,
        events: Sequence[ConversationEvent],
    ) -> List[ConversationEvent]:
        if not events:
            return []

        persisted: List[ConversationEvent] = []
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO agent_conversations (
                    conversation_id, pending_action, recent_context, summary,
                    last_sequence, created_at, updated_at
                )
                VALUES (?, NULL, NULL, '', 0, ?, ?)
                ON CONFLICT(conversation_id) DO NOTHING
                """,
                (conversation_id, now, now),
            )
            row = connection.execute(
                "SELECT last_sequence FROM agent_conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            next_sequence = int(row["last_sequence"]) + 1

            for event in events:
                stored = replace(
                    event,
                    sequence=next_sequence,
                    created_at=event.created_at or now,
                    payload=deepcopy(event.payload),
                )
                connection.execute(
                    """
                    INSERT INTO agent_conversation_events (
                        conversation_id, sequence, turn_id, event_type,
                        role, payload, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conversation_id,
                        stored.sequence,
                        stored.turn_id,
                        stored.event_type,
                        stored.role,
                        _dump_json(stored.payload),
                        stored.created_at,
                    ),
                )
                persisted.append(stored)
                next_sequence += 1

            connection.execute(
                """
                UPDATE agent_conversations
                SET last_sequence = ?, updated_at = ?
                WHERE conversation_id = ?
                """,
                (next_sequence - 1, now, conversation_id),
            )
            connection.commit()
        return persisted

    def get_recent_events(
        self,
        conversation_id: str,
        limit: int = 40,
    ) -> List[ConversationEvent]:
        if limit <= 0:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, turn_id, event_type, role, payload, created_at
                FROM agent_conversation_events
                WHERE conversation_id = ?
                ORDER BY sequence DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return [
            ConversationEvent(
                sequence=int(row["sequence"]),
                turn_id=row["turn_id"],
                event_type=row["event_type"],
                role=row["role"],
                payload=_load_json_object(row["payload"]),
                created_at=row["created_at"],
            )
            for row in reversed(rows)
        ]

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_conversations (
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
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_conversation_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    turn_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    role TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(conversation_id, sequence),
                    FOREIGN KEY(conversation_id)
                        REFERENCES agent_conversations(conversation_id)
                        ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_conversation_events_recent
                ON agent_conversation_events(conversation_id, sequence DESC)
                """
            )
            connection.commit()

    def _get_state_value(
        self,
        conversation_id: str,
        column: str,
    ) -> Optional[Dict[str, Any]]:
        _validate_state_column(column)
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {column} FROM agent_conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None or row[column] is None:
            return None
        return _load_json_object(row[column])

    def _set_state_value(
        self,
        conversation_id: str,
        column: str,
        value: Dict[str, Any],
    ) -> None:
        _validate_state_column(column)
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                f"""
                INSERT INTO agent_conversations (
                    conversation_id, {column}, summary,
                    last_sequence, created_at, updated_at
                )
                VALUES (?, ?, '', 0, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    {column} = excluded.{column},
                    updated_at = excluded.updated_at
                """,
                (conversation_id, _dump_json(value), now, now),
            )
            connection.commit()

    def _clear_state_value(self, conversation_id: str, column: str) -> None:
        _validate_state_column(column)
        with self._connect() as connection:
            connection.execute(
                f"""
                UPDATE agent_conversations
                SET {column} = NULL, updated_at = ?
                WHERE conversation_id = ?
                """,
                (_utc_now(), conversation_id),
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _validate_state_column(column: str) -> None:
    if column not in {"pending_action", "recent_context"}:
        raise ValueError(f"Unsupported conversation state column: {column}")


def _dump_json(value: Dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_json_object(value: str) -> Dict[str, Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("Persisted conversation payload must be a JSON object.")
    return decoded


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
