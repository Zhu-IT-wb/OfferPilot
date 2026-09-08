import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol


@dataclass(frozen=True)
class OperationRecord:
    operation_key: str
    fingerprint: str
    status: str
    outcome: Optional[Dict[str, Any]] = None
    thread_id: str = ""
    owner_id: str = ""
    run_id: str = ""
    tool_name: str = ""


@dataclass(frozen=True)
class IngressReceiptRecord:
    source: str
    request_id: str
    status: str
    fingerprint: str = ""
    thread_id: str = ""
    run_id: str = ""
    response: Optional[Dict[str, Any]] = None


class AgentRuntimeStore(Protocol):
    def get_operation(self, operation_key: str) -> Optional[OperationRecord]: ...
    def find_unresolved_operation(
        self,
        *,
        thread_id: str,
        owner_id: str,
        tool_name: str,
        fingerprint: str,
        include_partial: bool = False,
    ) -> Optional[OperationRecord]: ...
    def find_reconciled_applied_operation(
        self,
        *,
        thread_id: str,
        owner_id: str,
        tool_name: str,
        fingerprint: str,
    ) -> Optional[OperationRecord]: ...
    def begin_operation(
        self,
        operation_key: str,
        fingerprint: str,
        *,
        thread_id: str = "",
        owner_id: str = "",
        run_id: str = "",
        tool_name: str = "",
    ) -> OperationRecord: ...
    def finish_operation(
        self,
        operation_key: str,
        status: str,
        outcome: Dict[str, Any],
    ) -> None: ...
    def resolve_operation(
        self,
        operation_key: str,
        expected_statuses: tuple[str, ...],
        status: str,
        outcome: Dict[str, Any],
    ) -> bool: ...
    def get_receipt(self, source: str, request_id: str) -> Optional[Dict[str, Any]]: ...
    def get_receipt_record(
        self, source: str, request_id: str
    ) -> Optional[IngressReceiptRecord]: ...
    def begin_receipt(
        self,
        source: str,
        request_id: str,
        fingerprint: str = "",
        thread_id: str = "",
        run_id: str = "",
    ) -> bool: ...
    def finish_receipt(
        self,
        source: str,
        request_id: str,
        response: Dict[str, Any],
    ) -> None: ...
    def stage_receipt_response(
        self,
        source: str,
        request_id: str,
        response: Dict[str, Any],
    ) -> None: ...
    def record_event(
        self,
        event_id: str,
        thread_id: str,
        run_id: str,
        event_type: str,
        payload: Dict[str, Any],
        tool_call_id: Optional[str] = None,
    ) -> None: ...


class InMemoryAgentRuntimeStore:
    def __init__(self) -> None:
        self.operations: Dict[str, OperationRecord] = {}
        self.receipts: Dict[tuple[str, str], IngressReceiptRecord] = {}
        self.events: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def get_operation(self, operation_key: str) -> Optional[OperationRecord]:
        with self._lock:
            return self.operations.get(operation_key)

    def find_unresolved_operation(
        self,
        *,
        thread_id: str,
        owner_id: str,
        tool_name: str,
        fingerprint: str,
        include_partial: bool = False,
    ) -> Optional[OperationRecord]:
        unresolved_statuses = {"started", "unknown"}
        if include_partial:
            unresolved_statuses.add("partial")
        with self._lock:
            matches = [
                record
                for record in self.operations.values()
                if record.thread_id == thread_id
                and record.owner_id == owner_id
                and record.tool_name == tool_name
                and record.fingerprint == fingerprint
                and record.status in unresolved_statuses
            ]
            return matches[-1] if matches else None

    def find_reconciled_applied_operation(
        self,
        *,
        thread_id: str,
        owner_id: str,
        tool_name: str,
        fingerprint: str,
    ) -> Optional[OperationRecord]:
        with self._lock:
            matches = [
                record
                for record in self.operations.values()
                if record.thread_id == thread_id
                and record.owner_id == owner_id
                and record.tool_name == tool_name
                and record.fingerprint == fingerprint
                and _is_reconciled_applied(record)
            ]
            return matches[-1] if matches else None

    def begin_operation(
        self,
        operation_key: str,
        fingerprint: str,
        *,
        thread_id: str = "",
        owner_id: str = "",
        run_id: str = "",
        tool_name: str = "",
    ) -> OperationRecord:
        with self._lock:
            existing = self.operations.get(operation_key)
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    raise ValueError("Operation key was reused with different arguments.")
                return existing
            record = OperationRecord(
                operation_key,
                fingerprint,
                "started",
                thread_id=thread_id,
                owner_id=owner_id,
                run_id=run_id,
                tool_name=tool_name,
            )
            self.operations[operation_key] = record
            return record

    def finish_operation(
        self,
        operation_key: str,
        status: str,
        outcome: Dict[str, Any],
    ) -> None:
        with self._lock:
            existing = self.operations[operation_key]
            self.operations[operation_key] = OperationRecord(
                operation_key,
                existing.fingerprint,
                status,
                dict(outcome),
                thread_id=existing.thread_id,
                owner_id=existing.owner_id,
                run_id=existing.run_id,
                tool_name=existing.tool_name,
            )

    def resolve_operation(
        self,
        operation_key: str,
        expected_statuses: tuple[str, ...],
        status: str,
        outcome: Dict[str, Any],
    ) -> bool:
        with self._lock:
            existing = self.operations.get(operation_key)
            if existing is None or existing.status not in set(expected_statuses):
                return False
            self.operations[operation_key] = OperationRecord(
                operation_key,
                existing.fingerprint,
                status,
                dict(outcome),
                thread_id=existing.thread_id,
                owner_id=existing.owner_id,
                run_id=existing.run_id,
                tool_name=existing.tool_name,
            )
            return True

    def get_receipt(self, source: str, request_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            record = self.receipts.get((source, request_id))
            if record is None or record.status != "completed" or record.response is None:
                return None
            return dict(record.response)

    def get_receipt_record(
        self, source: str, request_id: str
    ) -> Optional[IngressReceiptRecord]:
        with self._lock:
            return self.receipts.get((source, request_id))

    def begin_receipt(
        self,
        source: str,
        request_id: str,
        fingerprint: str = "",
        thread_id: str = "",
        run_id: str = "",
    ) -> bool:
        with self._lock:
            key = (source, request_id)
            if key in self.receipts:
                return False
            self.receipts[key] = IngressReceiptRecord(
                source=source,
                request_id=request_id,
                status="processing",
                fingerprint=fingerprint,
                thread_id=thread_id,
                run_id=run_id,
            )
            return True

    def finish_receipt(
        self,
        source: str,
        request_id: str,
        response: Dict[str, Any],
    ) -> None:
        with self._lock:
            existing = self.receipts.get((source, request_id))
            self.receipts[(source, request_id)] = IngressReceiptRecord(
                source=source,
                request_id=request_id,
                status="completed",
                fingerprint=existing.fingerprint if existing else "",
                thread_id=existing.thread_id if existing else str(response.get("thread_id") or ""),
                run_id=existing.run_id if existing else str(response.get("run_id") or ""),
                response=dict(response),
            )

    def stage_receipt_response(
        self,
        source: str,
        request_id: str,
        response: Dict[str, Any],
    ) -> None:
        with self._lock:
            existing = self.receipts.get((source, request_id))
            self.receipts[(source, request_id)] = IngressReceiptRecord(
                source=source,
                request_id=request_id,
                status=existing.status if existing else "processing",
                fingerprint=existing.fingerprint if existing else "",
                thread_id=existing.thread_id if existing else str(response.get("thread_id") or ""),
                run_id=existing.run_id if existing else str(response.get("run_id") or ""),
                response=dict(response),
            )

    def record_event(
        self,
        event_id: str,
        thread_id: str,
        run_id: str,
        event_type: str,
        payload: Dict[str, Any],
        tool_call_id: Optional[str] = None,
    ) -> None:
        with self._lock:
            self.events.setdefault(
                event_id,
                {
                    "event_id": event_id,
                    "thread_id": thread_id,
                    "run_id": run_id,
                    "event_type": event_type,
                    "tool_call_id": tool_call_id,
                    "payload": dict(payload),
                },
            )


class SQLiteAgentRuntimeStore:
    def __init__(self, database_path: str) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def get_operation(self, operation_key: str) -> Optional[OperationRecord]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT operation_key, fingerprint, status, outcome, "
                "thread_id, owner_id, run_id, tool_name "
                "FROM agent_tool_operations WHERE operation_key = ?",
                (operation_key,),
            ).fetchone()
        return _operation_from_row(row) if row is not None else None

    def find_unresolved_operation(
        self,
        *,
        thread_id: str,
        owner_id: str,
        tool_name: str,
        fingerprint: str,
        include_partial: bool = False,
    ) -> Optional[OperationRecord]:
        status_clause = (
            "AND status IN ('started', 'unknown', 'partial') "
            if include_partial
            else "AND status IN ('started', 'unknown') "
        )
        query = (
            "SELECT operation_key, fingerprint, status, outcome, "
            "thread_id, owner_id, run_id, tool_name "
            "FROM agent_tool_operations WHERE thread_id = ? AND owner_id = ? "
            "AND tool_name = ? AND fingerprint = ? "
            f"{status_clause}"
            "ORDER BY updated_at DESC LIMIT 1"
        )
        with self._connect() as connection:
            row = connection.execute(
                query,
                (thread_id, owner_id, tool_name, fingerprint),
            ).fetchone()
        return _operation_from_row(row) if row is not None else None

    def find_reconciled_applied_operation(
        self,
        *,
        thread_id: str,
        owner_id: str,
        tool_name: str,
        fingerprint: str,
    ) -> Optional[OperationRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT operation_key, fingerprint, status, outcome, "
                "thread_id, owner_id, run_id, tool_name "
                "FROM agent_tool_operations WHERE thread_id = ? AND owner_id = ? "
                "AND tool_name = ? AND fingerprint = ? AND status = 'success' "
                "ORDER BY updated_at DESC",
                (thread_id, owner_id, tool_name, fingerprint),
            ).fetchall()
        for row in rows:
            record = _operation_from_row(row)
            if _is_reconciled_applied(record):
                return record
        return None

    def begin_operation(
        self,
        operation_key: str,
        fingerprint: str,
        *,
        thread_id: str = "",
        owner_id: str = "",
        run_id: str = "",
        tool_name: str = "",
    ) -> OperationRecord:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT operation_key, fingerprint, status, outcome, "
                "thread_id, owner_id, run_id, tool_name "
                "FROM agent_tool_operations WHERE operation_key = ?",
                (operation_key,),
            ).fetchone()
            if row is not None:
                record = _operation_from_row(row)
                if record.fingerprint != fingerprint:
                    raise ValueError("Operation key was reused with different arguments.")
                connection.commit()
                return record
            connection.execute(
                "INSERT INTO agent_tool_operations "
                "(operation_key, fingerprint, status, outcome, thread_id, owner_id, "
                "run_id, tool_name, created_at, updated_at) "
                "VALUES (?, ?, 'started', NULL, ?, ?, ?, ?, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                (operation_key, fingerprint, thread_id, owner_id, run_id, tool_name),
            )
            connection.commit()
        return OperationRecord(
            operation_key,
            fingerprint,
            "started",
            thread_id=thread_id,
            owner_id=owner_id,
            run_id=run_id,
            tool_name=tool_name,
        )

    def finish_operation(
        self,
        operation_key: str,
        status: str,
        outcome: Dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE agent_tool_operations SET status = ?, outcome = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE operation_key = ?",
                (status, _dump(outcome), operation_key),
            )
            connection.commit()

    def resolve_operation(
        self,
        operation_key: str,
        expected_statuses: tuple[str, ...],
        status: str,
        outcome: Dict[str, Any],
    ) -> bool:
        if not expected_statuses:
            return False
        placeholders = ", ".join("?" for _ in expected_statuses)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE agent_tool_operations SET status = ?, outcome = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE operation_key = ? "
                f"AND status IN ({placeholders})",
                (
                    status,
                    _dump(outcome),
                    operation_key,
                    *expected_statuses,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def get_receipt(self, source: str, request_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT response FROM agent_ingress_receipts "
                "WHERE source = ? AND request_id = ? AND status = 'completed'",
                (source, request_id),
            ).fetchone()
        if row is None or not row["response"]:
            return None
        return json.loads(row["response"])

    def get_receipt_record(
        self, source: str, request_id: str
    ) -> Optional[IngressReceiptRecord]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT source, request_id, status, fingerprint, thread_id, run_id, response "
                "FROM agent_ingress_receipts WHERE source = ? AND request_id = ?",
                (source, request_id),
            ).fetchone()
        if row is None:
            return None
        return IngressReceiptRecord(
            source=row["source"],
            request_id=row["request_id"],
            status=row["status"],
            fingerprint=row["fingerprint"] or "",
            thread_id=row["thread_id"] or "",
            run_id=row["run_id"] or "",
            response=json.loads(row["response"]) if row["response"] else None,
        )

    def begin_receipt(
        self,
        source: str,
        request_id: str,
        fingerprint: str = "",
        thread_id: str = "",
        run_id: str = "",
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO agent_ingress_receipts "
                "(source, request_id, status, fingerprint, thread_id, run_id, response, "
                "created_at, updated_at) VALUES (?, ?, 'processing', ?, ?, ?, NULL, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                (source, request_id, fingerprint, thread_id, run_id),
            )
            connection.commit()
        return cursor.rowcount == 1

    def finish_receipt(
        self,
        source: str,
        request_id: str,
        response: Dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO agent_ingress_receipts "
                "(source, request_id, status, response, created_at, updated_at) "
                "VALUES (?, ?, 'completed', ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
                "ON CONFLICT(source, request_id) DO UPDATE SET "
                "status = 'completed', response = excluded.response, "
                "updated_at = CURRENT_TIMESTAMP",
                (source, request_id, _dump(response)),
            )
            connection.commit()

    def stage_receipt_response(
        self,
        source: str,
        request_id: str,
        response: Dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO agent_ingress_receipts "
                "(source, request_id, status, thread_id, run_id, response, "
                "created_at, updated_at) VALUES (?, ?, 'processing', ?, ?, ?, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
                "ON CONFLICT(source, request_id) DO UPDATE SET "
                "response = excluded.response, updated_at = CURRENT_TIMESTAMP",
                (
                    source,
                    request_id,
                    str(response.get("thread_id") or ""),
                    str(response.get("run_id") or ""),
                    _dump(response),
                ),
            )
            connection.commit()

    def record_event(
        self,
        event_id: str,
        thread_id: str,
        run_id: str,
        event_type: str,
        payload: Dict[str, Any],
        tool_call_id: Optional[str] = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO agent_runtime_events "
                "(event_id, thread_id, run_id, event_type, tool_call_id, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (
                    event_id,
                    thread_id,
                    run_id,
                    event_type,
                    tool_call_id,
                    _dump(payload),
                ),
            )
            connection.commit()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_tool_operations (
                    operation_key TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    outcome TEXT,
                    thread_id TEXT NOT NULL DEFAULT '',
                    owner_id TEXT NOT NULL DEFAULT '',
                    run_id TEXT NOT NULL DEFAULT '',
                    tool_name TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            for column in ("thread_id", "owner_id", "run_id", "tool_name"):
                self._ensure_column(
                    connection,
                    "agent_tool_operations",
                    column,
                    "TEXT NOT NULL DEFAULT ''",
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_ingress_receipts (
                    source TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    fingerprint TEXT NOT NULL DEFAULT '',
                    thread_id TEXT NOT NULL DEFAULT '',
                    run_id TEXT NOT NULL DEFAULT '',
                    response TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(source, request_id)
                )
                """
            )
            self._ensure_column(
                connection,
                "agent_ingress_receipts",
                "fingerprint",
                "TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_column(
                connection,
                "agent_ingress_receipts",
                "thread_id",
                "TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_column(
                connection,
                "agent_ingress_receipts",
                "run_id",
                "TEXT NOT NULL DEFAULT ''",
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_runtime_events (
                    event_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    tool_call_id TEXT,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_runtime_events_thread_created "
                "ON agent_runtime_events(thread_id, created_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_tool_operations_unresolved "
                "ON agent_tool_operations(thread_id, owner_id, tool_name, fingerprint, status)"
            )
            connection.commit()

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        declaration: str,
    ) -> None:
        columns = {
            str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection


def _operation_from_row(row: sqlite3.Row) -> OperationRecord:
    return OperationRecord(
        operation_key=row["operation_key"],
        fingerprint=row["fingerprint"],
        status=row["status"],
        outcome=json.loads(row["outcome"]) if row["outcome"] else None,
        thread_id=row["thread_id"] or "",
        owner_id=row["owner_id"] or "",
        run_id=row["run_id"] or "",
        tool_name=row["tool_name"] or "",
    )


def _is_reconciled_applied(record: OperationRecord) -> bool:
    if record.status != "success" or not isinstance(record.outcome, dict):
        return False
    data = record.outcome.get("data")
    if not isinstance(data, dict):
        return False
    reconciliation = data.get("reconciliation")
    return (
        isinstance(reconciliation, dict)
        and reconciliation.get("outcome") == "applied"
    )


def _dump(value: Dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
