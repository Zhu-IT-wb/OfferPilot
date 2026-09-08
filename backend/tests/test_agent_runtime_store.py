import json
import sqlite3

from app.services.agent_runtime_store import SQLiteAgentRuntimeStore


def test_runtime_store_coexists_with_legacy_conversation_event_table(tmp_path) -> None:
    database_path = tmp_path / "runtime.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE agent_conversation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                turn_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                role TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(conversation_id, sequence)
            )
            """
        )
        connection.commit()

    store = SQLiteAgentRuntimeStore(str(database_path))
    store.record_event(
        event_id="event_1",
        thread_id="thread_1",
        run_id="run_1",
        event_type="tool_result",
        tool_call_id="call_1",
        payload={"status": "success"},
    )

    with sqlite3.connect(database_path) as connection:
        legacy_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(agent_conversation_events)")
        }
        runtime_row = connection.execute(
            "SELECT thread_id, run_id, tool_call_id, payload FROM agent_runtime_events"
        ).fetchone()

    assert "conversation_id" in legacy_columns
    assert "thread_id" not in legacy_columns
    assert runtime_row[:3] == ("thread_1", "run_1", "call_1")
    assert json.loads(runtime_row[3]) == {"status": "success"}


def test_runtime_store_persists_idempotent_operations_and_scoped_receipts(tmp_path) -> None:
    store = SQLiteAgentRuntimeStore(str(tmp_path / "runtime.db"))

    started = store.begin_operation(
        "operation_1",
        "fingerprint_1",
        thread_id="thread_a",
        owner_id="api:user_a",
        run_id="run_a",
        tool_name="sync_study_plan_to_calendar",
    )
    replayed = store.begin_operation("operation_1", "fingerprint_1")
    unresolved = store.find_unresolved_operation(
        thread_id="thread_a",
        owner_id="api:user_a",
        tool_name="sync_study_plan_to_calendar",
        fingerprint="fingerprint_1",
    )
    store.finish_operation("operation_1", "success", {"event_id": "evt_1"})

    assert started.status == "started"
    assert started.thread_id == "thread_a"
    assert started.owner_id == "api:user_a"
    assert started.run_id == "run_a"
    assert started.tool_name == "sync_study_plan_to_calendar"
    assert replayed.status == "started"
    assert unresolved is not None
    assert unresolved.operation_key == "operation_1"
    persisted_operation = store.get_operation("operation_1")
    assert persisted_operation.outcome == {"event_id": "evt_1"}
    assert persisted_operation.thread_id == "thread_a"
    assert (
        store.find_unresolved_operation(
            thread_id="thread_a",
            owner_id="api:user_a",
            tool_name="sync_study_plan_to_calendar",
            fingerprint="fingerprint_1",
        )
        is None
    )

    partial = store.begin_operation(
        "operation_partial",
        "fingerprint_partial",
        thread_id="thread_a",
        owner_id="api:user_a",
        run_id="run_partial",
        tool_name="create_application",
    )
    store.finish_operation(
        partial.operation_key,
        "partial",
        {"status": "partial", "data": {"application_id": "application_1"}},
    )
    partial_lookup = {
        "thread_id": "thread_a",
        "owner_id": "api:user_a",
        "tool_name": "create_application",
        "fingerprint": "fingerprint_partial",
    }
    assert store.find_unresolved_operation(**partial_lookup) is None
    unresolved_partial = store.find_unresolved_operation(
        **partial_lookup,
        include_partial=True,
    )
    assert unresolved_partial is not None
    assert unresolved_partial.status == "partial"

    assert store.begin_receipt("api:api:user_a", "request_1") is True
    assert store.begin_receipt("api:api:user_a", "request_1") is False
    assert store.begin_receipt("api:api:user_b", "request_1") is True
    store.stage_receipt_response(
        "api:api:user_a",
        "request_1",
        {"thread_id": "thread_a", "status": "completed"},
    )
    staged = store.get_receipt_record("api:api:user_a", "request_1")
    assert staged.status == "processing"
    assert staged.response == {"thread_id": "thread_a", "status": "completed"}
    store.finish_receipt(
        "api:api:user_a",
        "request_1",
        {"thread_id": "thread_a"},
    )

    assert store.get_receipt("api:api:user_a", "request_1") == {
        "thread_id": "thread_a"
    }
    assert store.get_receipt("api:api:user_b", "request_1") is None
