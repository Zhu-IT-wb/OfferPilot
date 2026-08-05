import json
import sqlite3
from dataclasses import asdict, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from app.models.project_training import (
    ProjectEvidence,
    ProjectProfile,
    ProjectProfileStatus,
    ProjectTopicProgress,
    ProjectTrainingAnswer,
    ProjectTrainingAnswerSource,
    ProjectTrainingDifficulty,
    ProjectTrainingEvaluationStatus,
    ProjectTrainingSession,
    ProjectTrainingSessionStatus,
    ProjectTrainingSummary,
    ProjectTrainingTurn,
    ProjectTrainingTurnStatus,
)
from app.repositories.project_training_repository import (
    InMemoryProjectTrainingRepository,
    ProjectTrainingSubmissionInProgressError,
    _answer_lease_expired,
    _inherit_code_evidence,
    _project_discovery_evidence,
    _profile_values,
    _project_content_hash,
    _source_values,
    build_project_evidence,
)


class SQLiteProjectTrainingRepository:
    """Relational SQLite adapter with owner-scoped keys and atomic answer commits."""

    def __init__(self, database_path: str) -> None:
        self.database_path = database_path
        Path(database_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create_project(self, owner_id: str, values: dict) -> ProjectProfile:
        project = InMemoryProjectTrainingRepository().create_project(owner_id, values)
        with self._transaction() as connection:
            self._store_project(connection, project)
        return project

    def update_project(self, owner_id: str, project_id: str, values: dict):
        with self._transaction() as connection:
            current = self._get_project(connection, owner_id, project_id)
            if current is None:
                return None
            updated = replace(
                current,
                version=current.version + 1,
                updated_at=datetime.now().astimezone(),
                **_profile_values(values),
            )
            updated.content_hash = _project_content_hash(updated)
            inherited = _inherit_code_evidence(
                updated,
                self._list_project_evidence(
                    connection, owner_id, project_id, current.version
                ),
            )
            self._store_project(connection, updated, inherited)
            return updated

    def create_discovered_project(self, owner_id, values, discovery_evidence):
        values = dict(values)
        discovery_job_id = str(values.get("source_discovery_job_id") or "")
        with self._transaction() as connection:
            existing = self._get_project_by_discovery_job(
                connection, owner_id, discovery_job_id
            )
            if existing is not None:
                return existing
            project_id = values.pop("project_id", None)
            if project_id:
                current = self._get_project(connection, owner_id, project_id)
                if current is None or current.status != ProjectProfileStatus.ACTIVE:
                    raise ValueError("要更新的项目不存在或已归档。")
                project = replace(
                    current, version=current.version + 1,
                    updated_at=datetime.now().astimezone(),
                    **_profile_values(values), **_source_values(values),
                )
            else:
                project = InMemoryProjectTrainingRepository().create_project(
                    owner_id, values
                )
                project.source_repository_url = values.get("source_repository_url", "")
                project.source_commit_sha = values.get("source_commit_sha", "")
                project.source_discovery_job_id = discovery_job_id
                project.content_hash = _project_content_hash(project)
            converted = _project_discovery_evidence(project, discovery_evidence)
            self._store_project(connection, project, converted)
            return project

    def get_project_by_discovery_job(self, owner_id, discovery_job_id):
        if not discovery_job_id:
            return None
        with self._connect() as connection:
            return self._get_project_by_discovery_job(
                connection, owner_id, discovery_job_id
            )

    def get_project(self, owner_id: str, project_id: str, version=None):
        with self._connect() as connection:
            return self._get_project(connection, owner_id, project_id, version)

    def list_projects(self, owner_id: str, include_archived: bool = False):
        query = "SELECT payload FROM project_profiles WHERE owner_id=?"
        parameters = [owner_id]
        if not include_archived:
            query += " AND status=?"
            parameters.append(ProjectProfileStatus.ACTIVE.value)
        query += " ORDER BY updated_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [_project(json.loads(row["payload"])) for row in rows]

    def list_project_versions(self, owner_id: str, project_id: str):
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT version FROM project_profile_versions
                WHERE owner_id=? AND project_id=? ORDER BY version
                """,
                (owner_id, project_id),
            ).fetchall()
        return [int(row["version"]) for row in rows]

    def list_project_evidence(self, owner_id: str, project_id: str, version: int):
        with self._connect() as connection:
            return self._list_project_evidence(
                connection, owner_id, project_id, version
            )

    def archive_project(self, owner_id: str, project_id: str):
        with self._transaction() as connection:
            project = self._get_project(connection, owner_id, project_id)
            if project is None:
                return None
            project.status = ProjectProfileStatus.ARCHIVED
            project.updated_at = datetime.now().astimezone()
            connection.execute(
                """
                UPDATE project_profiles SET status=?, updated_at=?, payload=?
                WHERE owner_id=? AND project_id=?
                """,
                (
                    project.status.value,
                    project.updated_at.isoformat(),
                    _dump(project),
                    owner_id,
                    project_id,
                ),
            )
            return project

    def create_session(
        self,
        owner_id,
        project,
        creation_id,
        target_role,
        goal,
        difficulty,
        max_turns,
        focus_topics=None,
    ):
        with self._transaction() as connection:
            existing = self._get_session_by_creation_id(
                connection, owner_id, creation_id
            )
            if existing is not None:
                turn = (
                    self._get_turn(connection, owner_id, existing.current_turn_id)
                    if existing.current_turn_id
                    else None
                )
                if existing.current_turn_id and turn is None:
                    raise RuntimeError("Project training session has no current turn.")
                return existing, turn, True
            temporary = InMemoryProjectTrainingRepository()
            session, turn, _ = temporary.create_session(
                owner_id,
                project,
                creation_id,
                target_role,
                goal,
                difficulty,
                max_turns,
                focus_topics,
            )
            self._insert_session(connection, session)
            self._insert_turn(connection, turn)
            return session, turn, False

    def get_session(self, owner_id: str, session_id: str):
        with self._connect() as connection:
            return self._get_session(connection, owner_id, session_id)

    def get_session_by_creation_id(self, owner_id: str, creation_id: str):
        with self._connect() as connection:
            return self._get_session_by_creation_id(connection, owner_id, creation_id)

    def list_sessions(self, owner_id: str, project_id: Optional[str] = None):
        query = "SELECT payload FROM project_training_sessions WHERE owner_id=?"
        parameters = [owner_id]
        if project_id is not None:
            query += " AND project_id=?"
            parameters.append(project_id)
        query += " ORDER BY created_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [_session(json.loads(row["payload"])) for row in rows]

    def get_turn(self, owner_id: str, turn_id: str):
        with self._connect() as connection:
            return self._get_turn(connection, owner_id, turn_id)

    def list_turns(self, owner_id: str, session_id: str):
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload FROM project_training_turns
                WHERE owner_id=? AND session_id=? ORDER BY sequence
                """,
                (owner_id, session_id),
            ).fetchall()
        return [_turn(json.loads(row["payload"])) for row in rows]

    def begin_answer(
        self,
        owner_id,
        session_id,
        turn_id,
        submission_id,
        answer_text,
        answer_source,
        submitted_at,
    ):
        with self._transaction() as connection:
            existing = self._get_answer_by_submission_id(
                connection, owner_id, submission_id
            )
            if existing is not None:
                if (
                    existing.session_id != session_id
                    or existing.turn_id != turn_id
                    or existing.answer_text != answer_text
                    or existing.answer_source != answer_source
                ):
                    raise ValueError("submission_id 已被另一个回答使用。")
                if existing.evaluation_status == ProjectTrainingEvaluationStatus.COMPLETED:
                    return existing, True
                if (
                    existing.evaluation_status == ProjectTrainingEvaluationStatus.PENDING
                    and not _answer_lease_expired(existing.submitted_at)
                ):
                    raise ProjectTrainingSubmissionInProgressError("这个回答正在评价中。")
            session = self._get_session(connection, owner_id, session_id)
            turn = self._get_turn(connection, owner_id, turn_id)
            if session is None or turn is None or turn.session_id != session_id:
                raise ValueError("当前项目训练题目不存在。")
            if (
                turn.status == ProjectTrainingTurnStatus.EVALUATING
                and turn.active_submission_started_at is not None
                and _answer_lease_expired(turn.active_submission_started_at)
            ):
                self._expire_active_answer(connection, owner_id, turn)
                turn.status = ProjectTrainingTurnStatus.PENDING
                turn.active_submission_id = None
                turn.active_submission_started_at = None
            if (
                session.status != ProjectTrainingSessionStatus.IN_PROGRESS
                or session.current_turn_id != turn_id
                or turn.status != ProjectTrainingTurnStatus.PENDING
            ):
                raise ValueError("这道题已经完成或不是当前题目。")
            if existing is None:
                answer = ProjectTrainingAnswer(
                    id=f"project_answer_{__import__('uuid').uuid4().hex}",
                    owner_id=owner_id,
                    session_id=session_id,
                    turn_id=turn_id,
                    submission_id=submission_id,
                    answer_text=answer_text,
                    answer_source=answer_source,
                    submitted_at=submitted_at,
                    evaluation_status=ProjectTrainingEvaluationStatus.PENDING,
                )
                connection.execute(
                    """
                    INSERT INTO project_training_answers(
                        owner_id, answer_id, session_id, turn_id, submission_id,
                        evaluation_status, submitted_at, payload
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        owner_id,
                        answer.id,
                        session_id,
                        turn_id,
                        submission_id,
                        answer.evaluation_status.value,
                        submitted_at.isoformat(),
                        _dump(answer),
                    ),
                )
            else:
                answer = existing
                answer.evaluation_status = ProjectTrainingEvaluationStatus.PENDING
                answer.submitted_at = submitted_at
                self._update_answer(connection, answer)
            turn.status = ProjectTrainingTurnStatus.EVALUATING
            turn.active_submission_id = submission_id
            turn.active_submission_started_at = submitted_at
            self._update_turn(connection, turn)
            return answer, False

    def complete_answer(self, answer, turn, session, next_turn, progress) -> None:
        with self._transaction() as connection:
            stored_turn = self._get_turn(connection, answer.owner_id, turn.id)
            stored_session = self._get_session(connection, answer.owner_id, session.id)
            if (
                stored_turn is None
                or stored_session is None
                or stored_turn.active_submission_id != answer.submission_id
                or stored_turn.status != ProjectTrainingTurnStatus.EVALUATING
                or stored_session.status != ProjectTrainingSessionStatus.IN_PROGRESS
                or stored_session.current_turn_id != turn.id
            ):
                raise ProjectTrainingSubmissionInProgressError(
                    "项目训练状态已变化，请刷新后重试。"
                )
            try:
                self._update_answer(connection, answer)
                self._update_turn(connection, turn)
                self._update_session(connection, session)
                connection.execute(
                    """
                    INSERT INTO project_topic_progress(owner_id, project_id, topic, payload)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(owner_id, project_id, topic)
                    DO UPDATE SET payload=excluded.payload
                    """,
                    (progress.owner_id, progress.project_id, progress.topic, _dump(progress)),
                )
                if next_turn is not None:
                    self._insert_turn(connection, next_turn)
            except sqlite3.IntegrityError as exc:
                raise ProjectTrainingSubmissionInProgressError(
                    "下一道项目题已经生成，请刷新页面。"
                ) from exc

    def fail_answer(self, answer) -> None:
        with self._transaction() as connection:
            stored = self._get_answer_by_submission_id(
                connection, answer.owner_id, answer.submission_id
            )
            if stored is None or stored.evaluation_status == ProjectTrainingEvaluationStatus.COMPLETED:
                return
            stored.evaluation_status = ProjectTrainingEvaluationStatus.FAILED
            self._update_answer(connection, stored)
            turn = self._get_turn(connection, answer.owner_id, answer.turn_id)
            if turn is not None and turn.active_submission_id == answer.submission_id:
                turn.status = ProjectTrainingTurnStatus.PENDING
                turn.active_submission_id = None
                turn.active_submission_started_at = None
                self._update_turn(connection, turn)

    def get_answer_by_submission_id(self, owner_id: str, submission_id: str):
        with self._connect() as connection:
            return self._get_answer_by_submission_id(connection, owner_id, submission_id)

    def list_answers(self, owner_id: str, session_id: str):
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload FROM project_training_answers
                WHERE owner_id=? AND session_id=? ORDER BY submitted_at
                """,
                (owner_id, session_id),
            ).fetchall()
        return [_answer(json.loads(row["payload"])) for row in rows]

    def get_topic_progress(self, owner_id: str, project_id: str, topic: str):
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload FROM project_topic_progress
                WHERE owner_id=? AND project_id=? AND topic=?
                """,
                (owner_id, project_id, topic),
            ).fetchone()
        return _progress(json.loads(row["payload"])) if row else None

    def close_session(self, owner_id, session_id, status, completed_at):
        with self._transaction() as connection:
            session = self._get_session(connection, owner_id, session_id)
            if session is None:
                return None
            turn = self._get_turn(
                connection, owner_id, session.current_turn_id or ""
            )
            if turn is not None and turn.status == ProjectTrainingTurnStatus.EVALUATING:
                raise ProjectTrainingSubmissionInProgressError(
                    "回答正在评价中，请等待评价完成后再结束训练。"
                )
            if session.status in {
                ProjectTrainingSessionStatus.COMPLETED,
                ProjectTrainingSessionStatus.ABANDONED,
            }:
                if session.status != status:
                    raise ValueError("已结束的项目训练不能变更状态。")
                return session
            session.status = status
            session.current_turn_id = None
            session.completed_at = completed_at
            self._update_session(connection, session)
            return session

    def save_summary(self, summary) -> None:
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO project_training_summaries(owner_id, session_id, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(owner_id, session_id) DO NOTHING
                """,
                (summary.owner_id, summary.session_id, _dump(summary)),
            )

    def get_summary(self, owner_id: str, session_id: str):
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload FROM project_training_summaries
                WHERE owner_id=? AND session_id=?
                """,
                (owner_id, session_id),
            ).fetchone()
        return _summary(json.loads(row["payload"])) if row else None

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS project_profiles (
                    owner_id TEXT NOT NULL, project_id TEXT NOT NULL,
                    current_version INTEGER NOT NULL, status TEXT NOT NULL,
                    updated_at TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(owner_id, project_id)
                );
                CREATE TABLE IF NOT EXISTS project_profile_versions (
                    owner_id TEXT NOT NULL, project_id TEXT NOT NULL,
                    version INTEGER NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(owner_id, project_id, version)
                );
                CREATE TABLE IF NOT EXISTS project_evidence (
                    owner_id TEXT NOT NULL, evidence_id TEXT NOT NULL,
                    project_id TEXT NOT NULL, project_version INTEGER NOT NULL,
                    evidence_order INTEGER NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(owner_id, evidence_id)
                );
                CREATE INDEX IF NOT EXISTS idx_project_evidence_version
                    ON project_evidence(owner_id, project_id, project_version, evidence_order);
                CREATE TABLE IF NOT EXISTS project_training_sessions (
                    owner_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    creation_id TEXT NOT NULL, project_id TEXT NOT NULL,
                    status TEXT NOT NULL, current_turn_id TEXT,
                    created_at TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(owner_id, session_id), UNIQUE(owner_id, creation_id)
                );
                CREATE TABLE IF NOT EXISTS project_training_turns (
                    owner_id TEXT NOT NULL, turn_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, sequence INTEGER NOT NULL,
                    status TEXT NOT NULL, active_submission_id TEXT, payload TEXT NOT NULL,
                    PRIMARY KEY(owner_id, turn_id), UNIQUE(owner_id, session_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS project_training_answers (
                    owner_id TEXT NOT NULL, answer_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, turn_id TEXT NOT NULL,
                    submission_id TEXT NOT NULL, evaluation_status TEXT NOT NULL,
                    submitted_at TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(owner_id, answer_id), UNIQUE(owner_id, submission_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS uq_project_active_answer_per_turn
                    ON project_training_answers(owner_id, turn_id)
                    WHERE evaluation_status != 'failed';
                CREATE TABLE IF NOT EXISTS project_topic_progress (
                    owner_id TEXT NOT NULL, project_id TEXT NOT NULL,
                    topic TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(owner_id, project_id, topic)
                );
                CREATE TABLE IF NOT EXISTS project_training_summaries (
                    owner_id TEXT NOT NULL, session_id TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(owner_id, session_id)
                );
                """
            )

    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _transaction(self):
        return _SQLiteTransaction(self._connect())

    def _store_project(self, connection, project: ProjectProfile, extra_evidence=None) -> None:
        payload = _dump(project)
        connection.execute(
            """
            INSERT INTO project_profiles(
                owner_id, project_id, current_version, status, updated_at, payload
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_id, project_id) DO UPDATE SET
                current_version=excluded.current_version, status=excluded.status,
                updated_at=excluded.updated_at, payload=excluded.payload
            """,
            (
                project.owner_id,
                project.id,
                project.version,
                project.status.value,
                project.updated_at.isoformat(),
                payload,
            ),
        )
        connection.execute(
            """
            INSERT INTO project_profile_versions(owner_id, project_id, version, payload)
            VALUES (?, ?, ?, ?)
            """,
            (project.owner_id, project.id, project.version, payload),
        )
        generated = build_project_evidence(project)
        additional = [
            replace(item, order=len(generated) + index)
            for index, item in enumerate(extra_evidence or [])
        ]
        for evidence in generated + additional:
            connection.execute(
                """
                INSERT INTO project_evidence(
                    owner_id, evidence_id, project_id, project_version,
                    evidence_order, payload
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence.owner_id,
                    evidence.id,
                    evidence.project_id,
                    evidence.project_version,
                    evidence.order,
                    _dump(evidence),
                ),
            )

    @staticmethod
    def _get_project(connection, owner_id, project_id, version=None):
        if version is None:
            row = connection.execute(
                "SELECT payload FROM project_profiles WHERE owner_id=? AND project_id=?",
                (owner_id, project_id),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT payload FROM project_profile_versions
                WHERE owner_id=? AND project_id=? AND version=?
                """,
                (owner_id, project_id, version),
            ).fetchone()
        return _project(json.loads(row["payload"])) if row else None

    @staticmethod
    def _get_project_by_discovery_job(connection, owner_id, discovery_job_id):
        if not discovery_job_id:
            return None
        rows = connection.execute(
            """
            SELECT payload FROM project_profile_versions
            WHERE owner_id=? ORDER BY version DESC
            """,
            (owner_id,),
        ).fetchall()
        for row in rows:
            project = _project(json.loads(row["payload"]))
            if project.source_discovery_job_id == discovery_job_id:
                return project
        return None

    @staticmethod
    def _list_project_evidence(connection, owner_id, project_id, version):
        rows = connection.execute(
            """
            SELECT payload FROM project_evidence
            WHERE owner_id=? AND project_id=? AND project_version=?
            ORDER BY evidence_order
            """, (owner_id, project_id, version)
        ).fetchall()
        result = []
        for row in rows:
            value = json.loads(row["payload"])
            value.setdefault("source_type", "profile_field")
            value.setdefault("source_path", "")
            value.setdefault("start_line", None)
            value.setdefault("end_line", None)
            value.setdefault("confidence", 1.0)
            value.setdefault("source_commit_sha", "")
            result.append(ProjectEvidence(**value))
        return result

    @staticmethod
    def _get_session(connection, owner_id, session_id):
        row = connection.execute(
            """
            SELECT payload FROM project_training_sessions
            WHERE owner_id=? AND session_id=?
            """,
            (owner_id, session_id),
        ).fetchone()
        return _session(json.loads(row["payload"])) if row else None

    @staticmethod
    def _get_session_by_creation_id(connection, owner_id, creation_id):
        row = connection.execute(
            """
            SELECT payload FROM project_training_sessions
            WHERE owner_id=? AND creation_id=?
            """,
            (owner_id, creation_id),
        ).fetchone()
        return _session(json.loads(row["payload"])) if row else None

    @staticmethod
    def _get_turn(connection, owner_id, turn_id):
        row = connection.execute(
            """
            SELECT payload FROM project_training_turns
            WHERE owner_id=? AND turn_id=?
            """,
            (owner_id, turn_id),
        ).fetchone()
        return _turn(json.loads(row["payload"])) if row else None

    @staticmethod
    def _get_answer_by_submission_id(connection, owner_id, submission_id):
        row = connection.execute(
            """
            SELECT payload FROM project_training_answers
            WHERE owner_id=? AND submission_id=?
            """,
            (owner_id, submission_id),
        ).fetchone()
        return _answer(json.loads(row["payload"])) if row else None

    @staticmethod
    def _insert_session(connection, session):
        connection.execute(
            """
            INSERT INTO project_training_sessions(
                owner_id, session_id, creation_id, project_id, status,
                current_turn_id, created_at, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.owner_id,
                session.id,
                session.creation_id,
                session.project_id,
                session.status.value,
                session.current_turn_id,
                session.created_at.isoformat(),
                _dump(session),
            ),
        )

    @staticmethod
    def _update_session(connection, session):
        connection.execute(
            """
            UPDATE project_training_sessions
            SET status=?, current_turn_id=?, payload=?
            WHERE owner_id=? AND session_id=?
            """,
            (
                session.status.value,
                session.current_turn_id,
                _dump(session),
                session.owner_id,
                session.id,
            ),
        )

    @staticmethod
    def _insert_turn(connection, turn):
        connection.execute(
            """
            INSERT INTO project_training_turns(
                owner_id, turn_id, session_id, sequence, status,
                active_submission_id, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                turn.owner_id,
                turn.id,
                turn.session_id,
                turn.sequence,
                turn.status.value,
                turn.active_submission_id,
                _dump(turn),
            ),
        )

    @staticmethod
    def _update_turn(connection, turn):
        connection.execute(
            """
            UPDATE project_training_turns
            SET status=?, active_submission_id=?, payload=?
            WHERE owner_id=? AND turn_id=?
            """,
            (
                turn.status.value,
                turn.active_submission_id,
                _dump(turn),
                turn.owner_id,
                turn.id,
            ),
        )

    @staticmethod
    def _update_answer(connection, answer):
        connection.execute(
            """
            UPDATE project_training_answers
            SET evaluation_status=?, submitted_at=?, payload=?
            WHERE owner_id=? AND answer_id=?
            """,
            (
                answer.evaluation_status.value,
                answer.submitted_at.isoformat(),
                _dump(answer),
                answer.owner_id,
                answer.id,
            ),
        )

    def _expire_active_answer(self, connection, owner_id, turn):
        if not turn.active_submission_id:
            return
        stale = self._get_answer_by_submission_id(
            connection, owner_id, turn.active_submission_id
        )
        if stale is not None:
            stale.evaluation_status = ProjectTrainingEvaluationStatus.FAILED
            self._update_answer(connection, stale)


class _SQLiteTransaction:
    def __init__(self, connection) -> None:
        self.connection = connection

    def __enter__(self):
        self.connection.execute("BEGIN IMMEDIATE")
        return self.connection

    def __exit__(self, exc_type, _exc, _traceback):
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()
        return False


def _project(value: dict) -> ProjectProfile:
    data = dict(value)
    data.setdefault("source_repository_url", "")
    data.setdefault("source_commit_sha", "")
    data.setdefault("source_discovery_job_id", "")
    data["status"] = ProjectProfileStatus(data["status"])
    data["created_at"] = datetime.fromisoformat(data["created_at"])
    data["updated_at"] = datetime.fromisoformat(data["updated_at"])
    return ProjectProfile(**data)


def _session(value: dict) -> ProjectTrainingSession:
    data = dict(value)
    data.setdefault("focus_topics", [])
    data["difficulty"] = ProjectTrainingDifficulty(data["difficulty"])
    data["status"] = ProjectTrainingSessionStatus(data["status"])
    for field in ("created_at", "started_at", "completed_at"):
        data[field] = datetime.fromisoformat(data[field]) if data.get(field) else None
    return ProjectTrainingSession(**data)


def _turn(value: dict) -> ProjectTrainingTurn:
    data = dict(value)
    data["status"] = ProjectTrainingTurnStatus(data["status"])
    for field in ("created_at", "completed_at", "active_submission_started_at"):
        data[field] = datetime.fromisoformat(data[field]) if data.get(field) else None
    return ProjectTrainingTurn(**data)


def _answer(value: dict) -> ProjectTrainingAnswer:
    data = dict(value)
    data["answer_source"] = ProjectTrainingAnswerSource(data["answer_source"])
    data["evaluation_status"] = ProjectTrainingEvaluationStatus(
        data["evaluation_status"]
    )
    data["submitted_at"] = datetime.fromisoformat(data["submitted_at"])
    return ProjectTrainingAnswer(**data)


def _progress(value: dict) -> ProjectTopicProgress:
    data = dict(value)
    data["last_practiced_at"] = (
        datetime.fromisoformat(data["last_practiced_at"])
        if data.get("last_practiced_at")
        else None
    )
    return ProjectTopicProgress(**data)


def _summary(value: dict) -> ProjectTrainingSummary:
    data = dict(value)
    data["created_at"] = datetime.fromisoformat(data["created_at"])
    return ProjectTrainingSummary(**data)


def _dump(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    return value
