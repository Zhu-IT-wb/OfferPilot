import json
import sqlite3
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional

from app.models.interview_knowledge import (
    KnowledgeAnswerSource,
    KnowledgeAssignment,
    KnowledgeAssignmentStatus,
    KnowledgeAssignmentType,
    KnowledgeAttempt,
    KnowledgeDeliveryType,
    KnowledgeDifficulty,
    KnowledgeMasteryStatus,
    KnowledgeProgress,
    KnowledgeQuestion,
    KnowledgeRubricKind,
    KnowledgeRubricPoint,
    KnowledgeSubscription,
)
from app.repositories.interview_knowledge_repository import (
    ConcurrentKnowledgeProgressUpdateError,
    DuplicateKnowledgeSubmissionError,
    KnowledgeAssignmentInProgressError,
)
from app.services.knowledge_catalog import load_knowledge_catalog


class SQLiteInterviewKnowledgeRepository:
    _ATTEMPT_LEASE = timedelta(minutes=15)

    def __init__(
        self,
        db_path: str,
        questions: Optional[List[KnowledgeQuestion]] = None,
        initialize_questions: bool = True,
        read_only: bool = False,
    ) -> None:
        self.db_path = Path(db_path)
        self._read_only = read_only
        if read_only:
            self._connect().close()
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        if initialize_questions:
            self._upsert_questions(
                load_knowledge_catalog().questions if questions is None else questions,
                only_if_empty=True,
            )

    def upsert_questions(self, questions: List[KnowledgeQuestion]) -> None:
        self._upsert_questions(questions)

    def _upsert_questions(
        self, questions: List[KnowledgeQuestion], *, only_if_empty: bool = False
    ) -> None:
        now = datetime.now().isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            # Keep the emptiness check and first seed in one transaction.
            if only_if_empty and connection.execute(
                "SELECT 1 FROM knowledge_questions LIMIT 1"
            ).fetchone():
                return
            connection.execute("UPDATE knowledge_questions SET enabled = 0")
            connection.executemany(
                """
                INSERT INTO knowledge_questions (
                    id, module_id, module_title, module_order, chapter_id, chapter_title,
                    chapter_order, question_order, prompt, difficulty, frequency,
                    short_reference_answer, full_reference_answer, rubric_points, hint,
                    source_title, source_url, source_chapter, source_page_start,
                    source_page_end, source_file, source_heading, content_hash, keywords,
                    enabled, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    module_id=excluded.module_id, module_title=excluded.module_title,
                    module_order=excluded.module_order, chapter_id=excluded.chapter_id,
                    chapter_title=excluded.chapter_title, chapter_order=excluded.chapter_order,
                    question_order=excluded.question_order, prompt=excluded.prompt,
                    difficulty=excluded.difficulty, frequency=excluded.frequency,
                    short_reference_answer=excluded.short_reference_answer,
                    full_reference_answer=excluded.full_reference_answer,
                    rubric_points=excluded.rubric_points, hint=excluded.hint,
                    source_title=excluded.source_title, source_url=excluded.source_url,
                    source_chapter=excluded.source_chapter,
                    source_page_start=excluded.source_page_start,
                    source_page_end=excluded.source_page_end,
                    source_file=excluded.source_file,
                    source_heading=excluded.source_heading,
                    content_hash=excluded.content_hash, keywords=excluded.keywords,
                    enabled=excluded.enabled,
                    updated_at=excluded.updated_at
                """,
                [
                    (
                        item.id, item.module_id, item.module_title, item.module_order,
                        item.chapter_id, item.chapter_title, item.chapter_order,
                        item.question_order, item.prompt, item.difficulty.value,
                        item.frequency, item.short_reference_answer,
                        item.full_reference_answer,
                        json.dumps([point.to_dict() for point in item.rubric_points], ensure_ascii=False),
                        item.hint, item.source_title, item.source_url, item.source_chapter,
                        item.source_page_start, item.source_page_end, item.source_file,
                        item.source_heading, item.content_hash,
                        json.dumps(item.keywords, ensure_ascii=False), int(item.enabled), now,
                    )
                    for item in questions
                ],
            )
            connection.commit()

    def list_questions(self) -> List[KnowledgeQuestion]:
        rows = self._fetch_all(
            """SELECT * FROM knowledge_questions WHERE enabled = 1
               ORDER BY module_order, chapter_order, question_order, id"""
        )
        return [self._question_from_row(row) for row in rows]

    def get_question(self, question_id: str) -> Optional[KnowledgeQuestion]:
        rows = self._fetch_all(
            "SELECT * FROM knowledge_questions WHERE id = ? AND enabled = 1",
            (question_id,),
        )
        return self._question_from_row(rows[0]) if rows else None

    def list_assignments(
        self, owner_id: str, assigned_on: Optional[date] = None
    ) -> List[KnowledgeAssignment]:
        query = "SELECT * FROM knowledge_assignments WHERE owner_id = ?"
        params: tuple[Any, ...] = (owner_id,)
        if assigned_on is not None:
            query += " AND assigned_on = ?"
            params += (assigned_on.isoformat(),)
        query += " ORDER BY assigned_on, created_at, id"
        return [self._assignment_from_row(row) for row in self._fetch_all(query, params)]

    def create_assignment(
        self,
        owner_id: str,
        question_id: str,
        assigned_on: date,
        assignment_type: KnowledgeAssignmentType,
        recommendation_reason: str,
    ) -> KnowledgeAssignment:
        assignment = KnowledgeAssignment(
            id=f"knowledge_assignment_{uuid.uuid4().hex}",
            owner_id=owner_id,
            question_id=question_id,
            assigned_on=assigned_on,
            assignment_type=assignment_type,
            recommendation_reason=recommendation_reason,
        )
        self._execute(
            """INSERT OR IGNORE INTO knowledge_assignments (
                id, owner_id, question_id, assigned_on, assignment_type,
                recommendation_reason, status, attempt_id, completed_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                assignment.id, owner_id, question_id, assigned_on.isoformat(),
                assignment_type.value, recommendation_reason, assignment.status.value,
                None, None, datetime.now().isoformat(),
            ),
        )
        return next(
            item for item in self.list_assignments(owner_id, assigned_on)
            if item.question_id == question_id
        )

    def save_assignment(self, assignment: KnowledgeAssignment) -> None:
        self._execute(
            """UPDATE knowledge_assignments SET assignment_type=?, recommendation_reason=?,
               status=?, attempt_id=?, completed_at=? WHERE id=? AND owner_id=?""",
            (
                assignment.assignment_type.value, assignment.recommendation_reason,
                assignment.status.value, assignment.attempt_id,
                assignment.completed_at.isoformat() if assignment.completed_at else None,
                assignment.id, assignment.owner_id,
            ),
        )

    def create_attempt(
        self,
        owner_id: str,
        question_id: str,
        assignment_id: Optional[str],
        answer_text: str,
        answer_source: KnowledgeAnswerSource,
        submitted_at: datetime,
        submission_id: Optional[str] = None,
    ) -> KnowledgeAttempt:
        attempt = KnowledgeAttempt(
            id=f"knowledge_attempt_{uuid.uuid4().hex}", owner_id=owner_id,
            question_id=question_id, assignment_id=assignment_id,
            answer_text=answer_text, answer_source=answer_source,
            submitted_at=submitted_at, submission_id=submission_id,
        )
        try:
            self._execute(
                """INSERT INTO knowledge_attempts (
                    id, owner_id, question_id, assignment_id, answer_text, answer_source,
                    submitted_at, submission_id, evaluation_status, evaluation_payload, score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    attempt.id, owner_id, question_id, assignment_id, answer_text,
                    answer_source.value, submitted_at.isoformat(), submission_id,
                    attempt.evaluation_status,
                    "{}", None,
                ),
            )
        except sqlite3.IntegrityError as exc:
            if submission_id and self.get_attempt_by_submission_id(owner_id, submission_id):
                raise DuplicateKnowledgeSubmissionError(
                    "Knowledge submission already exists."
                ) from exc
            raise
        return attempt

    def begin_attempt(
        self,
        owner_id: str,
        question_id: str,
        assignment_id: str,
        answer_text: str,
        answer_source: KnowledgeAnswerSource,
        submitted_at: datetime,
        submission_id: Optional[str] = None,
        existing_attempt_id: Optional[str] = None,
    ) -> KnowledgeAttempt:
        attempt = KnowledgeAttempt(
            id=existing_attempt_id or f"knowledge_attempt_{uuid.uuid4().hex}",
            owner_id=owner_id,
            question_id=question_id,
            assignment_id=assignment_id,
            answer_text=answer_text,
            answer_source=answer_source,
            submitted_at=submitted_at,
            submission_id=submission_id,
        )
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM knowledge_assignments WHERE id=? AND owner_id=?",
                    (assignment_id, owner_id),
                ).fetchone()
                if row is None or row["question_id"] != question_id:
                    raise ValueError("Knowledge assignment was not found.")
                active_attempt_id = row["active_attempt_id"]
                if active_attempt_id:
                    if not self._active_attempt_is_stale(
                        row["active_attempt_started_at"]
                    ):
                        raise KnowledgeAssignmentInProgressError(
                            "This knowledge assignment is already being evaluated."
                        )
                    connection.execute(
                        """UPDATE knowledge_attempts SET evaluation_status='failed'
                           WHERE id=? AND owner_id=? AND evaluation_status='pending'""",
                        (active_attempt_id, owner_id),
                    )
                    connection.execute(
                        """UPDATE knowledge_assignments
                           SET active_attempt_id=NULL, active_attempt_started_at=NULL
                           WHERE id=? AND owner_id=? AND active_attempt_id=?""",
                        (assignment_id, owner_id, active_attempt_id),
                    )
                if existing_attempt_id:
                    attempt_row = connection.execute(
                        "SELECT * FROM knowledge_attempts WHERE id=? AND owner_id=?",
                        (existing_attempt_id, owner_id),
                    ).fetchone()
                    if (
                        attempt_row is None
                        or attempt_row["assignment_id"] != assignment_id
                        or attempt_row["question_id"] != question_id
                        or attempt_row["evaluation_status"] not in {"pending", "failed"}
                    ):
                        raise ValueError("Knowledge attempt cannot be retried.")
                    if (
                        attempt_row["evaluation_status"] == "pending"
                        and not self._active_attempt_is_stale(
                            attempt_row["submitted_at"]
                        )
                    ):
                        raise KnowledgeAssignmentInProgressError(
                            "This answer is already being evaluated."
                        )
                    attempt.submission_id = attempt_row["submission_id"]
                    connection.execute(
                        """UPDATE knowledge_attempts SET submitted_at=?,
                           evaluation_status='pending', evaluation_payload='{}', score=NULL
                           WHERE id=? AND owner_id=?""",
                        (submitted_at.isoformat(), existing_attempt_id, owner_id),
                    )
                else:
                    connection.execute(
                        """INSERT INTO knowledge_attempts (
                            id, owner_id, question_id, assignment_id, answer_text,
                            answer_source, submitted_at, submission_id,
                            evaluation_status, evaluation_payload, score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            attempt.id,
                            owner_id,
                            question_id,
                            assignment_id,
                            answer_text,
                            answer_source.value,
                            submitted_at.isoformat(),
                            submission_id,
                            "pending",
                            "{}",
                            None,
                        ),
                    )
                updated = connection.execute(
                    """UPDATE knowledge_assignments
                       SET active_attempt_id=?, active_attempt_started_at=?
                       WHERE id=? AND owner_id=? AND active_attempt_id IS NULL""",
                    (
                        attempt.id,
                        datetime.now(timezone.utc).isoformat(),
                        assignment_id,
                        owner_id,
                    ),
                )
                if updated.rowcount != 1:
                    raise KnowledgeAssignmentInProgressError(
                        "This knowledge assignment is already being evaluated."
                    )
                connection.commit()
        except sqlite3.IntegrityError as exc:
            if submission_id and self.get_attempt_by_submission_id(owner_id, submission_id):
                raise DuplicateKnowledgeSubmissionError(
                    "Knowledge submission already exists."
                ) from exc
            raise
        return attempt

    def fail_attempt(self, attempt: KnowledgeAttempt) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """UPDATE knowledge_attempts SET evaluation_status='failed'
                   WHERE id=? AND owner_id=?""",
                (attempt.id, attempt.owner_id),
            )
            connection.execute(
                """UPDATE knowledge_assignments
                   SET active_attempt_id=NULL, active_attempt_started_at=NULL
                   WHERE id=? AND owner_id=? AND active_attempt_id=?""",
                (attempt.assignment_id, attempt.owner_id, attempt.id),
            )
            connection.commit()

    def complete_attempt(
        self,
        attempt: KnowledgeAttempt,
        assignment: KnowledgeAssignment,
        progress: KnowledgeProgress,
        expected_progress_attempt_count: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """SELECT 1 FROM knowledge_assignments
                   WHERE id=? AND owner_id=? AND active_attempt_id=?""",
                (assignment.id, assignment.owner_id, attempt.id),
            ).fetchone()
            if active is None:
                raise KnowledgeAssignmentInProgressError(
                    "This knowledge assignment is no longer owned by this attempt."
                )
            connection.execute(
                """UPDATE knowledge_attempts SET answer_text=?, answer_source=?,
                   evaluation_status=?, evaluation_payload=?, score=?
                   WHERE id=? AND owner_id=?""",
                (
                    attempt.answer_text,
                    attempt.answer_source.value,
                    attempt.evaluation_status,
                    json.dumps(attempt.evaluation_payload, ensure_ascii=False),
                    attempt.score,
                    attempt.id,
                    attempt.owner_id,
                ),
            )
            progress_values = (
                progress.owner_id,
                progress.question_id,
                progress.mastery_status.value,
                progress.mastery_score,
                progress.attempt_count,
                progress.high_score_streak,
                progress.last_score,
                progress.last_attempt_at.isoformat() if progress.last_attempt_at else None,
                progress.next_review_on.isoformat() if progress.next_review_on else None,
                json.dumps(progress.last_detected_gaps, ensure_ascii=False),
                datetime.now().isoformat(),
            )
            updated_progress = connection.execute(
                """INSERT INTO knowledge_progress (
                    owner_id, question_id, mastery_status, mastery_score, attempt_count,
                    high_score_streak, last_score, last_attempt_at, next_review_on,
                    last_detected_gaps, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner_id, question_id) DO UPDATE SET
                    mastery_status=excluded.mastery_status,
                    mastery_score=excluded.mastery_score,
                    attempt_count=excluded.attempt_count,
                    high_score_streak=excluded.high_score_streak,
                    last_score=excluded.last_score,
                    last_attempt_at=excluded.last_attempt_at,
                    next_review_on=excluded.next_review_on,
                    last_detected_gaps=excluded.last_detected_gaps,
                    updated_at=excluded.updated_at
                WHERE knowledge_progress.attempt_count=?""",
                (*progress_values, expected_progress_attempt_count),
            )
            if updated_progress.rowcount != 1:
                raise ConcurrentKnowledgeProgressUpdateError(
                    "Knowledge progress changed during evaluation."
                )
            completed = connection.execute(
                """UPDATE knowledge_assignments SET status=?, attempt_id=?, completed_at=?,
                   active_attempt_id=NULL, active_attempt_started_at=NULL
                   WHERE id=? AND owner_id=? AND active_attempt_id=?""",
                (
                    assignment.status.value,
                    assignment.attempt_id,
                    assignment.completed_at.isoformat() if assignment.completed_at else None,
                    assignment.id,
                    assignment.owner_id,
                    attempt.id,
                ),
            )
            if completed.rowcount != 1:
                raise KnowledgeAssignmentInProgressError(
                    "This knowledge assignment is no longer owned by this attempt."
                )
            connection.commit()

    def save_attempt(self, attempt: KnowledgeAttempt) -> None:
        self._execute(
            """UPDATE knowledge_attempts SET answer_text=?, answer_source=?,
               evaluation_status=?, evaluation_payload=?, score=?
               WHERE id=? AND owner_id=?""",
            (
                attempt.answer_text, attempt.answer_source.value, attempt.evaluation_status,
                json.dumps(attempt.evaluation_payload, ensure_ascii=False), attempt.score,
                attempt.id, attempt.owner_id,
            ),
        )

    def get_attempt(self, owner_id: str, attempt_id: str) -> Optional[KnowledgeAttempt]:
        rows = self._fetch_all(
            "SELECT * FROM knowledge_attempts WHERE owner_id=? AND id=?",
            (owner_id, attempt_id),
        )
        return self._attempt_from_row(rows[0]) if rows else None

    def get_attempt_by_submission_id(
        self, owner_id: str, submission_id: str
    ) -> Optional[KnowledgeAttempt]:
        rows = self._fetch_all(
            "SELECT * FROM knowledge_attempts WHERE owner_id=? AND submission_id=?",
            (owner_id, submission_id),
        )
        return self._attempt_from_row(rows[0]) if rows else None

    def list_attempts(
        self, owner_id: str, question_id: Optional[str] = None
    ) -> List[KnowledgeAttempt]:
        query = "SELECT * FROM knowledge_attempts WHERE owner_id=?"
        params: tuple[Any, ...] = (owner_id,)
        if question_id is not None:
            query += " AND question_id=?"
            params += (question_id,)
        query += " ORDER BY submitted_at, id"
        return [self._attempt_from_row(row) for row in self._fetch_all(query, params)]

    def get_progress(self, owner_id: str, question_id: str) -> Optional[KnowledgeProgress]:
        rows = self._fetch_all(
            "SELECT * FROM knowledge_progress WHERE owner_id=? AND question_id=?",
            (owner_id, question_id),
        )
        return self._progress_from_row(rows[0]) if rows else None

    def list_progress(self, owner_id: str) -> List[KnowledgeProgress]:
        return [
            self._progress_from_row(row)
            for row in self._fetch_all(
                "SELECT * FROM knowledge_progress WHERE owner_id=?", (owner_id,)
            )
        ]

    def save_progress(self, progress: KnowledgeProgress) -> None:
        self._execute(
            """INSERT INTO knowledge_progress (
                owner_id, question_id, mastery_status, mastery_score, attempt_count,
                high_score_streak, last_score, last_attempt_at, next_review_on,
                last_detected_gaps, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_id, question_id) DO UPDATE SET
                mastery_status=excluded.mastery_status, mastery_score=excluded.mastery_score,
                attempt_count=excluded.attempt_count,
                high_score_streak=excluded.high_score_streak, last_score=excluded.last_score,
                last_attempt_at=excluded.last_attempt_at,
                next_review_on=excluded.next_review_on,
                last_detected_gaps=excluded.last_detected_gaps,
                updated_at=excluded.updated_at""",
            (
                progress.owner_id, progress.question_id, progress.mastery_status.value,
                progress.mastery_score, progress.attempt_count, progress.high_score_streak,
                progress.last_score,
                progress.last_attempt_at.isoformat() if progress.last_attempt_at else None,
                progress.next_review_on.isoformat() if progress.next_review_on else None,
                json.dumps(progress.last_detected_gaps, ensure_ascii=False),
                datetime.now().isoformat(),
            ),
        )

    def save_subscription(self, subscription: KnowledgeSubscription) -> None:
        self._execute(
            """INSERT INTO knowledge_subscriptions (
                owner_id, feishu_open_id, enabled, timezone, morning_time, noon_time,
                evening_time, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_id) DO UPDATE SET
                feishu_open_id=excluded.feishu_open_id, enabled=excluded.enabled,
                timezone=excluded.timezone, morning_time=excluded.morning_time,
                noon_time=excluded.noon_time, evening_time=excluded.evening_time,
                updated_at=excluded.updated_at""",
            (
                subscription.owner_id, subscription.feishu_open_id,
                int(subscription.enabled), subscription.timezone,
                subscription.morning_time, subscription.noon_time,
                subscription.evening_time, datetime.now().isoformat(),
            ),
        )

    def get_subscription(self, owner_id: str) -> Optional[KnowledgeSubscription]:
        rows = self._fetch_all(
            "SELECT * FROM knowledge_subscriptions WHERE owner_id=?", (owner_id,)
        )
        return self._subscription_from_row(rows[0]) if rows else None

    def list_enabled_subscriptions(self) -> List[KnowledgeSubscription]:
        return [
            self._subscription_from_row(row)
            for row in self._fetch_all(
                "SELECT * FROM knowledge_subscriptions WHERE enabled=1"
            )
        ]

    def has_delivery(
        self, owner_id: str, delivery_on: date, delivery_type: KnowledgeDeliveryType
    ) -> bool:
        return bool(
            self._fetch_all(
                """SELECT 1 FROM knowledge_deliveries
                   WHERE owner_id=? AND delivery_on=? AND delivery_type=?""",
                (owner_id, delivery_on.isoformat(), delivery_type.value),
            )
        )

    def record_delivery(
        self, owner_id: str, delivery_on: date, delivery_type: KnowledgeDeliveryType
    ) -> None:
        self._execute(
            """INSERT OR IGNORE INTO knowledge_deliveries
               (owner_id, delivery_on, delivery_type, delivered_at) VALUES (?, ?, ?, ?)""",
            (owner_id, delivery_on.isoformat(), delivery_type.value, datetime.now().isoformat()),
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge_questions (
                    id TEXT PRIMARY KEY, module_id TEXT NOT NULL, module_title TEXT NOT NULL,
                    module_order INTEGER NOT NULL, chapter_id TEXT NOT NULL,
                    chapter_title TEXT NOT NULL, chapter_order INTEGER NOT NULL,
                    question_order INTEGER NOT NULL, prompt TEXT NOT NULL,
                    difficulty TEXT NOT NULL, frequency INTEGER NOT NULL,
                    short_reference_answer TEXT NOT NULL, full_reference_answer TEXT NOT NULL,
                    rubric_points TEXT NOT NULL, hint TEXT NOT NULL,
                    source_title TEXT NOT NULL, source_url TEXT NOT NULL,
                    source_chapter TEXT NOT NULL, source_page_start INTEGER,
                    source_page_end INTEGER, source_file TEXT NOT NULL DEFAULT '',
                    source_heading TEXT NOT NULL DEFAULT '',
                    content_hash TEXT NOT NULL DEFAULT '',
                    keywords TEXT NOT NULL DEFAULT '[]',
                    enabled INTEGER NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_assignments (
                    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, question_id TEXT NOT NULL,
                    assigned_on TEXT NOT NULL, assignment_type TEXT NOT NULL,
                    recommendation_reason TEXT NOT NULL, status TEXT NOT NULL,
                    attempt_id TEXT, completed_at TEXT, created_at TEXT NOT NULL,
                    active_attempt_id TEXT, active_attempt_started_at TEXT,
                    UNIQUE(owner_id, question_id, assigned_on)
                );
                CREATE INDEX IF NOT EXISTS idx_knowledge_assignments_owner_date
                    ON knowledge_assignments(owner_id, assigned_on);
                CREATE TABLE IF NOT EXISTS knowledge_attempts (
                    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, question_id TEXT NOT NULL,
                    assignment_id TEXT, answer_text TEXT NOT NULL, answer_source TEXT NOT NULL,
                    submitted_at TEXT NOT NULL, submission_id TEXT,
                    evaluation_status TEXT NOT NULL,
                    evaluation_payload TEXT NOT NULL, score INTEGER
                );
                CREATE TABLE IF NOT EXISTS knowledge_progress (
                    owner_id TEXT NOT NULL, question_id TEXT NOT NULL,
                    mastery_status TEXT NOT NULL, mastery_score INTEGER NOT NULL,
                    attempt_count INTEGER NOT NULL, high_score_streak INTEGER NOT NULL,
                    last_score INTEGER, last_attempt_at TEXT, next_review_on TEXT,
                    last_detected_gaps TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(owner_id, question_id)
                );
                CREATE TABLE IF NOT EXISTS knowledge_subscriptions (
                    owner_id TEXT PRIMARY KEY, feishu_open_id TEXT NOT NULL,
                    enabled INTEGER NOT NULL, timezone TEXT NOT NULL,
                    morning_time TEXT NOT NULL, noon_time TEXT NOT NULL,
                    evening_time TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_deliveries (
                    owner_id TEXT NOT NULL, delivery_on TEXT NOT NULL,
                    delivery_type TEXT NOT NULL, delivered_at TEXT NOT NULL,
                    PRIMARY KEY(owner_id, delivery_on, delivery_type)
                );
                """
            )
            attempt_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(knowledge_attempts)")
            }
            if "submission_id" not in attempt_columns:
                connection.execute(
                    "ALTER TABLE knowledge_attempts ADD COLUMN submission_id TEXT"
                )
            assignment_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(knowledge_assignments)")
            }
            assignment_column_migrations = {
                "active_attempt_id": "TEXT",
                "active_attempt_started_at": "TEXT",
            }
            for column, definition in assignment_column_migrations.items():
                if column not in assignment_columns:
                    connection.execute(
                        f"ALTER TABLE knowledge_assignments ADD COLUMN {column} {definition}"
                    )
            question_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(knowledge_questions)")
            }
            question_column_migrations = {
                "source_file": "TEXT NOT NULL DEFAULT ''",
                "source_heading": "TEXT NOT NULL DEFAULT ''",
                "content_hash": "TEXT NOT NULL DEFAULT ''",
                "keywords": "TEXT NOT NULL DEFAULT '[]'",
            }
            for column, definition in question_column_migrations.items():
                if column not in question_columns:
                    connection.execute(
                        f"ALTER TABLE knowledge_questions ADD COLUMN {column} {definition}"
                    )
            connection.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_attempt_submission
                   ON knowledge_attempts(owner_id, submission_id)
                   WHERE submission_id IS NOT NULL"""
            )
            connection.commit()

    def _execute(self, query: str, params: tuple[Any, ...]) -> None:
        with self._connect() as connection:
            connection.execute(query, params)
            connection.commit()

    def _fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> List[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(query, params).fetchall()

    def _connect(self) -> sqlite3.Connection:
        if self._read_only:
            connection = sqlite3.connect(
                f"{self.db_path.resolve().as_uri()}?mode=ro", uri=True
            )
        else:
            connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @classmethod
    def _active_attempt_is_stale(cls, started_at: Optional[str]) -> bool:
        if not started_at:
            return True
        try:
            parsed = datetime.fromisoformat(started_at)
        except ValueError:
            return True
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - parsed.astimezone(timezone.utc) >= cls._ATTEMPT_LEASE

    @staticmethod
    def _question_from_row(row: sqlite3.Row) -> KnowledgeQuestion:
        return KnowledgeQuestion(
            id=row["id"], module_id=row["module_id"], module_title=row["module_title"],
            module_order=row["module_order"], chapter_id=row["chapter_id"],
            chapter_title=row["chapter_title"], chapter_order=row["chapter_order"],
            question_order=row["question_order"], prompt=row["prompt"],
            difficulty=KnowledgeDifficulty(row["difficulty"]), frequency=row["frequency"],
            short_reference_answer=row["short_reference_answer"],
            full_reference_answer=row["full_reference_answer"], hint=row["hint"],
            rubric_points=[
                KnowledgeRubricPoint(
                    id=item["id"], kind=KnowledgeRubricKind(item["kind"]),
                    label=item["label"], description=item["description"],
                    weight=item.get("weight", 1),
                ) for item in json.loads(row["rubric_points"])
            ],
            source_title=row["source_title"], source_url=row["source_url"],
            source_chapter=row["source_chapter"],
            source_page_start=row["source_page_start"],
            source_page_end=row["source_page_end"], enabled=bool(row["enabled"]),
            source_file=row["source_file"], source_heading=row["source_heading"],
            content_hash=row["content_hash"], keywords=json.loads(row["keywords"]),
        )

    @staticmethod
    def _assignment_from_row(row: sqlite3.Row) -> KnowledgeAssignment:
        return KnowledgeAssignment(
            id=row["id"], owner_id=row["owner_id"], question_id=row["question_id"],
            assigned_on=date.fromisoformat(row["assigned_on"]),
            assignment_type=KnowledgeAssignmentType(row["assignment_type"]),
            recommendation_reason=row["recommendation_reason"],
            status=KnowledgeAssignmentStatus(row["status"]), attempt_id=row["attempt_id"],
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            active_attempt_id=row["active_attempt_id"],
            active_attempt_started_at=(
                datetime.fromisoformat(row["active_attempt_started_at"])
                if row["active_attempt_started_at"]
                else None
            ),
        )

    @staticmethod
    def _attempt_from_row(row: sqlite3.Row) -> KnowledgeAttempt:
        return KnowledgeAttempt(
            id=row["id"], owner_id=row["owner_id"], question_id=row["question_id"],
            assignment_id=row["assignment_id"], answer_text=row["answer_text"],
            answer_source=KnowledgeAnswerSource(row["answer_source"]),
            submitted_at=datetime.fromisoformat(row["submitted_at"]),
            submission_id=row["submission_id"],
            evaluation_status=row["evaluation_status"],
            evaluation_payload=json.loads(row["evaluation_payload"]), score=row["score"],
        )

    @staticmethod
    def _progress_from_row(row: sqlite3.Row) -> KnowledgeProgress:
        return KnowledgeProgress(
            owner_id=row["owner_id"], question_id=row["question_id"],
            mastery_status=KnowledgeMasteryStatus(row["mastery_status"]),
            mastery_score=row["mastery_score"], attempt_count=row["attempt_count"],
            high_score_streak=row["high_score_streak"], last_score=row["last_score"],
            last_attempt_at=datetime.fromisoformat(row["last_attempt_at"]) if row["last_attempt_at"] else None,
            next_review_on=date.fromisoformat(row["next_review_on"]) if row["next_review_on"] else None,
            last_detected_gaps=json.loads(row["last_detected_gaps"]),
        )

    @staticmethod
    def _subscription_from_row(row: sqlite3.Row) -> KnowledgeSubscription:
        return KnowledgeSubscription(
            owner_id=row["owner_id"], feishu_open_id=row["feishu_open_id"],
            enabled=bool(row["enabled"]), timezone=row["timezone"],
            morning_time=row["morning_time"], noon_time=row["noon_time"],
            evening_time=row["evening_time"],
        )
