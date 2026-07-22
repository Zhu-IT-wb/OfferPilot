import json
import sqlite3
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, List, Optional

from app.models.leetcode import (
    LeetCodeAssignment,
    LeetCodeAssignmentStatus,
    LeetCodeAssignmentType,
    LeetCodeDeliveryType,
    LeetCodeDifficulty,
    LeetCodePracticeResult,
    LeetCodeProblem,
    LeetCodeProgress,
    LeetCodeMasteryStatus,
    LeetCodeSubscription,
)
from app.services.leetcode_catalog import load_hot100_snapshot


class SQLiteLeetCodeRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self.upsert_problems(load_hot100_snapshot().problems)

    def upsert_problems(self, problems: List[LeetCodeProblem]) -> None:
        with self._connect() as connection:
            sources = sorted({problem.source for problem in problems})
            if sources:
                placeholders = ", ".join("?" for _ in sources)
                connection.execute(
                    f"UPDATE leetcode_problems SET active = 0 WHERE source IN ({placeholders})",
                    sources,
                )
            connection.executemany(
                """
                INSERT INTO leetcode_problems (
                    id, frontend_id, title_zh, title_en, slug, difficulty, topics,
                    category, category_order, problem_order, url, source, active, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(id) DO UPDATE SET
                    frontend_id = excluded.frontend_id,
                    title_zh = excluded.title_zh,
                    title_en = excluded.title_en,
                    slug = excluded.slug,
                    difficulty = excluded.difficulty,
                    topics = excluded.topics,
                    category = excluded.category,
                    category_order = excluded.category_order,
                    problem_order = excluded.problem_order,
                    url = excluded.url,
                    source = excluded.source,
                    active = 1,
                    updated_at = excluded.updated_at
                """,
                [
                    (
                        problem.id,
                        problem.frontend_id,
                        problem.title_zh,
                        problem.title_en,
                        problem.slug,
                        problem.difficulty.value,
                        json.dumps(problem.topics, ensure_ascii=False),
                        problem.category,
                        problem.category_order,
                        problem.problem_order,
                        problem.url,
                        problem.source,
                        datetime.now().isoformat(),
                    )
                    for problem in problems
                ],
            )
            connection.commit()

    def list_problems(self) -> List[LeetCodeProblem]:
        rows = self._fetch_all(
            """
            SELECT id, frontend_id, title_zh, title_en, slug, difficulty, topics,
                   category, category_order, problem_order, url, source
            FROM leetcode_problems
            WHERE active = 1
            ORDER BY category_order, problem_order
            """
        )
        return [self._problem_from_row(row) for row in rows]

    def list_assignments(
        self,
        owner_id: str,
        assigned_on: Optional[date] = None,
    ) -> List[LeetCodeAssignment]:
        query = """
            SELECT id, owner_id, problem_id, assigned_on, assignment_type,
                   recommendation_reason, status, result, postpone_count,
                   feedback_reminded_at, completed_at
            FROM leetcode_assignments
            WHERE owner_id = ?
        """
        parameters: tuple[Any, ...] = (owner_id,)
        if assigned_on is not None:
            query += " AND assigned_on = ?"
            parameters += (assigned_on.isoformat(),)
        query += " ORDER BY assigned_on, created_at, id"
        return [self._assignment_from_row(row) for row in self._fetch_all(query, parameters)]

    def create_assignment(
        self,
        owner_id: str,
        problem_id: str,
        assigned_on: date,
        assignment_type: LeetCodeAssignmentType,
        recommendation_reason: str,
    ) -> LeetCodeAssignment:
        assignment = LeetCodeAssignment(
            id=f"leetcode_assignment_{uuid.uuid4().hex}",
            owner_id=owner_id,
            problem_id=problem_id,
            assigned_on=assigned_on,
            assignment_type=assignment_type,
            recommendation_reason=recommendation_reason,
        )
        self._execute(
            """
            INSERT OR IGNORE INTO leetcode_assignments (
                id, owner_id, problem_id, assigned_on, assignment_type,
                recommendation_reason, status, result, postpone_count,
                feedback_reminded_at, completed_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                assignment.id,
                owner_id,
                problem_id,
                assigned_on.isoformat(),
                assignment_type.value,
                recommendation_reason,
                assignment.status.value,
                None,
                0,
                None,
                None,
                datetime.now().isoformat(),
            ),
        )
        return next(
            item
            for item in self.list_assignments(owner_id=owner_id, assigned_on=assigned_on)
            if item.problem_id == problem_id
        )

    def save_assignment(self, assignment: LeetCodeAssignment) -> None:
        self._execute(
            """
            UPDATE leetcode_assignments
            SET assignment_type = ?, recommendation_reason = ?, status = ?, result = ?,
                postpone_count = ?, feedback_reminded_at = ?, completed_at = ?
            WHERE id = ? AND owner_id = ?
            """,
            (
                assignment.assignment_type.value,
                assignment.recommendation_reason,
                assignment.status.value,
                assignment.result.value if assignment.result else None,
                assignment.postpone_count,
                assignment.feedback_reminded_at.isoformat() if assignment.feedback_reminded_at else None,
                assignment.completed_at.isoformat() if assignment.completed_at else None,
                assignment.id,
                assignment.owner_id,
            ),
        )

    def get_progress(self, owner_id: str, problem_id: str) -> Optional[LeetCodeProgress]:
        rows = self._fetch_all(
            """
            SELECT owner_id, problem_id, mastery_status, independent_streak, attempt_count,
                   last_result, last_practiced_on, next_review_on
            FROM leetcode_progress WHERE owner_id = ? AND problem_id = ?
            """,
            (owner_id, problem_id),
        )
        return self._progress_from_row(rows[0]) if rows else None

    def list_progress(self, owner_id: str) -> List[LeetCodeProgress]:
        rows = self._fetch_all(
            """
            SELECT owner_id, problem_id, mastery_status, independent_streak, attempt_count,
                   last_result, last_practiced_on, next_review_on
            FROM leetcode_progress WHERE owner_id = ?
            """,
            (owner_id,),
        )
        return [self._progress_from_row(row) for row in rows]

    def save_progress(self, progress: LeetCodeProgress) -> None:
        self._execute(
            """
            INSERT INTO leetcode_progress (
                owner_id, problem_id, mastery_status, independent_streak, attempt_count,
                last_result, last_practiced_on, next_review_on, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_id, problem_id) DO UPDATE SET
                mastery_status = excluded.mastery_status,
                independent_streak = excluded.independent_streak,
                attempt_count = excluded.attempt_count,
                last_result = excluded.last_result,
                last_practiced_on = excluded.last_practiced_on,
                next_review_on = excluded.next_review_on,
                updated_at = excluded.updated_at
            """,
            (
                progress.owner_id,
                progress.problem_id,
                progress.mastery_status.value,
                progress.independent_streak,
                progress.attempt_count,
                progress.last_result.value if progress.last_result else None,
                progress.last_practiced_on.isoformat() if progress.last_practiced_on else None,
                progress.next_review_on.isoformat() if progress.next_review_on else None,
                datetime.now().isoformat(),
            ),
        )

    def save_subscription(self, subscription: LeetCodeSubscription) -> None:
        self._execute(
            """
            INSERT INTO leetcode_subscriptions (
                owner_id, feishu_open_id, enabled, timezone, morning_time, noon_time,
                evening_time, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_id) DO UPDATE SET
                feishu_open_id = excluded.feishu_open_id,
                enabled = excluded.enabled,
                timezone = excluded.timezone,
                morning_time = excluded.morning_time,
                noon_time = excluded.noon_time,
                evening_time = excluded.evening_time,
                updated_at = excluded.updated_at
            """,
            (
                subscription.owner_id,
                subscription.feishu_open_id,
                int(subscription.enabled),
                subscription.timezone,
                subscription.morning_time,
                subscription.noon_time,
                subscription.evening_time,
                datetime.now().isoformat(),
            ),
        )

    def get_subscription(self, owner_id: str) -> Optional[LeetCodeSubscription]:
        rows = self._fetch_all(
            """
            SELECT owner_id, feishu_open_id, enabled, timezone, morning_time, noon_time,
                   evening_time
            FROM leetcode_subscriptions WHERE owner_id = ?
            """,
            (owner_id,),
        )
        return self._subscription_from_row(rows[0]) if rows else None

    def list_enabled_subscriptions(self) -> List[LeetCodeSubscription]:
        rows = self._fetch_all(
            """
            SELECT owner_id, feishu_open_id, enabled, timezone, morning_time, noon_time,
                   evening_time
            FROM leetcode_subscriptions WHERE enabled = 1 ORDER BY owner_id
            """
        )
        return [self._subscription_from_row(row) for row in rows]

    def has_delivery(
        self, owner_id: str, delivery_on: date, delivery_type: LeetCodeDeliveryType
    ) -> bool:
        rows = self._fetch_all(
            """
            SELECT 1 FROM leetcode_deliveries
            WHERE owner_id = ? AND delivery_on = ? AND delivery_type = ?
            """,
            (owner_id, delivery_on.isoformat(), delivery_type.value),
        )
        return bool(rows)

    def record_delivery(
        self, owner_id: str, delivery_on: date, delivery_type: LeetCodeDeliveryType
    ) -> None:
        self._execute(
            """
            INSERT OR IGNORE INTO leetcode_deliveries (
                owner_id, delivery_on, delivery_type, delivered_at
            ) VALUES (?, ?, ?, ?)
            """,
            (owner_id, delivery_on.isoformat(), delivery_type.value, datetime.now().isoformat()),
        )

    def get_delivery_attempts(
        self, owner_id: str, delivery_on: date, delivery_type: LeetCodeDeliveryType
    ) -> int:
        rows = self._fetch_all(
            """
            SELECT attempt_count FROM leetcode_delivery_attempts
            WHERE owner_id = ? AND delivery_on = ? AND delivery_type = ?
            """,
            (owner_id, delivery_on.isoformat(), delivery_type.value),
        )
        return int(rows[0]["attempt_count"]) if rows else 0

    def record_delivery_attempt(
        self, owner_id: str, delivery_on: date, delivery_type: LeetCodeDeliveryType
    ) -> None:
        self._execute(
            """
            INSERT INTO leetcode_delivery_attempts (
                owner_id, delivery_on, delivery_type, attempt_count, updated_at
            ) VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(owner_id, delivery_on, delivery_type) DO UPDATE SET
                attempt_count = leetcode_delivery_attempts.attempt_count + 1,
                updated_at = excluded.updated_at
            """,
            (owner_id, delivery_on.isoformat(), delivery_type.value, datetime.now().isoformat()),
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS leetcode_problems (
                    id TEXT PRIMARY KEY,
                    frontend_id TEXT NOT NULL UNIQUE,
                    title_zh TEXT NOT NULL,
                    title_en TEXT NOT NULL,
                    slug TEXT NOT NULL UNIQUE,
                    difficulty TEXT NOT NULL,
                    topics TEXT NOT NULL,
                    category TEXT NOT NULL,
                    category_order INTEGER NOT NULL,
                    problem_order INTEGER NOT NULL,
                    url TEXT NOT NULL,
                    source TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS leetcode_assignments (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    problem_id TEXT NOT NULL,
                    assigned_on TEXT NOT NULL,
                    assignment_type TEXT NOT NULL,
                    recommendation_reason TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    postpone_count INTEGER NOT NULL DEFAULT 0,
                    feedback_reminded_at TEXT,
                    completed_at TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(owner_id, problem_id, assigned_on)
                );
                CREATE INDEX IF NOT EXISTS idx_leetcode_assignments_owner_date
                    ON leetcode_assignments(owner_id, assigned_on);
                CREATE TABLE IF NOT EXISTS leetcode_progress (
                    owner_id TEXT NOT NULL,
                    problem_id TEXT NOT NULL,
                    mastery_status TEXT NOT NULL,
                    independent_streak INTEGER NOT NULL DEFAULT 0,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_result TEXT,
                    last_practiced_on TEXT,
                    next_review_on TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(owner_id, problem_id)
                );
                CREATE TABLE IF NOT EXISTS leetcode_subscriptions (
                    owner_id TEXT PRIMARY KEY,
                    feishu_open_id TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    timezone TEXT NOT NULL DEFAULT 'Asia/Shanghai',
                    morning_time TEXT NOT NULL DEFAULT '08:00',
                    noon_time TEXT NOT NULL DEFAULT '12:00',
                    evening_time TEXT NOT NULL DEFAULT '18:00',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS leetcode_deliveries (
                    owner_id TEXT NOT NULL,
                    delivery_on TEXT NOT NULL,
                    delivery_type TEXT NOT NULL,
                    delivered_at TEXT NOT NULL,
                    PRIMARY KEY(owner_id, delivery_on, delivery_type)
                );
                CREATE TABLE IF NOT EXISTS leetcode_delivery_attempts (
                    owner_id TEXT NOT NULL,
                    delivery_on TEXT NOT NULL,
                    delivery_type TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(owner_id, delivery_on, delivery_type)
                );
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(leetcode_subscriptions)")
            }
            if "noon_time" not in columns:
                connection.execute(
                    "ALTER TABLE leetcode_subscriptions "
                    "ADD COLUMN noon_time TEXT NOT NULL DEFAULT '12:00'"
                )
                connection.execute(
                    "UPDATE leetcode_subscriptions SET morning_time = '08:00' "
                    "WHERE morning_time = '09:00'"
                )
                connection.execute(
                    "UPDATE leetcode_subscriptions SET evening_time = '18:00' "
                    "WHERE evening_time = '21:00'"
                )
            connection.commit()

    def _execute(self, query: str, parameters: tuple[Any, ...]) -> None:
        with self._connect() as connection:
            connection.execute(query, parameters)
            connection.commit()

    def _fetch_all(self, query: str, parameters: tuple[Any, ...] = ()) -> List[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(query, parameters).fetchall()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _problem_from_row(row: sqlite3.Row) -> LeetCodeProblem:
        return LeetCodeProblem(
            id=row["id"],
            frontend_id=row["frontend_id"],
            title_zh=row["title_zh"],
            title_en=row["title_en"],
            slug=row["slug"],
            difficulty=LeetCodeDifficulty(row["difficulty"]),
            topics=json.loads(row["topics"]),
            category=row["category"],
            category_order=row["category_order"],
            problem_order=row["problem_order"],
            url=row["url"],
            source=row["source"],
        )

    @staticmethod
    def _assignment_from_row(row: sqlite3.Row) -> LeetCodeAssignment:
        return LeetCodeAssignment(
            id=row["id"],
            owner_id=row["owner_id"],
            problem_id=row["problem_id"],
            assigned_on=date.fromisoformat(row["assigned_on"]),
            assignment_type=LeetCodeAssignmentType(row["assignment_type"]),
            recommendation_reason=row["recommendation_reason"],
            status=LeetCodeAssignmentStatus(row["status"]),
            result=LeetCodePracticeResult(row["result"]) if row["result"] else None,
            postpone_count=row["postpone_count"],
            feedback_reminded_at=(
                datetime.fromisoformat(row["feedback_reminded_at"])
                if row["feedback_reminded_at"]
                else None
            ),
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        )

    @staticmethod
    def _progress_from_row(row: sqlite3.Row) -> LeetCodeProgress:
        return LeetCodeProgress(
            owner_id=row["owner_id"],
            problem_id=row["problem_id"],
            mastery_status=LeetCodeMasteryStatus(row["mastery_status"]),
            independent_streak=row["independent_streak"],
            attempt_count=row["attempt_count"],
            last_result=LeetCodePracticeResult(row["last_result"]) if row["last_result"] else None,
            last_practiced_on=date.fromisoformat(row["last_practiced_on"]) if row["last_practiced_on"] else None,
            next_review_on=date.fromisoformat(row["next_review_on"]) if row["next_review_on"] else None,
        )

    @staticmethod
    def _subscription_from_row(row: sqlite3.Row) -> LeetCodeSubscription:
        return LeetCodeSubscription(
            owner_id=row["owner_id"],
            feishu_open_id=row["feishu_open_id"],
            enabled=bool(row["enabled"]),
            timezone=row["timezone"],
            morning_time=row["morning_time"],
            noon_time=row["noon_time"],
            evening_time=row["evening_time"],
        )
