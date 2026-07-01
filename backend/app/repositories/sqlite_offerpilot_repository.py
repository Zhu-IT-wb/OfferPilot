import json
import sqlite3
from pathlib import Path
from typing import Any, List, Optional

from app.models.application import (
    Application,
    ApplicationStatus,
    application_status_from_round,
)
from app.models.interview_review import InterviewReview, InterviewReviewStatus
from app.models.interview_schedule import InterviewSchedule, InterviewScheduleStatus
from app.models.task import Task, TaskPriority, TaskStatus, TaskType
from app.repositories.offerpilot_repository import _default_tasks


class SQLiteOfferPilotRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def get_runtime_setting(self, key: str) -> Optional[str]:
        value = self._fetch_value(
            "SELECT value FROM runtime_settings WHERE key = ?",
            (key,),
        )
        return value if isinstance(value, str) else None

    def set_runtime_setting(self, key: str, value: str) -> None:
        self._execute(
            """
            INSERT INTO runtime_settings (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def list_today_tasks(self) -> List[Task]:
        active_statuses = (
            TaskStatus.PENDING.value,
            TaskStatus.IN_PROGRESS.value,
            TaskStatus.POSTPONED.value,
        )
        placeholders = ", ".join("?" for _ in active_statuses)
        rows = self._fetch_all(
            f"""
            SELECT id, title, task_type, status, priority
            FROM tasks
            WHERE status IN ({placeholders})
            ORDER BY id
            """,
            active_statuses,
        )
        return [self._task_from_row(row) for row in rows]

    def create_application(
        self,
        company: str,
        role: str,
        interview_time: Optional[str] = None,
        round_name: Optional[str] = None,
        jd_keywords: Optional[List[str]] = None,
    ) -> Application:
        application = Application(
            id=self._next_id("applications", "app"),
            company=company,
            role=role,
            status=application_status_from_round(round_name),
            interview_time=interview_time,
            round=round_name,
            jd_keywords=jd_keywords or [],
        )
        self._execute(
            """
            INSERT INTO applications (
                id, company, role, status, interview_time, round, jd_keywords
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                application.id,
                application.company,
                application.role,
                application.status.value,
                application.interview_time,
                application.round,
                json.dumps(application.jd_keywords, ensure_ascii=False),
            ),
        )
        return application

    def list_applications(self, company: Optional[str] = None) -> List[Application]:
        rows = self._fetch_all(
            """
            SELECT id, company, role, status, interview_time, round, jd_keywords
            FROM applications
            ORDER BY id
            """
        )
        applications = [self._application_from_row(row) for row in rows]
        if not company:
            return applications

        normalized_company = self._normalize(company)
        return [
            application
            for application in applications
            if normalized_company in self._normalize(application.company)
        ]

    def update_application(
        self,
        company: str,
        status: Optional[ApplicationStatus] = None,
        interview_time: Optional[str] = None,
        round_name: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Optional[Application]:
        application = self._find_application(company)
        if application is None:
            return None

        if status is not None:
            application.status = status
        if interview_time is not None:
            application.interview_time = interview_time
        if round_name is not None:
            application.round = round_name
        if role is not None:
            application.role = role

        self._execute(
            """
            UPDATE applications
            SET role = ?, status = ?, interview_time = ?, round = ?
            WHERE id = ?
            """,
            (
                application.role,
                application.status.value,
                application.interview_time,
                application.round,
                application.id,
            ),
        )
        return application

    def create_interview_schedule(
        self,
        company: str,
        round_name: str,
        application_id: Optional[str] = None,
        role: Optional[str] = None,
        start_time: Optional[str] = None,
        start_at: Optional[str] = None,
        reminder_minutes: int = 30,
        raw_message: str = "",
    ) -> InterviewSchedule:
        schedule = InterviewSchedule(
            id=self._next_id("interview_schedules", "schedule"),
            application_id=application_id,
            company=company,
            role=role,
            round=round_name,
            start_time=start_time,
            start_at=start_at,
            reminder_minutes=reminder_minutes,
            raw_message=raw_message,
        )
        self._execute(
            """
            INSERT INTO interview_schedules (
                id, application_id, company, role, round, start_time, start_at,
                reminder_minutes, status, calendar_event_id, raw_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                schedule.id,
                schedule.application_id,
                schedule.company,
                schedule.role,
                schedule.round,
                schedule.start_time,
                schedule.start_at,
                schedule.reminder_minutes,
                schedule.status.value,
                schedule.calendar_event_id,
                schedule.raw_message,
            ),
        )
        return schedule

    def list_interview_schedules(self, company: Optional[str] = None) -> List[InterviewSchedule]:
        rows = self._fetch_all(
            """
            SELECT id, application_id, company, role, round, start_time, start_at,
                   reminder_minutes, status, calendar_event_id, raw_message
            FROM interview_schedules
            ORDER BY id
            """
        )
        schedules = [self._interview_schedule_from_row(row) for row in rows]
        if not company:
            return schedules

        normalized_company = self._normalize(company)
        return [
            schedule
            for schedule in schedules
            if normalized_company in self._normalize(schedule.company)
        ]

    def update_interview_schedule_calendar_event(
        self,
        schedule_id: str,
        calendar_event_id: str,
    ) -> Optional[InterviewSchedule]:
        schedules = [
            schedule
            for schedule in self.list_interview_schedules()
            if schedule.id == schedule_id
        ]
        if not schedules:
            return None

        schedule = schedules[0]
        schedule.calendar_event_id = calendar_event_id
        self._execute(
            """
            UPDATE interview_schedules
            SET calendar_event_id = ?
            WHERE id = ?
            """,
            (calendar_event_id, schedule_id),
        )
        return schedule

    def complete_task(
        self,
        task_title: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Optional[Task]:
        return self._update_task_status(
            task_title=task_title,
            task_type=task_type,
            status=TaskStatus.PASSED,
        )

    def postpone_task(
        self,
        task_title: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Optional[Task]:
        return self._update_task_status(
            task_title=task_title,
            task_type=task_type,
            status=TaskStatus.POSTPONED,
        )

    def create_interview_review(
        self,
        company: Optional[str] = None,
        round_name: Optional[str] = None,
        topics: Optional[List[str]] = None,
        raw_message: str = "",
    ) -> InterviewReview:
        review = InterviewReview(
            id=self._next_id("interview_reviews", "review"),
            company=company,
            round=round_name,
            topics=topics or [],
            raw_message=raw_message,
        )
        self._execute(
            """
            INSERT INTO interview_reviews (
                id, company, round, topics, raw_message, status
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                review.id,
                review.company,
                review.round,
                json.dumps(review.topics, ensure_ascii=False),
                review.raw_message,
                review.status.value,
            ),
        )
        return review

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS applications (
                    id TEXT PRIMARY KEY,
                    company TEXT NOT NULL,
                    role TEXT NOT NULL,
                    status TEXT NOT NULL,
                    interview_time TEXT,
                    round TEXT,
                    jd_keywords TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS interview_reviews (
                    id TEXT PRIMARY KEY,
                    company TEXT,
                    round TEXT,
                    topics TEXT NOT NULL DEFAULT '[]',
                    raw_message TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS interview_schedules (
                    id TEXT PRIMARY KEY,
                    application_id TEXT,
                    company TEXT NOT NULL,
                    role TEXT,
                    round TEXT NOT NULL,
                    start_time TEXT,
                    start_at TEXT,
                    reminder_minutes INTEGER NOT NULL DEFAULT 30,
                    status TEXT NOT NULL,
                    calendar_event_id TEXT,
                    raw_message TEXT NOT NULL DEFAULT ''
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.commit()

        self._ensure_column("interview_schedules", "start_at", "TEXT")
        self._seed_default_tasks()

    def _ensure_column(self, table: str, column: str, column_type: str) -> None:
        rows = self._fetch_all(f"PRAGMA table_info({table})")
        column_names = {row["name"] for row in rows}
        if column in column_names:
            return

        self._execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}", ())

    def _seed_default_tasks(self) -> None:
        task_count = self._fetch_value("SELECT COUNT(*) FROM tasks")
        if task_count:
            return

        for task in _default_tasks():
            self._execute(
                """
                INSERT INTO tasks (id, title, task_type, status, priority)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    task.id,
                    task.title,
                    task.task_type.value,
                    task.status.value,
                    task.priority.value,
                ),
            )

    def _update_task_status(
        self,
        task_title: Optional[str],
        task_type: Optional[str],
        status: TaskStatus,
    ) -> Optional[Task]:
        task = self._find_task(task_title=task_title, task_type=task_type)
        if task is None:
            return None

        self._execute("UPDATE tasks SET status = ? WHERE id = ?", (status.value, task.id))
        task.status = status
        return task

    def _find_task(
        self,
        task_title: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Optional[Task]:
        rows = self._fetch_all(
            """
            SELECT id, title, task_type, status, priority
            FROM tasks
            WHERE status != ?
            ORDER BY id
            """,
            (TaskStatus.PASSED.value,),
        )
        tasks = [self._task_from_row(row) for row in rows]

        for task in tasks:
            if task_title and self._normalize(task_title) in self._normalize(task.title):
                return task

        if task_type:
            for task in tasks:
                if task.task_type.value == task_type:
                    return task

        return None

    def _find_application(self, company: str) -> Optional[Application]:
        matches = self.list_applications(company=company)
        return matches[-1] if matches else None

    def _next_id(self, table: str, prefix: str) -> str:
        count = self._fetch_value(f"SELECT COUNT(*) FROM {table}")
        return f"{prefix}_{count + 1}"

    def _fetch_value(self, query: str, parameters: tuple[Any, ...] = ()) -> Any:
        with self._connect() as connection:
            row = connection.execute(query, parameters).fetchone()
        return row[0] if row is not None else None

    def _fetch_all(self, query: str, parameters: tuple[Any, ...] = ()) -> List[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(query, parameters).fetchall()

    def _execute(self, query: str, parameters: tuple[Any, ...]) -> None:
        with self._connect() as connection:
            connection.execute(query, parameters)
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _task_from_row(row: sqlite3.Row) -> Task:
        return Task(
            id=row["id"],
            title=row["title"],
            task_type=TaskType(row["task_type"]),
            status=TaskStatus(row["status"]),
            priority=TaskPriority(row["priority"]),
        )

    @staticmethod
    def _application_from_row(row: sqlite3.Row) -> Application:
        return Application(
            id=row["id"],
            company=row["company"],
            role=row["role"],
            status=ApplicationStatus(row["status"]),
            interview_time=row["interview_time"],
            round=row["round"],
            jd_keywords=json.loads(row["jd_keywords"] or "[]"),
        )

    @staticmethod
    def _interview_review_from_row(row: sqlite3.Row) -> InterviewReview:
        return InterviewReview(
            id=row["id"],
            company=row["company"],
            round=row["round"],
            topics=json.loads(row["topics"] or "[]"),
            raw_message=row["raw_message"],
            status=InterviewReviewStatus(row["status"]),
        )

    @staticmethod
    def _interview_schedule_from_row(row: sqlite3.Row) -> InterviewSchedule:
        return InterviewSchedule(
            id=row["id"],
            application_id=row["application_id"],
            company=row["company"],
            role=row["role"],
            round=row["round"],
            start_time=row["start_time"],
            start_at=row["start_at"],
            reminder_minutes=row["reminder_minutes"],
            status=InterviewScheduleStatus(row["status"]),
            calendar_event_id=row["calendar_event_id"],
            raw_message=row["raw_message"],
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return value.replace(" ", "").lower()
