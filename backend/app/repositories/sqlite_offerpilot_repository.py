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


_PLANNED_APPLICATION_MIGRATION_SETTING = "migration.application_planned_to_submitted.v1"


# 使用 SQLite 持久化 OfferPilot 业务数据。
class SQLiteOfferPilotRepository:
    # 初始化当前组件所需的依赖和配置。
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    # 获取 runtime setting。
    def get_runtime_setting(self, key: str) -> Optional[str]:
        value = self._fetch_value(
            "SELECT value FROM runtime_settings WHERE key = ?",
            (key,),
        )
        return value if isinstance(value, str) else None

    # 保存或更新 runtime setting。
    def set_runtime_setting(self, key: str, value: str) -> None:
        self._execute(
            """
            INSERT INTO runtime_settings (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    # 查询并格式化今天的秋招任务。
    def list_today_tasks(self, owner_id: str = "local_user") -> List[Task]:
        owner_id = self._normalize_owner_id(owner_id)
        self._ensure_default_tasks_for_owner(owner_id)
        active_statuses = (
            TaskStatus.PENDING.value,
            TaskStatus.IN_PROGRESS.value,
            TaskStatus.POSTPONED.value,
        )
        placeholders = ", ".join("?" for _ in active_statuses)
        rows = self._fetch_all(
            f"""
            SELECT id, title, task_type, status, priority, owner_id
            FROM tasks
            WHERE owner_id = ? AND status IN ({placeholders})
            ORDER BY id
            """,
            (owner_id, *active_statuses),
        )
        return [self._task_from_row(row) for row in rows]

    # 创建投递记录，并按需同步面试日程和飞书多维表格。
    def create_application(
        self,
        company: str,
        role: str,
        interview_time: Optional[str] = None,
        round_name: Optional[str] = None,
        jd_keywords: Optional[List[str]] = None,
        owner_id: str = "local_user",
    ) -> Application:
        owner_id = self._normalize_owner_id(owner_id)
        application = Application(
            id=self._next_id("applications", "app"),
            company=company,
            role=role,
            owner_id=owner_id,
            status=(
                application_status_from_round(round_name)
                if round_name
                else ApplicationStatus.SUBMITTED
            ),
            interview_time=interview_time,
            round=round_name,
            jd_keywords=jd_keywords or [],
        )
        self._execute(
            """
            INSERT INTO applications (
                id, owner_id, company, role, status, interview_time, round, jd_keywords
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                application.id,
                application.owner_id,
                application.company,
                application.role,
                application.status.value,
                application.interview_time,
                application.round,
                json.dumps(application.jd_keywords, ensure_ascii=False),
            ),
        )
        return application

    # 查询 applications 列表。
    def list_applications(
        self,
        company: Optional[str] = None,
        owner_id: str = "local_user",
    ) -> List[Application]:
        owner_id = self._normalize_owner_id(owner_id)
        self._claim_legacy_rows_for_owner("applications", owner_id)
        rows = self._fetch_all(
            """
            SELECT id, owner_id, company, role, status, interview_time, round, jd_keywords
            FROM applications
            WHERE owner_id = ?
            ORDER BY id
            """,
            (owner_id,),
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

    # 更新投递进度，并按需同步日历和多维表格。
    def update_application(
        self,
        company: str,
        status: Optional[ApplicationStatus] = None,
        interview_time: Optional[str] = None,
        round_name: Optional[str] = None,
        role: Optional[str] = None,
        owner_id: str = "local_user",
    ) -> Optional[Application]:
        owner_id = self._normalize_owner_id(owner_id)
        application = self._find_application(company, owner_id=owner_id)
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
            WHERE id = ? AND owner_id = ?
            """,
            (
                application.role,
                application.status.value,
                application.interview_time,
                application.round,
                application.id,
                owner_id,
            ),
        )
        return application

    # 更新 application by id。
    def update_application_by_id(
        self,
        application_id: str,
        company: Optional[str] = None,
        role: Optional[str] = None,
        status: Optional[ApplicationStatus] = None,
        interview_time: Optional[str] = None,
        round_name: Optional[str] = None,
        jd_keywords: Optional[List[str]] = None,
        clear_interview_fields: bool = False,
        owner_id: str = "local_user",
    ) -> Optional[Application]:
        owner_id = self._normalize_owner_id(owner_id)
        application = self._find_application_by_id(application_id, owner_id=owner_id)
        if application is None:
            return None

        if company is not None:
            application.company = company
        if role is not None:
            application.role = role
        if status is not None:
            application.status = status
        if interview_time is not None:
            application.interview_time = interview_time
        if round_name is not None:
            application.round = round_name
        if jd_keywords is not None:
            application.jd_keywords = list(jd_keywords)
        if clear_interview_fields:
            application.interview_time = None
            application.round = None

        self._execute(
            """
            UPDATE applications
            SET company = ?, role = ?, status = ?, interview_time = ?, round = ?, jd_keywords = ?
            WHERE id = ? AND owner_id = ?
            """,
            (
                application.company,
                application.role,
                application.status.value,
                application.interview_time,
                application.round,
                json.dumps(application.jd_keywords, ensure_ascii=False),
                application.id,
                owner_id,
            ),
        )
        return application

    # 创建 interview schedule。
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
        owner_id: str = "local_user",
    ) -> InterviewSchedule:
        owner_id = self._normalize_owner_id(owner_id)
        schedule = InterviewSchedule(
            id=self._next_id("interview_schedules", "schedule"),
            owner_id=owner_id,
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
                id, owner_id, application_id, company, role, round, start_time, start_at,
                reminder_minutes, status, calendar_event_id, raw_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                schedule.id,
                schedule.owner_id,
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

    # 查询 interview schedules 列表。
    def list_interview_schedules(
        self,
        company: Optional[str] = None,
        owner_id: str = "local_user",
    ) -> List[InterviewSchedule]:
        owner_id = self._normalize_owner_id(owner_id)
        self._claim_legacy_rows_for_owner("interview_schedules", owner_id)
        rows = self._fetch_all(
            """
            SELECT id, owner_id, application_id, company, role, round, start_time, start_at,
                   reminder_minutes, status, calendar_event_id, raw_message
            FROM interview_schedules
            WHERE owner_id = ?
            ORDER BY id
            """,
            (owner_id,),
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

    # 更新 interview schedule calendar event。
    def update_interview_schedule_calendar_event(
        self,
        schedule_id: str,
        calendar_event_id: Optional[str],
        owner_id: str = "local_user",
    ) -> Optional[InterviewSchedule]:
        owner_id = self._normalize_owner_id(owner_id)
        schedules = [
            schedule
            for schedule in self.list_interview_schedules(owner_id=owner_id)
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
            WHERE id = ? AND owner_id = ?
            """,
            (calendar_event_id, schedule_id, owner_id),
        )
        return schedule

    # 按稳定 ID 更新面试安排，保留原日历事件关联。
    def update_interview_schedule(
        self,
        schedule_id: str,
        start_time: Optional[str] = None,
        start_at: Optional[str] = None,
        round_name: Optional[str] = None,
        reminder_minutes: Optional[int] = None,
        owner_id: str = "local_user",
    ) -> Optional[InterviewSchedule]:
        owner_id = self._normalize_owner_id(owner_id)
        schedules = [
            schedule
            for schedule in self.list_interview_schedules(owner_id=owner_id)
            if schedule.id == schedule_id
        ]
        if not schedules:
            return None

        schedule = schedules[0]
        if start_time is not None:
            schedule.start_time = start_time
        if start_at is not None:
            schedule.start_at = start_at
        if round_name is not None:
            schedule.round = round_name
        if reminder_minutes is not None:
            schedule.reminder_minutes = reminder_minutes
        schedule.status = InterviewScheduleStatus.SCHEDULED
        self._execute(
            """
            UPDATE interview_schedules
            SET round = ?, start_time = ?, start_at = ?, reminder_minutes = ?, status = ?
            WHERE id = ? AND owner_id = ?
            """,
            (
                schedule.round,
                schedule.start_time,
                schedule.start_at,
                schedule.reminder_minutes,
                schedule.status.value,
                schedule.id,
                owner_id,
            ),
        )
        return schedule

    # 按稳定 ID 将面试安排标记为已取消。
    def cancel_interview_schedule(
        self,
        schedule_id: str,
        owner_id: str = "local_user",
    ) -> Optional[InterviewSchedule]:
        owner_id = self._normalize_owner_id(owner_id)
        schedules = [
            schedule
            for schedule in self.list_interview_schedules(owner_id=owner_id)
            if schedule.id == schedule_id
        ]
        if not schedules:
            return None

        schedule = schedules[0]
        schedule.status = InterviewScheduleStatus.CANCELLED
        self._execute(
            """
            UPDATE interview_schedules
            SET status = ?
            WHERE id = ? AND owner_id = ?
            """,
            (schedule.status.value, schedule.id, owner_id),
        )
        return schedule

    # 将指定任务标记为已完成。
    def complete_task(
        self,
        task_title: Optional[str] = None,
        task_type: Optional[str] = None,
        owner_id: str = "local_user",
    ) -> Optional[Task]:
        return self._update_task_status(
            task_title=task_title,
            task_type=task_type,
            owner_id=owner_id,
            status=TaskStatus.PASSED,
        )

    # 将指定任务延期处理。
    def postpone_task(
        self,
        task_title: Optional[str] = None,
        task_type: Optional[str] = None,
        owner_id: str = "local_user",
    ) -> Optional[Task]:
        return self._update_task_status(
            task_title=task_title,
            task_type=task_type,
            owner_id=owner_id,
            status=TaskStatus.POSTPONED,
        )

    # 记录一次面试复盘。
    def create_interview_review(
        self,
        company: Optional[str] = None,
        round_name: Optional[str] = None,
        topics: Optional[List[str]] = None,
        raw_message: str = "",
        owner_id: str = "local_user",
    ) -> InterviewReview:
        owner_id = self._normalize_owner_id(owner_id)
        review = InterviewReview(
            id=self._next_id("interview_reviews", "review"),
            owner_id=owner_id,
            company=company,
            round=round_name,
            topics=topics or [],
            raw_message=raw_message,
        )
        self._execute(
            """
            INSERT INTO interview_reviews (
                id, owner_id, company, round, topics, raw_message, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                review.id,
                review.owner_id,
                review.company,
                review.round,
                json.dumps(review.topics, ensure_ascii=False),
                review.raw_message,
                review.status.value,
            ),
        )
        return review

    # 处理 initialize 相关逻辑。
    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL DEFAULT 'local_user',
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
                    owner_id TEXT NOT NULL DEFAULT 'local_user',
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
                    owner_id TEXT NOT NULL DEFAULT 'local_user',
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
                    owner_id TEXT NOT NULL DEFAULT 'local_user',
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

        self._ensure_column("tasks", "owner_id", "TEXT NOT NULL DEFAULT 'local_user'")
        self._ensure_column("applications", "owner_id", "TEXT NOT NULL DEFAULT 'local_user'")
        self._ensure_column("interview_reviews", "owner_id", "TEXT NOT NULL DEFAULT 'local_user'")
        self._ensure_column("interview_schedules", "owner_id", "TEXT NOT NULL DEFAULT 'local_user'")
        self._ensure_column("interview_schedules", "start_at", "TEXT")
        self._migrate_confirmed_planned_applications()
        self._remove_legacy_leetcode_sample_task()
        self._ensure_indexes()
        self._seed_default_tasks()

    # 确保 column 存在或已配置。
    def _ensure_column(self, table: str, column: str, column_type: str) -> None:
        rows = self._fetch_all(f"PRAGMA table_info({table})")
        column_names = {row["name"] for row in rows}
        if column in column_names:
            return

        self._execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}", ())

    # 旧版本只在用户确认后建记录，因此持久化的 planned 实际都已完成投递。
    def _migrate_confirmed_planned_applications(self) -> None:
        if self.get_runtime_setting(_PLANNED_APPLICATION_MIGRATION_SETTING) == "complete":
            return
        self._execute(
            "UPDATE applications SET status = ? WHERE status = ?",
            (ApplicationStatus.SUBMITTED.value, ApplicationStatus.PLANNED.value),
        )
        self.set_runtime_setting(_PLANNED_APPLICATION_MIGRATION_SETTING, "complete")

    # 只移除旧版本仍未完成的固定示例，不影响用户已完成或自行创建的任务。
    def _remove_legacy_leetcode_sample_task(self) -> None:
        self._execute(
            """
            DELETE FROM tasks
            WHERE title = ? AND task_type = ? AND priority = ?
              AND status IN (?, ?, ?)
            """,
            (
                "LeetCode 206. 反转链表",
                TaskType.LEETCODE.value,
                TaskPriority.HIGH.value,
                TaskStatus.PENDING.value,
                TaskStatus.IN_PROGRESS.value,
                TaskStatus.POSTPONED.value,
            ),
        )

    # 创建常用 owner 查询索引，保证多用户后查询不退化。
    def _ensure_indexes(self) -> None:
        self._execute("CREATE INDEX IF NOT EXISTS idx_tasks_owner_id ON tasks(owner_id)", ())
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_applications_owner_id ON applications(owner_id)",
            (),
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_interview_reviews_owner_id ON interview_reviews(owner_id)",
            (),
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_interview_schedules_owner_id ON interview_schedules(owner_id)",
            (),
        )

    # 处理 seed_default_tasks 相关逻辑。
    def _seed_default_tasks(self) -> None:
        self._ensure_default_tasks_for_owner("local_user")

    # 为指定用户补齐默认任务。
    def _ensure_default_tasks_for_owner(self, owner_id: str) -> None:
        owner_id = self._normalize_owner_id(owner_id)
        task_count = self._fetch_value(
            "SELECT COUNT(*) FROM tasks WHERE owner_id = ?",
            (owner_id,),
        )
        if task_count:
            return
        for task in _default_tasks():
            self._execute(
                """
                INSERT INTO tasks (id, owner_id, title, task_type, status, priority)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self._next_id("tasks", "task"),
                    owner_id,
                    task.title,
                    task.task_type.value,
                    task.status.value,
                    task.priority.value,
                ),
            )

    # 把升级前默认归属 local_user 的旧记录迁到第一个真实访问用户名下。
    def _claim_legacy_rows_for_owner(self, table: str, owner_id: str) -> None:
        owner_id = self._normalize_owner_id(owner_id)
        if owner_id == "local_user":
            return

        owner_count = self._fetch_value(
            f"SELECT COUNT(*) FROM {table} WHERE owner_id = ?",
            (owner_id,),
        )
        if owner_count:
            return

        legacy_count = self._fetch_value(
            f"SELECT COUNT(*) FROM {table} WHERE owner_id = ?",
            ("local_user",),
        )
        if not legacy_count:
            return

        self._execute(
            f"UPDATE {table} SET owner_id = ? WHERE owner_id = ?",
            (owner_id, "local_user"),
        )

    # 更新 task status。
    def _update_task_status(
        self,
        task_title: Optional[str],
        task_type: Optional[str],
        owner_id: str,
        status: TaskStatus,
    ) -> Optional[Task]:
        owner_id = self._normalize_owner_id(owner_id)
        task = self._find_task(task_title=task_title, task_type=task_type, owner_id=owner_id)
        if task is None:
            return None

        self._execute(
            "UPDATE tasks SET status = ? WHERE id = ? AND owner_id = ?",
            (status.value, task.id, owner_id),
        )
        task.status = status
        return task

    # 查找 task。
    def _find_task(
        self,
        task_title: Optional[str] = None,
        task_type: Optional[str] = None,
        owner_id: str = "local_user",
    ) -> Optional[Task]:
        owner_id = self._normalize_owner_id(owner_id)
        self._ensure_default_tasks_for_owner(owner_id)
        rows = self._fetch_all(
            """
            SELECT id, title, task_type, status, priority, owner_id
            FROM tasks
            WHERE owner_id = ? AND status != ?
            ORDER BY id
            """,
            (owner_id, TaskStatus.PASSED.value),
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

    # 查找 application。
    def _find_application(self, company: str, owner_id: str = "local_user") -> Optional[Application]:
        matches = self.list_applications(company=company, owner_id=owner_id)
        return matches[-1] if matches else None

    # 查找 application by id。
    def _find_application_by_id(
        self,
        application_id: str,
        owner_id: str = "local_user",
    ) -> Optional[Application]:
        owner_id = self._normalize_owner_id(owner_id)
        rows = self._fetch_all(
            """
            SELECT id, owner_id, company, role, status, interview_time, round, jd_keywords
            FROM applications
            WHERE id = ? AND owner_id = ?
            """,
            (application_id, owner_id),
        )
        if not rows:
            return None
        return self._application_from_row(rows[0])

    # 处理 next_id 相关逻辑。
    def _next_id(self, table: str, prefix: str) -> str:
        count = self._fetch_value(f"SELECT COUNT(*) FROM {table}")
        return f"{prefix}_{count + 1}"

    # 处理 fetch_value 相关逻辑。
    def _fetch_value(self, query: str, parameters: tuple[Any, ...] = ()) -> Any:
        with self._connect() as connection:
            row = connection.execute(query, parameters).fetchone()
        return row[0] if row is not None else None

    # 处理 fetch_all 相关逻辑。
    def _fetch_all(self, query: str, parameters: tuple[Any, ...] = ()) -> List[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(query, parameters).fetchall()

    # 处理 execute 相关逻辑。
    def _execute(self, query: str, parameters: tuple[Any, ...]) -> None:
        with self._connect() as connection:
            connection.execute(query, parameters)
            connection.commit()

    # 处理 connect 相关逻辑。
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    # 处理 task_from_row 相关逻辑。
    @staticmethod
    def _task_from_row(row: sqlite3.Row) -> Task:
        return Task(
            id=row["id"],
            title=row["title"],
            task_type=TaskType(row["task_type"]),
            owner_id=row["owner_id"],
            status=TaskStatus(row["status"]),
            priority=TaskPriority(row["priority"]),
        )

    # 处理 application_from_row 相关逻辑。
    @staticmethod
    def _application_from_row(row: sqlite3.Row) -> Application:
        return Application(
            id=row["id"],
            company=row["company"],
            role=row["role"],
            owner_id=row["owner_id"],
            status=ApplicationStatus(row["status"]),
            interview_time=row["interview_time"],
            round=row["round"],
            jd_keywords=json.loads(row["jd_keywords"] or "[]"),
        )

    # 处理 interview_review_from_row 相关逻辑。
    @staticmethod
    def _interview_review_from_row(row: sqlite3.Row) -> InterviewReview:
        return InterviewReview(
            id=row["id"],
            owner_id=row["owner_id"],
            company=row["company"],
            round=row["round"],
            topics=json.loads(row["topics"] or "[]"),
            raw_message=row["raw_message"],
            status=InterviewReviewStatus(row["status"]),
        )

    # 处理 interview_schedule_from_row 相关逻辑。
    @staticmethod
    def _interview_schedule_from_row(row: sqlite3.Row) -> InterviewSchedule:
        return InterviewSchedule(
            id=row["id"],
            owner_id=row["owner_id"],
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

    # 处理 normalize 相关逻辑。
    @staticmethod
    def _normalize(value: str) -> str:
        return value.replace(" ", "").lower()

    # 标准化 owner_id，避免空字符串造成数据串到匿名空间。
    @staticmethod
    def _normalize_owner_id(owner_id: str) -> str:
        normalized = (owner_id or "").strip()
        return normalized or "local_user"
