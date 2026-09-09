import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, List, Optional

from app.models.application import (
    Application,
    ApplicationStatus,
    application_status_from_round,
)
from app.models.interview_review import InterviewReview, InterviewReviewStatus
from app.models.interview_schedule import InterviewSchedule, InterviewScheduleStatus
from app.models.study import (
    StudyPlan,
    StudyPlanStatus,
    StudyPreferences,
    StudyPriority,
    StudySession,
    StudySessionStatus,
    StudySessionSyncStatus,
    StudyWindow,
    UnscheduledStudyItem,
)
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

    def list_runtime_settings(self, prefix: str = "") -> dict[str, str]:
        rows = self._fetch_all(
            """
            SELECT key, value
            FROM runtime_settings
            WHERE substr(key, 1, ?) = ?
            ORDER BY key
            """,
            (len(prefix), prefix),
        )
        return {str(row["key"]): str(row["value"]) for row in rows}

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
        base_location: Optional[str] = None,
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
            base_location=base_location.strip() if base_location else None,
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
                id, owner_id, company, role, base_location, status, interview_time, round, jd_keywords
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                application.id,
                application.owner_id,
                application.company,
                application.role,
                application.base_location,
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
        rows = self._fetch_all(
            """
            SELECT id, owner_id, company, role, base_location, status, interview_time, round, jd_keywords
            FROM applications
            WHERE owner_id = ? AND deleted_at IS NULL
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

    def delete_application(
        self,
        application_id: str,
        owner_id: str = "local_user",
    ) -> bool:
        owner_id = self._normalize_owner_id(owner_id)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE applications
                SET deleted_at = COALESCE(deleted_at, CURRENT_TIMESTAMP)
                WHERE id = ? AND owner_id = ?
                """,
                (application_id, owner_id),
            )
            return cursor.rowcount > 0

    def is_application_deleted(
        self,
        application_id: str,
        owner_id: str = "local_user",
    ) -> bool:
        return bool(
            self._fetch_value(
                """
                SELECT 1 FROM applications
                WHERE id = ? AND owner_id = ? AND deleted_at IS NOT NULL
                """,
                (application_id, self._normalize_owner_id(owner_id)),
            )
        )

    # 更新投递进度，并按需同步日历和多维表格。
    def update_application(
        self,
        company: str,
        status: Optional[ApplicationStatus] = None,
        interview_time: Optional[str] = None,
        round_name: Optional[str] = None,
        role: Optional[str] = None,
        base_location: Optional[str] = None,
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
        if base_location is not None:
            application.base_location = base_location.strip() or None

        self._execute(
            """
            UPDATE applications
            SET role = ?, base_location = ?, status = ?, interview_time = ?, round = ?
            WHERE id = ? AND owner_id = ? AND deleted_at IS NULL
            """,
            (
                application.role,
                application.base_location,
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
        base_location: Optional[str] = None,
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
        if base_location is not None:
            application.base_location = base_location.strip() or None
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
            SET company = ?, role = ?, base_location = ?, status = ?, interview_time = ?, round = ?, jd_keywords = ?
            WHERE id = ? AND owner_id = ? AND deleted_at IS NULL
            """,
            (
                application.company,
                application.role,
                application.base_location,
                application.status.value,
                application.interview_time,
                application.round,
                json.dumps(application.jd_keywords, ensure_ascii=False),
                application.id,
                owner_id,
            ),
        )
        return application

    # 将误归属的投递记录迁移到明确的用户空间。
    def reassign_application_owner(
        self,
        application_id: str,
        from_owner_id: str,
        to_owner_id: str,
    ) -> bool:
        from_owner_id = self._normalize_owner_id(from_owner_id)
        to_owner_id = self._normalize_owner_id(to_owner_id)
        application = self._find_application_by_id(
            application_id,
            owner_id=from_owner_id,
        )
        if application is None:
            return False
        self._execute(
            "UPDATE applications SET owner_id = ? WHERE id = ? AND owner_id = ? AND deleted_at IS NULL",
            (to_owner_id, application_id, from_owner_id),
        )
        self._execute(
            """
            UPDATE interview_schedules
            SET owner_id = ?
            WHERE application_id = ? AND owner_id = ?
            """,
            (to_owner_id, application_id, from_owner_id),
        )
        return True

    # 按全局唯一 application id 查询当前归属。
    def find_application_owner_id(self, application_id: str) -> Optional[str]:
        owner_id = self._fetch_value(
            "SELECT owner_id FROM applications WHERE id = ?",
            (application_id,),
        )
        return str(owner_id) if owner_id else None

    # 通过旧版 Bitable 正向映射精确查找投递记录及其归属。
    def find_application_by_bitable_record_id(
        self,
        table_id: str,
        record_id: str,
    ) -> Optional[tuple[str, str]]:
        setting_prefix = f"feishu.offerpilot_bitable_record_id.{table_id}."
        rows = self._fetch_all(
            """
            SELECT applications.id, applications.owner_id
            FROM runtime_settings
            JOIN applications ON applications.id = substr(runtime_settings.key, ?)
            WHERE substr(runtime_settings.key, 1, ?) = ? AND runtime_settings.value = ?
              AND applications.deleted_at IS NULL
            ORDER BY runtime_settings.key
            LIMIT 1
            """,
            (len(setting_prefix) + 1, len(setting_prefix), setting_prefix, record_id),
        )
        if not rows:
            return None
        return str(rows[0]["id"]), str(rows[0]["owner_id"])

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
        if (
            application_id is not None
            and self._find_application_by_id(application_id, owner_id=owner_id) is None
        ):
            raise ValueError(
                "Interview schedule application does not belong to the current owner."
            )
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
        rows = self._fetch_all(
            """
            SELECT id, owner_id, application_id, company, role, round, start_time, start_at,
                   reminder_minutes, status, calendar_event_id, raw_message
            FROM interview_schedules
            WHERE owner_id = ?
              AND NOT EXISTS (
                  SELECT 1 FROM applications
                  WHERE applications.id = interview_schedules.application_id
                    AND applications.deleted_at IS NOT NULL
              )
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
        rows = self._fetch_all(
            """
            SELECT id, owner_id, application_id, company, role, round, start_time, start_at,
                   reminder_minutes, status, calendar_event_id, raw_message
            FROM interview_schedules
            WHERE id = ? AND owner_id = ?
            """,
            (schedule_id, owner_id),
        )
        if not rows:
            return None

        schedule = self._interview_schedule_from_row(rows[0])
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

    def get_study_preferences(
        self,
        owner_id: str = "local_user",
    ) -> Optional[StudyPreferences]:
        owner_id = self._normalize_owner_id(owner_id)
        rows = self._fetch_all(
            """
            SELECT owner_id, timezone, weekday_windows, weekend_windows,
                   daily_max_minutes, session_minutes, updated_at
            FROM study_preferences
            WHERE owner_id = ?
            """,
            (owner_id,),
        )
        return self._study_preferences_from_row(rows[0]) if rows else None

    def save_study_preferences(
        self,
        preferences: StudyPreferences,
    ) -> StudyPreferences:
        preferences.owner_id = self._normalize_owner_id(preferences.owner_id)
        self._execute(
            """
            INSERT INTO study_preferences (
                owner_id, timezone, weekday_windows, weekend_windows,
                daily_max_minutes, session_minutes, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_id) DO UPDATE SET
                timezone = excluded.timezone,
                weekday_windows = excluded.weekday_windows,
                weekend_windows = excluded.weekend_windows,
                daily_max_minutes = excluded.daily_max_minutes,
                session_minutes = excluded.session_minutes,
                updated_at = excluded.updated_at
            """,
            (
                preferences.owner_id,
                preferences.timezone,
                json.dumps(
                    [window.to_dict() for window in preferences.weekday_windows],
                    ensure_ascii=False,
                ),
                json.dumps(
                    [window.to_dict() for window in preferences.weekend_windows],
                    ensure_ascii=False,
                ),
                preferences.daily_max_minutes,
                preferences.session_minutes,
                preferences.updated_at.isoformat(),
            ),
        )
        return preferences

    def create_study_plan(self, plan: StudyPlan) -> StudyPlan:
        plan.owner_id = self._normalize_owner_id(plan.owner_id)
        try:
            self._execute(
                """
                INSERT INTO study_plans (
                    id, owner_id, goal, range_start, range_end,
                    source_interview_ids, unscheduled_items, status, version,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan.id,
                    plan.owner_id,
                    plan.goal,
                    plan.range_start.isoformat(),
                    plan.range_end.isoformat(),
                    json.dumps(plan.source_interview_ids, ensure_ascii=False),
                    json.dumps(
                        [item.to_dict() for item in plan.unscheduled_items],
                        ensure_ascii=False,
                    ),
                    plan.status.value,
                    plan.version,
                    plan.created_at.isoformat(),
                    plan.updated_at.isoformat(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Study plan already exists: {plan.id}") from exc
        return plan

    def get_study_plan(
        self,
        plan_id: str,
        owner_id: str = "local_user",
    ) -> Optional[StudyPlan]:
        owner_id = self._normalize_owner_id(owner_id)
        rows = self._fetch_all(
            """
            SELECT id, owner_id, goal, range_start, range_end,
                   source_interview_ids, unscheduled_items, status, version,
                   created_at, updated_at
            FROM study_plans
            WHERE id = ? AND owner_id = ?
            """,
            (plan_id, owner_id),
        )
        return self._study_plan_from_row(rows[0]) if rows else None

    def list_study_plans(
        self,
        owner_id: str = "local_user",
        status: Optional[StudyPlanStatus] = None,
    ) -> List[StudyPlan]:
        owner_id = self._normalize_owner_id(owner_id)
        query = """
            SELECT id, owner_id, goal, range_start, range_end,
                   source_interview_ids, unscheduled_items, status, version,
                   created_at, updated_at
            FROM study_plans
            WHERE owner_id = ?
        """
        parameters: tuple[Any, ...] = (owner_id,)
        if status is not None:
            query += " AND status = ?"
            parameters += (status.value,)
        query += " ORDER BY created_at DESC, id DESC"
        return [self._study_plan_from_row(row) for row in self._fetch_all(query, parameters)]

    def update_study_plan(self, plan: StudyPlan) -> Optional[StudyPlan]:
        plan.owner_id = self._normalize_owner_id(plan.owner_id)
        if self.get_study_plan(plan.id, owner_id=plan.owner_id) is None:
            return None
        self._execute(
            """
            UPDATE study_plans
            SET goal = ?, range_start = ?, range_end = ?, source_interview_ids = ?,
                unscheduled_items = ?, status = ?, version = ?, created_at = ?,
                updated_at = ?
            WHERE id = ? AND owner_id = ?
            """,
            (
                plan.goal,
                plan.range_start.isoformat(),
                plan.range_end.isoformat(),
                json.dumps(plan.source_interview_ids, ensure_ascii=False),
                json.dumps(
                    [item.to_dict() for item in plan.unscheduled_items],
                    ensure_ascii=False,
                ),
                plan.status.value,
                plan.version,
                plan.created_at.isoformat(),
                plan.updated_at.isoformat(),
                plan.id,
                plan.owner_id,
            ),
        )
        return plan

    def delete_study_plan(
        self,
        plan_id: str,
        owner_id: str = "local_user",
    ) -> bool:
        owner_id = self._normalize_owner_id(owner_id)
        if self.get_study_plan(plan_id, owner_id=owner_id) is None:
            return False
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM study_sessions WHERE plan_id = ? AND owner_id = ?",
                (plan_id, owner_id),
            )
            connection.execute(
                "DELETE FROM study_plans WHERE id = ? AND owner_id = ?",
                (plan_id, owner_id),
            )
            connection.commit()
        return True

    def create_study_session(self, session: StudySession) -> StudySession:
        session.owner_id = self._normalize_owner_id(session.owner_id)
        if self.get_study_plan(session.plan_id, owner_id=session.owner_id) is None:
            raise ValueError(f"Study plan does not exist: {session.plan_id}")
        try:
            self._execute(
                """
                INSERT INTO study_sessions (
                    id, owner_id, plan_id, topic, start_at, end_at, priority,
                    source_refs, rationale, status, sync_status, calendar_event_id,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.owner_id,
                    session.plan_id,
                    session.topic,
                    session.start_at.isoformat(),
                    session.end_at.isoformat(),
                    int(session.priority),
                    json.dumps(session.source_refs, ensure_ascii=False),
                    session.rationale,
                    session.status.value,
                    session.sync_status.value,
                    session.calendar_event_id,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Study session already exists: {session.id}") from exc
        return session

    def get_study_session(
        self,
        session_id: str,
        owner_id: str = "local_user",
    ) -> Optional[StudySession]:
        owner_id = self._normalize_owner_id(owner_id)
        rows = self._fetch_all(
            """
            SELECT id, owner_id, plan_id, topic, start_at, end_at, priority,
                   source_refs, rationale, status, sync_status, calendar_event_id,
                   created_at, updated_at
            FROM study_sessions
            WHERE id = ? AND owner_id = ?
            """,
            (session_id, owner_id),
        )
        return self._study_session_from_row(rows[0]) if rows else None

    def list_study_sessions(
        self,
        owner_id: str = "local_user",
        plan_id: Optional[str] = None,
        range_start: Optional[datetime] = None,
        range_end: Optional[datetime] = None,
        status: Optional[StudySessionStatus] = None,
    ) -> List[StudySession]:
        owner_id = self._normalize_owner_id(owner_id)
        query = """
            SELECT id, owner_id, plan_id, topic, start_at, end_at, priority,
                   source_refs, rationale, status, sync_status, calendar_event_id,
                   created_at, updated_at
            FROM study_sessions
            WHERE owner_id = ?
        """
        parameters: tuple[Any, ...] = (owner_id,)
        if plan_id is not None:
            query += " AND plan_id = ?"
            parameters += (plan_id,)
        if range_start is not None:
            query += " AND end_at > ?"
            parameters += (range_start.isoformat(),)
        if range_end is not None:
            query += " AND start_at < ?"
            parameters += (range_end.isoformat(),)
        if status is not None:
            query += " AND status = ?"
            parameters += (status.value,)
        query += " ORDER BY start_at, id"
        return [self._study_session_from_row(row) for row in self._fetch_all(query, parameters)]

    def update_study_session(self, session: StudySession) -> Optional[StudySession]:
        session.owner_id = self._normalize_owner_id(session.owner_id)
        if self.get_study_session(session.id, owner_id=session.owner_id) is None:
            return None
        if self.get_study_plan(session.plan_id, owner_id=session.owner_id) is None:
            raise ValueError(
                "Study session plan does not belong to the current owner."
            )
        self._execute(
            """
            UPDATE study_sessions
            SET plan_id = ?, topic = ?, start_at = ?, end_at = ?, priority = ?,
                source_refs = ?, rationale = ?, status = ?, sync_status = ?,
                calendar_event_id = ?, created_at = ?, updated_at = ?
            WHERE id = ? AND owner_id = ?
            """,
            (
                session.plan_id,
                session.topic,
                session.start_at.isoformat(),
                session.end_at.isoformat(),
                int(session.priority),
                json.dumps(session.source_refs, ensure_ascii=False),
                session.rationale,
                session.status.value,
                session.sync_status.value,
                session.calendar_event_id,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
                session.id,
                session.owner_id,
            ),
        )
        return session

    def delete_study_session(
        self,
        session_id: str,
        owner_id: str = "local_user",
    ) -> bool:
        owner_id = self._normalize_owner_id(owner_id)
        if self.get_study_session(session_id, owner_id=owner_id) is None:
            return False
        self._execute(
            "DELETE FROM study_sessions WHERE id = ? AND owner_id = ?",
            (session_id, owner_id),
        )
        return True

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
                    base_location TEXT,
                    status TEXT NOT NULL,
                    interview_time TEXT,
                    round TEXT,
                    jd_keywords TEXT NOT NULL DEFAULT '[]',
                    deleted_at TEXT
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
                CREATE TABLE IF NOT EXISTS study_preferences (
                    owner_id TEXT PRIMARY KEY,
                    timezone TEXT NOT NULL,
                    weekday_windows TEXT NOT NULL DEFAULT '[]',
                    weekend_windows TEXT NOT NULL DEFAULT '[]',
                    daily_max_minutes INTEGER NOT NULL,
                    session_minutes INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS study_plans (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    range_start TEXT NOT NULL,
                    range_end TEXT NOT NULL,
                    source_interview_ids TEXT NOT NULL DEFAULT '[]',
                    unscheduled_items TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS study_sessions (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    start_at TEXT NOT NULL,
                    end_at TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    source_refs TEXT NOT NULL DEFAULT '[]',
                    rationale TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    sync_status TEXT NOT NULL,
                    calendar_event_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(plan_id) REFERENCES study_plans(id)
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
        self._ensure_column("applications", "base_location", "TEXT")
        self._ensure_column("applications", "deleted_at", "TEXT")
        self._ensure_column("interview_reviews", "owner_id", "TEXT NOT NULL DEFAULT 'local_user'")
        self._ensure_column("interview_schedules", "owner_id", "TEXT NOT NULL DEFAULT 'local_user'")
        self._ensure_column("interview_schedules", "start_at", "TEXT")
        self._ensure_column(
            "study_plans",
            "unscheduled_items",
            "TEXT NOT NULL DEFAULT '[]'",
        )
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
            "UPDATE applications SET status = ? WHERE status = ? AND deleted_at IS NULL",
            (ApplicationStatus.SUBMITTED.value, ApplicationStatus.PLANNED.value),
        )
        self.set_runtime_setting(_PLANNED_APPLICATION_MIGRATION_SETTING, "complete")

    # 只移除旧版本仍未完成的固定示例，不影响用户已完成或自行创建的任务。
    def _remove_legacy_leetcode_sample_task(self) -> None:
        self._execute(
            """
            DELETE FROM tasks
            WHERE id = ? AND title = ? AND task_type = ? AND priority = ?
              AND status IN (?, ?, ?)
            """,
            (
                "task_1",
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
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_study_plans_owner_id ON study_plans(owner_id)",
            (),
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_study_sessions_owner_plan ON study_sessions(owner_id, plan_id)",
            (),
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_study_sessions_owner_start ON study_sessions(owner_id, start_at)",
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
            SELECT id, owner_id, company, role, base_location, status, interview_time, round, jd_keywords
            FROM applications
            WHERE id = ? AND owner_id = ? AND deleted_at IS NULL
            """,
            (application_id, owner_id),
        )
        if not rows:
            return None
        return self._application_from_row(rows[0])

    # 处理 next_id 相关逻辑。
    def _next_id(self, table: str, prefix: str) -> str:
        if table == "applications":
            highest_id = self._fetch_value(
                """
                SELECT COALESCE(MAX(CAST(substr(id, 5) AS INTEGER)), 0)
                FROM applications
                WHERE substr(id, 1, 4) = 'app_'
                """
            )
            return f"{prefix}_{highest_id + 1}"
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
            base_location=row["base_location"],
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

    @staticmethod
    def _study_preferences_from_row(row: sqlite3.Row) -> StudyPreferences:
        return StudyPreferences(
            owner_id=row["owner_id"],
            timezone=row["timezone"],
            weekday_windows=[
                StudyWindow(
                    start_time=window["start_time"],
                    end_time=window["end_time"],
                )
                for window in json.loads(row["weekday_windows"] or "[]")
            ],
            weekend_windows=[
                StudyWindow(
                    start_time=window["start_time"],
                    end_time=window["end_time"],
                )
                for window in json.loads(row["weekend_windows"] or "[]")
            ],
            daily_max_minutes=row["daily_max_minutes"],
            session_minutes=row["session_minutes"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _study_plan_from_row(row: sqlite3.Row) -> StudyPlan:
        return StudyPlan(
            id=row["id"],
            owner_id=row["owner_id"],
            goal=row["goal"],
            range_start=date.fromisoformat(row["range_start"]),
            range_end=date.fromisoformat(row["range_end"]),
            source_interview_ids=json.loads(row["source_interview_ids"] or "[]"),
            unscheduled_items=[
                UnscheduledStudyItem(
                    topic=item["topic"],
                    requested_minutes=int(item["requested_minutes"]),
                    remaining_minutes=int(item["remaining_minutes"]),
                    priority=StudyPriority(int(item["priority"])),
                    deadline=(
                        datetime.fromisoformat(item["deadline"])
                        if item.get("deadline")
                        else None
                    ),
                    reason=item["reason"],
                )
                for item in json.loads(row["unscheduled_items"] or "[]")
            ],
            status=StudyPlanStatus(row["status"]),
            version=row["version"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _study_session_from_row(row: sqlite3.Row) -> StudySession:
        return StudySession(
            id=row["id"],
            owner_id=row["owner_id"],
            plan_id=row["plan_id"],
            topic=row["topic"],
            start_at=datetime.fromisoformat(row["start_at"]),
            end_at=datetime.fromisoformat(row["end_at"]),
            priority=StudyPriority(row["priority"]),
            source_refs=json.loads(row["source_refs"] or "[]"),
            rationale=row["rationale"],
            status=StudySessionStatus(row["status"]),
            sync_status=StudySessionSyncStatus(row["sync_status"]),
            calendar_event_id=row["calendar_event_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
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
