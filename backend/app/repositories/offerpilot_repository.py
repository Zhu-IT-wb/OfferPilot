from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Protocol

from app.models.application import Application, ApplicationStatus, application_status_from_round
from app.models.interview_review import InterviewReview
from app.models.interview_schedule import InterviewSchedule, InterviewScheduleStatus
from app.models.study import (
    StudyPlan,
    StudyPlanStatus,
    StudyPreferences,
    StudySession,
    StudySessionStatus,
)
from app.models.task import Task, TaskPriority, TaskStatus, TaskType


# 定义 OfferPilot 数据读写边界。
class OfferPilotRepository(Protocol):
    # 获取 runtime setting。
    def get_runtime_setting(self, key: str) -> Optional[str]:
        ...

    # 保存或更新 runtime setting。
    def set_runtime_setting(self, key: str, value: str) -> None:
        ...

    def list_runtime_settings(self, prefix: str = "") -> dict[str, str]:
        ...

    # 查询并格式化今天的秋招任务。
    def list_today_tasks(self, owner_id: str = "local_user") -> List[Task]:
        ...

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
        ...

    # 查询 applications 列表。
    def list_applications(
        self,
        company: Optional[str] = None,
        owner_id: str = "local_user",
    ) -> List[Application]:
        ...

    def delete_application(
        self,
        application_id: str,
        owner_id: str = "local_user",
    ) -> bool:
        """Soft-delete an owned application; repeated deletion remains successful."""
        ...

    def is_application_deleted(
        self,
        application_id: str,
        owner_id: str = "local_user",
    ) -> bool:
        ...

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
        ...

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
        ...

    # 将误归属的投递记录迁移到明确的用户空间。
    def reassign_application_owner(
        self,
        application_id: str,
        from_owner_id: str,
        to_owner_id: str,
    ) -> bool:
        ...

    # 按全局唯一 application id 查询当前归属。
    def find_application_owner_id(self, application_id: str) -> Optional[str]:
        ...

    # 通过旧版 Bitable 正向映射精确查找投递记录及其归属。
    def find_application_by_bitable_record_id(
        self,
        table_id: str,
        record_id: str,
    ) -> Optional[tuple[str, str]]:
        ...

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
        ...

    # 查询 interview schedules 列表。
    def list_interview_schedules(
        self,
        company: Optional[str] = None,
        owner_id: str = "local_user",
    ) -> List[InterviewSchedule]:
        ...

    # 更新 interview schedule calendar event。
    def update_interview_schedule_calendar_event(
        self,
        schedule_id: str,
        calendar_event_id: Optional[str],
        owner_id: str = "local_user",
    ) -> Optional[InterviewSchedule]:
        ...

    # 按稳定 ID 更新面试安排。
    def update_interview_schedule(
        self,
        schedule_id: str,
        start_time: Optional[str] = None,
        start_at: Optional[str] = None,
        round_name: Optional[str] = None,
        reminder_minutes: Optional[int] = None,
        owner_id: str = "local_user",
    ) -> Optional[InterviewSchedule]:
        ...

    # 按稳定 ID 取消面试安排。
    def cancel_interview_schedule(
        self,
        schedule_id: str,
        owner_id: str = "local_user",
    ) -> Optional[InterviewSchedule]:
        ...

    # 将指定任务标记为已完成。
    def complete_task(
        self,
        task_title: Optional[str] = None,
        task_type: Optional[str] = None,
        owner_id: str = "local_user",
    ) -> Optional[Task]:
        ...

    # 将指定任务延期处理。
    def postpone_task(
        self,
        task_title: Optional[str] = None,
        task_type: Optional[str] = None,
        owner_id: str = "local_user",
    ) -> Optional[Task]:
        ...

    # 记录一次面试复盘。
    def create_interview_review(
        self,
        company: Optional[str] = None,
        round_name: Optional[str] = None,
        topics: Optional[List[str]] = None,
        raw_message: str = "",
        owner_id: str = "local_user",
    ) -> InterviewReview:
        ...

    def get_study_preferences(
        self,
        owner_id: str = "local_user",
    ) -> Optional[StudyPreferences]:
        ...

    def save_study_preferences(
        self,
        preferences: StudyPreferences,
    ) -> StudyPreferences:
        ...

    def create_study_plan(self, plan: StudyPlan) -> StudyPlan:
        ...

    def get_study_plan(
        self,
        plan_id: str,
        owner_id: str = "local_user",
    ) -> Optional[StudyPlan]:
        ...

    def list_study_plans(
        self,
        owner_id: str = "local_user",
        status: Optional[StudyPlanStatus] = None,
    ) -> List[StudyPlan]:
        ...

    def update_study_plan(self, plan: StudyPlan) -> Optional[StudyPlan]:
        ...

    def delete_study_plan(
        self,
        plan_id: str,
        owner_id: str = "local_user",
    ) -> bool:
        ...

    def create_study_session(self, session: StudySession) -> StudySession:
        ...

    def get_study_session(
        self,
        session_id: str,
        owner_id: str = "local_user",
    ) -> Optional[StudySession]:
        ...

    def list_study_sessions(
        self,
        owner_id: str = "local_user",
        plan_id: Optional[str] = None,
        range_start: Optional[datetime] = None,
        range_end: Optional[datetime] = None,
        status: Optional[StudySessionStatus] = None,
    ) -> List[StudySession]:
        ...

    def update_study_session(self, session: StudySession) -> Optional[StudySession]:
        ...

    def delete_study_session(
        self,
        session_id: str,
        owner_id: str = "local_user",
    ) -> bool:
        ...


# 处理 default_tasks 相关逻辑。
def _default_tasks() -> List[Task]:
    return [
        Task(
            id="task_1",
            title="HashMap 扩容机制",
            task_type=TaskType.INTERVIEW_QUESTION,
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
        ),
        Task(
            id="task_2",
            title="云聚图库 Caffeine + Redis 两级缓存设计",
            task_type=TaskType.PROJECT_DEEP_DIVE,
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
        ),
    ]


# 提供用于测试和本地调试的内存数据仓库。
@dataclass
class InMemoryOfferPilotRepository:
    tasks: List[Task] = field(default_factory=_default_tasks)
    applications: List[Application] = field(default_factory=list)
    interview_reviews: List[InterviewReview] = field(default_factory=list)
    interview_schedules: List[InterviewSchedule] = field(default_factory=list)
    runtime_settings: dict[str, str] = field(default_factory=dict)
    study_preferences: dict[str, StudyPreferences] = field(default_factory=dict)
    study_plans: List[StudyPlan] = field(default_factory=list)
    study_sessions: List[StudySession] = field(default_factory=list)
    _deleted_application_ids: set[str] = field(default_factory=set, init=False, repr=False)

    # 获取 runtime setting。
    def get_runtime_setting(self, key: str) -> Optional[str]:
        return self.runtime_settings.get(key)

    # 保存或更新 runtime setting。
    def set_runtime_setting(self, key: str, value: str) -> None:
        self.runtime_settings[key] = value

    def list_runtime_settings(self, prefix: str = "") -> dict[str, str]:
        return {
            key: value
            for key, value in self.runtime_settings.items()
            if key.startswith(prefix)
        }

    # 查询并格式化今天的秋招任务。
    def list_today_tasks(self, owner_id: str = "local_user") -> List[Task]:
        self._ensure_default_tasks_for_owner(owner_id)
        active_statuses = {
            TaskStatus.PENDING,
            TaskStatus.IN_PROGRESS,
            TaskStatus.POSTPONED,
        }
        return [
            task
            for task in self.tasks
            if task.owner_id == owner_id and task.status in active_statuses
        ]

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
        next_id = max(
            (
                int(application.id[4:])
                for application in self.applications
                if application.id.startswith("app_") and application.id[4:].isdigit()
            ),
            default=0,
        ) + 1
        application = Application(
            id=f"app_{next_id}",
            company=company,
            role=role,
            base_location=base_location,
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
        self.applications.append(application)
        return application

    # 查询 applications 列表。
    def list_applications(
        self,
        company: Optional[str] = None,
        owner_id: str = "local_user",
    ) -> List[Application]:
        applications = [
            application
            for application in self.applications
            if application.owner_id == owner_id
            and application.id not in self._deleted_application_ids
        ]
        if not company:
            return list(applications)

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
        if self.find_application_owner_id(application_id) != owner_id:
            return False
        self._deleted_application_ids.add(application_id)
        return True

    def is_application_deleted(
        self,
        application_id: str,
        owner_id: str = "local_user",
    ) -> bool:
        return (
            application_id in self._deleted_application_ids
            and self.find_application_owner_id(application_id) == owner_id
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
        return application

    # 将误归属的投递记录迁移到明确的用户空间。
    def reassign_application_owner(
        self,
        application_id: str,
        from_owner_id: str,
        to_owner_id: str,
    ) -> bool:
        application = self._find_application_by_id(
            application_id,
            owner_id=from_owner_id,
        )
        if application is None:
            return False
        application.owner_id = to_owner_id
        for schedule in self.interview_schedules:
            if schedule.application_id == application_id and schedule.owner_id == from_owner_id:
                schedule.owner_id = to_owner_id
        return True

    # 按全局唯一 application id 查询当前归属。
    def find_application_owner_id(self, application_id: str) -> Optional[str]:
        for application in self.applications:
            if application.id == application_id:
                return application.owner_id
        return None

    # 通过旧版 Bitable 正向映射精确查找投递记录及其归属。
    def find_application_by_bitable_record_id(
        self,
        table_id: str,
        record_id: str,
    ) -> Optional[tuple[str, str]]:
        setting_prefix = f"feishu.offerpilot_bitable_record_id.{table_id}."
        for key, value in self.runtime_settings.items():
            if not key.startswith(setting_prefix) or value != record_id:
                continue
            application_id = key[len(setting_prefix) :]
            owner_id = self.find_application_owner_id(application_id)
            if owner_id and not self.is_application_deleted(application_id, owner_id):
                return application_id, owner_id
        return None

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
        if (
            application_id is not None
            and self._find_application_by_id(application_id, owner_id=owner_id) is None
        ):
            raise ValueError(
                "Interview schedule application does not belong to the current owner."
            )
        schedule = InterviewSchedule(
            id=f"schedule_{len(self.interview_schedules) + 1}",
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
        self.interview_schedules.append(schedule)
        return schedule

    # 查询 interview schedules 列表。
    def list_interview_schedules(
        self,
        company: Optional[str] = None,
        owner_id: str = "local_user",
    ) -> List[InterviewSchedule]:
        schedules = [
            schedule
            for schedule in self.interview_schedules
            if schedule.owner_id == owner_id
            and schedule.application_id not in self._deleted_application_ids
        ]
        if not company:
            return list(schedules)

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
        for schedule in self.interview_schedules:
            if schedule.id == schedule_id and schedule.owner_id == owner_id:
                schedule.calendar_event_id = calendar_event_id
                return schedule
        return None

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
        for schedule in self.list_interview_schedules(owner_id=owner_id):
            if schedule.id != schedule_id or schedule.owner_id != owner_id:
                continue
            if start_time is not None:
                schedule.start_time = start_time
            if start_at is not None:
                schedule.start_at = start_at
            if round_name is not None:
                schedule.round = round_name
            if reminder_minutes is not None:
                schedule.reminder_minutes = reminder_minutes
            schedule.status = InterviewScheduleStatus.SCHEDULED
            return schedule
        return None

    # 按稳定 ID 将面试安排标记为已取消。
    def cancel_interview_schedule(
        self,
        schedule_id: str,
        owner_id: str = "local_user",
    ) -> Optional[InterviewSchedule]:
        for schedule in self.list_interview_schedules(owner_id=owner_id):
            if schedule.id == schedule_id and schedule.owner_id == owner_id:
                schedule.status = InterviewScheduleStatus.CANCELLED
                return schedule
        return None

    # 将指定任务标记为已完成。
    def complete_task(
        self,
        task_title: Optional[str] = None,
        task_type: Optional[str] = None,
        owner_id: str = "local_user",
    ) -> Optional[Task]:
        task = self._find_task(task_title=task_title, task_type=task_type, owner_id=owner_id)
        if task is None:
            return None

        task.status = TaskStatus.PASSED
        return task

    # 将指定任务延期处理。
    def postpone_task(
        self,
        task_title: Optional[str] = None,
        task_type: Optional[str] = None,
        owner_id: str = "local_user",
    ) -> Optional[Task]:
        task = self._find_task(task_title=task_title, task_type=task_type, owner_id=owner_id)
        if task is None:
            return None

        task.status = TaskStatus.POSTPONED
        return task

    # 记录一次面试复盘。
    def create_interview_review(
        self,
        company: Optional[str] = None,
        round_name: Optional[str] = None,
        topics: Optional[List[str]] = None,
        raw_message: str = "",
        owner_id: str = "local_user",
    ) -> InterviewReview:
        review = InterviewReview(
            id=f"review_{len(self.interview_reviews) + 1}",
            owner_id=owner_id,
            company=company,
            round=round_name,
            topics=topics or [],
            raw_message=raw_message,
        )
        self.interview_reviews.append(review)
        return review

    def get_study_preferences(
        self,
        owner_id: str = "local_user",
    ) -> Optional[StudyPreferences]:
        return self.study_preferences.get(owner_id)

    def save_study_preferences(
        self,
        preferences: StudyPreferences,
    ) -> StudyPreferences:
        self.study_preferences[preferences.owner_id] = preferences
        return preferences

    def create_study_plan(self, plan: StudyPlan) -> StudyPlan:
        if self.get_study_plan(plan.id, owner_id=plan.owner_id) is not None:
            raise ValueError(f"Study plan already exists: {plan.id}")
        self.study_plans.append(plan)
        return plan

    def get_study_plan(
        self,
        plan_id: str,
        owner_id: str = "local_user",
    ) -> Optional[StudyPlan]:
        return next(
            (
                plan
                for plan in self.study_plans
                if plan.id == plan_id and plan.owner_id == owner_id
            ),
            None,
        )

    def list_study_plans(
        self,
        owner_id: str = "local_user",
        status: Optional[StudyPlanStatus] = None,
    ) -> List[StudyPlan]:
        return [
            plan
            for plan in self.study_plans
            if plan.owner_id == owner_id and (status is None or plan.status == status)
        ]

    def update_study_plan(self, plan: StudyPlan) -> Optional[StudyPlan]:
        for index, existing in enumerate(self.study_plans):
            if existing.id == plan.id and existing.owner_id == plan.owner_id:
                self.study_plans[index] = plan
                return plan
        return None

    def delete_study_plan(
        self,
        plan_id: str,
        owner_id: str = "local_user",
    ) -> bool:
        plan = self.get_study_plan(plan_id, owner_id=owner_id)
        if plan is None:
            return False
        self.study_plans.remove(plan)
        self.study_sessions = [
            session
            for session in self.study_sessions
            if not (session.plan_id == plan_id and session.owner_id == owner_id)
        ]
        return True

    def create_study_session(self, session: StudySession) -> StudySession:
        if self.get_study_session(session.id, owner_id=session.owner_id) is not None:
            raise ValueError(f"Study session already exists: {session.id}")
        plan = self.get_study_plan(session.plan_id, owner_id=session.owner_id)
        if plan is None:
            raise ValueError(f"Study plan does not exist: {session.plan_id}")
        self.study_sessions.append(session)
        return session

    def get_study_session(
        self,
        session_id: str,
        owner_id: str = "local_user",
    ) -> Optional[StudySession]:
        return next(
            (
                session
                for session in self.study_sessions
                if session.id == session_id and session.owner_id == owner_id
            ),
            None,
        )

    def list_study_sessions(
        self,
        owner_id: str = "local_user",
        plan_id: Optional[str] = None,
        range_start: Optional[datetime] = None,
        range_end: Optional[datetime] = None,
        status: Optional[StudySessionStatus] = None,
    ) -> List[StudySession]:
        sessions = [
            session
            for session in self.study_sessions
            if session.owner_id == owner_id
            and (plan_id is None or session.plan_id == plan_id)
            and (status is None or session.status == status)
            and (range_start is None or session.end_at > range_start)
            and (range_end is None or session.start_at < range_end)
        ]
        return sorted(sessions, key=lambda session: (session.start_at, session.id))

    def update_study_session(self, session: StudySession) -> Optional[StudySession]:
        if self.get_study_plan(session.plan_id, owner_id=session.owner_id) is None:
            raise ValueError(
                "Study session plan does not belong to the current owner."
            )
        for index, existing in enumerate(self.study_sessions):
            if existing.id == session.id and existing.owner_id == session.owner_id:
                self.study_sessions[index] = session
                return session
        return None

    def delete_study_session(
        self,
        session_id: str,
        owner_id: str = "local_user",
    ) -> bool:
        session = self.get_study_session(session_id, owner_id=owner_id)
        if session is None:
            return False
        self.study_sessions.remove(session)
        return True

    # 查找 task。
    def _find_task(
        self,
        task_title: Optional[str] = None,
        task_type: Optional[str] = None,
        owner_id: str = "local_user",
    ) -> Optional[Task]:
        self._ensure_default_tasks_for_owner(owner_id)
        for task in self.tasks:
            if task.owner_id != owner_id:
                continue
            if task.status == TaskStatus.PASSED:
                continue
            if task_title and self._normalize(task_title) in self._normalize(task.title):
                return task

        if task_type:
            for task in self.tasks:
                if (
                    task.owner_id == owner_id
                    and task.status != TaskStatus.PASSED
                    and task.task_type.value == task_type
                ):
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
        if application_id in self._deleted_application_ids:
            return None
        for application in self.applications:
            if application.id == application_id and application.owner_id == owner_id:
                return application
        return None

    # 为指定用户补齐默认任务，避免新用户看不到基础准备任务。
    def _ensure_default_tasks_for_owner(self, owner_id: str) -> None:
        if any(task.owner_id == owner_id for task in self.tasks):
            return

        for task in _default_tasks():
            self.tasks.append(
                Task(
                    id=f"task_{len(self.tasks) + 1}",
                    title=task.title,
                    task_type=task.task_type,
                    owner_id=owner_id,
                    status=task.status,
                    priority=task.priority,
                )
            )

    # 处理 normalize 相关逻辑。
    @staticmethod
    def _normalize(value: str) -> str:
        return value.replace(" ", "").lower()
