from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from app.models.application import Application, ApplicationStatus, application_status_from_round
from app.models.interview_review import InterviewReview
from app.models.interview_schedule import InterviewSchedule, InterviewScheduleStatus
from app.models.task import Task, TaskPriority, TaskStatus, TaskType


# 定义 OfferPilot 数据读写边界。
class OfferPilotRepository(Protocol):
    # 获取 runtime setting。
    def get_runtime_setting(self, key: str) -> Optional[str]:
        ...

    # 保存或更新 runtime setting。
    def set_runtime_setting(self, key: str, value: str) -> None:
        ...

    # 查询并格式化今天的秋招任务。
    def list_today_tasks(self, owner_id: str = "local_user") -> List[Task]:
        ...

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
        ...

    # 查询 applications 列表。
    def list_applications(
        self,
        company: Optional[str] = None,
        owner_id: str = "local_user",
    ) -> List[Application]:
        ...

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
        ...

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


# 处理 default_tasks 相关逻辑。
def _default_tasks() -> List[Task]:
    return [
        Task(
            id="task_1",
            title="LeetCode 206. 反转链表",
            task_type=TaskType.LEETCODE,
            status=TaskStatus.PENDING,
            priority=TaskPriority.HIGH,
        ),
        Task(
            id="task_2",
            title="HashMap 扩容机制",
            task_type=TaskType.INTERVIEW_QUESTION,
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
        ),
        Task(
            id="task_3",
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

    # 获取 runtime setting。
    def get_runtime_setting(self, key: str) -> Optional[str]:
        return self.runtime_settings.get(key)

    # 保存或更新 runtime setting。
    def set_runtime_setting(self, key: str, value: str) -> None:
        self.runtime_settings[key] = value

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
        interview_time: Optional[str] = None,
        round_name: Optional[str] = None,
        jd_keywords: Optional[List[str]] = None,
        owner_id: str = "local_user",
    ) -> Application:
        application = Application(
            id=f"app_{len(self.applications) + 1}",
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
        self.applications.append(application)
        return application

    # 查询 applications 列表。
    def list_applications(
        self,
        company: Optional[str] = None,
        owner_id: str = "local_user",
    ) -> List[Application]:
        self._claim_legacy_records_for_owner(
            records=self.applications,
            owner_id=owner_id,
        )
        applications = [
            application
            for application in self.applications
            if application.owner_id == owner_id
        ]
        if not company:
            return list(applications)

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
        self._claim_legacy_records_for_owner(
            records=self.interview_schedules,
            owner_id=owner_id,
        )
        schedules = [
            schedule
            for schedule in self.interview_schedules
            if schedule.owner_id == owner_id
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
        for schedule in self.interview_schedules:
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
        for schedule in self.interview_schedules:
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

    # 把升级前默认归属 local_user 的旧记录迁到第一个真实访问用户名下。
    @staticmethod
    def _claim_legacy_records_for_owner(records: List, owner_id: str) -> None:
        if owner_id == "local_user":
            return
        if any(getattr(record, "owner_id", "local_user") == owner_id for record in records):
            return

        for record in records:
            if getattr(record, "owner_id", "local_user") == "local_user":
                record.owner_id = owner_id

    # 处理 normalize 相关逻辑。
    @staticmethod
    def _normalize(value: str) -> str:
        return value.replace(" ", "").lower()
