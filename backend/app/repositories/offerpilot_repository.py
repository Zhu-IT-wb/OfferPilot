from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from app.models.application import Application, ApplicationStatus, application_status_from_round
from app.models.interview_review import InterviewReview
from app.models.interview_schedule import InterviewSchedule
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
    def list_today_tasks(self) -> List[Task]:
        ...

    # 创建投递记录，并按需同步面试日程和飞书多维表格。
    def create_application(
        self,
        company: str,
        role: str,
        interview_time: Optional[str] = None,
        round_name: Optional[str] = None,
        jd_keywords: Optional[List[str]] = None,
    ) -> Application:
        ...

    # 查询 applications 列表。
    def list_applications(self, company: Optional[str] = None) -> List[Application]:
        ...

    # 更新投递进度，并按需同步日历和多维表格。
    def update_application(
        self,
        company: str,
        status: Optional[ApplicationStatus] = None,
        interview_time: Optional[str] = None,
        round_name: Optional[str] = None,
        role: Optional[str] = None,
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
    ) -> InterviewSchedule:
        ...

    # 查询 interview schedules 列表。
    def list_interview_schedules(self, company: Optional[str] = None) -> List[InterviewSchedule]:
        ...

    # 更新 interview schedule calendar event。
    def update_interview_schedule_calendar_event(
        self,
        schedule_id: str,
        calendar_event_id: str,
    ) -> Optional[InterviewSchedule]:
        ...

    # 将指定任务标记为已完成。
    def complete_task(
        self,
        task_title: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Optional[Task]:
        ...

    # 将指定任务延期处理。
    def postpone_task(
        self,
        task_title: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Optional[Task]:
        ...

    # 记录一次面试复盘。
    def create_interview_review(
        self,
        company: Optional[str] = None,
        round_name: Optional[str] = None,
        topics: Optional[List[str]] = None,
        raw_message: str = "",
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
    def list_today_tasks(self) -> List[Task]:
        active_statuses = {
            TaskStatus.PENDING,
            TaskStatus.IN_PROGRESS,
            TaskStatus.POSTPONED,
        }
        return [task for task in self.tasks if task.status in active_statuses]

    # 创建投递记录，并按需同步面试日程和飞书多维表格。
    def create_application(
        self,
        company: str,
        role: str,
        interview_time: Optional[str] = None,
        round_name: Optional[str] = None,
        jd_keywords: Optional[List[str]] = None,
    ) -> Application:
        application = Application(
            id=f"app_{len(self.applications) + 1}",
            company=company,
            role=role,
            status=application_status_from_round(round_name),
            interview_time=interview_time,
            round=round_name,
            jd_keywords=jd_keywords or [],
        )
        self.applications.append(application)
        return application

    # 查询 applications 列表。
    def list_applications(self, company: Optional[str] = None) -> List[Application]:
        if not company:
            return list(self.applications)

        normalized_company = self._normalize(company)
        return [
            application
            for application in self.applications
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
    ) -> Optional[Application]:
        application = self._find_application_by_id(application_id)
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
    ) -> InterviewSchedule:
        schedule = InterviewSchedule(
            id=f"schedule_{len(self.interview_schedules) + 1}",
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
    def list_interview_schedules(self, company: Optional[str] = None) -> List[InterviewSchedule]:
        if not company:
            return list(self.interview_schedules)

        normalized_company = self._normalize(company)
        return [
            schedule
            for schedule in self.interview_schedules
            if normalized_company in self._normalize(schedule.company)
        ]

    # 更新 interview schedule calendar event。
    def update_interview_schedule_calendar_event(
        self,
        schedule_id: str,
        calendar_event_id: str,
    ) -> Optional[InterviewSchedule]:
        for schedule in self.interview_schedules:
            if schedule.id == schedule_id:
                schedule.calendar_event_id = calendar_event_id
                return schedule
        return None

    # 将指定任务标记为已完成。
    def complete_task(
        self,
        task_title: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Optional[Task]:
        task = self._find_task(task_title=task_title, task_type=task_type)
        if task is None:
            return None

        task.status = TaskStatus.PASSED
        return task

    # 将指定任务延期处理。
    def postpone_task(
        self,
        task_title: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Optional[Task]:
        task = self._find_task(task_title=task_title, task_type=task_type)
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
    ) -> InterviewReview:
        review = InterviewReview(
            id=f"review_{len(self.interview_reviews) + 1}",
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
    ) -> Optional[Task]:
        for task in self.tasks:
            if task.status == TaskStatus.PASSED:
                continue
            if task_title and self._normalize(task_title) in self._normalize(task.title):
                return task

        if task_type:
            for task in self.tasks:
                if task.status != TaskStatus.PASSED and task.task_type.value == task_type:
                    return task

        return None

    # 查找 application。
    def _find_application(self, company: str) -> Optional[Application]:
        matches = self.list_applications(company=company)
        return matches[-1] if matches else None

    # 查找 application by id。
    def _find_application_by_id(self, application_id: str) -> Optional[Application]:
        for application in self.applications:
            if application.id == application_id:
                return application
        return None

    # 处理 normalize 相关逻辑。
    @staticmethod
    def _normalize(value: str) -> str:
        return value.replace(" ", "").lower()
