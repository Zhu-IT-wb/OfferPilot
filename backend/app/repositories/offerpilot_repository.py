from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from app.models.application import Application, ApplicationStatus, application_status_from_round
from app.models.interview_review import InterviewReview
from app.models.interview_schedule import InterviewSchedule
from app.models.task import Task, TaskPriority, TaskStatus, TaskType


class OfferPilotRepository(Protocol):
    def get_runtime_setting(self, key: str) -> Optional[str]:
        ...

    def set_runtime_setting(self, key: str, value: str) -> None:
        ...

    def list_today_tasks(self) -> List[Task]:
        ...

    def create_application(
        self,
        company: str,
        role: str,
        interview_time: Optional[str] = None,
        round_name: Optional[str] = None,
        jd_keywords: Optional[List[str]] = None,
    ) -> Application:
        ...

    def list_applications(self, company: Optional[str] = None) -> List[Application]:
        ...

    def update_application(
        self,
        company: str,
        status: Optional[ApplicationStatus] = None,
        interview_time: Optional[str] = None,
        round_name: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Optional[Application]:
        ...

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

    def list_interview_schedules(self, company: Optional[str] = None) -> List[InterviewSchedule]:
        ...

    def update_interview_schedule_calendar_event(
        self,
        schedule_id: str,
        calendar_event_id: str,
    ) -> Optional[InterviewSchedule]:
        ...

    def complete_task(
        self,
        task_title: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Optional[Task]:
        ...

    def postpone_task(
        self,
        task_title: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Optional[Task]:
        ...

    def create_interview_review(
        self,
        company: Optional[str] = None,
        round_name: Optional[str] = None,
        topics: Optional[List[str]] = None,
        raw_message: str = "",
    ) -> InterviewReview:
        ...


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


@dataclass
class InMemoryOfferPilotRepository:
    tasks: List[Task] = field(default_factory=_default_tasks)
    applications: List[Application] = field(default_factory=list)
    interview_reviews: List[InterviewReview] = field(default_factory=list)
    interview_schedules: List[InterviewSchedule] = field(default_factory=list)
    runtime_settings: dict[str, str] = field(default_factory=dict)

    def get_runtime_setting(self, key: str) -> Optional[str]:
        return self.runtime_settings.get(key)

    def set_runtime_setting(self, key: str, value: str) -> None:
        self.runtime_settings[key] = value

    def list_today_tasks(self) -> List[Task]:
        active_statuses = {
            TaskStatus.PENDING,
            TaskStatus.IN_PROGRESS,
            TaskStatus.POSTPONED,
        }
        return [task for task in self.tasks if task.status in active_statuses]

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

    def list_applications(self, company: Optional[str] = None) -> List[Application]:
        if not company:
            return list(self.applications)

        normalized_company = self._normalize(company)
        return [
            application
            for application in self.applications
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

    def list_interview_schedules(self, company: Optional[str] = None) -> List[InterviewSchedule]:
        if not company:
            return list(self.interview_schedules)

        normalized_company = self._normalize(company)
        return [
            schedule
            for schedule in self.interview_schedules
            if normalized_company in self._normalize(schedule.company)
        ]

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

    def _find_application(self, company: str) -> Optional[Application]:
        matches = self.list_applications(company=company)
        return matches[-1] if matches else None

    @staticmethod
    def _normalize(value: str) -> str:
        return value.replace(" ", "").lower()
