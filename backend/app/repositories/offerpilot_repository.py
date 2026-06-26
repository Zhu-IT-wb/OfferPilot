from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from app.models.application import Application, application_status_from_round
from app.models.interview_review import InterviewReview
from app.models.task import Task, TaskPriority, TaskStatus, TaskType


class OfferPilotRepository(Protocol):
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

    @staticmethod
    def _normalize(value: str) -> str:
        return value.replace(" ", "").lower()
