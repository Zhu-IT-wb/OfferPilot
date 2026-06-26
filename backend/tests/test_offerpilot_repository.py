from app.models.application import ApplicationStatus
from app.models.task import TaskStatus
from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository


def test_repository_lists_default_today_tasks() -> None:
    repository = InMemoryOfferPilotRepository()

    tasks = repository.list_today_tasks()

    assert len(tasks) == 3
    assert tasks[0].id == "task_1"
    assert tasks[0].to_dict()["task_type"] == "leetcode"


def test_repository_creates_application_with_round_status() -> None:
    repository = InMemoryOfferPilotRepository()

    application = repository.create_application(
        company="深信服",
        role="开发实习",
        interview_time="明天下午三点",
        round_name="一面",
        jd_keywords=["Java", "Redis"],
    )

    assert application.id == "app_1"
    assert application.status == ApplicationStatus.INTERVIEW_1
    assert application.to_dict()["jd_keywords"] == ["Java", "Redis"]
    assert repository.applications == [application]


def test_repository_completes_task_by_title() -> None:
    repository = InMemoryOfferPilotRepository()

    task = repository.complete_task(task_title="反转链表")

    assert task is not None
    assert task.id == "task_1"
    assert task.status == TaskStatus.PASSED
    assert len(repository.list_today_tasks()) == 2


def test_repository_creates_interview_review() -> None:
    repository = InMemoryOfferPilotRepository()

    review = repository.create_interview_review(
        company="深信服",
        round_name="一面",
        topics=["Redis", "HashMap"],
        raw_message="复盘深信服一面",
    )

    assert review.id == "review_1"
    assert review.company == "深信服"
    assert review.to_dict()["topics"] == ["Redis", "HashMap"]
