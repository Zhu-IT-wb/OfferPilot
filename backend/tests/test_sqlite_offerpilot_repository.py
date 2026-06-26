from app.models.application import ApplicationStatus
from app.models.task import TaskStatus
from app.repositories.sqlite_offerpilot_repository import SQLiteOfferPilotRepository
from app.schemas.agent import AgentActionName
from app.tools.offerpilot_tools import build_offerpilot_tool_registry


def test_sqlite_repository_seeds_default_tasks(tmp_path) -> None:
    repository = SQLiteOfferPilotRepository(str(tmp_path / "offerpilot.db"))

    tasks = repository.list_today_tasks()

    assert len(tasks) == 3
    assert tasks[0].id == "task_1"
    assert tasks[0].to_dict()["task_type"] == "leetcode"


def test_sqlite_repository_persists_application_across_instances(tmp_path) -> None:
    db_path = tmp_path / "offerpilot.db"
    repository = SQLiteOfferPilotRepository(str(db_path))

    application = repository.create_application(
        company="深信服",
        role="开发实习",
        interview_time="明天下午三点",
        round_name="一面",
        jd_keywords=["Java", "Redis"],
    )
    reopened_repository = SQLiteOfferPilotRepository(str(db_path))
    second_application = reopened_repository.create_application(
        company="字节跳动",
        role="后端开发",
    )

    assert application.id == "app_1"
    assert application.status == ApplicationStatus.INTERVIEW_1
    assert application.to_dict()["jd_keywords"] == ["Java", "Redis"]
    assert second_application.id == "app_2"


def test_sqlite_repository_completes_task_by_title(tmp_path) -> None:
    repository = SQLiteOfferPilotRepository(str(tmp_path / "offerpilot.db"))

    task = repository.complete_task(task_title="反转链表")

    assert task is not None
    assert task.id == "task_1"
    assert task.status == TaskStatus.PASSED
    assert len(repository.list_today_tasks()) == 2


def test_sqlite_repository_creates_interview_review(tmp_path) -> None:
    repository = SQLiteOfferPilotRepository(str(tmp_path / "offerpilot.db"))

    review = repository.create_interview_review(
        company="深信服",
        round_name="一面",
        topics=["Redis", "HashMap"],
        raw_message="复盘深信服一面",
    )

    assert review.id == "review_1"
    assert review.company == "深信服"
    assert review.to_dict()["topics"] == ["Redis", "HashMap"]


def test_tools_can_run_with_sqlite_repository(tmp_path) -> None:
    repository = SQLiteOfferPilotRepository(str(tmp_path / "offerpilot.db"))
    registry = build_offerpilot_tool_registry(repository)

    result = registry.run(
        AgentActionName.CREATE_APPLICATION.value,
        {
            "company": "深信服",
            "role": "开发实习",
            "interview_time": "明天下午三点",
            "round": "一面",
        },
    )

    assert result.success is True
    assert result.data["application"]["id"] == "app_1"
    assert result.data["application"]["status"] == "interview_1"
