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


def test_sqlite_repository_persists_runtime_setting(tmp_path) -> None:
    db_path = tmp_path / "offerpilot.db"
    repository = SQLiteOfferPilotRepository(str(db_path))

    repository.set_runtime_setting(
        "feishu.offerpilot_calendar_id",
        "feishu.cn_offerpilot@group.calendar.feishu.cn",
    )
    reopened_repository = SQLiteOfferPilotRepository(str(db_path))

    assert (
        reopened_repository.get_runtime_setting("feishu.offerpilot_calendar_id")
        == "feishu.cn_offerpilot@group.calendar.feishu.cn"
    )


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


def test_sqlite_repository_lists_applications_by_company(tmp_path) -> None:
    repository = SQLiteOfferPilotRepository(str(tmp_path / "offerpilot.db"))
    repository.create_application(company="深信服", role="AI 应用开发", round_name="二面")
    repository.create_application(company="美团", role="Java 后端")

    all_applications = repository.list_applications()
    filtered_applications = repository.list_applications(company="深信服")

    assert len(all_applications) == 2
    assert len(filtered_applications) == 1
    assert filtered_applications[0].company == "深信服"
    assert filtered_applications[0].status == ApplicationStatus.INTERVIEW_2


def test_sqlite_repository_updates_application_status(tmp_path) -> None:
    repository = SQLiteOfferPilotRepository(str(tmp_path / "offerpilot.db"))
    repository.create_application(company="深信服", role="AI 应用开发")

    application = repository.update_application(
        company="深信服",
        status=ApplicationStatus.INTERVIEW_1,
        interview_time="明天早上八点",
        round_name="一面",
    )
    reopened_repository = SQLiteOfferPilotRepository(str(tmp_path / "offerpilot.db"))
    applications = reopened_repository.list_applications(company="深信服")

    assert application is not None
    assert application.status == ApplicationStatus.INTERVIEW_1
    assert applications[0].interview_time == "明天早上八点"
    assert applications[0].round == "一面"


def test_sqlite_repository_creates_interview_schedule(tmp_path) -> None:
    repository = SQLiteOfferPilotRepository(str(tmp_path / "offerpilot.db"))
    application = repository.create_application(company="深信服", role="AI 应用开发")

    schedule = repository.create_interview_schedule(
        application_id=application.id,
        company="深信服",
        role="AI 应用开发",
        round_name="一面",
        start_time="明天早上八点",
        start_at="2026-06-30T08:00:00+08:00",
    )
    reopened_repository = SQLiteOfferPilotRepository(str(tmp_path / "offerpilot.db"))
    schedules = reopened_repository.list_interview_schedules(company="深信服")

    assert schedule.id == "schedule_1"
    assert len(schedules) == 1
    assert schedules[0].application_id == "app_1"
    assert schedules[0].start_time == "明天早上八点"
    assert schedules[0].start_at == "2026-06-30T08:00:00+08:00"


def test_sqlite_repository_updates_interview_schedule_calendar_event(tmp_path) -> None:
    repository = SQLiteOfferPilotRepository(str(tmp_path / "offerpilot.db"))
    schedule = repository.create_interview_schedule(
        company="深信服",
        role="AI 应用开发",
        round_name="一面",
        start_time="明天早上八点",
    )

    updated_schedule = repository.update_interview_schedule_calendar_event(
        schedule_id=schedule.id,
        calendar_event_id="evt_test_1",
    )
    reopened_repository = SQLiteOfferPilotRepository(str(tmp_path / "offerpilot.db"))
    schedules = reopened_repository.list_interview_schedules(company="深信服")

    assert updated_schedule is not None
    assert updated_schedule.calendar_event_id == "evt_test_1"
    assert schedules[0].calendar_event_id == "evt_test_1"


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
