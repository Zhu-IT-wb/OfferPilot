from app.models.application import ApplicationStatus
from app.models.interview_schedule import InterviewScheduleStatus
from app.models.task import TaskStatus
from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository


def test_repository_lists_default_today_tasks() -> None:
    repository = InMemoryOfferPilotRepository()

    tasks = repository.list_today_tasks()

    assert len(tasks) == 3
    assert tasks[0].id == "task_1"
    assert tasks[0].to_dict()["task_type"] == "leetcode"


def test_repository_saves_runtime_setting() -> None:
    repository = InMemoryOfferPilotRepository()

    repository.set_runtime_setting("feishu.offerpilot_calendar_id", "calendar_1")

    assert repository.get_runtime_setting("feishu.offerpilot_calendar_id") == "calendar_1"


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


def test_repository_creates_confirmed_application_as_submitted() -> None:
    repository = InMemoryOfferPilotRepository()

    application = repository.create_application(company="腾讯", role="Java 后端")

    assert application.status == ApplicationStatus.SUBMITTED


def test_repository_lists_applications_by_company() -> None:
    repository = InMemoryOfferPilotRepository()
    repository.create_application(company="深信服", role="AI 应用开发", round_name="二面")
    repository.create_application(company="美团", role="Java 后端")

    all_applications = repository.list_applications()
    filtered_applications = repository.list_applications(company="深信服")

    assert len(all_applications) == 2
    assert len(filtered_applications) == 1
    assert filtered_applications[0].company == "深信服"


def test_repository_updates_application_status() -> None:
    repository = InMemoryOfferPilotRepository()
    repository.create_application(company="深信服", role="AI 应用开发")

    application = repository.update_application(
        company="深信服",
        status=ApplicationStatus.INTERVIEW_1,
        interview_time="明天早上八点",
        round_name="一面",
    )

    assert application is not None
    assert application.status == ApplicationStatus.INTERVIEW_1
    assert application.interview_time == "明天早上八点"
    assert repository.applications[0].round == "一面"


def test_repository_creates_interview_schedule() -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="深信服", role="AI 应用开发")

    schedule = repository.create_interview_schedule(
        application_id=application.id,
        company="深信服",
        role="AI 应用开发",
        round_name="一面",
        start_time="明天早上八点",
        start_at="2026-06-30T08:00:00+08:00",
        raw_message="明天早上八点，深信服约我一面",
    )

    assert schedule.id == "schedule_1"
    assert schedule.application_id == "app_1"
    assert schedule.start_time == "明天早上八点"
    assert schedule.start_at == "2026-06-30T08:00:00+08:00"
    assert schedule.reminder_minutes == 30
    assert repository.list_interview_schedules(company="深信服") == [schedule]


def test_repository_updates_interview_schedule_calendar_event() -> None:
    repository = InMemoryOfferPilotRepository()
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

    assert updated_schedule is not None
    assert updated_schedule.calendar_event_id == "evt_test_1"
    assert repository.interview_schedules[0].calendar_event_id == "evt_test_1"


def test_repository_reschedules_interview_schedule_by_id() -> None:
    repository = InMemoryOfferPilotRepository()
    schedule = repository.create_interview_schedule(
        company="深信服",
        role="AI 应用开发",
        round_name="一面",
        start_time="明天早上八点",
        start_at="2026-07-17T08:00:00+08:00",
    )

    updated_schedule = repository.update_interview_schedule(
        schedule_id=schedule.id,
        start_time="后天下午三点",
        start_at="2026-07-18T15:00:00+08:00",
    )

    assert updated_schedule is not None
    assert updated_schedule.id == schedule.id
    assert updated_schedule.start_time == "后天下午三点"
    assert updated_schedule.start_at == "2026-07-18T15:00:00+08:00"
    assert updated_schedule.status == InterviewScheduleStatus.SCHEDULED


def test_repository_cancels_interview_schedule_by_id() -> None:
    repository = InMemoryOfferPilotRepository()
    schedule = repository.create_interview_schedule(
        company="深信服",
        role="AI 应用开发",
        round_name="一面",
        start_time="明天早上八点",
    )

    cancelled_schedule = repository.cancel_interview_schedule(schedule.id)

    assert cancelled_schedule is not None
    assert cancelled_schedule.status == InterviewScheduleStatus.CANCELLED
    assert repository.list_interview_schedules(company="深信服")[0].status == InterviewScheduleStatus.CANCELLED


def test_repository_can_clear_application_interview_fields_by_id() -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(
        company="深信服",
        role="AI 应用开发",
        round_name="一面",
        interview_time="明天下午三点",
    )

    updated = repository.update_application_by_id(
        application_id=application.id,
        status=ApplicationStatus.SUBMITTED,
        clear_interview_fields=True,
    )

    assert updated is not None
    assert updated.status == ApplicationStatus.SUBMITTED
    assert updated.interview_time is None
    assert updated.round is None


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
