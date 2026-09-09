import pytest

from app.models.application import ApplicationStatus
from app.models.interview_schedule import InterviewScheduleStatus
from app.models.task import TaskStatus
from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository
from app.repositories.sqlite_offerpilot_repository import SQLiteOfferPilotRepository


@pytest.fixture(params=["memory", "sqlite"])
def application_repository(request, tmp_path):
    if request.param == "sqlite":
        return SQLiteOfferPilotRepository(str(tmp_path / "offerpilot.db"))
    return InMemoryOfferPilotRepository()


def test_application_soft_deletion_is_owner_scoped_and_idempotent(application_repository) -> None:
    repository = application_repository
    application = repository.create_application("Acme", "Engineer", owner_id="owner_a")
    other_application = repository.create_application("Acme", "Engineer", owner_id="owner_b")
    mapping_key = f"feishu.offerpilot_bitable_record_id.table_1.{application.id}"
    repository.set_runtime_setting(mapping_key, "record_1")

    assert repository.delete_application("missing", owner_id="owner_a") is False
    assert repository.is_application_deleted("missing", owner_id="owner_a") is False
    assert repository.delete_application(application.id, owner_id="owner_b") is False
    assert repository.is_application_deleted(application.id, owner_id="owner_a") is False
    assert [item.id for item in repository.list_applications(owner_id="owner_a")] == [
        application.id
    ]
    assert repository.find_application_by_bitable_record_id("table_1", "record_1") == (
        application.id,
        "owner_a",
    )

    assert repository.delete_application(application.id, owner_id="owner_a") is True
    assert repository.delete_application(application.id, owner_id="owner_a") is True
    assert repository.delete_application(application.id, owner_id="owner_b") is False
    assert repository.is_application_deleted(application.id, owner_id="owner_a") is True
    assert repository.is_application_deleted(application.id, owner_id="owner_b") is False
    assert repository.find_application_owner_id(application.id) == "owner_a"
    assert repository.list_applications(owner_id="owner_a") == []
    assert repository.list_applications(company="Acme", owner_id="owner_a") == []
    assert [item.id for item in repository.list_applications(owner_id="owner_b")] == [
        other_application.id
    ]
    assert repository.find_application_by_bitable_record_id("table_1", "record_1") is None
    assert repository.get_runtime_setting(mapping_key) == "record_1"

    replacement = repository.create_application("Acme", "Engineer", owner_id="owner_a")
    repository.set_runtime_setting(
        f"feishu.offerpilot_bitable_record_id.table_1.{replacement.id}", "record_1"
    )
    assert repository.find_application_by_bitable_record_id("table_1", "record_1") == (
        replacement.id,
        "owner_a",
    )


def test_deleted_applications_cannot_be_updated_or_reassigned(application_repository) -> None:
    repository = application_repository
    application = repository.create_application("Acme", "Engineer")
    assert repository.delete_application(application.id) is True

    assert repository.update_application("Acme", status=ApplicationStatus.OFFER) is None
    assert repository.update_application_by_id(application.id, company="Changed") is None
    assert repository.reassign_application_owner(application.id, "local_user", "owner_b") is False
    assert repository.is_application_deleted(application.id) is True
    assert repository.find_application_owner_id(application.id) == "local_user"
    assert repository.list_applications() == []
    assert repository.list_applications(owner_id="owner_b") == []


def test_application_ids_never_reuse_soft_deleted_records(application_repository) -> None:
    repository = application_repository
    first = repository.create_application("Acme", "Engineer")
    second = repository.create_application("Acme", "Engineer")
    assert repository.delete_application(first.id) is True
    third = repository.create_application("Acme", "Engineer")
    assert repository.delete_application(second.id) is True
    assert repository.delete_application(third.id) is True
    fourth = repository.create_application("Acme", "Engineer")

    assert [first.id, second.id, third.id, fourth.id] == ["app_1", "app_2", "app_3", "app_4"]
    assert [item.id for item in repository.list_applications()] == [fourth.id]
    assert repository.find_application_owner_id(first.id) == "local_user"


def test_application_soft_deletion_hides_schedules_and_preserves_receipts(application_repository) -> None:
    repository = application_repository
    application = repository.create_application("Acme", "Engineer", owner_id="owner_a")
    linked = repository.create_interview_schedule(
        "Acme", "First", application_id=application.id, owner_id="owner_a"
    )
    standalone = repository.create_interview_schedule("Acme", "First", owner_id="owner_a")
    other_owner = repository.create_interview_schedule("Acme", "First", owner_id="owner_b")
    repository.update_interview_schedule_calendar_event(linked.id, "event_1", owner_id="owner_a")
    assert repository.delete_application(application.id, owner_id="owner_a") is True

    for company in (None, "Acme"):
        assert [
            item.id for item in repository.list_interview_schedules(company, owner_id="owner_a")
        ] == [standalone.id]
    assert [item.id for item in repository.list_interview_schedules(owner_id="owner_b")] == [
        other_owner.id
    ]
    assert repository.update_interview_schedule(linked.id, round_name="Second", owner_id="owner_a") is None
    assert repository.update_interview_schedule_calendar_event(linked.id, "wrong_owner", owner_id="owner_b") is None
    receipt = repository.update_interview_schedule_calendar_event(
        linked.id, "event_after_deletion", owner_id="owner_a"
    )
    assert receipt is not None
    assert receipt.calendar_event_id == "event_after_deletion"
    assert [item.id for item in repository.list_interview_schedules(owner_id="owner_a")] == [
        standalone.id
    ]
    assert repository.cancel_interview_schedule(linked.id, owner_id="owner_a") is None
    with pytest.raises(ValueError, match="current owner"):
        repository.create_interview_schedule(
            "Acme", "Second", application_id=application.id, owner_id="owner_a"
        )
    if isinstance(repository, InMemoryOfferPilotRepository):
        stored = next(item for item in repository.interview_schedules if item.id == linked.id)
        assert stored.calendar_event_id == "event_after_deletion"
        assert stored.status == InterviewScheduleStatus.SCHEDULED
        assert stored.round == "First"
    else:
        with repository._connect() as connection:
            stored = connection.execute(
                "SELECT calendar_event_id, status, round FROM interview_schedules WHERE id = ?",
                (linked.id,),
            ).fetchone()
        assert tuple(stored) == ("event_after_deletion", InterviewScheduleStatus.SCHEDULED.value, "First")


def test_memory_application_ids_advance_past_sparse_seed_ids() -> None:
    repository = InMemoryOfferPilotRepository()
    seeded = repository.create_application("Acme", "Engineer")
    seeded.id = "app_2"
    assert repository.delete_application(seeded.id) is True

    created = repository.create_application("Acme", "Engineer")

    assert created.id == "app_3"


def test_repository_lists_default_today_tasks() -> None:
    repository = InMemoryOfferPilotRepository()

    tasks = repository.list_today_tasks()

    assert len(tasks) == 2
    assert tasks[0].id == "task_1"
    assert tasks[0].to_dict()["task_type"] == "interview_question"


def test_repository_saves_runtime_setting() -> None:
    repository = InMemoryOfferPilotRepository()

    repository.set_runtime_setting("feishu.offerpilot_calendar_id", "calendar_1")

    assert repository.get_runtime_setting("feishu.offerpilot_calendar_id") == "calendar_1"


def test_repository_creates_application_with_round_status() -> None:
    repository = InMemoryOfferPilotRepository()

    application = repository.create_application(
        company="深信服",
        role="开发实习",
        base_location="深圳",
        interview_time="明天下午三点",
        round_name="一面",
        jd_keywords=["Java", "Redis"],
    )

    assert application.id == "app_1"
    assert application.status == ApplicationStatus.INTERVIEW_1
    assert application.to_dict()["jd_keywords"] == ["Java", "Redis"]
    assert application.to_dict()["base_location"] == "深圳"
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


def test_repository_updates_application_base_location() -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(
        company="字节",
        role="Agent 开发",
        base_location="杭州",
    )

    updated = repository.update_application_by_id(
        application_id=application.id,
        base_location=" 上海 ",
    )

    assert updated is not None
    assert updated.base_location == "上海"


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


def test_repository_rejects_cross_owner_application_schedule() -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(
        company="深信服",
        role="AI 应用开发",
        owner_id="feishu:ou_a",
    )

    with pytest.raises(ValueError, match="current owner"):
        repository.create_interview_schedule(
            application_id=application.id,
            company="深信服",
            role="AI 应用开发",
            round_name="一面",
            owner_id="feishu:ou_b",
        )


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

    task = repository.complete_task(task_title="HashMap")

    assert task is not None
    assert task.id == "task_1"
    assert task.status == TaskStatus.PASSED
    assert len(repository.list_today_tasks()) == 1


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
