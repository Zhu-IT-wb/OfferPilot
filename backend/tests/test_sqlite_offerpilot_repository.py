import sqlite3

import pytest

from app.models.application import ApplicationStatus
from app.models.interview_schedule import InterviewScheduleStatus
from app.models.task import TaskStatus
from app.repositories.sqlite_offerpilot_repository import SQLiteOfferPilotRepository
from app.tools.tool_names import AgentActionName
from app.tools.offerpilot_tools import build_offerpilot_tool_registry


def test_sqlite_repository_seeds_default_tasks(tmp_path) -> None:
    repository = SQLiteOfferPilotRepository(str(tmp_path / "offerpilot.db"))

    tasks = repository.list_today_tasks()

    assert len(tasks) == 2
    assert tasks[0].id == "task_1"
    assert tasks[0].to_dict()["task_type"] == "interview_question"


def test_sqlite_repository_removes_only_unfinished_legacy_leetcode_sample(tmp_path) -> None:
    db_path = tmp_path / "offerpilot.db"
    SQLiteOfferPilotRepository(str(db_path))
    with sqlite3.connect(db_path) as connection:
        connection.execute("DELETE FROM tasks WHERE id = ?", ("task_1",))
        connection.executemany(
            """
            INSERT INTO tasks (id, owner_id, title, task_type, status, priority)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                ("task_1", "local_user", "LeetCode 206. 反转链表", "leetcode", "pending", "high"),
                ("user_created", "user_1", "LeetCode 206. 反转链表", "leetcode", "pending", "high"),
                ("legacy_passed", "user_1", "LeetCode 206. 反转链表", "leetcode", "passed", "high"),
            ],
        )
        connection.commit()

    SQLiteOfferPilotRepository(str(db_path))

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT id FROM tasks WHERE owner_id = ? ORDER BY id",
            ("user_1",),
        ).fetchall()
    assert rows == [("legacy_passed",), ("user_created",)]


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
        base_location="深圳",
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
    assert reopened_repository.list_applications(company="深信服")[0].base_location == "深圳"
    assert second_application.id == "app_2"
    assert second_application.status == ApplicationStatus.SUBMITTED


def test_sqlite_repository_reassigns_application_and_linked_schedule_owner(tmp_path) -> None:
    repository = SQLiteOfferPilotRepository(str(tmp_path / "offerpilot.db"))
    application = repository.create_application(company="美团", role="AI 全栈工程师")
    schedule = repository.create_interview_schedule(
        company="美团",
        role="AI 全栈工程师",
        round_name="一面",
        application_id=application.id,
    )

    reassigned = repository.reassign_application_owner(
        application_id=application.id,
        from_owner_id="local_user",
        to_owner_id="feishu:ou_test",
    )

    assert reassigned is True
    assert repository.find_application_owner_id(application.id) == "feishu:ou_test"
    assert repository.list_applications(owner_id="local_user") == []
    assert [item.id for item in repository.list_applications(owner_id="feishu:ou_test")] == [
        application.id
    ]
    assert repository.list_interview_schedules(owner_id="local_user") == []
    assert [
        item.id
        for item in repository.list_interview_schedules(owner_id="feishu:ou_test")
    ] == [schedule.id]


def test_sqlite_repository_never_implicitly_claims_another_owners_rows(tmp_path) -> None:
    repository = SQLiteOfferPilotRepository(str(tmp_path / "offerpilot.db"))
    application = repository.create_application(
        company="腾讯",
        role="后端开发",
        owner_id="local_user",
    )
    schedule = repository.create_interview_schedule(
        company="腾讯",
        role="后端开发",
        round_name="一面",
        application_id=application.id,
        owner_id="local_user",
    )

    assert repository.list_applications(owner_id="feishu:ou_new_user") == []
    assert repository.list_interview_schedules(owner_id="feishu:ou_new_user") == []
    assert [item.id for item in repository.list_applications(owner_id="local_user")] == [
        application.id
    ]
    assert [
        item.id for item in repository.list_interview_schedules(owner_id="local_user")
    ] == [schedule.id]


def test_sqlite_repository_migrates_legacy_confirmed_planned_application(tmp_path) -> None:
    db_path = tmp_path / "offerpilot.db"
    repository = SQLiteOfferPilotRepository(str(db_path))
    application = repository.create_application(company="腾讯", role="Java 后端")
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE applications SET status = 'planned' WHERE id = ?",
            (application.id,),
        )
        connection.execute(
            "DELETE FROM runtime_settings WHERE key = 'migration.application_planned_to_submitted.v1'"
        )
        connection.commit()

    reopened = SQLiteOfferPilotRepository(str(db_path))

    assert reopened.list_applications()[0].status == ApplicationStatus.SUBMITTED

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE applications SET status = 'planned' WHERE id = ?",
            (application.id,),
        )
        connection.commit()

    reopened_after_migration = SQLiteOfferPilotRepository(str(db_path))

    assert reopened_after_migration.list_applications()[0].status == ApplicationStatus.PLANNED


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


def test_sqlite_repository_migrates_and_persists_application_base_location(tmp_path) -> None:
    db_path = tmp_path / "offerpilot.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE applications (
                id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL DEFAULT 'local_user',
                company TEXT NOT NULL,
                role TEXT NOT NULL,
                status TEXT NOT NULL,
                interview_time TEXT,
                round TEXT,
                jd_keywords TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO applications (
                id, owner_id, company, role, status, interview_time, round, jd_keywords
            ) VALUES ('app_1', 'local_user', '字节', 'Agent 开发', 'submitted', NULL, NULL, '[]')
            """
        )
        connection.commit()

    repository = SQLiteOfferPilotRepository(str(db_path))
    legacy = repository.list_applications(company="字节")[0]
    updated = repository.update_application_by_id(
        application_id=legacy.id,
        base_location="上海",
    )
    reopened = SQLiteOfferPilotRepository(str(db_path))

    assert legacy.base_location is None
    assert updated is not None
    assert reopened.list_applications(company="字节")[0].base_location == "上海"
    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(applications)").fetchall()
        }
    assert "base_location" in columns


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


def test_sqlite_repository_rejects_cross_owner_application_schedule(tmp_path) -> None:
    repository = SQLiteOfferPilotRepository(str(tmp_path / "offerpilot.db"))
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


def test_sqlite_repository_reschedules_interview_schedule_by_id(tmp_path) -> None:
    db_path = tmp_path / "offerpilot.db"
    repository = SQLiteOfferPilotRepository(str(db_path))
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
    reopened_repository = SQLiteOfferPilotRepository(str(db_path))
    persisted = reopened_repository.list_interview_schedules(company="深信服")[0]

    assert updated_schedule is not None
    assert persisted.id == schedule.id
    assert persisted.start_time == "后天下午三点"
    assert persisted.start_at == "2026-07-18T15:00:00+08:00"


def test_sqlite_repository_cancels_interview_schedule_by_id(tmp_path) -> None:
    db_path = tmp_path / "offerpilot.db"
    repository = SQLiteOfferPilotRepository(str(db_path))
    schedule = repository.create_interview_schedule(
        company="深信服",
        role="AI 应用开发",
        round_name="一面",
        start_time="明天早上八点",
    )

    cancelled_schedule = repository.cancel_interview_schedule(schedule.id)
    reopened_repository = SQLiteOfferPilotRepository(str(db_path))
    persisted = reopened_repository.list_interview_schedules(company="深信服")[0]

    assert cancelled_schedule is not None
    assert cancelled_schedule.status == InterviewScheduleStatus.CANCELLED
    assert persisted.status == InterviewScheduleStatus.CANCELLED


def test_sqlite_repository_can_clear_application_interview_fields_by_id(tmp_path) -> None:
    db_path = tmp_path / "offerpilot.db"
    repository = SQLiteOfferPilotRepository(str(db_path))
    application = repository.create_application(
        company="深信服",
        role="AI 应用开发",
        round_name="一面",
        interview_time="明天下午三点",
    )

    repository.update_application_by_id(
        application_id=application.id,
        status=ApplicationStatus.SUBMITTED,
        clear_interview_fields=True,
    )
    reopened_repository = SQLiteOfferPilotRepository(str(db_path))
    persisted = reopened_repository.list_applications(company="深信服")[0]

    assert persisted.status == ApplicationStatus.SUBMITTED
    assert persisted.interview_time is None
    assert persisted.round is None


def test_sqlite_repository_completes_task_by_title(tmp_path) -> None:
    repository = SQLiteOfferPilotRepository(str(tmp_path / "offerpilot.db"))

    task = repository.complete_task(task_title="HashMap")

    assert task is not None
    assert task.id == "task_1"
    assert task.status == TaskStatus.PASSED
    assert len(repository.list_today_tasks()) == 1


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
