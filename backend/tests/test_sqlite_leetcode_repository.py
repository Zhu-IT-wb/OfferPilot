import sqlite3
from datetime import date

from app.models.leetcode import (
    LeetCodeDeliveryType,
    LeetCodePracticeResult,
    LeetCodeSubscription,
)
from app.repositories.sqlite_leetcode_repository import SQLiteLeetCodeRepository
from app.services.leetcode_recommendation import LeetCodeRecommendationWorkflow


def test_sqlite_leetcode_state_survives_restart_and_is_owner_isolated(tmp_path) -> None:
    database = tmp_path / "offerpilot.db"
    repository = SQLiteLeetCodeRepository(str(database))
    workflow = LeetCodeRecommendationWorkflow(repository)
    today = date(2026, 7, 22)

    recommendations = workflow.get_today("feishu:ou_1", today)
    feedback = workflow.record_result(
        owner_id="feishu:ou_1",
        assignment_id=recommendations[0].assignment.id,
        result=LeetCodePracticeResult.WITH_HINT,
        practiced_on=today,
    )
    repository.save_subscription(
        LeetCodeSubscription(owner_id="feishu:ou_1", feishu_open_id="ou_1")
    )
    repository.record_delivery("feishu:ou_1", today, LeetCodeDeliveryType.MORNING)
    repository.record_delivery_attempt("feishu:ou_1", today, LeetCodeDeliveryType.EVENING)

    reopened = SQLiteLeetCodeRepository(str(database))

    assert len(reopened.list_problems()) == 100
    assert len(reopened.list_assignments("feishu:ou_1", today)) == 3
    assert reopened.list_assignments("feishu:ou_2", today) == []
    persisted_progress = reopened.get_progress("feishu:ou_1", recommendations[0].problem.id)
    assert persisted_progress is not None
    assert persisted_progress.next_review_on == feedback.progress.next_review_on
    subscription = reopened.get_subscription("feishu:ou_1")
    assert subscription.feishu_open_id == "ou_1"
    assert (subscription.morning_time, subscription.noon_time, subscription.evening_time) == (
        "08:00",
        "12:00",
        "18:00",
    )
    assert reopened.has_delivery(
        "feishu:ou_1", today, LeetCodeDeliveryType.MORNING
    ) is True
    assert reopened.get_delivery_attempts(
        "feishu:ou_1", today, LeetCodeDeliveryType.EVENING
    ) == 1


def test_sqlite_migrates_the_legacy_push_schedule_once(tmp_path) -> None:
    database = tmp_path / "offerpilot.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE leetcode_subscriptions (
                owner_id TEXT PRIMARY KEY,
                feishu_open_id TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                timezone TEXT NOT NULL DEFAULT 'Asia/Shanghai',
                morning_time TEXT NOT NULL DEFAULT '09:00',
                evening_time TEXT NOT NULL DEFAULT '21:00',
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO leetcode_subscriptions (
                owner_id, feishu_open_id, morning_time, evening_time, updated_at
            ) VALUES ('feishu:ou_legacy', 'ou_legacy', '09:00', '21:00', '2026-07-22')
            """
        )

    repository = SQLiteLeetCodeRepository(str(database))
    subscription = repository.get_subscription("feishu:ou_legacy")

    assert subscription is not None
    assert (subscription.morning_time, subscription.noon_time, subscription.evening_time) == (
        "08:00",
        "12:00",
        "18:00",
    )

    subscription.morning_time = "09:00"
    repository.save_subscription(subscription)
    reopened = SQLiteLeetCodeRepository(str(database))
    assert reopened.get_subscription("feishu:ou_legacy").morning_time == "09:00"
