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
    assert reopened.get_subscription("feishu:ou_1").feishu_open_id == "ou_1"
    assert reopened.has_delivery(
        "feishu:ou_1", today, LeetCodeDeliveryType.MORNING
    ) is True
    assert reopened.get_delivery_attempts(
        "feishu:ou_1", today, LeetCodeDeliveryType.EVENING
    ) == 1
