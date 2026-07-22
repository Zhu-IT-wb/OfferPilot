from datetime import date, timedelta

from app.models.leetcode import (
    LeetCodeAssignmentStatus,
    LeetCodeAssignmentType,
    LeetCodeDifficulty,
    LeetCodeMasteryStatus,
    LeetCodePracticeResult,
    LeetCodeProblem,
)
from app.repositories.leetcode_repository import InMemoryLeetCodeRepository
from app.services.leetcode_recommendation import LeetCodeRecommendationWorkflow


def _problem(
    frontend_id: str,
    title: str,
    difficulty: LeetCodeDifficulty,
    category_order: int,
    problem_order: int,
) -> LeetCodeProblem:
    slug = title.lower().replace(" ", "-")
    return LeetCodeProblem(
        id=f"leetcode_{frontend_id}",
        frontend_id=frontend_id,
        title_zh=title,
        title_en=title,
        slug=slug,
        difficulty=difficulty,
        topics=["数组"],
        category="哈希" if category_order == 0 else "双指针",
        category_order=category_order,
        problem_order=problem_order,
        url=f"https://leetcode.cn/problems/{slug}/",
    )


def test_new_user_gets_three_stable_new_problems_in_learning_order() -> None:
    repository = InMemoryLeetCodeRepository(
        problems=[
            _problem("49", "Group Anagrams", LeetCodeDifficulty.MEDIUM, 0, 1),
            _problem("1", "Two Sum", LeetCodeDifficulty.EASY, 0, 2),
            _problem("128", "Longest Consecutive", LeetCodeDifficulty.MEDIUM, 0, 3),
            _problem("283", "Move Zeroes", LeetCodeDifficulty.EASY, 1, 4),
        ]
    )
    workflow = LeetCodeRecommendationWorkflow(repository)
    today = date(2026, 7, 22)

    first = workflow.get_today(owner_id="feishu:ou_1", today=today)
    repeated = workflow.get_today(owner_id="feishu:ou_1", today=today)

    assert [item.problem.frontend_id for item in first] == ["1", "49", "128"]
    assert [item.assignment.assignment_type.value for item in first] == ["new", "new", "new"]
    assert [item.assignment.id for item in repeated] == [item.assignment.id for item in first]


def test_solution_feedback_creates_one_due_review_and_two_new_problems_next_day() -> None:
    repository = InMemoryLeetCodeRepository(
        problems=[
            _problem(str(index), f"Problem {index}", LeetCodeDifficulty.EASY, 0, index)
            for index in range(1, 7)
        ]
    )
    workflow = LeetCodeRecommendationWorkflow(repository)
    first_day = date(2026, 7, 22)
    first_recommendations = workflow.get_today("feishu:ou_1", first_day)

    feedback = workflow.record_result(
        owner_id="feishu:ou_1",
        assignment_id=first_recommendations[0].assignment.id,
        result=LeetCodePracticeResult.WITH_SOLUTION,
        practiced_on=first_day,
    )
    for recommendation in first_recommendations[1:]:
        workflow.record_result(
            owner_id="feishu:ou_1",
            assignment_id=recommendation.assignment.id,
            result=LeetCodePracticeResult.SKIPPED,
            practiced_on=first_day,
        )
    next_recommendations = workflow.get_today("feishu:ou_1", first_day + timedelta(days=1))

    assert feedback.progress is not None
    assert feedback.progress.next_review_on == first_day + timedelta(days=1)
    assert [item.assignment.assignment_type for item in next_recommendations].count(
        LeetCodeAssignmentType.NEW
    ) == 2
    assert [item.assignment.assignment_type for item in next_recommendations].count(
        LeetCodeAssignmentType.REVIEW
    ) == 1
    review = next(
        item for item in next_recommendations if item.assignment.assignment_type == LeetCodeAssignmentType.REVIEW
    )
    assert review.problem.id == first_recommendations[0].problem.id


def test_feedback_results_update_assignment_and_review_schedule() -> None:
    expected = {
        LeetCodePracticeResult.INDEPENDENT: (
            LeetCodeAssignmentStatus.COMPLETED,
            LeetCodeMasteryStatus.REVIEWING,
            1,
            7,
        ),
        LeetCodePracticeResult.WITH_HINT: (
            LeetCodeAssignmentStatus.COMPLETED,
            LeetCodeMasteryStatus.LEARNING,
            0,
            3,
        ),
        LeetCodePracticeResult.WITH_SOLUTION: (
            LeetCodeAssignmentStatus.COMPLETED,
            LeetCodeMasteryStatus.LEARNING,
            0,
            1,
        ),
        LeetCodePracticeResult.FAILED: (
            LeetCodeAssignmentStatus.COMPLETED,
            LeetCodeMasteryStatus.LEARNING,
            0,
            1,
        ),
    }
    today = date(2026, 7, 22)

    for result, (status, mastery, streak, review_days) in expected.items():
        repository = InMemoryLeetCodeRepository(
            problems=[_problem("1", "Two Sum", LeetCodeDifficulty.EASY, 0, 1)]
        )
        workflow = LeetCodeRecommendationWorkflow(repository)
        assignment = workflow.get_today("owner", today)[0].assignment

        feedback = workflow.record_result("owner", assignment.id, result, today)

        assert feedback.assignment.status == status
        assert feedback.progress is not None
        assert feedback.progress.mastery_status == mastery
        assert feedback.progress.independent_streak == streak
        assert feedback.progress.next_review_on == today + timedelta(days=review_days)


def test_postponed_and_skipped_results_update_progress_without_counting_an_attempt() -> None:
    today = date(2026, 7, 22)
    for result, expected_status, expected_review_on in [
        (
            LeetCodePracticeResult.POSTPONED,
            LeetCodeAssignmentStatus.POSTPONED,
            today + timedelta(days=1),
        ),
        (LeetCodePracticeResult.SKIPPED, LeetCodeAssignmentStatus.SKIPPED, None),
    ]:
        repository = InMemoryLeetCodeRepository(
            problems=[_problem("1", "Two Sum", LeetCodeDifficulty.EASY, 0, 1)]
        )
        workflow = LeetCodeRecommendationWorkflow(repository)
        assignment = workflow.get_today("owner", today)[0].assignment

        feedback = workflow.record_result("owner", assignment.id, result, today)

        assert feedback.assignment.status == expected_status
        assert feedback.progress is not None
        assert feedback.progress.attempt_count == 0
        assert feedback.progress.last_result == result
        assert feedback.progress.next_review_on == expected_review_on


def test_three_consecutive_independent_results_mark_problem_mastered() -> None:
    repository = InMemoryLeetCodeRepository(
        problems=[_problem("1", "Two Sum", LeetCodeDifficulty.EASY, 0, 1)]
    )
    workflow = LeetCodeRecommendationWorkflow(repository)
    practice_days = [date(2026, 7, 22), date(2026, 7, 29), date(2026, 8, 19)]

    for index, practice_day in enumerate(practice_days):
        if index == 0:
            assignment = workflow.get_today("owner", practice_day)[0].assignment
        else:
            assignment = repository.create_assignment(
                owner_id="owner",
                problem_id="leetcode_1",
                assigned_on=practice_day,
                assignment_type=LeetCodeAssignmentType.REVIEW,
                recommendation_reason="到期复习",
            )
        feedback = workflow.record_result(
            "owner",
            assignment.id,
            LeetCodePracticeResult.INDEPENDENT,
            practice_day,
        )

    assert feedback.progress is not None
    assert feedback.progress.independent_streak == 3
    assert feedback.progress.mastery_status == LeetCodeMasteryStatus.MASTERED
    assert feedback.progress.next_review_on is None


def test_three_postponements_auto_skip_and_release_next_day_slot() -> None:
    repository = InMemoryLeetCodeRepository(
        problems=[
            _problem(str(index), f"Problem {index}", LeetCodeDifficulty.EASY, 0, index)
            for index in range(1, 7)
        ]
    )
    workflow = LeetCodeRecommendationWorkflow(repository)
    today = date(2026, 7, 22)
    carried_problem_id = workflow.get_today("owner", today)[0].problem.id

    for offset in range(3):
        current_day = today + timedelta(days=offset)
        assignment = next(
            item.assignment
            for item in workflow.get_today("owner", current_day)
            if item.problem.id == carried_problem_id
        )
        feedback = workflow.record_result(
            "owner", assignment.id, LeetCodePracticeResult.POSTPONED, current_day
        )

    assert feedback.assignment.status == LeetCodeAssignmentStatus.SKIPPED
    following = workflow.get_today("owner", today + timedelta(days=3))
    assert carried_problem_id not in {item.problem.id for item in following}


def test_postponed_problem_is_carryover_not_a_duplicate_due_review() -> None:
    repository = InMemoryLeetCodeRepository(
        problems=[
            _problem(str(index), f"Problem {index}", LeetCodeDifficulty.EASY, 0, index)
            for index in range(1, 7)
        ]
    )
    workflow = LeetCodeRecommendationWorkflow(repository)
    first_day = date(2026, 7, 22)
    first = workflow.get_today("owner", first_day)
    workflow.record_result(
        "owner",
        first[0].assignment.id,
        LeetCodePracticeResult.POSTPONED,
        first_day,
    )
    for recommendation in first[1:]:
        workflow.record_result(
            "owner",
            recommendation.assignment.id,
            LeetCodePracticeResult.SKIPPED,
            first_day,
        )

    recommendations = workflow.get_today("owner", first_day + timedelta(days=1))

    assert len(recommendations) == 3
    assert [item.assignment.assignment_type for item in recommendations].count(
        LeetCodeAssignmentType.CARRYOVER
    ) == 1
    assert [item.assignment.assignment_type for item in recommendations].count(
        LeetCodeAssignmentType.NEW
    ) == 2
    assert len({item.problem.id for item in recommendations}) == 3


def test_carryover_uses_at_most_two_slots_when_no_review_is_due() -> None:
    repository = InMemoryLeetCodeRepository(
        problems=[
            _problem(str(index), f"Problem {index}", LeetCodeDifficulty.EASY, 0, index)
            for index in range(1, 8)
        ]
    )
    workflow = LeetCodeRecommendationWorkflow(repository)
    first_day = date(2026, 7, 22)
    workflow.get_today("owner", first_day)

    recommendations = workflow.get_today("owner", first_day + timedelta(days=1))

    assert len(recommendations) == 3
    assert [item.assignment.assignment_type for item in recommendations].count(
        LeetCodeAssignmentType.CARRYOVER
    ) == 2
    assert [item.assignment.assignment_type for item in recommendations].count(
        LeetCodeAssignmentType.NEW
    ) == 1


def test_hot100_exhaustion_returns_fewer_than_three_without_duplicates() -> None:
    repository = InMemoryLeetCodeRepository(
        problems=[
            _problem("1", "Two Sum", LeetCodeDifficulty.EASY, 0, 1),
            _problem("2", "Add Two Numbers", LeetCodeDifficulty.MEDIUM, 0, 2),
        ]
    )

    recommendations = LeetCodeRecommendationWorkflow(repository).get_today(
        "owner", date(2026, 7, 22)
    )

    assert len(recommendations) == 2
    assert len({item.problem.id for item in recommendations}) == 2


def test_new_problem_shortage_uses_an_extra_non_mastered_review() -> None:
    repository = InMemoryLeetCodeRepository(
        problems=[
            _problem(str(index), f"Problem {index}", LeetCodeDifficulty.EASY, 0, index)
            for index in range(1, 4)
        ]
    )
    workflow = LeetCodeRecommendationWorkflow(repository)
    first_day = date(2026, 7, 22)
    first = workflow.get_today("owner", first_day)
    workflow.record_result(
        "owner", first[0].assignment.id, LeetCodePracticeResult.INDEPENDENT, first_day
    )
    for recommendation in first[1:]:
        workflow.record_result(
            "owner", recommendation.assignment.id, LeetCodePracticeResult.SKIPPED, first_day
        )

    recommendations = workflow.get_today("owner", first_day + timedelta(days=1))

    assert len(recommendations) == 1
    assert recommendations[0].assignment.assignment_type == LeetCodeAssignmentType.REVIEW
    assert recommendations[0].problem.id == first[0].problem.id


def test_duplicate_feedback_is_idempotent() -> None:
    repository = InMemoryLeetCodeRepository(
        problems=[_problem("1", "Two Sum", LeetCodeDifficulty.EASY, 0, 1)]
    )
    workflow = LeetCodeRecommendationWorkflow(repository)
    today = date(2026, 7, 22)
    assignment = workflow.get_today("owner", today)[0].assignment

    first = workflow.record_result(
        "owner", assignment.id, LeetCodePracticeResult.INDEPENDENT, today
    )
    repeated = workflow.record_result(
        "owner", assignment.id, LeetCodePracticeResult.INDEPENDENT, today
    )

    assert first.progress is not None and repeated.progress is not None
    assert repeated.progress.attempt_count == 1
    assert repeated.progress.independent_streak == 1
