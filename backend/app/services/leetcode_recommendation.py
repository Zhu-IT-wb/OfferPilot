from datetime import date, datetime, time, timedelta
from typing import List, Optional

from app.models.leetcode import (
    LeetCodeAssignment,
    LeetCodeAssignmentType,
    LeetCodeAssignmentStatus,
    LeetCodeDifficulty,
    LeetCodeFeedback,
    LeetCodeMasteryStatus,
    LeetCodePracticeResult,
    LeetCodeProgress,
    LeetCodeRecommendation,
)
from app.repositories.leetcode_repository import LeetCodeRepository


_DIFFICULTY_ORDER = {
    LeetCodeDifficulty.EASY: 0,
    LeetCodeDifficulty.MEDIUM: 1,
    LeetCodeDifficulty.HARD: 2,
}


class LeetCodeRecommendationWorkflow:
    def __init__(self, repository: LeetCodeRepository) -> None:
        self.repository = repository

    def get_today(self, owner_id: str, today: date) -> List[LeetCodeRecommendation]:
        current = self.repository.list_assignments(owner_id=owner_id, assigned_on=today)
        if not current:
            self._create_today_assignments(owner_id=owner_id, today=today)
            current = self.repository.list_assignments(owner_id=owner_id, assigned_on=today)
        problem_by_id = {problem.id: problem for problem in self.repository.list_problems()}
        return [
            LeetCodeRecommendation(assignment=assignment, problem=problem_by_id[assignment.problem_id])
            for assignment in current
            if assignment.problem_id in problem_by_id
        ]

    def record_result(
        self,
        owner_id: str,
        assignment_id: str,
        result: LeetCodePracticeResult,
        practiced_on: date,
    ) -> LeetCodeFeedback:
        assignment = next(
            (
                item
                for item in self.repository.list_assignments(owner_id=owner_id)
                if item.id == assignment_id
            ),
            None,
        )
        if assignment is None:
            raise ValueError("LeetCode assignment was not found.")

        if assignment.result is not None:
            if assignment.result != result:
                raise ValueError("LeetCode assignment already has a different result.")
            return LeetCodeFeedback(
                assignment=assignment,
                progress=self.repository.get_progress(owner_id, assignment.problem_id),
            )

        if result == LeetCodePracticeResult.POSTPONED:
            assignment.postpone_count += 1
            assignment.result = result
            assignment.status = (
                LeetCodeAssignmentStatus.SKIPPED
                if assignment.postpone_count >= 3
                else LeetCodeAssignmentStatus.POSTPONED
            )
            self.repository.save_assignment(assignment)
            progress = self._record_non_attempt_result(
                assignment=assignment,
                result=(
                    LeetCodePracticeResult.SKIPPED
                    if assignment.status == LeetCodeAssignmentStatus.SKIPPED
                    else result
                ),
                next_review_on=(
                    None
                    if assignment.status == LeetCodeAssignmentStatus.SKIPPED
                    else practiced_on + timedelta(days=1)
                ),
            )
            return LeetCodeFeedback(assignment=assignment, progress=progress)
        if result == LeetCodePracticeResult.SKIPPED:
            assignment.result = result
            assignment.status = LeetCodeAssignmentStatus.SKIPPED
            self.repository.save_assignment(assignment)
            progress = self._record_non_attempt_result(
                assignment=assignment,
                result=result,
                next_review_on=None,
            )
            return LeetCodeFeedback(assignment=assignment, progress=progress)

        assignment.result = result
        assignment.status = LeetCodeAssignmentStatus.COMPLETED
        assignment.completed_at = datetime.combine(practiced_on, time.min)
        self.repository.save_assignment(assignment)

        progress = self.repository.get_progress(owner_id, assignment.problem_id) or LeetCodeProgress(
            owner_id=owner_id,
            problem_id=assignment.problem_id,
        )
        progress.attempt_count += 1
        progress.last_result = result
        progress.last_practiced_on = practiced_on
        if result == LeetCodePracticeResult.INDEPENDENT:
            progress.independent_streak += 1
            if progress.independent_streak >= 3:
                progress.mastery_status = LeetCodeMasteryStatus.MASTERED
                progress.next_review_on = None
            else:
                progress.mastery_status = LeetCodeMasteryStatus.REVIEWING
                review_days = 7 if progress.independent_streak == 1 else 21
                progress.next_review_on = practiced_on + timedelta(days=review_days)
        else:
            progress.independent_streak = 0
            progress.mastery_status = LeetCodeMasteryStatus.LEARNING
            review_days = 3 if result == LeetCodePracticeResult.WITH_HINT else 1
            progress.next_review_on = practiced_on + timedelta(days=review_days)
        self.repository.save_progress(progress)
        return LeetCodeFeedback(assignment=assignment, progress=progress)

    def _record_non_attempt_result(
        self,
        assignment: LeetCodeAssignment,
        result: LeetCodePracticeResult,
        next_review_on: Optional[date],
    ) -> LeetCodeProgress:
        progress = self.repository.get_progress(
            assignment.owner_id, assignment.problem_id
        ) or LeetCodeProgress(
            owner_id=assignment.owner_id,
            problem_id=assignment.problem_id,
        )
        progress.last_result = result
        progress.next_review_on = next_review_on
        self.repository.save_progress(progress)
        return progress

    def _create_today_assignments(self, owner_id: str, today: date) -> None:
        all_assignments = self.repository.list_assignments(owner_id=owner_id)
        prior_pending = sorted(
            [
                assignment
                for assignment in all_assignments
                if assignment.assigned_on < today
                and assignment.status in {
                    LeetCodeAssignmentStatus.PENDING,
                    LeetCodeAssignmentStatus.POSTPONED,
                }
            ],
            key=lambda assignment: (assignment.assigned_on, assignment.id),
        )
        prior_pending_problem_ids = {
            assignment.problem_id for assignment in prior_pending
        }
        review_candidates = sorted(
            [
                progress
                for progress in self.repository.list_progress(owner_id)
                if progress.mastery_status != LeetCodeMasteryStatus.MASTERED
                and progress.next_review_on is not None
                and progress.problem_id not in prior_pending_problem_ids
            ],
            key=lambda progress: (
                progress.next_review_on or today,
                _review_result_priority(progress.last_result),
                progress.problem_id,
            ),
        )
        due_progress = [
            progress
            for progress in review_candidates
            if progress.next_review_on is not None and progress.next_review_on <= today
        ]
        seen_problem_ids = {assignment.problem_id for assignment in all_assignments}
        problems = sorted(
            self.repository.list_problems(),
            key=lambda problem: (
                problem.category_order,
                _DIFFICULTY_ORDER[problem.difficulty],
                problem.problem_order,
            ),
        )
        problem_by_id = {problem.id: problem for problem in problems}
        selected_problem_ids = set()
        new_slot_count = 2 if due_progress else 3
        carryover_slot_count = min(2, new_slot_count)

        for previous in prior_pending[:carryover_slot_count]:
            previous.status = LeetCodeAssignmentStatus.SKIPPED
            self.repository.save_assignment(previous)
            carried = self.repository.create_assignment(
                owner_id=owner_id,
                problem_id=previous.problem_id,
                assigned_on=today,
                assignment_type=LeetCodeAssignmentType.CARRYOVER,
                recommendation_reason="昨天尚未完成，今天优先继续。",
            )
            carried.postpone_count = previous.postpone_count
            self.repository.save_assignment(carried)
            selected_problem_ids.add(previous.problem_id)

        remaining_new_slots = new_slot_count - len(selected_problem_ids)
        new_problems = [item for item in problems if item.id not in seen_problem_ids]
        for problem in new_problems[:remaining_new_slots]:
            self.repository.create_assignment(
                owner_id=owner_id,
                problem_id=problem.id,
                assigned_on=today,
                assignment_type=LeetCodeAssignmentType.NEW,
                recommendation_reason=f"按 Hot 100 的{problem.category}专题顺序学习。",
            )
            selected_problem_ids.add(problem.id)

        remaining_slots = 3 - len(selected_problem_ids)
        for progress in review_candidates:
            if remaining_slots <= 0 or progress.problem_id in selected_problem_ids:
                continue
            problem = problem_by_id.get(progress.problem_id)
            if problem is None:
                continue
            self.repository.create_assignment(
                owner_id=owner_id,
                problem_id=problem.id,
                assigned_on=today,
                assignment_type=LeetCodeAssignmentType.REVIEW,
                recommendation_reason=(
                    _review_reason(progress.last_result)
                    if progress.next_review_on is not None and progress.next_review_on <= today
                    else "新题不足，提前安排一次巩固复习。"
                ),
            )
            selected_problem_ids.add(problem.id)
            remaining_slots -= 1


def _review_result_priority(result: Optional[LeetCodePracticeResult]) -> int:
    return {
        LeetCodePracticeResult.FAILED: 0,
        LeetCodePracticeResult.WITH_SOLUTION: 1,
        LeetCodePracticeResult.WITH_HINT: 2,
        LeetCodePracticeResult.INDEPENDENT: 3,
    }.get(result, 4)


def _review_reason(result: Optional[LeetCodePracticeResult]) -> str:
    labels = {
        LeetCodePracticeResult.FAILED: "上次尝试未完成，今天再次练习。",
        LeetCodePracticeResult.WITH_SOLUTION: "上次看题解完成，今天安排复习。",
        LeetCodePracticeResult.WITH_HINT: "上次提示后完成，今天安排复习。",
        LeetCodePracticeResult.INDEPENDENT: "已到间隔复习时间，今天再次验证。",
    }
    return labels.get(result, "已到复习时间，今天再次练习。")
