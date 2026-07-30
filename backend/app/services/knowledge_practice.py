import asyncio
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import Optional

from app.models.interview_knowledge import (
    KnowledgeAnswerSource,
    KnowledgeAssignment,
    KnowledgeAssignmentStatus,
    KnowledgeAttempt,
    KnowledgeMasteryStatus,
    KnowledgeProgress,
)
from app.repositories.interview_knowledge_repository import (
    ConcurrentKnowledgeProgressUpdateError,
    DuplicateKnowledgeSubmissionError,
    InterviewKnowledgeRepository,
    KnowledgeAssignmentInProgressError,
)
from app.services.knowledge_evaluation import (
    KnowledgeEvaluationResult,
    KnowledgeEvaluationService,
    skipped_evaluation,
)


@dataclass(frozen=True)
class KnowledgePracticeOutcome:
    assignment: KnowledgeAssignment
    attempt: KnowledgeAttempt
    evaluation: KnowledgeEvaluationResult
    progress: KnowledgeProgress


class KnowledgeSubmissionInProgressError(RuntimeError):
    pass


class KnowledgePracticeWorkflow:
    def __init__(
        self,
        repository: InterviewKnowledgeRepository,
        evaluation_service: KnowledgeEvaluationService,
    ) -> None:
        self.repository = repository
        self.evaluation_service = evaluation_service

    async def submit_answer(
        self,
        owner_id: str,
        assignment_id: str,
        answer_text: str,
        answer_source: KnowledgeAnswerSource,
        submitted_at: datetime,
        submission_id: str = "",
    ) -> KnowledgePracticeOutcome:
        assignment = next(
            (
                item
                for item in self.repository.list_assignments(owner_id)
                if item.id == assignment_id
            ),
            None,
        )
        if assignment is None:
            raise ValueError("Knowledge assignment was not found.")
        question = self.repository.get_question(assignment.question_id)
        if question is None:
            raise ValueError("Knowledge question was not found.")
        normalized_answer = answer_text.strip()
        if answer_source != KnowledgeAnswerSource.SKIPPED and not normalized_answer:
            raise ValueError("Answer text is required.")
        existing_attempt = (
            self.repository.get_attempt_by_submission_id(owner_id, submission_id)
            if submission_id
            else None
        )
        if existing_attempt is not None:
            if (
                existing_attempt.assignment_id != assignment.id
                or existing_attempt.answer_text != normalized_answer
                or existing_attempt.answer_source != answer_source
            ):
                raise ValueError("Submission ID was already used for another answer.")
            if existing_attempt.evaluation_status == "completed":
                return self._completed_outcome(
                    owner_id, assignment, existing_attempt
                )
        try:
            attempt = self.repository.begin_attempt(
                owner_id=owner_id,
                question_id=question.id,
                assignment_id=assignment.id,
                answer_text=normalized_answer,
                answer_source=answer_source,
                submitted_at=submitted_at,
                submission_id=submission_id or None,
                existing_attempt_id=(existing_attempt.id if existing_attempt else None),
            )
        except DuplicateKnowledgeSubmissionError as exc:
            concurrent_attempt = self.repository.get_attempt_by_submission_id(
                owner_id, submission_id
            )
            if (
                concurrent_attempt is not None
                and concurrent_attempt.evaluation_status == "completed"
            ):
                return self._completed_outcome(
                    owner_id, assignment, concurrent_attempt
                )
            raise KnowledgeSubmissionInProgressError(
                "This answer is already being evaluated."
            ) from exc
        except KnowledgeAssignmentInProgressError as exc:
            raise KnowledgeSubmissionInProgressError(str(exc)) from exc
        try:
            evaluation = (
                skipped_evaluation(question)
                if answer_source == KnowledgeAnswerSource.SKIPPED
                else await self.evaluation_service.evaluate(question, normalized_answer)
            )
        except asyncio.CancelledError:
            self.repository.fail_attempt(attempt)
            raise
        except Exception:
            self.repository.fail_attempt(attempt)
            raise
        attempt.score = evaluation.score
        attempt.evaluation_status = "completed"
        attempt.evaluation_payload = evaluation.to_dict()
        completed_assignment = replace(
            assignment,
            status=KnowledgeAssignmentStatus.COMPLETED,
            attempt_id=attempt.id,
            completed_at=submitted_at,
            active_attempt_id=None,
            active_attempt_started_at=None,
        )
        progress = None
        for _ in range(5):
            current_progress = self.repository.get_progress(owner_id, question.id)
            expected_attempt_count = (
                current_progress.attempt_count if current_progress is not None else 0
            )
            progress = self._next_progress(
                current=current_progress,
                owner_id=owner_id,
                question_id=question.id,
                evaluation=evaluation,
                practiced_on=submitted_at.date(),
                submitted_at=submitted_at,
            )
            try:
                self.repository.complete_attempt(
                    attempt=attempt,
                    assignment=completed_assignment,
                    progress=progress,
                    expected_progress_attempt_count=expected_attempt_count,
                )
                break
            except ConcurrentKnowledgeProgressUpdateError:
                continue
            except KnowledgeAssignmentInProgressError as exc:
                self.repository.fail_attempt(attempt)
                raise KnowledgeSubmissionInProgressError(str(exc)) from exc
        else:
            self.repository.fail_attempt(attempt)
            raise KnowledgeSubmissionInProgressError(
                "Knowledge progress changed too frequently; please retry."
            )
        return KnowledgePracticeOutcome(
            assignment=completed_assignment,
            attempt=attempt,
            evaluation=evaluation,
            progress=progress,
        )

    def _completed_outcome(
        self,
        owner_id: str,
        assignment: KnowledgeAssignment,
        attempt: KnowledgeAttempt,
    ) -> KnowledgePracticeOutcome:
        progress = self.repository.get_progress(owner_id, assignment.question_id)
        if progress is None:
            raise ValueError("Knowledge progress was not found.")
        return KnowledgePracticeOutcome(
            assignment=assignment,
            attempt=attempt,
            evaluation=KnowledgeEvaluationResult.from_dict(attempt.evaluation_payload),
            progress=progress,
        )

    def _next_progress(
        self,
        current: Optional[KnowledgeProgress],
        owner_id: str,
        question_id: str,
        evaluation: KnowledgeEvaluationResult,
        practiced_on: date,
        submitted_at: datetime,
    ) -> KnowledgeProgress:
        progress = replace(current) if current is not None else KnowledgeProgress(
            owner_id=owner_id, question_id=question_id
        )
        score = evaluation.score
        previous_attempt_on = (
            progress.last_attempt_at.date() if progress.last_attempt_at else None
        )
        previous_next_review_on = progress.next_review_on
        progress.attempt_count += 1
        progress.last_score = score
        progress.mastery_score = (
            score
            if progress.attempt_count == 1
            else round(progress.mastery_score * 0.6 + score * 0.4)
        )
        progress.last_attempt_at = submitted_at
        progress.last_detected_gaps = [
            *evaluation.missing_required_labels,
            *[
                f"事实错误：{item.explanation}"
                for item in evaluation.factual_errors
            ],
        ]
        if score < 60:
            progress.high_score_streak = 0
            progress.mastery_status = KnowledgeMasteryStatus.LEARNING
            progress.next_review_on = practiced_on + timedelta(days=1)
        elif score < 80:
            progress.high_score_streak = 0
            progress.mastery_status = KnowledgeMasteryStatus.REVIEWING
            progress.next_review_on = practiced_on + timedelta(days=3)
        else:
            spaced_attempt = (
                progress.high_score_streak == 0
                or previous_next_review_on is None
                or practiced_on >= previous_next_review_on
            )
            if previous_attempt_on != practiced_on and spaced_attempt:
                progress.high_score_streak += 1
            else:
                progress.high_score_streak = max(progress.high_score_streak, 1)
            if progress.high_score_streak >= 2:
                progress.mastery_status = KnowledgeMasteryStatus.MASTERED
                progress.next_review_on = practiced_on + timedelta(days=30)
            else:
                progress.mastery_status = KnowledgeMasteryStatus.REVIEWING
                progress.next_review_on = (
                    previous_next_review_on
                    if progress.high_score_streak > 0
                    and not spaced_attempt
                    and previous_next_review_on is not None
                    else practiced_on + timedelta(days=7 if score < 90 else 14)
                )
        return progress
