from dataclasses import dataclass
from datetime import date, datetime, timedelta

from app.models.interview_knowledge import (
    KnowledgeAnswerSource,
    KnowledgeAssignment,
    KnowledgeAssignmentStatus,
    KnowledgeAttempt,
    KnowledgeMasteryStatus,
    KnowledgeProgress,
)
from app.repositories.interview_knowledge_repository import (
    DuplicateKnowledgeSubmissionError,
    InterviewKnowledgeRepository,
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
            if existing_attempt.evaluation_status == "pending":
                raise KnowledgeSubmissionInProgressError(
                    "This answer is already being evaluated."
                )
            attempt = existing_attempt
            attempt.submitted_at = submitted_at
        else:
            try:
                attempt = self.repository.create_attempt(
                    owner_id=owner_id,
                    question_id=question.id,
                    assignment_id=assignment.id,
                    answer_text=normalized_answer,
                    answer_source=answer_source,
                    submitted_at=submitted_at,
                    submission_id=submission_id or None,
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
        try:
            evaluation = (
                skipped_evaluation(question)
                if answer_source == KnowledgeAnswerSource.SKIPPED
                else await self.evaluation_service.evaluate(question, normalized_answer)
            )
        except Exception:
            attempt.evaluation_status = "failed"
            self.repository.save_attempt(attempt)
            raise
        attempt.score = evaluation.score
        attempt.evaluation_status = "completed"
        attempt.evaluation_payload = evaluation.to_dict()
        self.repository.save_attempt(attempt)
        progress = self._update_progress(
            owner_id=owner_id,
            question_id=question.id,
            evaluation=evaluation,
            practiced_on=submitted_at.date(),
            submitted_at=submitted_at,
        )
        assignment.status = KnowledgeAssignmentStatus.COMPLETED
        assignment.attempt_id = attempt.id
        assignment.completed_at = submitted_at
        self.repository.save_assignment(assignment)
        return KnowledgePracticeOutcome(
            assignment=assignment,
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

    def _update_progress(
        self,
        owner_id: str,
        question_id: str,
        evaluation: KnowledgeEvaluationResult,
        practiced_on: date,
        submitted_at: datetime,
    ) -> KnowledgeProgress:
        progress = self.repository.get_progress(owner_id, question_id) or KnowledgeProgress(
            owner_id=owner_id,
            question_id=question_id,
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
        progress.last_detected_gaps = list(evaluation.missing_required_labels)
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
        self.repository.save_progress(progress)
        return progress
