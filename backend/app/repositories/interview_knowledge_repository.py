from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Protocol, Tuple

from app.models.interview_knowledge import (
    KnowledgeAnswerSource,
    KnowledgeAssignment,
    KnowledgeAssignmentStatus,
    KnowledgeAssignmentType,
    KnowledgeAttempt,
    KnowledgeDeliveryType,
    KnowledgeProgress,
    KnowledgeQuestion,
    KnowledgeSubscription,
)


class DuplicateKnowledgeSubmissionError(RuntimeError):
    pass


class KnowledgeAssignmentInProgressError(RuntimeError):
    pass


class ConcurrentKnowledgeProgressUpdateError(RuntimeError):
    pass


class InterviewKnowledgeRepository(Protocol):
    def upsert_questions(self, questions: List[KnowledgeQuestion]) -> None: ...
    def list_questions(self) -> List[KnowledgeQuestion]: ...
    def get_question(self, question_id: str) -> Optional[KnowledgeQuestion]: ...
    def list_assignments(
        self, owner_id: str, assigned_on: Optional[date] = None
    ) -> List[KnowledgeAssignment]: ...
    def create_assignment(
        self,
        owner_id: str,
        question_id: str,
        assigned_on: date,
        assignment_type: KnowledgeAssignmentType,
        recommendation_reason: str,
    ) -> KnowledgeAssignment: ...
    def save_assignment(self, assignment: KnowledgeAssignment) -> None: ...
    def create_attempt(
        self,
        owner_id: str,
        question_id: str,
        assignment_id: Optional[str],
        answer_text: str,
        answer_source: KnowledgeAnswerSource,
        submitted_at: datetime,
        submission_id: Optional[str] = None,
    ) -> KnowledgeAttempt: ...
    def begin_attempt(
        self,
        owner_id: str,
        question_id: str,
        assignment_id: str,
        answer_text: str,
        answer_source: KnowledgeAnswerSource,
        submitted_at: datetime,
        submission_id: Optional[str] = None,
        existing_attempt_id: Optional[str] = None,
    ) -> KnowledgeAttempt: ...
    def fail_attempt(self, attempt: KnowledgeAttempt) -> None: ...
    def complete_attempt(
        self,
        attempt: KnowledgeAttempt,
        assignment: KnowledgeAssignment,
        progress: KnowledgeProgress,
        expected_progress_attempt_count: int,
    ) -> None: ...
    def save_attempt(self, attempt: KnowledgeAttempt) -> None: ...
    def get_attempt(self, owner_id: str, attempt_id: str) -> Optional[KnowledgeAttempt]: ...
    def get_attempt_by_submission_id(
        self, owner_id: str, submission_id: str
    ) -> Optional[KnowledgeAttempt]: ...
    def list_attempts(
        self, owner_id: str, question_id: Optional[str] = None
    ) -> List[KnowledgeAttempt]: ...
    def get_progress(self, owner_id: str, question_id: str) -> Optional[KnowledgeProgress]: ...
    def list_progress(self, owner_id: str) -> List[KnowledgeProgress]: ...
    def save_progress(self, progress: KnowledgeProgress) -> None: ...
    def save_subscription(self, subscription: KnowledgeSubscription) -> None: ...
    def get_subscription(self, owner_id: str) -> Optional[KnowledgeSubscription]: ...
    def list_enabled_subscriptions(self) -> List[KnowledgeSubscription]: ...
    def has_delivery(
        self, owner_id: str, delivery_on: date, delivery_type: KnowledgeDeliveryType
    ) -> bool: ...
    def record_delivery(
        self, owner_id: str, delivery_on: date, delivery_type: KnowledgeDeliveryType
    ) -> None: ...


@dataclass
class InMemoryInterviewKnowledgeRepository:
    questions: List[KnowledgeQuestion] = field(default_factory=list)
    assignments: List[KnowledgeAssignment] = field(default_factory=list)
    attempts: List[KnowledgeAttempt] = field(default_factory=list)
    progress: Dict[Tuple[str, str], KnowledgeProgress] = field(default_factory=dict)
    subscriptions: Dict[str, KnowledgeSubscription] = field(default_factory=dict)
    deliveries: set[Tuple[str, date, KnowledgeDeliveryType]] = field(default_factory=set)

    def upsert_questions(self, questions: List[KnowledgeQuestion]) -> None:
        self.questions = list(questions)

    def list_questions(self) -> List[KnowledgeQuestion]:
        return sorted(
            [question for question in self.questions if question.enabled],
            key=lambda item: (
                item.module_order,
                item.chapter_order,
                item.question_order,
                item.id,
            ),
        )

    def get_question(self, question_id: str) -> Optional[KnowledgeQuestion]:
        return next((item for item in self.list_questions() if item.id == question_id), None)

    def list_assignments(
        self, owner_id: str, assigned_on: Optional[date] = None
    ) -> List[KnowledgeAssignment]:
        return [
            item
            for item in self.assignments
            if item.owner_id == owner_id
            and (assigned_on is None or item.assigned_on == assigned_on)
        ]

    def create_assignment(
        self,
        owner_id: str,
        question_id: str,
        assigned_on: date,
        assignment_type: KnowledgeAssignmentType,
        recommendation_reason: str,
    ) -> KnowledgeAssignment:
        existing = next(
            (
                item
                for item in self.assignments
                if item.owner_id == owner_id
                and item.question_id == question_id
                and item.assigned_on == assigned_on
            ),
            None,
        )
        if existing is not None:
            return existing
        assignment = KnowledgeAssignment(
            id=f"knowledge_assignment_{len(self.assignments) + 1}",
            owner_id=owner_id,
            question_id=question_id,
            assigned_on=assigned_on,
            assignment_type=assignment_type,
            recommendation_reason=recommendation_reason,
        )
        self.assignments.append(assignment)
        return assignment

    def save_assignment(self, assignment: KnowledgeAssignment) -> None:
        for index, existing in enumerate(self.assignments):
            if existing.id == assignment.id and existing.owner_id == assignment.owner_id:
                self.assignments[index] = assignment
                return

    def create_attempt(
        self,
        owner_id: str,
        question_id: str,
        assignment_id: Optional[str],
        answer_text: str,
        answer_source: KnowledgeAnswerSource,
        submitted_at: datetime,
        submission_id: Optional[str] = None,
    ) -> KnowledgeAttempt:
        if submission_id and self.get_attempt_by_submission_id(owner_id, submission_id):
            raise DuplicateKnowledgeSubmissionError(
                "Knowledge submission already exists."
            )
        attempt = KnowledgeAttempt(
            id=f"knowledge_attempt_{len(self.attempts) + 1}",
            owner_id=owner_id,
            question_id=question_id,
            assignment_id=assignment_id,
            answer_text=answer_text,
            answer_source=answer_source,
            submitted_at=submitted_at,
            submission_id=submission_id,
        )
        self.attempts.append(attempt)
        return attempt

    def begin_attempt(
        self,
        owner_id: str,
        question_id: str,
        assignment_id: str,
        answer_text: str,
        answer_source: KnowledgeAnswerSource,
        submitted_at: datetime,
        submission_id: Optional[str] = None,
        existing_attempt_id: Optional[str] = None,
    ) -> KnowledgeAttempt:
        assignment = next(
            (
                item
                for item in self.assignments
                if item.id == assignment_id and item.owner_id == owner_id
            ),
            None,
        )
        if assignment is None or assignment.question_id != question_id:
            raise ValueError("Knowledge assignment was not found.")
        if assignment.active_attempt_id is not None:
            raise KnowledgeAssignmentInProgressError(
                "This knowledge assignment is already being evaluated."
            )
        if existing_attempt_id:
            attempt = self.get_attempt(owner_id, existing_attempt_id)
            if (
                attempt is None
                or attempt.assignment_id != assignment_id
                or attempt.evaluation_status not in {"pending", "failed"}
            ):
                raise ValueError("Knowledge attempt cannot be retried.")
            if attempt.evaluation_status == "pending":
                raise KnowledgeAssignmentInProgressError(
                    "This answer is already being evaluated."
                )
            attempt.submitted_at = submitted_at
            attempt.evaluation_status = "pending"
            attempt.evaluation_payload = {}
            attempt.score = None
        else:
            attempt = self.create_attempt(
                owner_id=owner_id,
                question_id=question_id,
                assignment_id=assignment_id,
                answer_text=answer_text,
                answer_source=answer_source,
                submitted_at=submitted_at,
                submission_id=submission_id,
            )
        assignment.active_attempt_id = attempt.id
        assignment.active_attempt_started_at = datetime.now().astimezone()
        return attempt

    def fail_attempt(self, attempt: KnowledgeAttempt) -> None:
        attempt.evaluation_status = "failed"
        assignment = next(
            (
                item
                for item in self.assignments
                if item.id == attempt.assignment_id and item.owner_id == attempt.owner_id
            ),
            None,
        )
        if assignment is not None and assignment.active_attempt_id == attempt.id:
            assignment.active_attempt_id = None
            assignment.active_attempt_started_at = None

    def complete_attempt(
        self,
        attempt: KnowledgeAttempt,
        assignment: KnowledgeAssignment,
        progress: KnowledgeProgress,
        expected_progress_attempt_count: int,
    ) -> None:
        stored_assignment = next(
            (
                item
                for item in self.assignments
                if item.id == assignment.id and item.owner_id == assignment.owner_id
            ),
            None,
        )
        if (
            stored_assignment is None
            or stored_assignment.active_attempt_id != attempt.id
        ):
            raise KnowledgeAssignmentInProgressError(
                "This knowledge assignment is no longer owned by this attempt."
            )
        stored_progress = self.progress.get((progress.owner_id, progress.question_id))
        stored_attempt_count = stored_progress.attempt_count if stored_progress else 0
        if stored_attempt_count != expected_progress_attempt_count:
            raise ConcurrentKnowledgeProgressUpdateError(
                "Knowledge progress changed during evaluation."
            )
        self.save_attempt(attempt)
        self.save_progress(progress)
        self.save_assignment(assignment)

    def save_attempt(self, attempt: KnowledgeAttempt) -> None:
        return None

    def get_attempt(self, owner_id: str, attempt_id: str) -> Optional[KnowledgeAttempt]:
        return next(
            (
                attempt
                for attempt in self.attempts
                if attempt.owner_id == owner_id and attempt.id == attempt_id
            ),
            None,
        )

    def get_attempt_by_submission_id(
        self, owner_id: str, submission_id: str
    ) -> Optional[KnowledgeAttempt]:
        return next(
            (
                attempt
                for attempt in self.attempts
                if attempt.owner_id == owner_id
                and attempt.submission_id == submission_id
            ),
            None,
        )

    def list_attempts(
        self, owner_id: str, question_id: Optional[str] = None
    ) -> List[KnowledgeAttempt]:
        return [
            attempt
            for attempt in self.attempts
            if attempt.owner_id == owner_id
            and (question_id is None or attempt.question_id == question_id)
        ]

    def get_progress(self, owner_id: str, question_id: str) -> Optional[KnowledgeProgress]:
        return self.progress.get((owner_id, question_id))

    def list_progress(self, owner_id: str) -> List[KnowledgeProgress]:
        return [item for (item_owner, _), item in self.progress.items() if item_owner == owner_id]

    def save_progress(self, progress: KnowledgeProgress) -> None:
        self.progress[(progress.owner_id, progress.question_id)] = progress

    def save_subscription(self, subscription: KnowledgeSubscription) -> None:
        self.subscriptions[subscription.owner_id] = subscription

    def get_subscription(self, owner_id: str) -> Optional[KnowledgeSubscription]:
        return self.subscriptions.get(owner_id)

    def list_enabled_subscriptions(self) -> List[KnowledgeSubscription]:
        return [item for item in self.subscriptions.values() if item.enabled]

    def has_delivery(
        self, owner_id: str, delivery_on: date, delivery_type: KnowledgeDeliveryType
    ) -> bool:
        return (owner_id, delivery_on, delivery_type) in self.deliveries

    def record_delivery(
        self, owner_id: str, delivery_on: date, delivery_type: KnowledgeDeliveryType
    ) -> None:
        self.deliveries.add((owner_id, delivery_on, delivery_type))
