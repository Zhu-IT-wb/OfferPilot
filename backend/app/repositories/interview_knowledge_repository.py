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
        by_id = {question.id: question for question in self.questions}
        by_id.update({question.id: question for question in questions})
        self.questions = list(by_id.values())

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
        return None

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
