from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Protocol, Tuple

from app.models.leetcode import (
    LeetCodeAssignment,
    LeetCodeAssignmentStatus,
    LeetCodeAssignmentType,
    LeetCodeDeliveryType,
    LeetCodePracticeResult,
    LeetCodeProblem,
    LeetCodeProgress,
    LeetCodeSubscription,
)


class LeetCodeRepository(Protocol):
    def upsert_problems(self, problems: List[LeetCodeProblem]) -> None: ...
    def list_problems(self) -> List[LeetCodeProblem]: ...
    def list_assignments(
        self,
        owner_id: str,
        assigned_on: Optional[date] = None,
    ) -> List[LeetCodeAssignment]: ...
    def create_assignment(
        self,
        owner_id: str,
        problem_id: str,
        assigned_on: date,
        assignment_type: LeetCodeAssignmentType,
        recommendation_reason: str,
    ) -> LeetCodeAssignment: ...
    def save_assignment(self, assignment: LeetCodeAssignment) -> None: ...
    def get_progress(self, owner_id: str, problem_id: str) -> Optional[LeetCodeProgress]: ...
    def list_progress(self, owner_id: str) -> List[LeetCodeProgress]: ...
    def save_progress(self, progress: LeetCodeProgress) -> None: ...
    def save_subscription(self, subscription: LeetCodeSubscription) -> None: ...
    def get_subscription(self, owner_id: str) -> Optional[LeetCodeSubscription]: ...
    def list_enabled_subscriptions(self) -> List[LeetCodeSubscription]: ...
    def has_delivery(
        self, owner_id: str, delivery_on: date, delivery_type: LeetCodeDeliveryType
    ) -> bool: ...
    def record_delivery(
        self, owner_id: str, delivery_on: date, delivery_type: LeetCodeDeliveryType
    ) -> None: ...
    def get_delivery_attempts(
        self, owner_id: str, delivery_on: date, delivery_type: LeetCodeDeliveryType
    ) -> int: ...
    def record_delivery_attempt(
        self, owner_id: str, delivery_on: date, delivery_type: LeetCodeDeliveryType
    ) -> None: ...


@dataclass
class InMemoryLeetCodeRepository:
    problems: List[LeetCodeProblem] = field(default_factory=list)
    assignments: List[LeetCodeAssignment] = field(default_factory=list)
    progress: Dict[Tuple[str, str], LeetCodeProgress] = field(default_factory=dict)
    subscriptions: Dict[str, LeetCodeSubscription] = field(default_factory=dict)
    deliveries: set[Tuple[str, date, LeetCodeDeliveryType]] = field(default_factory=set)
    delivery_attempts: Dict[Tuple[str, date, LeetCodeDeliveryType], int] = field(
        default_factory=dict
    )

    def upsert_problems(self, problems: List[LeetCodeProblem]) -> None:
        incoming_sources = {problem.source for problem in problems}
        by_id = {
            problem.id: problem
            for problem in self.problems
            if problem.source not in incoming_sources
        }
        by_id.update({problem.id: problem for problem in problems})
        self.problems = list(by_id.values())

    def list_problems(self) -> List[LeetCodeProblem]:
        return list(self.problems)

    def list_assignments(
        self,
        owner_id: str,
        assigned_on: Optional[date] = None,
    ) -> List[LeetCodeAssignment]:
        return [
            assignment
            for assignment in self.assignments
            if assignment.owner_id == owner_id
            and (assigned_on is None or assignment.assigned_on == assigned_on)
        ]

    def create_assignment(
        self,
        owner_id: str,
        problem_id: str,
        assigned_on: date,
        assignment_type: LeetCodeAssignmentType,
        recommendation_reason: str,
    ) -> LeetCodeAssignment:
        for assignment in self.assignments:
            if (
                assignment.owner_id == owner_id
                and assignment.problem_id == problem_id
                and assignment.assigned_on == assigned_on
            ):
                return assignment
        assignment = LeetCodeAssignment(
            id=f"leetcode_assignment_{len(self.assignments) + 1}",
            owner_id=owner_id,
            problem_id=problem_id,
            assigned_on=assigned_on,
            assignment_type=assignment_type,
            recommendation_reason=recommendation_reason,
        )
        self.assignments.append(assignment)
        return assignment

    def save_assignment(self, assignment: LeetCodeAssignment) -> None:
        return None

    def get_progress(self, owner_id: str, problem_id: str) -> Optional[LeetCodeProgress]:
        return self.progress.get((owner_id, problem_id))

    def list_progress(self, owner_id: str) -> List[LeetCodeProgress]:
        return [item for (item_owner, _), item in self.progress.items() if item_owner == owner_id]

    def save_progress(self, progress: LeetCodeProgress) -> None:
        self.progress[(progress.owner_id, progress.problem_id)] = progress

    def save_subscription(self, subscription: LeetCodeSubscription) -> None:
        self.subscriptions[subscription.owner_id] = subscription

    def get_subscription(self, owner_id: str) -> Optional[LeetCodeSubscription]:
        return self.subscriptions.get(owner_id)

    def list_enabled_subscriptions(self) -> List[LeetCodeSubscription]:
        return [subscription for subscription in self.subscriptions.values() if subscription.enabled]

    def has_delivery(
        self, owner_id: str, delivery_on: date, delivery_type: LeetCodeDeliveryType
    ) -> bool:
        return (owner_id, delivery_on, delivery_type) in self.deliveries

    def record_delivery(
        self, owner_id: str, delivery_on: date, delivery_type: LeetCodeDeliveryType
    ) -> None:
        self.deliveries.add((owner_id, delivery_on, delivery_type))

    def get_delivery_attempts(
        self, owner_id: str, delivery_on: date, delivery_type: LeetCodeDeliveryType
    ) -> int:
        return self.delivery_attempts.get((owner_id, delivery_on, delivery_type), 0)

    def record_delivery_attempt(
        self, owner_id: str, delivery_on: date, delivery_type: LeetCodeDeliveryType
    ) -> None:
        key = (owner_id, delivery_on, delivery_type)
        self.delivery_attempts[key] = self.delivery_attempts.get(key, 0) + 1
