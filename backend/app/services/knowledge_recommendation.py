from dataclasses import dataclass
from datetime import date
from typing import List

from app.models.interview_knowledge import (
    KnowledgeAssignment,
    KnowledgeAssignmentType,
    KnowledgeMasteryStatus,
    KnowledgeQuestion,
)
from app.repositories.interview_knowledge_repository import InterviewKnowledgeRepository


@dataclass(frozen=True)
class KnowledgeRecommendation:
    assignment: KnowledgeAssignment
    question: KnowledgeQuestion


class KnowledgeRecommendationWorkflow:
    def __init__(self, repository: InterviewKnowledgeRepository) -> None:
        self.repository = repository

    def get_today(
        self, owner_id: str, today: date, limit: int = 5
    ) -> List[KnowledgeRecommendation]:
        assignments = [
            item
            for item in self.repository.list_assignments(owner_id, today)
            if item.assignment_type != KnowledgeAssignmentType.MANUAL
        ]
        if not assignments:
            self._create_today_assignments(owner_id, today, limit)
            assignments = [
                item
                for item in self.repository.list_assignments(owner_id, today)
                if item.assignment_type != KnowledgeAssignmentType.MANUAL
            ]
        questions = {question.id: question for question in self.repository.list_questions()}
        return [
            KnowledgeRecommendation(assignment=item, question=questions[item.question_id])
            for item in assignments
            if item.question_id in questions
        ]

    def _create_today_assignments(self, owner_id: str, today: date, limit: int) -> None:
        questions = self.repository.list_questions()
        question_ids = {question.id for question in questions}
        all_assignments = self.repository.list_assignments(owner_id)
        seen = {item.question_id for item in all_assignments}
        progress = [
            item for item in self.repository.list_progress(owner_id)
            if item.question_id in question_ids
        ]
        due = sorted(
            [
                item for item in progress
                if item.next_review_on is not None and item.next_review_on <= today
            ],
            key=lambda item: (
                item.next_review_on or today,
                item.mastery_score,
                item.question_id,
            ),
        )
        selected: set[str] = set()
        for item in due[:2]:
            self._assign(
                owner_id, item.question_id, today,
                KnowledgeAssignmentType.DUE_REVIEW,
                "已到间隔复习时间，今天再次验证。",
            )
            selected.add(item.question_id)

        weak = sorted(
            [
                item for item in progress
                if item.question_id not in selected
                and item.mastery_status != KnowledgeMasteryStatus.MASTERED
                and (item.last_score is None or item.last_score < 60)
            ],
            key=lambda item: (item.mastery_score, item.question_id),
        )
        if weak and len(selected) < limit:
            self._assign(
                owner_id, weak[0].question_id, today,
                KnowledgeAssignmentType.WEAKNESS,
                "上次回答存在关键遗漏，今天重点巩固。",
            )
            selected.add(weak[0].question_id)

        for question in questions:
            if len(selected) >= limit:
                break
            if question.id in selected or question.id in seen:
                continue
            self._assign(
                owner_id, question.id, today,
                KnowledgeAssignmentType.NEW,
                f"继续学习{question.module_title} · {question.chapter_title}。",
            )
            selected.add(question.id)

    def _assign(
        self,
        owner_id: str,
        question_id: str,
        today: date,
        assignment_type: KnowledgeAssignmentType,
        reason: str,
    ) -> None:
        self.repository.create_assignment(
            owner_id=owner_id,
            question_id=question_id,
            assigned_on=today,
            assignment_type=assignment_type,
            recommendation_reason=reason,
        )
