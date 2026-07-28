import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from app.models.interview_knowledge import (
    KnowledgeDifficulty,
    KnowledgeQuestion,
    KnowledgeRubricKind,
    KnowledgeRubricPoint,
)


_CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "interview_knowledge.json"


@dataclass(frozen=True)
class KnowledgeCatalog:
    version: str
    questions: List[KnowledgeQuestion]

    @property
    def modules(self) -> List[str]:
        ordered = sorted(
            {(item.module_order, item.module_id) for item in self.questions}
        )
        return [module_id for _, module_id in ordered]


def load_knowledge_catalog(path: Path = _CATALOG_PATH) -> KnowledgeCatalog:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError("Interview knowledge catalog must contain questions.")
    questions = [_question_from_dict(item) for item in raw_questions]
    ids = [question.id for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Interview knowledge question IDs must be unique.")
    questions.sort(
        key=lambda item: (
            item.module_order,
            item.chapter_order,
            item.question_order,
            item.id,
        )
    )
    return KnowledgeCatalog(version=str(payload.get("version", "")), questions=questions)


def _question_from_dict(value: Dict[str, Any]) -> KnowledgeQuestion:
    points = [
        KnowledgeRubricPoint(
            id=str(point["id"]),
            kind=KnowledgeRubricKind(point["kind"]),
            label=str(point["label"]),
            description=str(point["description"]),
            weight=int(point.get("weight", 1)),
        )
        for point in value.get("rubric_points", [])
    ]
    if not any(point.kind == KnowledgeRubricKind.REQUIRED for point in points):
        raise ValueError(f"Question {value.get('id')} has no required rubric points.")
    return KnowledgeQuestion(
        id=str(value["id"]),
        module_id=str(value["module_id"]),
        module_title=str(value["module_title"]),
        module_order=int(value["module_order"]),
        chapter_id=str(value["chapter_id"]),
        chapter_title=str(value["chapter_title"]),
        chapter_order=int(value["chapter_order"]),
        question_order=int(value["question_order"]),
        prompt=str(value["prompt"]),
        difficulty=KnowledgeDifficulty(value["difficulty"]),
        frequency=int(value.get("frequency", 1)),
        short_reference_answer=str(value["short_reference_answer"]),
        full_reference_answer=str(value["full_reference_answer"]),
        rubric_points=points,
        hint=str(value.get("hint", "")),
        source_title=str(value.get("source_title", "")),
        source_url=str(value.get("source_url", "")),
        source_chapter=str(value.get("source_chapter", "")),
        source_page_start=value.get("source_page_start"),
        source_page_end=value.get("source_page_end"),
        enabled=bool(value.get("enabled", True)),
        source_file=str(value.get("source_file", "")),
        source_heading=str(value.get("source_heading", "")),
        content_hash=str(value.get("content_hash", "")),
        keywords=[str(item) for item in value.get("keywords", [])],
    )
