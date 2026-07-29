import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.models.interview_knowledge import KnowledgeQuestion
from app.services.knowledge_markdown import markdown_table_rows
from app.services.llm_service import LLMConfigurationError, LLMRequestError, LLMService


class KnowledgeEvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class KnowledgeEvaluationResult:
    score: int
    matched_required_point_ids: List[str]
    matched_bonus_point_ids: List[str]
    triggered_misconception_ids: List[str]
    missing_required_labels: List[str]
    misconception_labels: List[str]
    evidence: List[Dict[str, str]]
    feedback: str
    suggested_improvement: str
    organization_score: int
    clarity_score: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "matched_required_point_ids": self.matched_required_point_ids,
            "matched_bonus_point_ids": self.matched_bonus_point_ids,
            "triggered_misconception_ids": self.triggered_misconception_ids,
            "missing_required_labels": self.missing_required_labels,
            "misconception_labels": self.misconception_labels,
            "evidence": self.evidence,
            "feedback": self.feedback,
            "suggested_improvement": self.suggested_improvement,
            "organization_score": self.organization_score,
            "clarity_score": self.clarity_score,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "KnowledgeEvaluationResult":
        return cls(**{field: payload[field] for field in cls.__dataclass_fields__})


class KnowledgeEvaluationService:
    def __init__(self, llm_service: Optional[LLMService] = None) -> None:
        self.llm_service = llm_service or LLMService()

    async def evaluate(
        self, question: KnowledgeQuestion, answer_text: str
    ) -> KnowledgeEvaluationResult:
        if not answer_text.strip():
            return skipped_evaluation(question)
        base_prompt = _evaluation_prompt(question, answer_text)
        payload = None
        parse_error: Optional[Exception] = None
        for attempt in range(2):
            retry_instruction = (
                "\n\n上一次响应不是完整合法的 JSON。请缩短反馈和证据，只输出完整 JSON。"
                if attempt
                else ""
            )
            try:
                result = await self.llm_service.generate_text(
                    prompt=base_prompt + retry_instruction,
                    system_prompt=(
                        "你是严格但友善的计算机面试知识评分器。只根据给定评分点判断，"
                        "不得发明评分点；必须输出 JSON，不要输出 Markdown 以外的解释。"
                    ),
                    temperature=0.0,
                    max_tokens=2400,
                    response_format={"type": "json_object"},
                    thinking={"type": "disabled"},
                )
            except (LLMConfigurationError, LLMRequestError) as exc:
                raise KnowledgeEvaluationError(
                    "AI 评价暂时不可用，你的答案已保留，请稍后重试。"
                ) from exc
            try:
                payload = _parse_json_object(result.content)
                break
            except (ValueError, json.JSONDecodeError) as exc:
                parse_error = exc
        if payload is None:
            raise KnowledgeEvaluationError(
                "AI 评价返回了无效 JSON，请稍后重试。"
            ) from parse_error
        return _validated_evaluation(question, answer_text, payload)


def skipped_evaluation(question: KnowledgeQuestion) -> KnowledgeEvaluationResult:
    return KnowledgeEvaluationResult(
        score=0,
        matched_required_point_ids=[],
        matched_bonus_point_ids=[],
        triggered_misconception_ids=[],
        missing_required_labels=[point.label for point in question.required_points],
        misconception_labels=[],
        evidence=[],
        feedback="本次选择了“我不会”，先阅读参考答案，明天会再次复习。",
        suggested_improvement="阅读后尝试用自己的话复述，不要只记忆原文。",
        organization_score=0,
        clarity_score=0,
    )


def _validated_evaluation(
    question: KnowledgeQuestion,
    answer_text: str,
    payload: Dict[str, Any],
) -> KnowledgeEvaluationResult:
    required = {point.id: point for point in question.required_points}
    bonus = {point.id: point for point in question.bonus_points}
    misconceptions = {point.id: point for point in question.misconception_points}
    claimed_required = _known_ids(payload.get("matched_required_point_ids"), required)
    claimed_bonus = _known_ids(payload.get("matched_bonus_point_ids"), bonus)
    claimed_misconceptions = _known_ids(
        payload.get("triggered_misconception_ids"), misconceptions
    )
    raw_evidence = payload.get("evidence", [])
    if isinstance(raw_evidence, dict):
        evidence_items = [
            {"point_id": point_id, "quote": quote}
            for point_id, quote in raw_evidence.items()
        ]
    elif isinstance(raw_evidence, list):
        evidence_items = raw_evidence
    else:
        evidence_items = []
    evidence = []
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        point_id = item.get("point_id")
        quote = item.get("quote")
        if (
            isinstance(point_id, str)
            and point_id in {*required, *bonus, *misconceptions}
            and isinstance(quote, str)
            and quote
            and _is_supported_evidence_quote(quote, answer_text)
        ):
            evidence.append({"point_id": point_id, "quote": quote})
    evidenced_ids = {item["point_id"] for item in evidence}
    matched_required = [item for item in claimed_required if item in evidenced_ids]
    matched_bonus = [item for item in claimed_bonus if item in evidenced_ids]
    triggered = [item for item in claimed_misconceptions if item in evidenced_ids]
    organization = _bounded_int(payload.get("organization_score"), 0, 10)
    clarity = _bounded_int(payload.get("clarity_score"), 0, 10)
    total_required_weight = sum(max(point.weight, 1) for point in required.values())
    matched_weight = sum(max(required[item].weight, 1) for item in matched_required)
    coverage_score = round(60 * matched_weight / total_required_weight)
    accuracy_score = 0 if not matched_required else max(0, 20 - 10 * len(triggered))
    bonus_score = min(2 * len(matched_bonus), 10)
    score = max(
        0,
        min(100, coverage_score + accuracy_score + organization + clarity + bonus_score),
    )
    return KnowledgeEvaluationResult(
        score=score,
        matched_required_point_ids=matched_required,
        matched_bonus_point_ids=matched_bonus,
        triggered_misconception_ids=triggered,
        missing_required_labels=[
            point.label for point in question.required_points if point.id not in matched_required
        ],
        misconception_labels=[misconceptions[item].label for item in triggered],
        evidence=evidence,
        feedback=str(payload.get("feedback", "")).strip() or "已完成评价。",
        suggested_improvement=(
            str(payload.get("suggested_improvement", "")).strip()
            or "对照遗漏点重新组织一次答案。"
        ),
        organization_score=organization,
        clarity_score=clarity,
    )


def _known_ids(value: Any, allowed: Dict[str, Any]) -> List[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, str) and item in allowed and item not in result:
            result.append(item)
    return result


def _bounded_int(value: Any, lower: int, upper: int) -> int:
    try:
        return min(max(int(value), lower), upper)
    except (TypeError, ValueError):
        return lower


def _is_supported_evidence_quote(quote: str, answer_text: str) -> bool:
    if quote in answer_text:
        return True
    normalized_quote = _without_whitespace(quote)
    for cells in markdown_table_rows(answer_text):
        for start in range(len(cells)):
            for end in range(start + 1, len(cells) + 1):
                if _without_whitespace("；".join(cells[start:end])) == normalized_quote:
                    return True
    return False


def _without_whitespace(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _parse_json_object(content: str) -> Dict[str, Any]:
    stripped = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("Evaluation payload must be an object.")
    return payload


def _evaluation_prompt(question: KnowledgeQuestion, answer_text: str) -> str:
    points = [point.to_dict() for point in question.rubric_points]
    return (
        "请评价用户回答。只能使用下列评分点 ID。证据 quote 必须逐字来自用户回答。\n\n"
        f"题目：{question.prompt}\n"
        f"评分点：{json.dumps(points, ensure_ascii=False)}\n"
        f"用户回答：{answer_text}\n\n"
        "严格输出一个 JSON 对象，字段为：matched_required_point_ids、"
        "matched_bonus_point_ids、triggered_misconception_ids、evidence、"
        "organization_score(0-10)、clarity_score(0-10)、feedback、"
        "suggested_improvement。evidence 必须是数组，数组元素格式必须为"
        '{"point_id":"评分点ID","quote":"逐字来自用户回答的最短原文片段"}；'
        "不要把 evidence 输出成以评分点 ID 为键的对象。每个声称命中的评分点"
        "都必须有一条 evidence，quote 控制在 4～30 个字，没有逐字证据就不要"
        "声称命中。feedback 和 suggested_improvement 各不超过 120 个字。"
    )
