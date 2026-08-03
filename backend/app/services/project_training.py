import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.agents.interview_agent import (
    InterviewAgentError,
    InterviewAgentGraph,
    ProjectInterviewDecision,
)
from app.models.project_training import (
    ProjectProfileStatus,
    ProjectTopicProgress,
    ProjectTrainingAnswer,
    ProjectTrainingAnswerSource,
    ProjectTrainingDifficulty,
    ProjectTrainingEvaluationStatus,
    ProjectTrainingSession,
    ProjectTrainingSessionStatus,
    ProjectTrainingSummary,
    ProjectTrainingTurn,
    ProjectTrainingTurnStatus,
    PROJECT_TRAINING_THEME_ORDER,
)
from app.repositories.project_training_repository import (
    ProjectTrainingRepository,
    ProjectTrainingSubmissionInProgressError,
)
from app.services.llm_service import LLMService


class ProjectTrainingWorkflow:
    def __init__(
        self,
        repository: ProjectTrainingRepository,
        llm_service: Optional[LLMService] = None,
    ) -> None:
        self.repository = repository
        self.llm_service = llm_service or LLMService()
        self.interview_agent = InterviewAgentGraph(self.llm_service)

    def start(
        self,
        owner_id: str,
        project_id: str,
        creation_id: str,
        target_role: str = "",
        goal: str = "",
        difficulty: ProjectTrainingDifficulty = ProjectTrainingDifficulty.MEDIUM,
        max_turns: int = 6,
        focus_topics: Optional[List[str]] = None,
    ) -> "ProjectTrainingOutcome":
        project = self.repository.get_project(owner_id, project_id)
        if project is None:
            raise ValueError("项目档案不存在。")
        if project.status == ProjectProfileStatus.ARCHIVED:
            raise ValueError("已归档项目不能开始新的训练。")
        session, turn, resumed = self.repository.create_session(
            owner_id=owner_id,
            project=project,
            creation_id=creation_id,
            target_role=target_role,
            goal=goal,
            difficulty=difficulty,
            max_turns=max_turns,
            focus_topics=focus_topics or [],
        )
        return ProjectTrainingOutcome(session=session, current_turn=turn, resumed=resumed)

    def resume(self, owner_id: str, session_id: str) -> "ProjectTrainingOutcome":
        session = self.repository.get_session(owner_id, session_id)
        if session is None:
            raise ValueError("项目训练不存在。")
        turn = self.repository.get_turn(owner_id, session.current_turn_id or "")
        return ProjectTrainingOutcome(session=session, current_turn=turn, resumed=True)

    async def submit_answer(
        self,
        owner_id: str,
        session_id: str,
        turn_id: str,
        answer_text: str,
        answer_source: ProjectTrainingAnswerSource,
        submission_id: str,
        submitted_at: datetime,
    ) -> "ProjectAnswerOutcome":
        normalized_answer = answer_text.strip()
        if not normalized_answer:
            raise ValueError("回答内容不能为空。")
        existing = self.repository.get_answer_by_submission_id(owner_id, submission_id)
        if existing is not None:
            if (
                existing.session_id != session_id
                or existing.turn_id != turn_id
                or existing.answer_text != normalized_answer
                or existing.answer_source != answer_source
            ):
                raise ValueError("submission_id 已被另一个回答使用。")
            if existing.evaluation_status == ProjectTrainingEvaluationStatus.COMPLETED:
                return self._completed_answer_outcome(owner_id, existing)
        answer, completed = self.repository.begin_answer(
            owner_id=owner_id,
            session_id=session_id,
            turn_id=turn_id,
            submission_id=submission_id,
            answer_text=normalized_answer,
            answer_source=answer_source,
            submitted_at=submitted_at,
        )
        if completed:
            return self._completed_answer_outcome(owner_id, answer)
        session = self.repository.get_session(owner_id, session_id)
        turn = self.repository.get_turn(owner_id, turn_id)
        if session is None or turn is None:
            self.repository.fail_answer(answer)
            raise ValueError("项目训练不存在。")
        project = self.repository.get_project(
            owner_id, session.project_id, version=session.project_version
        )
        if project is None:
            self.repository.fail_answer(answer)
            raise ValueError("项目历史版本不存在。")
        evidence = self.repository.list_project_evidence(
            owner_id, project.id, project.version
        )
        recent_turns = _recent_turn_payloads(
            self.repository.list_turns(owner_id, session_id),
            self.repository.list_answers(owner_id, session_id),
        )
        try:
            decision = await self.interview_agent.run(
                project=project,
                evidence=_select_evidence(evidence, turn, normalized_answer),
                session=session,
                turn=turn,
                answer_text=normalized_answer,
                recent_turns=recent_turns,
            )
        except BaseException:
            self.repository.fail_answer(answer)
            raise
        now = datetime.now().astimezone()
        answer.evaluation_status = ProjectTrainingEvaluationStatus.COMPLETED
        answer.evaluation_payload = decision.evaluation_payload()
        answer.overall_score = decision.overall_score
        turn.status = ProjectTrainingTurnStatus.COMPLETED
        turn.answer_id = answer.id
        turn.completed_at = now
        turn.active_submission_id = None
        turn.active_submission_started_at = None
        progress = _updated_topic_progress(
            self.repository.get_topic_progress(
                owner_id, session.project_id, turn.theme
            ),
            owner_id=owner_id,
            project_id=session.project_id,
            topic=turn.theme,
            score=decision.overall_score,
            gaps=decision.missing_details,
            practiced_at=now,
        )
        next_turn = None
        if turn.sequence >= session.max_turns:
            session.status = ProjectTrainingSessionStatus.COMPLETED
            session.current_turn_id = None
            session.completed_at = now
        else:
            next_turn = _next_turn(
                owner_id=owner_id,
                session=session,
                completed_turn=turn,
                decision=decision,
                previous_turns=self.repository.list_turns(owner_id, session_id),
                evidence=evidence,
            )
            session.current_turn_id = next_turn.id
        try:
            self.repository.complete_answer(answer, turn, session, next_turn, progress)
        except ProjectTrainingSubmissionInProgressError:
            latest = self.repository.get_answer_by_submission_id(
                owner_id, submission_id
            )
            if (
                latest is not None
                and latest.evaluation_status
                == ProjectTrainingEvaluationStatus.COMPLETED
            ):
                return self._completed_answer_outcome(owner_id, latest)
            self.repository.fail_answer(answer)
            raise
        if session.status == ProjectTrainingSessionStatus.COMPLETED:
            self._ensure_summary(owner_id, session)
        return ProjectAnswerOutcome(
            session=session,
            completed_turn=turn,
            answer=answer,
            evaluation=decision.evaluation_payload(),
            current_turn=next_turn,
            progress=progress,
        )

    def finish(self, owner_id: str, session_id: str) -> ProjectTrainingSummary:
        session = self.repository.get_session(owner_id, session_id)
        if session is None:
            raise ValueError("项目训练不存在。")
        if session.status == ProjectTrainingSessionStatus.ABANDONED:
            raise ValueError("已放弃的训练不能生成正式总结。")
        completed_answers = [
            item
            for item in self.repository.list_answers(owner_id, session_id)
            if item.evaluation_status == ProjectTrainingEvaluationStatus.COMPLETED
        ]
        if not completed_answers:
            raise ValueError("至少回答一题后才能结束训练。")
        if session.status != ProjectTrainingSessionStatus.COMPLETED:
            session = self.repository.close_session(
                owner_id,
                session_id,
                ProjectTrainingSessionStatus.COMPLETED,
                datetime.now().astimezone(),
            )
            if session is None:
                raise ValueError("项目训练不存在。")
        return self._ensure_summary(owner_id, session)

    def abandon(self, owner_id: str, session_id: str) -> ProjectTrainingSession:
        session = self.repository.get_session(owner_id, session_id)
        if session is None:
            raise ValueError("项目训练不存在。")
        if session.status == ProjectTrainingSessionStatus.COMPLETED:
            raise ValueError("已完成的训练不能放弃。")
        closed = self.repository.close_session(
            owner_id,
            session_id,
            ProjectTrainingSessionStatus.ABANDONED,
            datetime.now().astimezone(),
        )
        if closed is None:
            raise ValueError("项目训练不存在。")
        return closed

    def summary(self, owner_id: str, session_id: str) -> ProjectTrainingSummary:
        summary = self.repository.get_summary(owner_id, session_id)
        if summary is None:
            session = self.repository.get_session(owner_id, session_id)
            if session is None or session.status != ProjectTrainingSessionStatus.COMPLETED:
                raise ValueError("训练总结尚未生成。")
            summary = self._ensure_summary(owner_id, session)
        return summary

    def _completed_answer_outcome(
        self, owner_id: str, answer: ProjectTrainingAnswer
    ) -> "ProjectAnswerOutcome":
        session = self.repository.get_session(owner_id, answer.session_id)
        turn = self.repository.get_turn(owner_id, answer.turn_id)
        if session is None or turn is None:
            raise ValueError("项目训练记录不完整。")
        current = self.repository.get_turn(owner_id, session.current_turn_id or "")
        progress = self.repository.get_topic_progress(
            owner_id, session.project_id, turn.theme
        ) or ProjectTopicProgress(owner_id, session.project_id, turn.theme)
        return ProjectAnswerOutcome(
            session=session,
            completed_turn=turn,
            answer=answer,
            evaluation=dict(answer.evaluation_payload),
            current_turn=current,
            progress=progress,
        )

    def _ensure_summary(
        self, owner_id: str, session: ProjectTrainingSession
    ) -> ProjectTrainingSummary:
        existing = self.repository.get_summary(owner_id, session.id)
        if existing is not None:
            return existing
        answers = [
            item
            for item in self.repository.list_answers(owner_id, session.id)
            if item.evaluation_status == ProjectTrainingEvaluationStatus.COMPLETED
        ]
        turns = {item.id: item for item in self.repository.list_turns(owner_id, session.id)}
        topic_values: Dict[str, List[int]] = {}
        strengths: List[str] = []
        improvements: List[str] = []
        unsupported: List[str] = []
        contradictions: List[str] = []
        for answer in answers:
            turn = turns.get(answer.turn_id)
            if turn is not None and answer.overall_score is not None:
                topic_values.setdefault(turn.theme, []).append(answer.overall_score)
            payload = answer.evaluation_payload
            strengths.extend(payload.get("strengths", []))
            improvements.extend(payload.get("missing_details", []))
            unsupported.extend(payload.get("unsupported_claims", []))
            contradictions.extend(payload.get("contradictions", []))
        topic_scores = {
            topic: round(sum(values) / len(values)) for topic, values in topic_values.items()
        }
        overall = round(sum(topic_scores.values()) / len(topic_scores)) if topic_scores else 0
        recommended = [
            topic for topic, _ in sorted(topic_scores.items(), key=lambda item: item[1])[:3]
        ]
        summary = ProjectTrainingSummary(
            session_id=session.id,
            owner_id=owner_id,
            project_id=session.project_id,
            project_version=session.project_version,
            overall_score=overall,
            topic_scores=topic_scores,
            strengths=_unique(strengths, 8),
            improvement_areas=_unique(improvements, 10),
            unsupported_claims=_unique(unsupported, 8),
            contradictions=_unique(contradictions, 8),
            recommended_topics=recommended,
            created_at=datetime.now().astimezone(),
        )
        self.repository.save_summary(summary)
        return summary

@dataclass(frozen=True)
class ProjectTrainingOutcome:
    session: ProjectTrainingSession
    current_turn: Optional[ProjectTrainingTurn]
    resumed: bool = False


@dataclass(frozen=True)
class ProjectAnswerOutcome:
    session: ProjectTrainingSession
    completed_turn: ProjectTrainingTurn
    answer: ProjectTrainingAnswer
    evaluation: Dict[str, Any]
    current_turn: Optional[ProjectTrainingTurn]
    progress: ProjectTopicProgress


def _next_turn(
    owner_id: str,
    session: ProjectTrainingSession,
    completed_turn: ProjectTrainingTurn,
    decision: ProjectInterviewDecision,
    previous_turns: List[ProjectTrainingTurn],
    evidence,
) -> ProjectTrainingTurn:
    candidate = dict(decision.candidate_next_turn)
    trailing_same_theme = 0
    for item in reversed(previous_turns):
        if item.theme != completed_turn.theme:
            break
        trailing_same_theme += 1
    should_follow_up = bool(
        set(decision.follow_up_signals)
        - {"topic_complete"}
    ) and trailing_same_theme < 2
    if not should_follow_up or not candidate.get("question_text"):
        theme = _next_uncovered_theme(previous_turns, session.focus_topics)
        candidate = _deterministic_theme_question(theme, evidence)
    return ProjectTrainingTurn(
        id=f"project_turn_{uuid.uuid4().hex}",
        owner_id=owner_id,
        session_id=session.id,
        parent_turn_id=completed_turn.id,
        sequence=completed_turn.sequence + 1,
        theme=str(candidate["theme"]),
        question_kind=str(candidate["question_kind"]),
        question_text=str(candidate["question_text"]),
        generation_reason=str(candidate["generation_reason"]),
        source_evidence_ids=list(candidate.get("source_evidence_ids", [])),
        hypothetical=bool(candidate.get("hypothetical", False)),
        status=ProjectTrainingTurnStatus.PENDING,
        created_at=datetime.now().astimezone(),
    )


def _next_uncovered_theme(
    turns: List[ProjectTrainingTurn], focus_topics: Optional[List[str]] = None
) -> str:
    covered = {item.theme for item in turns}
    order = [
        item for item in (focus_topics or []) if item in PROJECT_TRAINING_THEME_ORDER
    ]
    order.extend(item for item in PROJECT_TRAINING_THEME_ORDER if item not in order)
    return next((item for item in order if item not in covered), "technical_depth")


def _deterministic_theme_question(theme: str, evidence) -> Dict[str, Any]:
    questions = {
        "ownership": "这个项目里你个人独立负责了哪些部分？请区分团队成果和你的贡献。",
        "architecture": "请描述核心架构和一次请求经过的关键链路。",
        "technical_depth": "项目中最难的技术问题是什么，你是怎样定位并解决的？",
        "tradeoff": "这个项目做过哪些关键技术选型？当时有哪些替代方案和取舍？",
        "reliability": "系统出现关键依赖故障时如何降级、恢复并避免数据不一致？",
        "metrics": "你如何量化项目结果？请给出优化前后指标和测量方式。",
        "project_overview": "请进一步说明项目背景、目标和最终结果。",
    }
    matching = [item for item in evidence if theme in item.topic_tags][:2]
    return {
        "theme": theme,
        "question_kind": "evidence" if matching else "design",
        "question_text": questions.get(theme, questions["technical_depth"]),
        "generation_reason": "按项目训练主题计划切换下一主题",
        "source_evidence_ids": [item.id for item in matching],
        "hypothetical": False,
    }


def _updated_topic_progress(
    current: Optional[ProjectTopicProgress],
    owner_id: str,
    project_id: str,
    topic: str,
    score: int,
    gaps: List[str],
    practiced_at: datetime,
) -> ProjectTopicProgress:
    if current is None:
        return ProjectTopicProgress(
            owner_id=owner_id,
            project_id=project_id,
            topic=topic,
            attempt_count=1,
            mastery_score=score,
            last_score=score,
            last_practiced_at=practiced_at,
            last_detected_gaps=list(gaps),
        )
    count = current.attempt_count + 1
    current.mastery_score = round(
        (current.mastery_score * current.attempt_count + score) / count
    )
    current.attempt_count = count
    current.last_score = score
    current.last_practiced_at = practiced_at
    current.last_detected_gaps = list(gaps)
    return current


def _recent_turn_payloads(turns, answers) -> List[Dict[str, str]]:
    answers_by_turn = {item.turn_id: item for item in answers}
    result = []
    for turn in turns[-2:]:
        answer = answers_by_turn.get(turn.id)
        result.append(
            {
                "question": turn.question_text,
                "answer": answer.answer_text if answer is not None else "",
            }
        )
    return result


def _select_evidence(evidence, turn, answer_text):
    terms = {item.casefold() for item in answer_text.split() if len(item) > 1}
    ranked = sorted(
        evidence,
        key=lambda item: (
            turn.theme not in item.topic_tags,
            -sum(term in item.content.casefold() for term in terms),
            item.order,
        ),
    )
    return ranked[:12]


def _unique(values: List[str], limit: int) -> List[str]:
    result = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result
