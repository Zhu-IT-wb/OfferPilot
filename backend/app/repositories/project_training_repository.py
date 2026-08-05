import hashlib
import json
import re
import threading
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Protocol, Tuple

from app.models.project_training import (
    ProjectEvidence,
    ProjectProfile,
    ProjectProfileStatus,
    ProjectProfileVersion,
    ProjectTrainingDifficulty,
    ProjectTrainingAnswer,
    ProjectTrainingAnswerSource,
    ProjectTrainingEvaluationStatus,
    ProjectTrainingSession,
    ProjectTrainingSessionStatus,
    ProjectTrainingTurn,
    ProjectTrainingTurnStatus,
    ProjectTopicProgress,
    ProjectTrainingSummary,
)


class ProjectTrainingSubmissionInProgressError(RuntimeError):
    pass


PROJECT_PROFILE_FIELDS = (
    "name",
    "target_role",
    "background",
    "responsibilities",
    "tech_stack",
    "architecture",
    "key_decisions",
    "technical_challenges",
    "metrics",
    "outcomes",
    "resume_description",
    "supplemental_text",
)


class ProjectTrainingRepository(Protocol):
    def create_project(self, owner_id: str, values: dict) -> ProjectProfile: ...
    def update_project(
        self, owner_id: str, project_id: str, values: dict
    ) -> Optional[ProjectProfile]: ...
    def create_discovered_project(
        self, owner_id: str, values: dict, discovery_evidence: list
    ) -> ProjectProfile: ...
    def get_project_by_discovery_job(
        self, owner_id: str, discovery_job_id: str
    ) -> Optional[ProjectProfile]: ...
    def get_project(
        self, owner_id: str, project_id: str, version: Optional[int] = None
    ) -> Optional[ProjectProfile]: ...
    def list_projects(
        self, owner_id: str, include_archived: bool = False
    ) -> List[ProjectProfile]: ...
    def list_project_versions(self, owner_id: str, project_id: str) -> List[int]: ...
    def list_project_evidence(
        self, owner_id: str, project_id: str, version: int
    ) -> List[ProjectEvidence]: ...
    def archive_project(self, owner_id: str, project_id: str) -> Optional[ProjectProfile]: ...
    def create_session(
        self,
        owner_id: str,
        project: ProjectProfile,
        creation_id: str,
        target_role: str,
        goal: str,
        difficulty: ProjectTrainingDifficulty,
        max_turns: int,
        focus_topics: List[str],
    ) -> Tuple[ProjectTrainingSession, Optional[ProjectTrainingTurn], bool]: ...
    def get_session(
        self, owner_id: str, session_id: str
    ) -> Optional[ProjectTrainingSession]: ...
    def get_session_by_creation_id(
        self, owner_id: str, creation_id: str
    ) -> Optional[ProjectTrainingSession]: ...
    def list_sessions(
        self, owner_id: str, project_id: Optional[str] = None
    ) -> List[ProjectTrainingSession]: ...
    def get_turn(self, owner_id: str, turn_id: str) -> Optional[ProjectTrainingTurn]: ...
    def list_turns(self, owner_id: str, session_id: str) -> List[ProjectTrainingTurn]: ...
    def begin_answer(
        self,
        owner_id: str,
        session_id: str,
        turn_id: str,
        submission_id: str,
        answer_text: str,
        answer_source: ProjectTrainingAnswerSource,
        submitted_at: datetime,
    ) -> Tuple[ProjectTrainingAnswer, bool]: ...
    def complete_answer(
        self,
        answer: ProjectTrainingAnswer,
        turn: ProjectTrainingTurn,
        session: ProjectTrainingSession,
        next_turn: Optional[ProjectTrainingTurn],
        progress: ProjectTopicProgress,
    ) -> None: ...
    def fail_answer(self, answer: ProjectTrainingAnswer) -> None: ...
    def get_answer_by_submission_id(
        self, owner_id: str, submission_id: str
    ) -> Optional[ProjectTrainingAnswer]: ...
    def list_answers(
        self, owner_id: str, session_id: str
    ) -> List[ProjectTrainingAnswer]: ...
    def get_topic_progress(
        self, owner_id: str, project_id: str, topic: str
    ) -> Optional[ProjectTopicProgress]: ...
    def close_session(
        self,
        owner_id: str,
        session_id: str,
        status: ProjectTrainingSessionStatus,
        completed_at: datetime,
    ) -> Optional[ProjectTrainingSession]: ...
    def save_summary(self, summary: ProjectTrainingSummary) -> None: ...
    def get_summary(
        self, owner_id: str, session_id: str
    ) -> Optional[ProjectTrainingSummary]: ...


@dataclass
class InMemoryProjectTrainingRepository:
    projects: Dict[Tuple[str, str], ProjectProfile] = field(default_factory=dict)
    project_versions: Dict[Tuple[str, str, int], ProjectProfile] = field(
        default_factory=dict
    )
    evidence: Dict[Tuple[str, str, int], List[ProjectEvidence]] = field(
        default_factory=dict
    )
    sessions: Dict[Tuple[str, str], ProjectTrainingSession] = field(default_factory=dict)
    turns: Dict[Tuple[str, str], ProjectTrainingTurn] = field(default_factory=dict)
    answers: Dict[Tuple[str, str], ProjectTrainingAnswer] = field(default_factory=dict)
    progress: Dict[Tuple[str, str, str], ProjectTopicProgress] = field(default_factory=dict)
    summaries: Dict[Tuple[str, str], ProjectTrainingSummary] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def create_project(self, owner_id: str, values: dict) -> ProjectProfile:
        now = datetime.now().astimezone()
        project = ProjectProfile(
            id=f"project_{uuid.uuid4().hex}",
            owner_id=owner_id,
            status=ProjectProfileStatus.ACTIVE,
            version=1,
            created_at=now,
            updated_at=now,
            **_profile_values(values),
        )
        project.content_hash = _project_content_hash(project)
        self._store_project_version(project)
        return replace(project)

    def update_project(
        self, owner_id: str, project_id: str, values: dict
    ) -> Optional[ProjectProfile]:
        current = self.projects.get((owner_id, project_id))
        if current is None:
            return None
        updated = replace(
            current,
            version=current.version + 1,
            updated_at=datetime.now().astimezone(),
            **_profile_values(values),
        )
        updated.content_hash = _project_content_hash(updated)
        inherited = _inherit_code_evidence(
            updated,
            self.evidence.get((owner_id, project_id, current.version), []),
        )
        self._store_project_version(updated, extra_evidence=inherited)
        return replace(updated)

    def create_discovered_project(self, owner_id, values, discovery_evidence):
        with self.lock:
            return self._create_discovered_project(
                owner_id, values, discovery_evidence
            )

    def _create_discovered_project(self, owner_id, values, discovery_evidence):
        values = dict(values)
        discovery_job_id = str(values.get("source_discovery_job_id") or "")
        existing = self.get_project_by_discovery_job(owner_id, discovery_job_id)
        if existing is not None:
            return existing
        project_id = values.pop("project_id", None)
        if project_id:
            current = self.projects.get((owner_id, project_id))
            if current is None or current.status != ProjectProfileStatus.ACTIVE:
                raise ValueError("要更新的项目不存在或已归档。")
            now = datetime.now().astimezone()
            project = replace(
                current, version=current.version + 1, updated_at=now,
                **_profile_values(values), **_source_values(values),
            )
        else:
            now = datetime.now().astimezone()
            project = ProjectProfile(
                id=f"project_{uuid.uuid4().hex}", owner_id=owner_id,
                status=ProjectProfileStatus.ACTIVE, version=1,
                created_at=now, updated_at=now,
                **_profile_values(values), **_source_values(values),
            )
        project.content_hash = _project_content_hash(project)
        evidence = _project_discovery_evidence(project, discovery_evidence)
        self._store_project_version(project, extra_evidence=evidence)
        return replace(project)

    def get_project_by_discovery_job(self, owner_id, discovery_job_id):
        if not discovery_job_id:
            return None
        item = next(
            (
                project
                for (stored_owner, _, _), project in self.project_versions.items()
                if stored_owner == owner_id
                and project.source_discovery_job_id == discovery_job_id
            ),
            None,
        )
        return replace(item) if item else None

    def get_project(
        self, owner_id: str, project_id: str, version: Optional[int] = None
    ) -> Optional[ProjectProfile]:
        if version is None:
            project = self.projects.get((owner_id, project_id))
        else:
            project = self.project_versions.get((owner_id, project_id, version))
        return replace(project) if project is not None else None

    def list_projects(
        self, owner_id: str, include_archived: bool = False
    ) -> List[ProjectProfile]:
        projects = [
            replace(project)
            for (stored_owner, _), project in self.projects.items()
            if stored_owner == owner_id
            and (include_archived or project.status == ProjectProfileStatus.ACTIVE)
        ]
        return sorted(projects, key=lambda item: item.updated_at, reverse=True)

    def list_project_versions(self, owner_id: str, project_id: str) -> List[int]:
        return sorted(
            version
            for stored_owner, stored_project, version in self.project_versions
            if stored_owner == owner_id and stored_project == project_id
        )

    def list_project_evidence(
        self, owner_id: str, project_id: str, version: int
    ) -> List[ProjectEvidence]:
        return list(self.evidence.get((owner_id, project_id, version), []))

    def archive_project(self, owner_id: str, project_id: str) -> Optional[ProjectProfile]:
        current = self.projects.get((owner_id, project_id))
        if current is None:
            return None
        current.status = ProjectProfileStatus.ARCHIVED
        current.updated_at = datetime.now().astimezone()
        return replace(current)

    def create_session(
        self,
        owner_id: str,
        project: ProjectProfile,
        creation_id: str,
        target_role: str,
        goal: str,
        difficulty: ProjectTrainingDifficulty,
        max_turns: int,
        focus_topics: Optional[List[str]] = None,
    ) -> Tuple[ProjectTrainingSession, ProjectTrainingTurn, bool]:
        existing = self.get_session_by_creation_id(owner_id, creation_id)
        if existing is not None:
            turn = (
                self.get_turn(owner_id, existing.current_turn_id)
                if existing.current_turn_id
                else None
            )
            if existing.current_turn_id and turn is None:
                raise RuntimeError("Project training session has no current turn.")
            return existing, turn, True
        now = datetime.now().astimezone()
        session_id = f"project_session_{uuid.uuid4().hex}"
        turn = ProjectTrainingTurn(
            id=f"project_turn_{uuid.uuid4().hex}",
            owner_id=owner_id,
            session_id=session_id,
            parent_turn_id=None,
            sequence=1,
            theme="project_overview",
            question_kind="opening",
            question_text=(
                f"请用 1～2 分钟介绍一下「{project.name}」，重点说明项目背景、"
                "你的个人职责、核心设计和最终结果。"
            ),
            generation_reason="项目训练固定开场题",
            source_evidence_ids=[],
            hypothetical=False,
            status=ProjectTrainingTurnStatus.PENDING,
            created_at=now,
        )
        session = ProjectTrainingSession(
            id=session_id,
            owner_id=owner_id,
            project_id=project.id,
            project_version=project.version,
            creation_id=creation_id,
            target_role=target_role or project.target_role,
            goal=goal,
            difficulty=difficulty,
            max_turns=max_turns,
            focus_topics=list(focus_topics or []),
            status=ProjectTrainingSessionStatus.IN_PROGRESS,
            current_turn_id=turn.id,
            created_at=now,
            started_at=now,
        )
        self.sessions[(owner_id, session.id)] = session
        self.turns[(owner_id, turn.id)] = turn
        return replace(session), replace(turn), False

    def get_session(
        self, owner_id: str, session_id: str
    ) -> Optional[ProjectTrainingSession]:
        session = self.sessions.get((owner_id, session_id))
        return replace(session) if session is not None else None

    def get_session_by_creation_id(
        self, owner_id: str, creation_id: str
    ) -> Optional[ProjectTrainingSession]:
        session = next(
            (
                item
                for (stored_owner, _), item in self.sessions.items()
                if stored_owner == owner_id and item.creation_id == creation_id
            ),
            None,
        )
        return replace(session) if session is not None else None

    def list_sessions(
        self, owner_id: str, project_id: Optional[str] = None
    ) -> List[ProjectTrainingSession]:
        sessions = [
            replace(item)
            for (stored_owner, _), item in self.sessions.items()
            if stored_owner == owner_id
            and (project_id is None or item.project_id == project_id)
        ]
        return sorted(sessions, key=lambda item: item.created_at, reverse=True)

    def get_turn(self, owner_id: str, turn_id: str) -> Optional[ProjectTrainingTurn]:
        turn = self.turns.get((owner_id, turn_id))
        return replace(turn) if turn is not None else None

    def list_turns(self, owner_id: str, session_id: str) -> List[ProjectTrainingTurn]:
        turns = [
            replace(item)
            for (stored_owner, _), item in self.turns.items()
            if stored_owner == owner_id and item.session_id == session_id
        ]
        return sorted(turns, key=lambda item: item.sequence)

    def begin_answer(
        self,
        owner_id: str,
        session_id: str,
        turn_id: str,
        submission_id: str,
        answer_text: str,
        answer_source: ProjectTrainingAnswerSource,
        submitted_at: datetime,
    ) -> Tuple[ProjectTrainingAnswer, bool]:
        existing = self.get_answer_by_submission_id(owner_id, submission_id)
        if existing is not None:
            if (
                existing.session_id != session_id
                or existing.turn_id != turn_id
                or existing.answer_text != answer_text
                or existing.answer_source != answer_source
            ):
                raise ValueError("submission_id 已被另一个回答使用。")
            if existing.evaluation_status == ProjectTrainingEvaluationStatus.COMPLETED:
                return existing, True
            if existing.evaluation_status == ProjectTrainingEvaluationStatus.PENDING:
                if not _answer_lease_expired(existing.submitted_at):
                    raise ProjectTrainingSubmissionInProgressError("这个回答正在评价中。")
                existing.evaluation_status = ProjectTrainingEvaluationStatus.FAILED
                stale_turn = self.turns.get((owner_id, existing.turn_id))
                if stale_turn is not None:
                    stale_turn.status = ProjectTrainingTurnStatus.PENDING
                    stale_turn.active_submission_id = None
                    stale_turn.active_submission_started_at = None
            existing.evaluation_status = ProjectTrainingEvaluationStatus.PENDING
            existing.submitted_at = submitted_at
            stored_turn = self.turns[(owner_id, turn_id)]
            stored_turn.status = ProjectTrainingTurnStatus.EVALUATING
            stored_turn.active_submission_id = submission_id
            stored_turn.active_submission_started_at = submitted_at
            self.answers[(owner_id, existing.id)] = existing
            return replace(existing), False
        session = self.sessions.get((owner_id, session_id))
        turn = self.turns.get((owner_id, turn_id))
        if session is None or turn is None or turn.session_id != session_id:
            raise ValueError("当前项目训练题目不存在。")
        if (
            turn.status == ProjectTrainingTurnStatus.EVALUATING
            and turn.active_submission_started_at is not None
            and _answer_lease_expired(turn.active_submission_started_at)
        ):
            stale = next(
                (
                    item
                    for (stored_owner, _), item in self.answers.items()
                    if stored_owner == owner_id
                    and item.submission_id == turn.active_submission_id
                ),
                None,
            )
            if stale is not None:
                stale.evaluation_status = ProjectTrainingEvaluationStatus.FAILED
            turn.status = ProjectTrainingTurnStatus.PENDING
            turn.active_submission_id = None
            turn.active_submission_started_at = None
        if session.current_turn_id != turn_id or turn.status != ProjectTrainingTurnStatus.PENDING:
            raise ValueError("这道题已经完成或不是当前题目。")
        answer = ProjectTrainingAnswer(
            id=f"project_answer_{uuid.uuid4().hex}",
            owner_id=owner_id,
            session_id=session_id,
            turn_id=turn_id,
            submission_id=submission_id,
            answer_text=answer_text,
            answer_source=answer_source,
            submitted_at=submitted_at,
            evaluation_status=ProjectTrainingEvaluationStatus.PENDING,
        )
        turn.status = ProjectTrainingTurnStatus.EVALUATING
        turn.active_submission_id = submission_id
        turn.active_submission_started_at = submitted_at
        self.answers[(owner_id, answer.id)] = answer
        return replace(answer), False

    def complete_answer(
        self,
        answer: ProjectTrainingAnswer,
        turn: ProjectTrainingTurn,
        session: ProjectTrainingSession,
        next_turn: Optional[ProjectTrainingTurn],
        progress: ProjectTopicProgress,
    ) -> None:
        stored_turn = self.turns.get((answer.owner_id, turn.id))
        stored_session = self.sessions.get((answer.owner_id, session.id))
        if (
            stored_turn is None
            or stored_session is None
            or stored_turn.active_submission_id != answer.submission_id
            or stored_turn.status != ProjectTrainingTurnStatus.EVALUATING
            or stored_session.status != ProjectTrainingSessionStatus.IN_PROGRESS
            or stored_session.current_turn_id != turn.id
        ):
            raise ProjectTrainingSubmissionInProgressError(
                "项目训练状态已变化，请刷新后重试。"
            )
        if next_turn is not None and any(
            item.session_id == next_turn.session_id
            and item.sequence == next_turn.sequence
            for (owner, _), item in self.turns.items()
            if owner == next_turn.owner_id
        ):
            raise ProjectTrainingSubmissionInProgressError(
                "下一道项目题已经生成，请刷新页面。"
            )
        self.answers[(answer.owner_id, answer.id)] = replace(answer)
        self.turns[(turn.owner_id, turn.id)] = replace(turn)
        self.sessions[(session.owner_id, session.id)] = replace(session)
        self.progress[(progress.owner_id, progress.project_id, progress.topic)] = replace(
            progress
        )
        if next_turn is not None:
            self.turns[(next_turn.owner_id, next_turn.id)] = replace(next_turn)

    def fail_answer(self, answer: ProjectTrainingAnswer) -> None:
        answer.evaluation_status = ProjectTrainingEvaluationStatus.FAILED
        self.answers[(answer.owner_id, answer.id)] = replace(answer)
        turn = self.turns.get((answer.owner_id, answer.turn_id))
        if turn is not None and turn.active_submission_id == answer.submission_id:
            turn.status = ProjectTrainingTurnStatus.PENDING
            turn.active_submission_id = None
            turn.active_submission_started_at = None

    def get_answer_by_submission_id(
        self, owner_id: str, submission_id: str
    ) -> Optional[ProjectTrainingAnswer]:
        answer = next(
            (
                item
                for (stored_owner, _), item in self.answers.items()
                if stored_owner == owner_id and item.submission_id == submission_id
            ),
            None,
        )
        return replace(answer) if answer is not None else None

    def list_answers(
        self, owner_id: str, session_id: str
    ) -> List[ProjectTrainingAnswer]:
        return sorted(
            [
                replace(item)
                for (stored_owner, _), item in self.answers.items()
                if stored_owner == owner_id and item.session_id == session_id
            ],
            key=lambda item: item.submitted_at,
        )

    def get_topic_progress(
        self, owner_id: str, project_id: str, topic: str
    ) -> Optional[ProjectTopicProgress]:
        item = self.progress.get((owner_id, project_id, topic))
        return replace(item) if item is not None else None

    def close_session(
        self,
        owner_id: str,
        session_id: str,
        status: ProjectTrainingSessionStatus,
        completed_at: datetime,
    ) -> Optional[ProjectTrainingSession]:
        session = self.sessions.get((owner_id, session_id))
        if session is None:
            return None
        turn = self.turns.get((owner_id, session.current_turn_id or ""))
        if turn is not None and turn.status == ProjectTrainingTurnStatus.EVALUATING:
            raise ProjectTrainingSubmissionInProgressError(
                "回答正在评价中，请等待评价完成后再结束训练。"
            )
        if session.status in {
            ProjectTrainingSessionStatus.COMPLETED,
            ProjectTrainingSessionStatus.ABANDONED,
        }:
            if session.status != status:
                raise ValueError("已结束的项目训练不能变更状态。")
            return replace(session)
        session.status = status
        session.current_turn_id = None
        session.completed_at = completed_at
        return replace(session)

    def save_summary(self, summary: ProjectTrainingSummary) -> None:
        self.summaries[(summary.owner_id, summary.session_id)] = replace(summary)

    def get_summary(
        self, owner_id: str, session_id: str
    ) -> Optional[ProjectTrainingSummary]:
        summary = self.summaries.get((owner_id, session_id))
        return replace(summary) if summary is not None else None

    def _store_project_version(self, project: ProjectProfile, extra_evidence=None) -> None:
        snapshot = replace(project)
        self.projects[(project.owner_id, project.id)] = snapshot
        self.project_versions[(project.owner_id, project.id, project.version)] = replace(
            snapshot
        )
        generated = build_project_evidence(snapshot)
        additional = list(extra_evidence or [])
        self.evidence[(project.owner_id, project.id, project.version)] = generated + [
            replace(item, order=len(generated) + index)
            for index, item in enumerate(additional)
        ]


def build_project_evidence(project: ProjectProfile) -> List[ProjectEvidence]:
    entries = []
    structured = [
        ("background", "项目背景", [project.background], ["project_overview"]),
        ("responsibilities", "个人职责", project.responsibilities, ["ownership"]),
        ("architecture", "核心架构", [project.architecture], ["architecture"]),
        ("key_decisions", "关键决策", project.key_decisions, ["tradeoff"]),
        (
            "technical_challenges",
            "技术难点",
            project.technical_challenges,
            ["technical_depth", "troubleshooting"],
        ),
        ("metrics", "量化指标", project.metrics, ["metrics"]),
        ("outcomes", "项目结果", project.outcomes, ["metrics"]),
        (
            "resume_description",
            "简历描述",
            [project.resume_description],
            ["project_overview", "communication"],
        ),
    ]
    for source_field, heading, values, tags in structured:
        for value in values:
            normalized = str(value).strip()
            if normalized:
                entries.append((source_field, heading, normalized, tags))
    entries.extend(_supplemental_entries(project.supplemental_text))
    result = []
    for order, (source_field, heading, content, tags) in enumerate(entries):
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        evidence_id = hashlib.sha256(
            f"{project.id}:{project.version}:{source_field}:{order}:{digest}".encode(
                "utf-8"
            )
        ).hexdigest()[:24]
        result.append(
            ProjectEvidence(
                id=f"evidence_{evidence_id}",
                owner_id=project.owner_id,
                project_id=project.id,
                project_version=project.version,
                source_field=source_field,
                heading=heading,
                content=content,
                topic_tags=list(tags),
                content_hash=digest,
                order=order,
            )
        )
    return result


def _supplemental_entries(text: str):
    result = []
    heading = "补充材料"
    buffer: List[str] = []
    for line in text.splitlines() + [""]:
        stripped = line.strip()
        if stripped.startswith("#"):
            if buffer:
                result.append(
                    ("supplemental_text", heading, "\n".join(buffer).strip()[:1200], [])
                )
                buffer = []
            heading = stripped.lstrip("#").strip() or "补充材料"
        elif not stripped:
            if buffer:
                result.append(
                    ("supplemental_text", heading, "\n".join(buffer).strip()[:1200], [])
                )
                buffer = []
        else:
            buffer.append(stripped)
    return result


def _profile_values(values: dict) -> dict:
    result = {}
    for field_name in PROJECT_PROFILE_FIELDS:
        value = values.get(field_name, [] if field_name in _LIST_FIELDS else "")
        if field_name in _LIST_FIELDS:
            result[field_name] = [str(item).strip() for item in value if str(item).strip()]
        else:
            result[field_name] = str(value).strip()
    return result


_LIST_FIELDS = {
    "responsibilities",
    "tech_stack",
    "key_decisions",
    "technical_challenges",
    "metrics",
    "outcomes",
}


def _project_content_hash(project: ProjectProfile) -> str:
    payload = {field: getattr(project, field) for field in PROJECT_PROFILE_FIELDS}
    payload.update(_source_values(project.__dict__))
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _source_values(values: dict) -> dict:
    return {
        field: str(values.get(field) or "").strip()
        for field in (
            "source_repository_url", "source_commit_sha", "source_discovery_job_id"
        )
    }


def _project_discovery_evidence(project, items):
    result = []
    for index, item in enumerate(items):
        content = str(item.excerpt).strip()
        if not content:
            continue
        result.append(ProjectEvidence(
            id=f"evidence_{hashlib.sha256(f'{project.id}:{project.version}:{item.id}'.encode()).hexdigest()[:24]}",
            owner_id=project.owner_id, project_id=project.id,
            project_version=project.version, source_field=item.target_field,
            heading=("用户补充" if item.source_type == "user_statement" else item.file_path),
            content=content, topic_tags=[item.topic] if item.topic else [],
            content_hash=item.content_hash, order=index, source_type=item.source_type,
            source_path=item.file_path, start_line=item.start_line,
            end_line=item.end_line, confidence=item.confidence,
            source_commit_sha=item.commit_sha,
            source_evidence_id=item.id, grounded_claim=item.claim,
        ))
    return result


def _inherit_code_evidence(project, items):
    return [
        replace(
            item,
            id=f"evidence_{hashlib.sha256(f'{project.id}:{project.version}:inherited:{item.id}'.encode()).hexdigest()[:24]}",
            project_version=project.version,
        )
        for item in items
        if item.source_type == "code"
    ]


def _answer_lease_expired(started_at: datetime) -> bool:
    normalized = (
        started_at.replace(tzinfo=timezone.utc)
        if started_at.tzinfo is None
        else started_at.astimezone(timezone.utc)
    )
    return datetime.now(timezone.utc) - normalized >= timedelta(minutes=15)
