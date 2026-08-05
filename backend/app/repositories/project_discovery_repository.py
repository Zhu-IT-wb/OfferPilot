import json
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

from app.models.project_discovery import (
    ACTIVE_DISCOVERY_STATUSES,
    ProjectDiscoveryAnswer,
    ProjectDiscoveryEvidence,
    ProjectDiscoveryJob,
    ProjectDiscoveryQuestion,
    ProjectDiscoveryStage,
    ProjectDiscoveryStatus,
)


MAX_ACTIVE_DISCOVERY_JOBS_PER_OWNER = 5
MAX_ACTIVE_DISCOVERY_JOBS_GLOBAL = 100


class ProjectDiscoveryRepository(Protocol):
    def create_or_get(
        self,
        owner_id: str,
        repository_url: str,
        request_id: str,
        project_id: Optional[str],
    ) -> ProjectDiscoveryJob: ...
    def get(self, owner_id: str, job_id: str) -> Optional[ProjectDiscoveryJob]: ...
    def get_unscoped(self, job_id: str) -> Optional[ProjectDiscoveryJob]: ...
    def list(self, owner_id: str) -> List[ProjectDiscoveryJob]: ...
    def save(self, job: ProjectDiscoveryJob) -> None: ...
    def claim(self, job_id: str) -> Optional[ProjectDiscoveryJob]: ...
    def save_leased(self, job: ProjectDiscoveryJob, lease_token: str) -> bool: ...
    def complete_analysis(
        self,
        job: ProjectDiscoveryJob,
        evidence: List[ProjectDiscoveryEvidence],
        lease_token: str,
    ) -> bool: ...
    def requeue_expired(
        self, job_id: str, lease_token: str
    ) -> Optional[ProjectDiscoveryJob]: ...
    def release_lease(self, job_id: str, lease_token: str) -> bool: ...
    def recoverable(self) -> List[ProjectDiscoveryJob]: ...
    def replace_evidence(
        self, job: ProjectDiscoveryJob, evidence: List[ProjectDiscoveryEvidence]
    ) -> None: ...
    def list_evidence(
        self, owner_id: str, job_id: str
    ) -> List[ProjectDiscoveryEvidence]: ...
    def save_answer(self, answer: ProjectDiscoveryAnswer) -> ProjectDiscoveryAnswer: ...
    def answer_by_submission(
        self, owner_id: str, submission_id: str
    ) -> Optional[ProjectDiscoveryAnswer]: ...


@dataclass
class InMemoryProjectDiscoveryRepository:
    jobs: Dict[Tuple[str, str], ProjectDiscoveryJob] = field(default_factory=dict)
    evidence: Dict[Tuple[str, str], List[ProjectDiscoveryEvidence]] = field(
        default_factory=dict
    )
    answers: Dict[Tuple[str, str], ProjectDiscoveryAnswer] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def create_or_get(self, owner_id, repository_url, request_id, project_id=None):
        with self.lock:
            return self._create_or_get(
                owner_id, repository_url, request_id, project_id
            )

    def _create_or_get(self, owner_id, repository_url, request_id, project_id=None):
        for (stored_owner, _), item in self.jobs.items():
            if stored_owner == owner_id and item.request_id == request_id:
                if item.repository_url != repository_url or item.project_id != project_id:
                    raise ValueError("request_id 已被另一个导入请求使用。")
                return _copy_job(item)
        for (stored_owner, _), item in self.jobs.items():
            if (
                stored_owner == owner_id
                and item.repository_url == repository_url
                and item.status in ACTIVE_DISCOVERY_STATUSES
            ):
                return _copy_job(item)
        active = [
            item for item in self.jobs.values()
            if item.status in ACTIVE_DISCOVERY_STATUSES
        ]
        if sum(item.owner_id == owner_id for item in active) >= MAX_ACTIVE_DISCOVERY_JOBS_PER_OWNER:
            raise ValueError("每位用户最多同时进行 5 个项目分析任务。")
        if len(active) >= MAX_ACTIVE_DISCOVERY_JOBS_GLOBAL:
            raise ValueError("项目分析队列已满，请稍后重试。")
        now = datetime.now(timezone.utc).astimezone()
        job = ProjectDiscoveryJob(
            id=f"discovery_{uuid.uuid4().hex}",
            owner_id=owner_id,
            request_id=request_id,
            repository_url=repository_url,
            project_id=project_id,
            status=ProjectDiscoveryStatus.QUEUED,
            stage=None,
            progress=0,
            commit_sha="",
            draft={},
            questions=[],
            warnings=[],
            stats={},
            error_code="",
            error_message="",
            retry_count=0,
            confirmation_id="",
            confirmed_project_id=None,
            created_at=now,
            updated_at=now,
        )
        self.jobs[(owner_id, job.id)] = job
        return _copy_job(job)

    def get(self, owner_id, job_id):
        item = self.jobs.get((owner_id, job_id))
        return _copy_job(item) if item else None

    def get_unscoped(self, job_id):
        item = next(
            (
                job
                for (_, stored_id), job in self.jobs.items()
                if stored_id == job_id
            ),
            None,
        )
        return _copy_job(item) if item else None

    def list(self, owner_id):
        return sorted(
            (
                _copy_job(item)
                for (stored_owner, _), item in self.jobs.items()
                if stored_owner == owner_id
            ),
            key=lambda item: item.created_at,
            reverse=True,
        )

    def save(self, job):
        job.updated_at = datetime.now(timezone.utc).astimezone()
        self.jobs[(job.owner_id, job.id)] = _copy_job(job)

    def claim(self, job_id):
        with self.lock:
            item = self.get_unscoped(job_id)
            if item is None or item.status != ProjectDiscoveryStatus.QUEUED:
                return None
            item.status = ProjectDiscoveryStatus.RUNNING
            item.stage = ProjectDiscoveryStage.CLONING
            item.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
            item.lease_token = uuid.uuid4().hex
            self.save(item)
            return item

    def save_leased(self, job, lease_token):
        with self.lock:
            current = self.jobs.get((job.owner_id, job.id))
            if (
                current is None
                or current.status != ProjectDiscoveryStatus.RUNNING
                or current.lease_token != lease_token
            ):
                return False
            self.save(job)
            return True

    def complete_analysis(self, job, evidence, lease_token):
        with self.lock:
            current = self.jobs.get((job.owner_id, job.id))
            if (
                current is None
                or current.status != ProjectDiscoveryStatus.RUNNING
                or current.lease_token != lease_token
            ):
                return False
            job.lease_token = ""
            self.replace_evidence(job, evidence)
            self.save(job)
            return True

    def requeue_expired(self, job_id, lease_token):
        with self.lock:
            item = self.get_unscoped(job_id)
            now = datetime.now(timezone.utc)
            if (
                item is None
                or item.status != ProjectDiscoveryStatus.RUNNING
                or item.lease_token != lease_token
                or item.lease_expires_at is None
                or item.lease_expires_at > now
            ):
                return None
            item.status = ProjectDiscoveryStatus.QUEUED
            item.stage = None
            item.lease_expires_at = None
            item.lease_token = ""
            self.save(item)
            return item

    def release_lease(self, job_id, lease_token):
        with self.lock:
            item = self.get_unscoped(job_id)
            if item is None or item.lease_token != lease_token:
                return False
            item.status = ProjectDiscoveryStatus.QUEUED
            item.stage = None
            item.lease_expires_at = None
            item.lease_token = ""
            self.save(item)
            return True

    def recoverable(self):
        now = datetime.now(timezone.utc)
        return [
            item
            for item in self.list_all()
            if item.status == ProjectDiscoveryStatus.QUEUED
            or (
                item.status == ProjectDiscoveryStatus.RUNNING
                and (item.lease_expires_at is None or item.lease_expires_at <= now)
            )
        ]

    def list_all(self):
        return [_copy_job(item) for item in self.jobs.values()]

    def replace_evidence(self, job, evidence):
        self.evidence[(job.owner_id, job.id)] = list(evidence)

    def list_evidence(self, owner_id, job_id):
        return list(self.evidence.get((owner_id, job_id), []))

    def save_answer(self, answer):
        existing = self.answer_by_submission(answer.owner_id, answer.submission_id)
        if existing:
            if (existing.job_id, existing.question_id, existing.answer_text) != (
                answer.job_id,
                answer.question_id,
                answer.answer_text,
            ):
                raise ValueError("submission_id 已被另一个回答使用。")
            return existing
        existing_question = next(
            (
                item
                for (stored_owner, _), item in self.answers.items()
                if stored_owner == answer.owner_id
                and item.job_id == answer.job_id
                and item.question_id == answer.question_id
            ),
            None,
        )
        if existing_question is not None:
            raise ValueError("这个补充问题已经回答。")
        self.answers[(answer.owner_id, answer.id)] = answer
        return answer

    def answer_by_submission(self, owner_id, submission_id):
        return next(
            (
                item
                for (stored_owner, _), item in self.answers.items()
                if stored_owner == owner_id and item.submission_id == submission_id
            ),
            None,
        )


class SQLiteProjectDiscoveryRepository:
    def __init__(self, database_path: str):
        self.database_path = database_path
        Path(database_path).expanduser().resolve().parent.mkdir(
            parents=True, exist_ok=True
        )
        with self._connect() as connection:
            connection.executescript("""
            CREATE TABLE IF NOT EXISTS project_discovery_jobs (
              owner_id TEXT NOT NULL, job_id TEXT NOT NULL, request_id TEXT NOT NULL,
              repository_url TEXT NOT NULL, project_id TEXT NOT NULL DEFAULT '',
              status TEXT NOT NULL, lease_expires_at TEXT,
              lease_token TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
              payload TEXT NOT NULL, PRIMARY KEY(owner_id, job_id),
              UNIQUE(owner_id, request_id));
            CREATE INDEX IF NOT EXISTS idx_discovery_active_url
              ON project_discovery_jobs(owner_id, repository_url, project_id, status);
            CREATE UNIQUE INDEX IF NOT EXISTS uq_discovery_active_url
              ON project_discovery_jobs(owner_id, repository_url)
              WHERE status IN ('queued','running','needs_input','ready');
            CREATE TABLE IF NOT EXISTS project_discovery_evidence (
              owner_id TEXT NOT NULL, evidence_id TEXT NOT NULL, job_id TEXT NOT NULL,
              payload TEXT NOT NULL, PRIMARY KEY(owner_id, evidence_id));
            CREATE INDEX IF NOT EXISTS idx_discovery_evidence_job
              ON project_discovery_evidence(owner_id, job_id);
            CREATE TABLE IF NOT EXISTS project_discovery_answers (
              owner_id TEXT NOT NULL, answer_id TEXT NOT NULL, job_id TEXT NOT NULL,
              submission_id TEXT NOT NULL, payload TEXT NOT NULL,
              PRIMARY KEY(owner_id, answer_id), UNIQUE(owner_id, submission_id));
            CREATE UNIQUE INDEX IF NOT EXISTS uq_discovery_answer_question
              ON project_discovery_answers(
                owner_id, job_id, json_extract(payload, '$.question_id')
              );
            """)
            _ensure_column(
                connection,
                "project_discovery_jobs",
                "lease_token",
                "TEXT NOT NULL DEFAULT ''",
            )

    def create_or_get(self, owner_id, repository_url, request_id, project_id=None):
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT payload FROM project_discovery_jobs
                WHERE owner_id=? AND request_id=?
                """,
                (owner_id, request_id),
            ).fetchone()
            if row:
                job = _job(json.loads(row["payload"]))
                if job.repository_url != repository_url or job.project_id != project_id:
                    connection.rollback()
                    raise ValueError("request_id 已被另一个导入请求使用。")
                connection.commit()
                return job
            placeholders = ",".join("?" for _ in ACTIVE_DISCOVERY_STATUSES)
            row = connection.execute(
                f"""
                SELECT payload FROM project_discovery_jobs
                WHERE owner_id=? AND repository_url=?
                  AND status IN ({placeholders})
                ORDER BY created_at DESC LIMIT 1
                """,
                (
                    owner_id,
                    repository_url,
                    *(item.value for item in ACTIVE_DISCOVERY_STATUSES),
                ),
            ).fetchone()
            if row:
                connection.commit()
                return _job(json.loads(row["payload"]))
            active_values = tuple(
                item.value for item in ACTIVE_DISCOVERY_STATUSES
            )
            owner_count = connection.execute(
                f"""
                SELECT COUNT(*) AS count FROM project_discovery_jobs
                WHERE owner_id=? AND status IN ({placeholders})
                """,
                (owner_id, *active_values),
            ).fetchone()["count"]
            if owner_count >= MAX_ACTIVE_DISCOVERY_JOBS_PER_OWNER:
                connection.rollback()
                raise ValueError("每位用户最多同时进行 5 个项目分析任务。")
            global_count = connection.execute(
                f"""
                SELECT COUNT(*) AS count FROM project_discovery_jobs
                WHERE status IN ({placeholders})
                """,
                active_values,
            ).fetchone()["count"]
            if global_count >= MAX_ACTIVE_DISCOVERY_JOBS_GLOBAL:
                connection.rollback()
                raise ValueError("项目分析队列已满，请稍后重试。")
            temporary = InMemoryProjectDiscoveryRepository()
            job = temporary.create_or_get(
                owner_id, repository_url, request_id, project_id
            )
            connection.execute(
                """
                INSERT INTO project_discovery_jobs(
                  owner_id, job_id, request_id, repository_url, project_id,
                  status, lease_expires_at, lease_token, created_at, payload
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    job.owner_id,
                    job.id,
                    job.request_id,
                    job.repository_url,
                    job.project_id or "",
                    job.status.value,
                    None,
                    "",
                    job.created_at.isoformat(),
                    _dump_job(job),
                ),
            )
            connection.commit()
            return job

    def save_leased(self, job, lease_token):
        job.updated_at = datetime.now(timezone.utc).astimezone()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE project_discovery_jobs
                SET status=?, lease_expires_at=?, lease_token=?, payload=?
                WHERE owner_id=? AND job_id=? AND status='running'
                  AND lease_token=?
                """,
                (
                    job.status.value,
                    job.lease_expires_at.isoformat()
                    if job.lease_expires_at
                    else None,
                    job.lease_token,
                    _dump_job(job),
                    job.owner_id,
                    job.id,
                    lease_token,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def complete_analysis(self, job, evidence, lease_token):
        job.updated_at = datetime.now(timezone.utc).astimezone()
        job.lease_token = ""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT lease_token, status FROM project_discovery_jobs
                WHERE owner_id=? AND job_id=?
                """,
                (job.owner_id, job.id),
            ).fetchone()
            if (
                row is None
                or row["status"] != ProjectDiscoveryStatus.RUNNING.value
                or row["lease_token"] != lease_token
            ):
                connection.rollback()
                return False
            connection.execute(
                """
                DELETE FROM project_discovery_evidence
                WHERE owner_id=? AND job_id=?
                """,
                (job.owner_id, job.id),
            )
            connection.executemany(
                """
                INSERT INTO project_discovery_evidence(
                  owner_id, evidence_id, job_id, payload
                ) VALUES(?,?,?,?)
                """,
                [
                    (
                        item.owner_id,
                        item.id,
                        item.job_id,
                        json.dumps(asdict(item), ensure_ascii=False),
                    )
                    for item in evidence
                ],
            )
            connection.execute(
                """
                UPDATE project_discovery_jobs
                SET status=?, lease_expires_at=NULL, lease_token='', payload=?
                WHERE owner_id=? AND job_id=? AND lease_token=?
                """,
                (
                    job.status.value,
                    _dump_job(job),
                    job.owner_id,
                    job.id,
                    lease_token,
                ),
            )
            connection.commit()
            return True

    def requeue_expired(self, job_id, lease_token):
        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload FROM project_discovery_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            job = _job(json.loads(row["payload"]))
            if (
                job.status != ProjectDiscoveryStatus.RUNNING
                or job.lease_token != lease_token
                or job.lease_expires_at is None
                or job.lease_expires_at > now
            ):
                connection.rollback()
                return None
            job.status = ProjectDiscoveryStatus.QUEUED
            job.stage = None
            job.lease_expires_at = None
            job.lease_token = ""
            job.updated_at = datetime.now(timezone.utc).astimezone()
            cursor = connection.execute(
                """
                UPDATE project_discovery_jobs
                SET status='queued', lease_expires_at=NULL, lease_token='', payload=?
                WHERE owner_id=? AND job_id=? AND status='running'
                  AND lease_token=? AND lease_expires_at<=?
                """,
                (
                    _dump_job(job),
                    job.owner_id,
                    job.id,
                    lease_token,
                    now.isoformat(),
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            connection.commit()
            return job

    def release_lease(self, job_id, lease_token):
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload FROM project_discovery_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                return False
            job = _job(json.loads(row["payload"]))
            if job.lease_token != lease_token:
                connection.rollback()
                return False
            job.status = ProjectDiscoveryStatus.QUEUED
            job.stage = None
            job.lease_expires_at = None
            job.lease_token = ""
            job.updated_at = datetime.now(timezone.utc).astimezone()
            cursor = connection.execute(
                """
                UPDATE project_discovery_jobs
                SET status='queued', lease_expires_at=NULL, lease_token='', payload=?
                WHERE owner_id=? AND job_id=? AND lease_token=?
                """,
                (_dump_job(job), job.owner_id, job.id, lease_token),
            )
            connection.commit()
            return cursor.rowcount == 1

    def get(self, owner_id, job_id):
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload FROM project_discovery_jobs
                WHERE owner_id=? AND job_id=?
                """,
                (owner_id, job_id),
            ).fetchone()
        return _job(json.loads(row["payload"])) if row else None

    def get_unscoped(self, job_id):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM project_discovery_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
        return _job(json.loads(row["payload"])) if row else None

    def list(self, owner_id):
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload FROM project_discovery_jobs
                WHERE owner_id=? ORDER BY created_at DESC
                """,
                (owner_id,),
            ).fetchall()
        return [_job(json.loads(row["payload"])) for row in rows]

    def save(self, job):
        job.updated_at = datetime.now(timezone.utc).astimezone()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE project_discovery_jobs
                SET status=?, lease_expires_at=?, lease_token=?, payload=?
                WHERE owner_id=? AND job_id=?
                """,
                (
                    job.status.value,
                    job.lease_expires_at.isoformat()
                    if job.lease_expires_at
                    else None,
                    job.lease_token,
                    _dump_job(job),
                    job.owner_id,
                    job.id,
                ),
            )
            connection.commit()

    def claim(self, job_id):
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload FROM project_discovery_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if not row:
                connection.rollback()
                return None
            job = _job(json.loads(row["payload"]))
            if job.status != ProjectDiscoveryStatus.QUEUED:
                connection.rollback()
                return None
            job.status = ProjectDiscoveryStatus.RUNNING
            job.stage = ProjectDiscoveryStage.CLONING
            job.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
            job.lease_token = uuid.uuid4().hex
            job.updated_at = datetime.now(timezone.utc).astimezone()
            connection.execute(
                """
                UPDATE project_discovery_jobs
                SET status=?, lease_expires_at=?, lease_token=?, payload=?
                WHERE owner_id=? AND job_id=?
                """,
                (
                    job.status.value,
                    job.lease_expires_at.isoformat(),
                    job.lease_token,
                    _dump_job(job),
                    job.owner_id,
                    job.id,
                ),
            )
            connection.commit()
            return job

    def recoverable(self):
        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            rows = connection.execute("SELECT payload FROM project_discovery_jobs WHERE status=? OR (status=? AND (lease_expires_at IS NULL OR lease_expires_at<=?))",
                                      (ProjectDiscoveryStatus.QUEUED.value, ProjectDiscoveryStatus.RUNNING.value, now.isoformat())).fetchall()
        return [_job(json.loads(row["payload"])) for row in rows]

    def replace_evidence(self, job, evidence):
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM project_discovery_evidence WHERE owner_id=? AND job_id=?", (job.owner_id, job.id))
            connection.executemany("INSERT INTO project_discovery_evidence(owner_id,evidence_id,job_id,payload) VALUES(?,?,?,?)",
                                   [(item.owner_id, item.id, item.job_id, json.dumps(asdict(item), ensure_ascii=False)) for item in evidence])
            connection.commit()

    def list_evidence(self, owner_id, job_id):
        with self._connect() as connection:
            rows = connection.execute("SELECT payload FROM project_discovery_evidence WHERE owner_id=? AND job_id=?", (owner_id, job_id)).fetchall()
        return [ProjectDiscoveryEvidence(**json.loads(row["payload"])) for row in rows]

    def save_answer(self, answer):
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload FROM project_discovery_answers WHERE owner_id=? AND submission_id=?",
                (answer.owner_id, answer.submission_id),
            ).fetchone()
            if row:
                existing = _answer(json.loads(row["payload"]))
                if (
                    existing.job_id,
                    existing.question_id,
                    existing.answer_text,
                ) != (answer.job_id, answer.question_id, answer.answer_text):
                    connection.rollback()
                    raise ValueError("submission_id 已被另一个回答使用。")
                connection.commit()
                return existing
            row = connection.execute(
                """
                SELECT payload FROM project_discovery_answers
                WHERE owner_id=? AND job_id=?
                  AND json_extract(payload, '$.question_id')=?
                """,
                (answer.owner_id, answer.job_id, answer.question_id),
            ).fetchone()
            if row:
                connection.rollback()
                raise ValueError("这个补充问题已经回答。")
            connection.execute("INSERT INTO project_discovery_answers(owner_id,answer_id,job_id,submission_id,payload) VALUES(?,?,?,?,?)",
                               (answer.owner_id, answer.id, answer.job_id, answer.submission_id, _dump_answer(answer)))
            connection.commit()
        return answer

    def answer_by_submission(self, owner_id, submission_id):
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM project_discovery_answers WHERE owner_id=? AND submission_id=?", (owner_id, submission_id)).fetchone()
        return _answer(json.loads(row["payload"])) if row else None

    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection


def _copy_job(job):
    return replace(job, draft=dict(job.draft), questions=list(job.questions), warnings=list(job.warnings), stats=dict(job.stats))


def _dump_job(job):
    value = asdict(job)
    value["status"] = job.status.value
    value["stage"] = job.stage.value if job.stage else None
    for name in ("created_at", "updated_at", "lease_expires_at"):
        value[name] = getattr(job, name).isoformat() if getattr(job, name) else None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _job(value):
    data = dict(value)
    data.setdefault("lease_token", "")
    data["status"] = ProjectDiscoveryStatus(data["status"])
    data["stage"] = ProjectDiscoveryStage(data["stage"]) if data.get("stage") else None
    data["questions"] = [ProjectDiscoveryQuestion(**item) for item in data.get("questions", [])]
    for name in ("created_at", "updated_at", "lease_expires_at"):
        data[name] = datetime.fromisoformat(data[name]) if data.get(name) else None
    return ProjectDiscoveryJob(**data)


def _dump_answer(answer):
    value = asdict(answer); value["submitted_at"] = answer.submitted_at.isoformat()
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _answer(value):
    data = dict(value); data["submitted_at"] = datetime.fromisoformat(data["submitted_at"])
    return ProjectDiscoveryAnswer(**data)


def _ensure_column(connection, table_name, column_name, definition):
    columns = {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in columns:
        connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"
        )
