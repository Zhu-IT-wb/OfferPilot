import asyncio
import hashlib
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.agents.tool_calling_agent import MAX_PERSISTED_MODEL_TURNS
from app.models.project_discovery import (
    ACTIVE_DISCOVERY_STATUSES,
    ProjectDiscoveryAnswer,
    ProjectDiscoveryEvidence,
    ProjectDiscoveryQuestion,
    ProjectDiscoveryStage,
    ProjectDiscoveryStatus,
)
from app.services.github_repository_source import normalize_public_github_url


class _RepositoryAnalysisServiceBackend:
    """把新 RepositoryAnalysisService 适配成 Workflow 内部协议。"""

    def __init__(self, analysis_service) -> None:
        """保存生产环境使用的新代码分析服务。"""

        self.analysis_service = analysis_service

    async def analyze(
        self,
        job,
        progress_callback,
        runtime_callback,
    ):
        """运行新服务并转换成统一的任务完成数据。"""

        analysis_run = await self.analysis_service.analyze(
            job.repository_url,
            progress_callback=progress_callback,
            runtime_callback=runtime_callback,
        )
        return _completed_from_analysis_run(
            job,
            analysis_run,
        )


class _LegacyProjectDiscoveryGraphBackend:
    """仅为旧测试保留的 ProjectDiscoveryGraph 协议适配器。"""

    def __init__(self, graph) -> None:
        """保存待兼容的旧 ProjectDiscoveryGraph。"""

        self.graph = graph

    async def analyze(
        self,
        job,
        progress_callback,
        runtime_callback,
    ):
        """运行旧 Graph，并忽略新 Runtime 才支持的轮次回调。"""

        state = await self.graph.run(
            job.owner_id,
            job.id,
            job.repository_url,
            progress_callback=progress_callback,
        )
        return _completed_from_legacy_state(
            state
        )


class ProjectImportWorkflow:
    def __init__(
        self,
        repository,
        project_repository,
        graph=None,
        analysis_service=None,
        lease_heartbeat_seconds=60,
    ):
        """保存任务仓库、项目仓库和唯一的代码分析实现。"""

        if graph is not None and analysis_service is not None:
            raise ValueError(
                "Configure either graph or analysis_service, not both."
            )
        if lease_heartbeat_seconds <= 0:
            raise ValueError(
                "lease_heartbeat_seconds must be positive."
            )

        self.repository = repository
        self.project_repository = project_repository
        self.graph = graph
        self.analysis_service = analysis_service
        self.lease_heartbeat_seconds = (
            lease_heartbeat_seconds
        )
        self._analysis_backend = (
            _RepositoryAnalysisServiceBackend(
                analysis_service
            )
            if analysis_service is not None
            else (
                _LegacyProjectDiscoveryGraphBackend(
                    graph
                )
                if graph is not None
                else None
            )
        )

    def start(self, owner_id: str, repository_url: str, request_id: str,
              project_id: Optional[str] = None):
        canonical = normalize_public_github_url(repository_url)
        request_id = str(request_id or "").strip()
        if len(request_id) < 8 or len(request_id) > 160:
            raise ValueError("request_id 长度必须在 8 到 160 个字符之间。")
        if project_id:
            project = self.project_repository.get_project(owner_id, project_id)
            if project is None or project.status.value != "active":
                raise ValueError("要重新分析的项目不存在或已归档。")
        return self.repository.create_or_get(
            owner_id, canonical, request_id, project_id
        )

    def resume(self, owner_id: str, job_id: str):
        return self.repository.get(owner_id, job_id)

    async def run(self, job_id: str):
        if self._analysis_backend is None:
            raise RuntimeError("Repository analysis is not configured.")

        job = self.repository.claim(job_id)
        if job is None:
            return self.repository.get_unscoped(job_id)
        lease_token = job.lease_token
        try:
            def progress(stage, value):
                """持久化代码下载、清点、分析和校验阶段。"""

                latest = self.repository.get(job.owner_id, job.id)
                if latest is None:
                    return
                if latest.status == ProjectDiscoveryStatus.CANCELLED:
                    raise RuntimeError("Project discovery was cancelled.")
                latest.status = ProjectDiscoveryStatus.RUNNING
                latest.stage = ProjectDiscoveryStage(stage)
                latest.progress = value
                latest.lease_expires_at = (
                    datetime.now(timezone.utc) + timedelta(minutes=5)
                )
                if not self.repository.save_leased(latest, lease_token):
                    raise RuntimeError("Project discovery lease was lost.")

            def runtime_progress(snapshot):
                """持久化 Agent 每轮累计用量并推进分析阶段进度。"""

                latest = self.repository.get(
                    job.owner_id,
                    job.id,
                )
                if latest is None:
                    return
                if latest.status == ProjectDiscoveryStatus.CANCELLED:
                    raise RuntimeError(
                        "Project discovery was cancelled."
                    )

                latest.status = ProjectDiscoveryStatus.RUNNING
                latest.stage = ProjectDiscoveryStage.ANALYZING
                latest.progress = max(
                    latest.progress,
                    min(
                        85,
                        30 + snapshot.model_turn_count * 2,
                    ),
                )
                latest.stats = {
                    **latest.stats,
                    "model_turn_count": (
                        snapshot.model_turn_count
                    ),
                    "tool_call_count": (
                        snapshot.tool_call_count
                    ),
                    "prompt_tokens": (
                        snapshot.usage.prompt_tokens
                    ),
                    "completion_tokens": (
                        snapshot.usage.completion_tokens
                    ),
                    "total_tokens": (
                        snapshot.usage.total_tokens
                    ),
                    "last_finish_reason": (
                        snapshot.finish_reason
                    ),
                }
                if snapshot.last_event is not None:
                    event = dict(snapshot.last_event)
                    if event.get("event") == "model_turn":
                        runtime_trace = [
                            item
                            for item in latest.stats.get(
                                "runtime_trace",
                                [],
                            )
                            if item.get("turn") != event.get("turn")
                        ]
                        runtime_trace.append(event)
                        latest.stats["runtime_trace"] = runtime_trace[
                            -MAX_PERSISTED_MODEL_TURNS:
                        ]
                latest.lease_expires_at = (
                    datetime.now(timezone.utc)
                    + timedelta(minutes=5)
                )

                if not self.repository.save_leased(
                    latest,
                    lease_token,
                ):
                    raise RuntimeError(
                        "Project discovery lease was lost."
                    )

            completed = await self._analyze_with_heartbeat(
                job,
                lease_token,
                progress,
                runtime_progress,
            )
            latest = self.repository.get(job.owner_id, job.id)
            if latest is not None and latest.status == ProjectDiscoveryStatus.CANCELLED:
                return latest
            job = latest or job
            job.commit_sha = completed["commit_sha"]
            job.draft = completed["draft"]
            if not job.draft.get("name"):
                job.draft["name"] = job.repository_url.rsplit("/", 1)[-1]
            job.questions = completed["questions"]
            job.warnings = completed["warnings"]
            job.stats = completed["stats"]
            job.stage = ProjectDiscoveryStage.VALIDATING
            job.progress = 100
            job.status = (
                ProjectDiscoveryStatus.NEEDS_INPUT
                if job.questions else ProjectDiscoveryStatus.READY
            )
            job.error_code = ""
            job.error_message = ""
            job.lease_expires_at = None
            job.lease_token = ""
            if not self.repository.complete_analysis(
                job, completed["evidence"], lease_token
            ):
                return self.repository.get_unscoped(job.id)
            return self.repository.get(job.owner_id, job.id)
        except asyncio.CancelledError:
            self.repository.release_lease(job.id, lease_token)
            raise
        except Exception as exc:
            latest = self.repository.get(job.owner_id, job.id)
            if latest is not None and latest.status == ProjectDiscoveryStatus.CANCELLED:
                return latest
            failed_job = latest or job
            failed_job.status = ProjectDiscoveryStatus.FAILED
            failed_job.error_code = type(exc).__name__
            failed_job.error_message = (
                str(exc)[:500]
                or "项目分析失败，请稍后重试。"
            )
            failed_job.lease_expires_at = None
            failed_job.lease_token = ""
            if not self.repository.save_leased(
                failed_job,
                lease_token,
            ):
                return self.repository.get_unscoped(job.id)
            return self.repository.get(job.owner_id, job.id)

    async def _analyze_with_heartbeat(
        self,
        job,
        lease_token,
        progress_callback,
        runtime_callback,
    ):
        """分析期间持续续租，并在租约丢失时取消 Agent。"""

        analysis_task = asyncio.create_task(
            self._analyze_repository(
                job,
                progress_callback,
                runtime_callback,
            )
        )
        heartbeat_task = asyncio.create_task(
            self._lease_heartbeat(
                job,
                lease_token,
            )
        )

        try:
            completed, _ = await asyncio.wait(
                {
                    analysis_task,
                    heartbeat_task,
                },
                return_when=asyncio.FIRST_COMPLETED,
            )

            if heartbeat_task in completed:
                await heartbeat_task
                raise RuntimeError(
                    "Project discovery heartbeat stopped unexpectedly."
                )

            return await analysis_task
        finally:
            for task in (
                analysis_task,
                heartbeat_task,
            ):
                if not task.done():
                    task.cancel()

            await asyncio.gather(
                analysis_task,
                heartbeat_task,
                return_exceptions=True,
            )

    async def _lease_heartbeat(
        self,
        job,
        lease_token,
    ) -> None:
        """按固定间隔刷新运行任务的租约到期时间。"""

        while True:
            await asyncio.sleep(
                self.lease_heartbeat_seconds
            )
            latest = self.repository.get(
                job.owner_id,
                job.id,
            )

            if latest is None:
                raise RuntimeError(
                    "Project discovery job no longer exists."
                )

            if latest.status == ProjectDiscoveryStatus.CANCELLED:
                raise RuntimeError(
                    "Project discovery was cancelled."
                )

            latest.status = ProjectDiscoveryStatus.RUNNING
            latest.lease_expires_at = (
                datetime.now(timezone.utc)
                + timedelta(minutes=5)
            )

            if not self.repository.save_leased(
                latest,
                lease_token,
            ):
                raise RuntimeError(
                    "Project discovery lease was lost."
                )

    async def _analyze_repository(
        self,
        job,
        progress_callback,
        runtime_callback,
    ):
        """通过统一后端协议执行一次项目代码分析。"""

        return await self._analysis_backend.analyze(
            job,
            progress_callback=progress_callback,
            runtime_callback=runtime_callback,
        )

    def answer(self, owner_id: str, job_id: str, question_id: str,
               answer_text: str, submission_id: str):
        job = self.repository.get(owner_id, job_id)
        if job is None:
            return None
        if job.status not in {
            ProjectDiscoveryStatus.NEEDS_INPUT, ProjectDiscoveryStatus.READY
        }:
            raise ValueError("当前分析任务不能补充回答。")
        question = next((item for item in job.questions if item.id == question_id), None)
        if question is None:
            raise ValueError("补充问题不存在。")
        text = str(answer_text or "").strip()
        if not text:
            raise ValueError("回答不能为空。")
        submission_id = str(submission_id or "").strip()
        if len(submission_id) < 8:
            raise ValueError("submission_id 至少需要 8 个字符。")
        answer = ProjectDiscoveryAnswer(
            id=f"discovery_answer_{uuid.uuid4().hex}", owner_id=owner_id,
            job_id=job.id, question_id=question.id,
            submission_id=submission_id, answer_text=text,
            submitted_at=datetime.now(timezone.utc).astimezone(),
        )
        stored = self.repository.save_answer(answer)
        already_answered = question.answered
        job.questions = [
            replace(item, answered=True) if item.id == question.id else item
            for item in job.questions
        ]
        if not already_answered:
            if not _is_unknown(text):
                current = list(job.draft.get(question.target_field, []))
                if text not in current:
                    current.append(text)
                job.draft[question.target_field] = current
            evidence = self.repository.list_evidence(owner_id, job.id)
            evidence_id = "discovery_evidence_" + hashlib.sha256(
                f"{owner_id}:{stored.submission_id}".encode()
            ).hexdigest()[:24]
            if not any(item.id == evidence_id for item in evidence):
                evidence.append(
                    ProjectDiscoveryEvidence(
                        id=evidence_id,
                        owner_id=owner_id,
                        job_id=job.id,
                        source_type="user_statement",
                        file_path="",
                        start_line=None,
                        end_line=None,
                        excerpt=stored.answer_text,
                        content_hash=hashlib.sha256(
                            stored.answer_text.encode()
                        ).hexdigest(),
                        topic=_topic_for_field(question.target_field),
                        target_field=question.target_field,
                        confidence=1.0,
                        commit_sha=job.commit_sha,
                    )
                )
            self.repository.replace_evidence(job, evidence)
        job.status = (
            ProjectDiscoveryStatus.READY
            if all(item.answered or not item.required for item in job.questions)
            else ProjectDiscoveryStatus.NEEDS_INPUT
        )
        self.repository.save(job)
        return job

    def confirm(self, owner_id: str, job_id: str, confirmation_id: str):
        job = self.repository.get(owner_id, job_id)
        if job is None:
            return None
        confirmation_id = str(confirmation_id or "").strip()
        if len(confirmation_id) < 8:
            raise ValueError("confirmation_id 至少需要 8 个字符。")
        if job.confirmed_project_id:
            confirmed = self.project_repository.get_project(
                owner_id, job.confirmed_project_id
            )
            if confirmed is not None:
                return confirmed
        existing = self.project_repository.get_project_by_discovery_job(
            owner_id, job.id
        )
        if existing is not None:
            job.status = ProjectDiscoveryStatus.CONFIRMED
            job.confirmed_project_id = existing.id
            job.confirmation_id = job.confirmation_id or confirmation_id
            self.repository.save(job)
            return existing
        if job.status != ProjectDiscoveryStatus.READY:
            raise ValueError("请先完成必答的个人职责问题。")
        values = dict(job.draft)
        values.update({
            "project_id": job.project_id,
            "source_repository_url": job.repository_url,
            "source_commit_sha": job.commit_sha,
            "source_discovery_job_id": job.id,
        })
        project = self.project_repository.create_discovered_project(
            owner_id, values,
            self.repository.list_evidence(owner_id, job.id),
        )
        job.status = ProjectDiscoveryStatus.CONFIRMED
        job.confirmation_id = confirmation_id
        job.confirmed_project_id = project.id
        self.repository.save(job)
        return project

    def retry(self, owner_id: str, job_id: str):
        job = self.repository.get(owner_id, job_id)
        if job is None:
            return None
        if job.status != ProjectDiscoveryStatus.FAILED:
            raise ValueError("只有失败的分析任务可以重试。")
        if job.retry_count >= 3:
            raise ValueError("这个任务已达到最多 3 次重试。")
        if any(
            item.id != job.id
            and item.repository_url == job.repository_url
            and item.status in ACTIVE_DISCOVERY_STATUSES
            for item in self.repository.list(owner_id)
        ):
            raise ValueError("这个仓库已有另一个分析任务正在进行。")
        job.retry_count += 1
        job.status = ProjectDiscoveryStatus.QUEUED
        job.stage = None
        job.progress = 0
        job.error_code = ""
        job.error_message = ""
        self.repository.save(job)
        return job

    def cancel(self, owner_id: str, job_id: str):
        job = self.repository.get(owner_id, job_id)
        if job is None:
            return None
        if job.status in {ProjectDiscoveryStatus.CONFIRMED, ProjectDiscoveryStatus.CANCELLED}:
            return job
        job.status = ProjectDiscoveryStatus.CANCELLED
        job.lease_expires_at = None
        job.lease_token = ""
        self.repository.save(job)
        return job


def _completed_from_analysis_run(job, analysis_run):
    """把新 Agent 服务结果转换成现有任务持久化协议。"""

    findings = list(
        analysis_run.analysis.findings
    )
    analyzed_paths = {
        finding["path"]
        for finding in findings
    }

    return {
        "commit_sha": analysis_run.commit_sha,
        "draft": dict(
            analysis_run.analysis.draft
        ),
        "questions": _build_discovery_questions(
            analysis_run.analysis.draft
        ),
        "warnings": list(
            analysis_run.analysis.warnings
        ),
        "stats": {
            "file_count": (
                analysis_run.safe_file_count
            ),
            "evidence_file_count": len(
                analyzed_paths
            ),
            "languages": {},
            "default_branch": (
                analysis_run.default_branch
            ),
            "model_turn_count": (
                analysis_run.model_turn_count
            ),
            "tool_call_count": (
                analysis_run.tool_call_count
            ),
            "prompt_tokens": (
                analysis_run.usage.prompt_tokens
            ),
            "completion_tokens": (
                analysis_run.usage.completion_tokens
            ),
            "total_tokens": (
                analysis_run.usage.total_tokens
            ),
            "completion_reason": (
                analysis_run.completion_reason
            ),
            "result_status": (
                analysis_run.result_status
            ),
            "runtime_trace": [
                dict(event)
                for event in analysis_run.trace[
                    -MAX_PERSISTED_MODEL_TURNS:
                ]
            ],
        },
        "evidence": _build_discovery_evidence(
            job,
            analysis_run,
        ),
    }


def _completed_from_legacy_state(state):
    """把旧 Graph 状态转换成与新分析服务相同的完成协议。"""

    snapshot = state["snapshot"]

    return {
        "commit_sha": snapshot.commit_sha,
        "draft": dict(state["draft"]),
        "questions": [
            ProjectDiscoveryQuestion(**item)
            for item in state["questions"]
        ],
        "warnings": list(
            state.get("warnings", [])
        ),
        "stats": {
            **state.get("inventory", {}),
            "selected_file_count": len(
                state.get("selected_paths", [])
            ),
            "analyzed_file_count": len(
                snapshot.files
            ),
        },
        "evidence": list(state["evidence"]),
    }


def _build_discovery_questions(draft):
    """为代码无法证明的个人职责、指标和结果生成补充问题。"""

    questions = [
        ProjectDiscoveryQuestion(
            id="responsibilities",
            target_field="responsibilities",
            prompt=(
                "你本人在这个项目中具体负责了哪些模块、"
                "设计或关键决策？"
            ),
            required=True,
        )
    ]

    optional_questions = (
        (
            "metrics",
            "这个项目是否有你能确认的性能、流量或质量指标？"
            "没有可以回答“暂无”。",
        ),
        (
            "outcomes",
            "这个项目最终带来了什么业务或工程结果？"
            "不知道可以回答“暂无”。",
        ),
    )

    for target_field, prompt in optional_questions:
        if draft.get(target_field):
            continue

        questions.append(
            ProjectDiscoveryQuestion(
                id=target_field,
                target_field=target_field,
                prompt=prompt,
                required=False,
            )
        )

    return questions


def _build_discovery_evidence(job, analysis_run):
    """把已验证 Finding 转换成可持久化的代码证据对象。"""

    evidence = []

    for finding in analysis_run.analysis.findings:
        excerpt = finding["quote"].strip()
        finding_id = finding["id"]
        evidence_id = (
            "discovery_evidence_"
            + hashlib.sha256(
                f"{job.id}:{finding_id}".encode()
            ).hexdigest()[:24]
        )

        evidence.append(
            ProjectDiscoveryEvidence(
                id=evidence_id,
                owner_id=job.owner_id,
                job_id=job.id,
                source_type="code",
                file_path=finding["path"],
                start_line=finding["start_line"],
                end_line=finding["end_line"],
                excerpt=excerpt,
                content_hash=hashlib.sha256(
                    excerpt.encode("utf-8")
                ).hexdigest(),
                topic=finding["topic"],
                target_field=(
                    finding["target_field"]
                ),
                confidence=finding["confidence"],
                commit_sha=(
                    analysis_run.commit_sha
                ),
                claim=finding["claim"],
            )
        )

    return evidence


def _is_unknown(text):
    normalized = text.strip().lower().replace("。", "")
    return normalized in {"暂无", "没有", "不知道", "不清楚", "none", "n/a"}


def _topic_for_field(field):
    return {
        "responsibilities": "ownership", "metrics": "metrics",
        "outcomes": "metrics",
    }.get(field, "project_overview")
