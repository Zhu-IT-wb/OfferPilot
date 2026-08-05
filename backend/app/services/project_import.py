import asyncio
import hashlib
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.models.project_discovery import (
    ACTIVE_DISCOVERY_STATUSES,
    ProjectDiscoveryAnswer,
    ProjectDiscoveryEvidence,
    ProjectDiscoveryQuestion,
    ProjectDiscoveryStage,
    ProjectDiscoveryStatus,
)
from app.services.github_repository_source import normalize_public_github_url


class ProjectImportWorkflow:
    def __init__(self, repository, project_repository, graph):
        self.repository = repository
        self.project_repository = project_repository
        self.graph = graph

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
        job = self.repository.claim(job_id)
        if job is None:
            return self.repository.get_unscoped(job_id)
        lease_token = job.lease_token
        if self.graph is None:
            raise RuntimeError("ProjectDiscoveryGraph is not configured.")
        try:
            def progress(stage, value):
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

            state = await self.graph.run(
                job.owner_id, job.id, job.repository_url,
                progress_callback=progress,
            )
            latest = self.repository.get(job.owner_id, job.id)
            if latest is not None and latest.status == ProjectDiscoveryStatus.CANCELLED:
                return latest
            snapshot = state["snapshot"]
            job.commit_sha = snapshot.commit_sha
            job.draft = state["draft"]
            if not job.draft.get("name"):
                job.draft["name"] = job.repository_url.rsplit("/", 1)[-1]
            job.questions = [
                ProjectDiscoveryQuestion(**item) for item in state["questions"]
            ]
            job.warnings = list(state.get("warnings", []))
            job.stats = {
                **state.get("inventory", {}),
                "selected_file_count": len(state.get("selected_paths", [])),
                "analyzed_file_count": len(snapshot.files),
            }
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
                job, state["evidence"], lease_token
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
            job.status = ProjectDiscoveryStatus.FAILED
            job.error_code = type(exc).__name__
            job.error_message = str(exc)[:500] or "项目分析失败，请稍后重试。"
            job.lease_expires_at = None
            job.lease_token = ""
            if not self.repository.save_leased(job, lease_token):
                return self.repository.get_unscoped(job.id)
            return self.repository.get(job.owner_id, job.id)

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


def _is_unknown(text):
    normalized = text.strip().lower().replace("。", "")
    return normalized in {"暂无", "没有", "不知道", "不清楚", "none", "n/a"}


def _topic_for_field(field):
    return {
        "responsibilities": "ownership", "metrics": "metrics",
        "outcomes": "metrics",
    }.get(field, "project_overview")
