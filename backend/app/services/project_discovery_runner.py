import asyncio
import logging
from typing import Optional, Set

from app.models.project_discovery import (
    ProjectDiscoveryStatus,
)


class ProjectDiscoveryQueueFull(RuntimeError):
    pass


logger = logging.getLogger(__name__)


class ProjectDiscoveryRunner:
    """Bounded persistent-job runner with periodic expired-lease recovery."""

    def __init__(
        self,
        workflow,
        max_concurrency: int = 1,
        queue_capacity: int = 100,
        recovery_interval_seconds: int = 30,
    ):
        self.workflow = workflow
        self._max_concurrency = max(1, max_concurrency)
        self._queue_capacity = max(self._max_concurrency, queue_capacity)
        self._recovery_interval_seconds = max(5, recovery_interval_seconds)
        self._queue: Optional[asyncio.Queue] = None
        self._workers: Set[asyncio.Task] = set()
        self._recovery_task: Optional[asyncio.Task] = None
        self._queued_ids: Set[str] = set()
        self._active_ids: Set[str] = set()
        self._executions = {}

    def start(self):
        self._ensure_started()
        self._recover_once()

    def enqueue(self, job_id: str):
        self._ensure_started()
        if job_id in self._queued_ids or job_id in self._active_ids:
            return
        try:
            self._queue.put_nowait(job_id)
        except asyncio.QueueFull as exc:
            raise ProjectDiscoveryQueueFull(
                "项目分析队列已满，请稍后重试。"
            ) from exc
        self._queued_ids.add(job_id)

    def cancel(self, job_id: str):
        execution = self._executions.get(job_id)
        if execution is not None and not execution.done():
            execution.cancel()

    def _ensure_started(self):
        if self._queue is not None:
            return
        self._queue = asyncio.Queue(maxsize=self._queue_capacity)
        self._workers = {
            asyncio.create_task(self._worker())
            for _ in range(self._max_concurrency)
        }
        self._recovery_task = asyncio.create_task(self._recovery_loop())

    async def _worker(self):
        while True:
            job_id = await self._queue.get()
            self._queued_ids.discard(job_id)
            self._active_ids.add(job_id)
            execution = asyncio.create_task(self.workflow.run(job_id))
            self._executions[job_id] = execution
            try:
                await execution
            except asyncio.CancelledError:
                # Python 3.9 has no Task.cancelling(). A cancelled child is a
                # normal per-job cancellation; shutdown cancels workers only
                # after all child executions have settled (see stop()).
                continue
            except Exception:
                logger.exception("Project discovery job failed outside workflow: %s", job_id)
            finally:
                self._executions.pop(job_id, None)
                self._active_ids.discard(job_id)
                self._queue.task_done()

    async def _recovery_loop(self):
        while True:
            await asyncio.sleep(self._recovery_interval_seconds)
            try:
                self._recover_once()
            except Exception:
                logger.exception("Project discovery recovery sweep failed.")

    def _recover_once(self):
        for job in self.workflow.repository.recoverable():
            if job.status == ProjectDiscoveryStatus.RUNNING:
                job = self.workflow.repository.requeue_expired(
                    job.id, job.lease_token
                )
                if job is None:
                    continue
            try:
                self.enqueue(job.id)
            except ProjectDiscoveryQueueFull:
                break

    async def stop(self):
        executions = list(self._executions.values())
        for execution in executions:
            execution.cancel()
        if executions:
            await asyncio.gather(*executions, return_exceptions=True)
        tasks = list(self._workers)
        if self._recovery_task is not None:
            tasks.append(self._recovery_task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._workers.clear()
        self._queued_ids.clear()
        self._active_ids.clear()
        self._executions.clear()
        self._recovery_task = None
        self._queue = None
