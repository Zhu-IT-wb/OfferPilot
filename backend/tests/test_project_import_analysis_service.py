import asyncio

import pytest

from app.models.project_discovery import (
    ProjectDiscoveryStatus,
)
from app.models.tool_calling import TokenUsage
from app.agents.tool_calling_agent import (
    ToolCallingAgentProgress,
    ToolCallingAgentLimits,
)
from app.project_analysis.repository_analysis_agent import (
    RepositoryAnalysisAgent,
)
from app.project_analysis.repository_analysis_result import (
    RepositoryAnalysisResult,
)
from app.project_analysis.repository_analysis_service import (
    RepositoryAnalysisRun,
    RepositoryAnalysisService,
)
from app.project_analysis.repository_workspace import (
    GitHubRepositoryWorkspace,
)
from app.repositories.project_discovery_repository import (
    InMemoryProjectDiscoveryRepository,
)
from app.repositories.project_training_repository import (
    InMemoryProjectTrainingRepository,
)
from app.services.project_discovery_dependencies import (
    build_project_discovery_services,
)
from app.services.project_import import (
    ProjectImportWorkflow,
)
from app.services.tool_calling_model import (
    DeepSeekToolCallingModel,
    OpenAIToolCallingModel,
)
from app.core.config import Settings


class FakeAnalysisService:
    """返回固定可信分析结果并触发真实任务进度回调。"""

    def __init__(self) -> None:
        """初始化分析调用记录。"""

        self.calls = []

    async def analyze(
        self,
        repository_url,
        progress_callback=None,
        runtime_callback=None,
    ):
        """模拟新代码分析服务完成一次仓库分析。"""

        self.calls.append(repository_url)

        if progress_callback is not None:
            progress_callback("cloning", 5)
            progress_callback("inventory", 20)
            progress_callback("analyzing", 30)
            progress_callback("validating", 90)
            progress_callback("validating", 100)

        return RepositoryAnalysisRun(
            repository_url=repository_url,
            commit_sha="b" * 40,
            default_branch="main",
            safe_file_count=12,
            analysis=RepositoryAnalysisResult(
                draft={
                    "name": "Checkout API",
                    "background": (
                        "项目提供订单结算接口。"
                    ),
                    "responsibilities": [],
                    "tech_stack": ["FastAPI"],
                    "architecture": (
                        "FastAPI 接收订单请求。"
                    ),
                    "key_decisions": [],
                    "technical_challenges": [],
                    "metrics": [],
                    "outcomes": [],
                    "resume_description": "",
                    "supplemental_text": "",
                },
                findings=[
                    {
                        "id": "F1",
                        "claim": "FastAPI",
                        "topic": "architecture",
                        "target_field": "tech_stack",
                        "path": "app/main.py",
                        "start_line": 1,
                        "end_line": 1,
                        "quote": (
                            "from fastapi import FastAPI"
                        ),
                        "confidence": 0.98,
                    },
                    {
                        "id": "F2",
                        "claim": (
                            "FastAPI 接收订单请求。"
                        ),
                        "topic": "architecture",
                        "target_field": "architecture",
                        "path": "app/routes.py",
                        "start_line": 10,
                        "end_line": 12,
                        "quote": (
                            "@app.post('/checkout')"
                        ),
                        "confidence": 0.9,
                    },
                ],
                evidence_by_field={
                    "tech_stack": ["F1"],
                    "architecture": ["F2"],
                },
                warnings=[
                    "生产指标需要用户补充。"
                ],
            ),
            usage=TokenUsage(
                prompt_tokens=900,
                completion_tokens=200,
                total_tokens=1100,
            ),
            model_turn_count=4,
            tool_call_count=6,
        )


class FailingAnalysisService:
    """模拟模型或仓库分析失败的新分析服务。"""

    async def analyze(
        self,
        repository_url,
        progress_callback=None,
        runtime_callback=None,
    ):
        """报告分析阶段后抛出固定异常。"""

        if progress_callback is not None:
            progress_callback("cloning", 5)
            progress_callback("analyzing", 30)

        raise RuntimeError(
            "analysis model unavailable"
        )


class ObservableFailingAnalysisService:
    """在报告 Agent 指标后模拟模型请求失败。"""

    async def analyze(
        self,
        repository_url,
        progress_callback=None,
        runtime_callback=None,
    ):
        """报告分析阶段和累计用量后抛出固定异常。"""

        if progress_callback is not None:
            progress_callback("cloning", 5)
            progress_callback("analyzing", 30)

        if runtime_callback is not None:
            runtime_callback(
                ToolCallingAgentProgress(
                    model_turn_count=4,
                    tool_call_count=7,
                    usage=TokenUsage(
                        prompt_tokens=90000,
                        completion_tokens=5000,
                        total_tokens=95000,
                    ),
                    finish_reason="tool_calls",
                    last_event={
                        "event": "model_turn",
                        "turn": 4,
                        "request_chars": 12000,
                        "finish_reason": "tool_calls",
                        "requested_tools": 1,
                        "total_tokens": 95000,
                        "decision": "execute_tools",
                    },
                )
            )

        raise RuntimeError("model request timed out")


class SlowAnalysisService:
    """让分析持续多个心跳周期后再返回成功结果。"""

    def __init__(self) -> None:
        """初始化用于生成最终结果的普通 Fake 服务。"""

        self.delegate = FakeAnalysisService()

    async def analyze(
        self,
        repository_url,
        progress_callback=None,
        runtime_callback=None,
    ):
        """暂停一段时间以验证独立租约心跳会持续执行。"""

        if progress_callback is not None:
            progress_callback("analyzing", 30)

        await asyncio.sleep(0.08)

        return await self.delegate.analyze(
            repository_url,
            progress_callback=progress_callback,
            runtime_callback=runtime_callback,
        )


class RecordingDiscoveryRepository(
    InMemoryProjectDiscoveryRepository
):
    """统计任务分析期间成功尝试保存租约的次数。"""

    def __init__(self) -> None:
        """初始化内存仓库和租约保存计数。"""

        super().__init__()
        self.save_leased_call_count = 0

    def save_leased(
        self,
        job,
        lease_token,
    ):
        """记录租约保存调用后执行真实内存 CAS。"""

        self.save_leased_call_count += 1
        return super().save_leased(
            job,
            lease_token,
        )


class LeaseLossRepository(
    InMemoryProjectDiscoveryRepository
):
    """在首次进度保存后模拟当前 Worker 丢失任务租约。"""

    def __init__(self) -> None:
        """初始化内存仓库和租约保存计数。"""

        super().__init__()
        self.save_leased_call_count = 0

    def save_leased(
        self,
        job,
        lease_token,
    ):
        """第一次保存成功，后续保存模拟 CAS 租约失败。"""

        self.save_leased_call_count += 1

        if self.save_leased_call_count > 1:
            return False

        return super().save_leased(
            job,
            lease_token,
        )


class CancellableSlowAnalysisService:
    """持续等待直到心跳因租约丢失而取消当前分析。"""

    def __init__(self) -> None:
        """初始化分析开始和取消状态。"""

        self.started = False
        self.cancelled = False

    async def analyze(
        self,
        repository_url,
        progress_callback=None,
        runtime_callback=None,
    ):
        """进入分析后无限等待，并记录收到的取消异常。"""

        self.started = True

        if progress_callback is not None:
            progress_callback("analyzing", 30)

        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def test_workflow_persists_new_agent_analysis_and_evidence() -> None:
    """验证新 Agent 服务结果能进入任务、项目档案和代码证据。"""

    jobs = InMemoryProjectDiscoveryRepository()
    projects = InMemoryProjectTrainingRepository()
    analysis_service = FakeAnalysisService()
    workflow = ProjectImportWorkflow(
        repository=jobs,
        project_repository=projects,
        analysis_service=analysis_service,
    )
    job = workflow.start(
        owner_id="feishu:owner",
        repository_url=(
            "https://github.com/example/checkout"
        ),
        request_id="new-agent-request-1234",
    )

    completed = asyncio.run(
        workflow.run(job.id)
    )

    assert analysis_service.calls == [
        "https://github.com/example/checkout"
    ]
    assert completed.status == (
        ProjectDiscoveryStatus.NEEDS_INPUT
    )
    assert completed.commit_sha == "b" * 40
    assert completed.draft["name"] == (
        "Checkout API"
    )
    assert completed.draft["tech_stack"] == [
        "FastAPI"
    ]
    assert completed.stats == {
        "file_count": 12,
        "evidence_file_count": 2,
        "languages": {},
        "default_branch": "main",
        "model_turn_count": 4,
        "tool_call_count": 6,
        "prompt_tokens": 900,
        "completion_tokens": 200,
        "total_tokens": 1100,
        "completion_reason": "model_stop",
        "result_status": "complete",
        "runtime_trace": [],
    }
    assert [
        question.id
        for question in completed.questions
    ] == [
        "responsibilities",
        "metrics",
        "outcomes",
    ]
    assert completed.warnings == [
        "生产指标需要用户补充。"
    ]

    evidence = jobs.list_evidence(
        "feishu:owner",
        job.id,
    )

    assert len(evidence) == 2
    assert evidence[0].source_type == "code"
    assert evidence[0].file_path == (
        "app/main.py"
    )
    assert evidence[0].commit_sha == "b" * 40
    assert evidence[0].claim == "FastAPI"
    assert evidence[0].content_hash

    answered = workflow.answer(
        owner_id="feishu:owner",
        job_id=job.id,
        question_id="responsibilities",
        answer_text="我负责订单接口和缓存设计。",
        submission_id="new-agent-answer-1234",
    )
    assert answered.status == (
        ProjectDiscoveryStatus.READY
    )

    project = workflow.confirm(
        owner_id="feishu:owner",
        job_id=job.id,
        confirmation_id=(
            "new-agent-confirmation-1234"
        ),
    )
    project_evidence = (
        projects.list_project_evidence(
            "feishu:owner",
            project.id,
            project.version,
        )
    )

    assert project.source_commit_sha == "b" * 40
    assert any(
        item.source_type == "code"
        and item.source_path == "app/main.py"
        and item.grounded_claim == "FastAPI"
        for item in project_evidence
    )
    assert any(
        item.source_type == "user_statement"
        for item in project_evidence
    )


def test_workflow_records_new_analysis_service_failure() -> None:
    """验证新 Agent 服务异常会转成可重试的失败任务。"""

    jobs = InMemoryProjectDiscoveryRepository()
    workflow = ProjectImportWorkflow(
        repository=jobs,
        project_repository=(
            InMemoryProjectTrainingRepository()
        ),
        analysis_service=FailingAnalysisService(),
    )
    job = workflow.start(
        owner_id="feishu:owner",
        repository_url=(
            "https://github.com/example/failing"
        ),
        request_id="new-agent-failing-1234",
    )

    failed = asyncio.run(
        workflow.run(job.id)
    )

    assert failed.status == (
        ProjectDiscoveryStatus.FAILED
    )
    assert failed.error_code == "RuntimeError"
    assert failed.error_message == (
        "analysis model unavailable"
    )
    assert failed.lease_token == ""
    assert failed.lease_expires_at is None


def test_workflow_preserves_latest_stage_and_usage_on_failure() -> None:
    """验证失败任务不会被最初的 cloning 状态覆盖运行指标。"""

    jobs = InMemoryProjectDiscoveryRepository()
    workflow = ProjectImportWorkflow(
        repository=jobs,
        project_repository=(
            InMemoryProjectTrainingRepository()
        ),
        analysis_service=(
            ObservableFailingAnalysisService()
        ),
    )
    job = workflow.start(
        owner_id="feishu:owner",
        repository_url=(
            "https://github.com/example/observable-failure"
        ),
        request_id="observable-failure-1234",
    )

    failed = asyncio.run(
        workflow.run(job.id)
    )

    assert failed.status == (
        ProjectDiscoveryStatus.FAILED
    )
    assert failed.stage.value == "analyzing"
    assert failed.progress > 30
    assert failed.stats == {
        "model_turn_count": 4,
        "tool_call_count": 7,
        "prompt_tokens": 90000,
        "completion_tokens": 5000,
        "total_tokens": 95000,
        "last_finish_reason": "tool_calls",
        "runtime_trace": [
            {
                "event": "model_turn",
                "turn": 4,
                "request_chars": 12000,
                "finish_reason": "tool_calls",
                "requested_tools": 1,
                "total_tokens": 95000,
                "decision": "execute_tools",
            }
        ],
    }


def test_retry_clears_previous_runtime_stats() -> None:
    """重新分析从干净的运行指标开始，不混入上一轮轨迹。"""

    jobs = InMemoryProjectDiscoveryRepository()
    workflow = ProjectImportWorkflow(
        repository=jobs,
        project_repository=InMemoryProjectTrainingRepository(),
        analysis_service=ObservableFailingAnalysisService(),
    )
    job = workflow.start(
        owner_id="feishu:owner",
        repository_url="https://github.com/example/retry-stats",
        request_id="retry-stats-1234",
    )
    failed = asyncio.run(workflow.run(job.id))

    assert failed.stats["runtime_trace"]

    retried = workflow.retry("feishu:owner", job.id)

    assert retried.status == ProjectDiscoveryStatus.QUEUED
    assert retried.stats == {}
    assert retried.retry_count == 1


def test_workflow_renews_lease_during_slow_agent_analysis() -> None:
    """验证长时间没有阶段变化时独立心跳仍会刷新租约。"""

    jobs = RecordingDiscoveryRepository()
    workflow = ProjectImportWorkflow(
        repository=jobs,
        project_repository=(
            InMemoryProjectTrainingRepository()
        ),
        analysis_service=SlowAnalysisService(),
        lease_heartbeat_seconds=0.01,
    )
    job = workflow.start(
        owner_id="feishu:owner",
        repository_url=(
            "https://github.com/example/slow"
        ),
        request_id="new-agent-slow-1234",
    )

    completed = asyncio.run(
        workflow.run(job.id)
    )

    assert completed.status == (
        ProjectDiscoveryStatus.NEEDS_INPUT
    )
    assert jobs.save_leased_call_count > 6


def test_unconfigured_workflow_does_not_claim_job_lease() -> None:
    """验证缺少分析器时任务保持排队状态而不会占住租约。"""

    jobs = InMemoryProjectDiscoveryRepository()
    workflow = ProjectImportWorkflow(
        repository=jobs,
        project_repository=(
            InMemoryProjectTrainingRepository()
        ),
    )
    job = workflow.start(
        owner_id="feishu:owner",
        repository_url=(
            "https://github.com/example/unconfigured"
        ),
        request_id="unconfigured-request-1234",
    )

    with pytest.raises(
        RuntimeError,
        match="not configured",
    ):
        asyncio.run(
            workflow.run(job.id)
        )

    stored = workflow.resume(
        "feishu:owner",
        job.id,
    )
    assert stored.status == (
        ProjectDiscoveryStatus.QUEUED
    )
    assert stored.lease_token == ""
    assert stored.lease_expires_at is None


def test_lease_loss_cancels_agent_without_persisting_result() -> None:
    """验证心跳丢失租约时取消 Agent 且不保存草稿或证据。"""

    jobs = LeaseLossRepository()
    analysis_service = (
        CancellableSlowAnalysisService()
    )
    workflow = ProjectImportWorkflow(
        repository=jobs,
        project_repository=(
            InMemoryProjectTrainingRepository()
        ),
        analysis_service=analysis_service,
        lease_heartbeat_seconds=0.01,
    )
    job = workflow.start(
        owner_id="feishu:owner",
        repository_url=(
            "https://github.com/example/lease-loss"
        ),
        request_id="lease-loss-request-1234",
    )

    returned = asyncio.run(
        workflow.run(job.id)
    )

    assert analysis_service.started is True
    assert analysis_service.cancelled is True
    assert returned.draft == {}
    assert returned.commit_sha == ""
    assert jobs.list_evidence(
        "feishu:owner",
        job.id,
    ) == []


def test_production_dependencies_use_repository_analysis_service() -> None:
    """验证生产依赖已经从旧 Graph 切换到新分析 Service。"""

    services = build_project_discovery_services(
        Settings(
            storage_backend="memory",
            llm_provider="deepseek",
            llm_api_key="test-key",
            llm_model="deepseek-chat",
            project_analysis_llm_timeout_seconds=180,
            project_analysis_max_model_turns=15,
            project_analysis_max_tool_calls=40,
            project_analysis_max_total_tokens=(
                450_000
            ),
            project_analysis_max_request_chars=(
                400_000
            ),
            project_analysis_max_findings=12,
            project_analysis_max_quote_chars=400,
        ),
        InMemoryProjectTrainingRepository(),
    )

    assert services.workflow.graph is None
    assert isinstance(
        services.workflow.analysis_service,
        RepositoryAnalysisService,
    )
    analysis_service = (
        services.workflow.analysis_service
    )
    assert isinstance(
        analysis_service._agent,
        RepositoryAnalysisAgent,
    )
    assert isinstance(
        analysis_service._agent._model,
        DeepSeekToolCallingModel,
    )
    assert (
        analysis_service._agent._model.timeout_seconds
        == 180
    )
    assert analysis_service._agent._limits == (
        ToolCallingAgentLimits(
            max_model_turns=15,
            max_tool_calls=40,
            max_total_tokens=450_000,
            max_request_chars=400_000,
        )
    )
    assert analysis_service._workspace_factory.func is (
        GitHubRepositoryWorkspace
    )
    assert (
        analysis_service._parser._max_findings
        == 12
    )
    assert (
        analysis_service._parser._max_quote_chars
        == 400
    )
    assert analysis_service._workspace_factory.keywords == {
        "clone_timeout_seconds": 120,
        "max_workspace_mb": 100,
    }


def test_project_analysis_dependencies_select_openai_adapter() -> None:
    services = build_project_discovery_services(
        Settings(
            storage_backend="memory",
            llm_provider="openai",
            llm_api_key="openai-key",
            llm_base_url="https://api.openai.com/v1",
            llm_model="gpt-4.1-mini",
        ),
        InMemoryProjectTrainingRepository(),
    )

    model = services.workflow.analysis_service._agent._model

    assert isinstance(model, OpenAIToolCallingModel)
    assert model.provider == "openai"
    assert model.base_url == "https://api.openai.com/v1"
