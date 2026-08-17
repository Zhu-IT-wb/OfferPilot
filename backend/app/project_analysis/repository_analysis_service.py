from dataclasses import dataclass
from typing import Callable, Optional

from app.agents.tool_calling_agent import (
    AgentProgressCallback,
    MAX_PERSISTED_MODEL_TURNS,
    ToolCallingAgentResult,
)
from app.models.tool_calling import TokenUsage
from app.project_analysis.repository_analysis_agent import (
    RepositoryAnalysisAgent,
)
from app.project_analysis.repository_analysis_result import (
    RepositoryAnalysisResult,
    RepositoryAnalysisResultParser,
)
from app.project_analysis.repository_workspace import (
    GitHubRepositoryWorkspace,
)


ProgressCallback = Callable[
    [str, int],
    None,
]

WorkspaceFactory = Callable[
    [str],
    GitHubRepositoryWorkspace,
]


class RepositoryAnalysisServiceError(RuntimeError):
    """项目仓库分析服务执行失败。"""


@dataclass(frozen=True)
class RepositoryAnalysisRun:
    """保存一次完整仓库分析的结果和运行指标。"""

    repository_url: str
    commit_sha: str
    default_branch: str
    safe_file_count: int
    analysis: RepositoryAnalysisResult
    usage: TokenUsage
    model_turn_count: int
    tool_call_count: int
    completion_reason: str = "model_stop"
    result_status: str = "complete"
    trace: tuple = ()


class RepositoryAnalysisService:
    """管理临时仓库并执行完整的 Agent 代码分析流程。"""

    def __init__(
        self,
        agent: RepositoryAnalysisAgent,
        parser: Optional[
            RepositoryAnalysisResultParser
        ] = None,
        workspace_factory: Optional[
            WorkspaceFactory
        ] = None,
    ) -> None:
        """保存 Agent、结果解析器和可替换的 Workspace 工厂。"""

        self._agent = agent
        self._parser = (
            parser
            or RepositoryAnalysisResultParser()
        )
        self._workspace_factory = (
            workspace_factory
            or GitHubRepositoryWorkspace
        )

    async def analyze(
        self,
        repository_url: str,
        progress_callback: Optional[
            ProgressCallback
        ] = None,
        runtime_callback: Optional[
            AgentProgressCallback
        ] = None,
    ) -> RepositoryAnalysisRun:
        """打开临时仓库、运行 Agent、验证结果并自动清理代码。"""

        self._notify_progress(
            progress_callback,
            "cloning",
            5,
        )

        async with self._workspace_factory(
            repository_url
        ) as workspace:
            self._notify_progress(
                progress_callback,
                "inventory",
                20,
            )

            safe_file_count = len(
                workspace.all_paths
            )

            if safe_file_count == 0:
                raise RepositoryAnalysisServiceError(
                    "Repository contains no safe files "
                    "that can be analyzed."
                )

            self._notify_progress(
                progress_callback,
                "analyzing",
                30,
            )

            agent_result = await self._agent.analyze(
                workspace,
                progress_callback=runtime_callback,
            )

            self._notify_progress(
                progress_callback,
                "validating",
                90,
            )

            analysis_result = (
                await self._parser.parse(
                    raw_content=agent_result.content,
                    workspace=workspace,
                )
            )

            run = RepositoryAnalysisRun(
                repository_url=(
                    workspace.repository_url
                ),
                commit_sha=workspace.commit_sha,
                default_branch=(
                    workspace.default_branch
                ),
                safe_file_count=safe_file_count,
                analysis=analysis_result,
                usage=agent_result.usage,
                model_turn_count=len(
                    agent_result.turns
                ),
                tool_call_count=(
                    agent_result.tool_call_count
                ),
                completion_reason=(
                    agent_result.completion_reason
                ),
                result_status=(
                    "partial"
                    if agent_result.completion_reason
                    in {
                        "emergency_finalize",
                        "partial_terminal_tool",
                    }
                    else "complete"
                ),
                trace=tuple([
                    event
                    for event in agent_result.trace
                    if event.get("event") == "model_turn"
                ][-MAX_PERSISTED_MODEL_TURNS:]),
            )

        self._notify_progress(
            progress_callback,
            "validating",
            100,
        )

        return run

    @staticmethod
    def _notify_progress(
        callback: Optional[
            ProgressCallback
        ],
        stage: str,
        progress: int,
    ) -> None:
        """存在进度回调时同步报告当前阶段和完成比例。"""

        if callback is not None:
            callback(
                stage,
                progress,
            )
