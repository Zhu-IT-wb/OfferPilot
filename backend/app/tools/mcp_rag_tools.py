from typing import Any, Dict, Optional, Protocol

from app.core.config import settings
from app.mcp.client import get_default_mcp_client
from app.rag.answering import GroundedAnswer, GroundedAnswerComposer
from app.schemas.agent import AgentActionName
from app.schemas.tool import ToolResult
from app.tools.registry import ToolRegistry


class MCPToolCaller(Protocol):
    async def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...


class CareerKnowledgeToolAdapter:
    def __init__(
        self,
        client: Optional[MCPToolCaller] = None,
        answer_composer: Optional[GroundedAnswerComposer] = None,
    ) -> None:
        self.client = client or get_default_mcp_client()
        self.answer_composer = answer_composer or GroundedAnswerComposer()

    async def search_knowledge(self, arguments: Dict[str, Any]) -> ToolResult:
        return await self._search(
            action=AgentActionName.SEARCH_CAREER_KNOWLEDGE,
            mcp_tool="search_knowledge",
            arguments=arguments,
        )

    async def search_project_evidence(self, arguments: Dict[str, Any]) -> ToolResult:
        return await self._search(
            action=AgentActionName.SEARCH_PROJECT_EVIDENCE,
            mcp_tool="search_project_evidence",
            arguments=arguments,
        )

    async def _search(
        self,
        action: AgentActionName,
        mcp_tool: str,
        arguments: Dict[str, Any],
    ) -> ToolResult:
        owner_id = str(arguments.get("owner_id") or "").strip()
        query = str(arguments.get("query") or arguments.get("raw_message") or "").strip()
        if not owner_id:
            return ToolResult(
                tool_name=action.value,
                success=False,
                message="无法确认当前用户身份，已拒绝检索私人项目证据。",
            )
        if not query:
            return ToolResult(
                tool_name=action.value,
                success=False,
                message="请说明你想检索或了解的问题。",
                data={"missing_slots": ["query"]},
            )
        request: Dict[str, Any] = {
            "query": query,
            "owner_id": owner_id,
            "top_k": max(1, min(int(arguments.get("top_k") or settings.rag_top_k), 10)),
        }
        for key in ("project_id", "project_version", "domains"):
            value = arguments.get(key)
            if value not in (None, "", []):
                request[key] = value
        result = await self.client.call_tool(mcp_tool, request)
        hits = [item for item in result.get("hits", []) if isinstance(item, dict)]
        answer_kind = (
            "project"
            if action == AgentActionName.SEARCH_PROJECT_EVIDENCE
            else "knowledge"
        )
        answer = await self.answer_composer.compose(
            query=query,
            hits=hits,
            answer_kind=answer_kind,
        )
        message = (
            _render_grounded_answer(answer, hits)
            if answer
            else _format_evidence_results(query, hits)
        )
        return ToolResult(
            tool_name=action.value,
            success=True,
            message=message,
            data={
                "query": query,
                "hits": hits,
                "evidence_ids": [item.get("evidence_id") for item in hits],
                "citations": _citation_metadata(hits),
                "grounded": True,
                "synthesized": answer is not None,
            },
        )


def register_mcp_rag_tools(
    registry: ToolRegistry,
    adapter: Optional[CareerKnowledgeToolAdapter] = None,
) -> None:
    if not settings.rag_enabled:
        return
    selected = adapter or CareerKnowledgeToolAdapter()
    registry.register(
        AgentActionName.SEARCH_CAREER_KNOWLEDGE.value,
        selected.search_knowledge,
        description=(
            "检索八股面试知识和当前用户的项目证据，并基于证据回答技术、项目或面试问题。"
            "这是只读工具；回答会返回可追溯 evidence_id。"
        ),
        mutating=False,
        optional_slots=["query", "domains", "project_id", "top_k"],
        examples=["解释一下 Redis 缓存穿透", "我的项目里为什么选择 LangGraph"],
    )
    registry.register(
        AgentActionName.SEARCH_PROJECT_EVIDENCE.value,
        selected.search_project_evidence,
        description=(
            "只检索当前用户的项目画像与源码分析证据，用于项目介绍、技术决策、难点和简历追问。"
        ),
        mutating=False,
        optional_slots=["query", "project_id", "project_version", "top_k"],
        examples=["Maggie 的会话持久化是怎么实现的", "这个项目有哪些可量化的技术亮点"],
    )


def _format_evidence_results(query: str, hits: list[Dict[str, Any]]) -> str:
    if not hits:
        return f"没有找到与“{query}”直接相关的可靠证据。你可以补充项目名称或更具体的技术关键词。"
    lines = ["检索到以下可追溯证据："]
    for index, hit in enumerate(hits, start=1):
        title = str(hit.get("title") or "未命名证据")
        excerpt = str(hit.get("text") or "").replace("\n", " ").strip()
        if len(excerpt) > 280:
            excerpt = excerpt[:277] + "..."
        source = str(hit.get("source_path") or hit.get("source_url") or "")
        line_text = ""
        if hit.get("start_line") is not None:
            line_text = f":{hit['start_line']}"
            if hit.get("end_line") is not None:
                line_text += f"-{hit['end_line']}"
        lines.append(f"{index}. {title}")
        if excerpt:
            lines.append(f"   {excerpt}")
        if source:
            lines.append(f"   来源：{source}{line_text}")
    return "\n".join(lines)


def _render_grounded_answer(
    answer: GroundedAnswer,
    hits: list[Dict[str, Any]],
) -> str:
    cited_indexes = set(answer.citation_indexes)
    if not cited_indexes:
        return answer.text
    sources = ["参考依据："]
    for index in sorted(cited_indexes):
        hit = hits[index - 1]
        title = str(hit.get("title") or "未命名证据")
        source = str(hit.get("source_path") or hit.get("source_url") or "")
        location = ""
        if source:
            location = f"（{source}"
            if hit.get("start_line") is not None:
                location += f":{hit['start_line']}"
                if hit.get("end_line") is not None:
                    location += f"-{hit['end_line']}"
            location += "）"
        sources.append(f"[{index}] {title}{location}")
    return f"{answer.text}\n\n" + "\n".join(sources)


def _citation_metadata(hits: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return [
        {
            "index": index,
            "evidence_id": hit.get("evidence_id"),
            "title": hit.get("title"),
            "source_path": hit.get("source_path"),
            "source_url": hit.get("source_url"),
            "start_line": hit.get("start_line"),
            "end_line": hit.get("end_line"),
        }
        for index, hit in enumerate(hits, start=1)
    ]
