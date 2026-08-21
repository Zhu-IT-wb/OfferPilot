import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.core.config import Settings, settings


class MCPClientError(RuntimeError):
    pass


@dataclass
class _Request:
    operation: str
    future: asyncio.Future
    name: str = ""
    arguments: Optional[Dict[str, Any]] = None


class PersistentMCPClient:
    def __init__(self, app_settings: Settings = settings) -> None:
        self.settings = app_settings
        self._queue: Optional[asyncio.Queue[_Request]] = None
        self._task: Optional[asyncio.Task] = None
        self._start_lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._start_lock:
            if self._task is not None and not self._task.done():
                return
            loop = asyncio.get_running_loop()
            ready = loop.create_future()
            self._queue = asyncio.Queue()
            self._task = asyncio.create_task(
                self._run(ready),
                name="offerpilot-career-knowledge-mcp",
            )
            await ready

    async def list_tools(self) -> list[dict]:
        result = await self._request("list_tools")
        return result

    async def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result = await self._request("call_tool", name=name, arguments=arguments or {})
        if not isinstance(result, dict):
            raise MCPClientError(f"MCP tool {name} returned a non-object result")
        return result

    async def close(self) -> None:
        task = self._task
        queue = self._queue
        if task is None or queue is None:
            return
        if task.done():
            self._task = None
            self._queue = None
            return
        future = asyncio.get_running_loop().create_future()
        await queue.put(_Request(operation="close", future=future))
        await future
        await task
        self._task = None
        self._queue = None

    async def _request(
        self,
        operation: str,
        name: str = "",
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        await self.start()
        if self._queue is None:
            raise MCPClientError("MCP request queue is unavailable")
        future = asyncio.get_running_loop().create_future()
        await self._queue.put(
            _Request(
                operation=operation,
                future=future,
                name=name,
                arguments=arguments,
            )
        )
        return await future

    async def _run(self, ready: asyncio.Future) -> None:
        current: Optional[_Request] = None
        try:
            parameters = StdioServerParameters(
                command=sys.executable,
                args=["-m", "app.mcp.career_knowledge_server"],
                cwd=str(Path(__file__).resolve().parents[2]),
                env=self._server_environment(),
                encoding="utf-8",
                encoding_error_handler="replace",
            )
            async with stdio_client(parameters) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    if not ready.done():
                        ready.set_result(None)
                    assert self._queue is not None
                    while True:
                        current = await self._queue.get()
                        if current.operation == "close":
                            current.future.set_result(None)
                            current = None
                            break
                        try:
                            if current.operation == "list_tools":
                                response = await session.list_tools()
                                value = [
                                    {
                                        "name": tool.name,
                                        "description": tool.description or "",
                                        "input_schema": tool.inputSchema,
                                    }
                                    for tool in response.tools
                                ]
                            elif current.operation == "call_tool":
                                response = await session.call_tool(
                                    current.name,
                                    arguments=current.arguments or {},
                                )
                                if response.isError:
                                    raise MCPClientError(
                                        _tool_error_message(current.name, response.content)
                                    )
                                value = _structured_result(response)
                            else:
                                raise MCPClientError(
                                    f"Unsupported MCP operation: {current.operation}"
                                )
                            current.future.set_result(value)
                        except Exception as exc:
                            if not current.future.done():
                                current.future.set_exception(exc)
                        finally:
                            current = None
        except Exception as exc:
            wrapped = exc if isinstance(exc, MCPClientError) else MCPClientError(str(exc))
            if not ready.done():
                ready.set_exception(wrapped)
            if current is not None and not current.future.done():
                current.future.set_exception(wrapped)
            self._fail_queued_requests(wrapped)
        finally:
            if not ready.done():
                ready.set_exception(MCPClientError("MCP client stopped before initialization"))

    def _server_environment(self) -> Dict[str, str]:
        return {
            "PYTHONUTF8": "1",
            "OFFERPILOT_STORAGE_BACKEND": self.settings.storage_backend,
            "OFFERPILOT_SQLITE_PATH": self.settings.sqlite_path,
            "OFFERPILOT_KNOWLEDGE_SOURCE_PATH": self.settings.knowledge_source_path,
            "OFFERPILOT_KNOWLEDGE_MARKDOWN_SYNC_ENABLED": "false",
            "OFFERPILOT_RAG_QDRANT_PATH": self.settings.rag_qdrant_path,
            "OFFERPILOT_RAG_COLLECTION_NAME": self.settings.rag_collection_name,
            "OFFERPILOT_RAG_DENSE_MODEL": self.settings.rag_dense_model,
            "OFFERPILOT_RAG_SPARSE_MODEL": self.settings.rag_sparse_model,
            "OFFERPILOT_RAG_FASTEMBED_CACHE_PATH": self.settings.rag_fastembed_cache_path,
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        }

    def _fail_queued_requests(self, exc: Exception) -> None:
        if self._queue is None:
            return
        while not self._queue.empty():
            request = self._queue.get_nowait()
            if not request.future.done():
                request.future.set_exception(exc)


def _structured_result(response: Any) -> Dict[str, Any]:
    if response.structuredContent is not None:
        return dict(response.structuredContent)
    for item in response.content:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return {"content": [getattr(item, "text", "") for item in response.content]}


def _tool_error_message(name: str, content: list[Any]) -> str:
    details = " ".join(
        str(getattr(item, "text", "")).strip()
        for item in content
        if getattr(item, "text", None)
    )
    return f"MCP tool {name} failed: {details or 'unknown error'}"


_default_client = PersistentMCPClient()


def get_default_mcp_client() -> PersistentMCPClient:
    return _default_client


async def close_default_mcp_client() -> None:
    await _default_client.close()
