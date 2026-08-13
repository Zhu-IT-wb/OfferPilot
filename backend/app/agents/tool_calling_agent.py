import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.models.tool_calling import (
    ModelOptions,
    ModelToolCall,
    ModelTurn,
    TokenUsage,
)
from app.services.tool_calling_model import ToolCallingModel
from app.tools.agent_tool import AgentToolResult
from app.tools.agent_tool_registry import AgentToolRegistry


class ToolCallingAgentError(RuntimeError):
    """Base error raised by the agent loop."""


class AgentProtocolError(ToolCallingAgentError):
    """The model returned an inconsistent response."""


class AgentLimitExceededError(ToolCallingAgentError):
    """The agent exceeded a configured safety limit."""


MAX_PERSISTED_MODEL_TURNS = 31


@dataclass(frozen=True)
class ToolCallingAgentLimits:
    """限制单次 Agent 任务的探索、上下文和应急收尾预算。"""

    max_model_turns: int = 30
    max_tool_calls: int = 80
    max_total_tokens: int = 1_500_000
    max_request_chars: int = 400_000
    soft_model_turns: int = 20
    soft_tool_calls: int = 50
    context_compact_chars: int = 240_000
    recent_tool_results: int = 6
    no_progress_limit: int = 3
    max_invalid_terminal_submissions: int = 2


@dataclass(frozen=True)
class ToolCallingAgentProgress:
    """描述 Agent 完成一轮模型请求后的可持久化运行指标。"""

    model_turn_count: int
    tool_call_count: int
    usage: TokenUsage
    finish_reason: str
    last_event: Optional[Dict[str, Any]] = None


AgentProgressCallback = Callable[[ToolCallingAgentProgress], None]


@dataclass(frozen=True)
class ToolCallingAgentResult:
    """保存 Agent 最终回答、轨迹和累计运行指标。"""

    content: str
    messages: List[Dict[str, Any]]
    turns: List[ModelTurn]
    usage: TokenUsage
    tool_call_count: int
    completion_reason: str = "model_stop"
    trace: List[Dict[str, Any]] = field(default_factory=list)


class ToolCallingAgent:
    """运行通用 Tool Calling 循环，并为 terminal 工具提供收敛控制。"""

    def __init__(
        self,
        model: ToolCallingModel,
        tool_registry: AgentToolRegistry,
        limits: Optional[ToolCallingAgentLimits] = None,
        terminal_tool_name: Optional[str] = None,
    ) -> None:
        self._model = model
        self._tool_registry = tool_registry
        self._limits = limits or ToolCallingAgentLimits()
        self._terminal_tool_name = terminal_tool_name

    async def run(
        self,
        messages: List[Dict[str, Any]],
        options: Optional[ModelOptions] = None,
        progress_callback: Optional[AgentProgressCallback] = None,
    ) -> ToolCallingAgentResult:
        """循环调用模型和工具，优先由模型通过 terminal 工具主动结束。"""

        if not messages:
            raise ValueError("Agent messages cannot be empty")

        model_options = options or ModelOptions()
        history = [dict(message) for message in messages]
        turns: List[ModelTurn] = []
        usage = TokenUsage(0, 0, 0)
        tool_call_count = 0
        trace: List[Dict[str, Any]] = []
        result_cache: Dict[str, AgentToolResult] = {}
        consecutive_no_progress = 0
        invalid_terminal_submissions = 0
        terminal_validation_errors: List[str] = []
        invalid_terminal_payloads = set()
        soft_warning_sent = False
        emergency_reason: Optional[str] = None
        tool_schemas = self._tool_registry.model_schemas()

        for _ in range(self._limits.max_model_turns):
            history, compacted = self._compact_history_if_needed(
                history,
                tool_schemas,
            )
            if compacted:
                trace.append(
                    {
                        "event": "context_compacted",
                        "turn": len(turns) + 1,
                        "request_chars": self._estimate_request_chars(
                            history,
                            tool_schemas,
                        ),
                    }
                )

            request_chars = self._estimate_request_chars(history, tool_schemas)
            if request_chars > self._limits.max_request_chars:
                if self._terminal_tool_name:
                    emergency_reason = "request_size_limit"
                    break
                raise self._limit_error(
                    "current request size budget",
                    request_chars,
                    self._limits.max_request_chars,
                    turns,
                    tool_call_count,
                )

            turn = await self._model.complete(
                messages=history,
                tools=tool_schemas,
                options=model_options,
            )
            turns.append(turn)
            usage = self._add_usage(usage, turn.usage)
            model_event = {
                "event": "model_turn",
                "turn": len(turns),
                "request_chars": request_chars,
                "finish_reason": turn.finish_reason,
                "requested_tools": len(turn.tool_calls),
                "total_tokens": usage.total_tokens,
                "decision": self._turn_decision(turn),
            }
            trace.append(model_event)
            self._notify_progress(
                progress_callback,
                ToolCallingAgentProgress(
                    len(turns),
                    tool_call_count,
                    usage,
                    turn.finish_reason,
                    model_event,
                ),
            )

            assistant_message = dict(turn.assistant_message)
            assistant_message.setdefault("role", "assistant")
            history.append(assistant_message)

            token_budget_exceeded = (
                usage.total_tokens > self._limits.max_total_tokens
            )
            terminal_submission_turn = self._is_terminal_submission_turn(
                turn
            )
            if token_budget_exceeded and not terminal_submission_turn:
                if self._terminal_tool_name:
                    # Discard the unexecuted assistant tool request so the
                    # emergency request does not contain orphan tool_use IDs.
                    history.pop()
                    emergency_reason = "cumulative_token_limit"
                    break
                raise self._limit_error(
                    "cumulative token budget",
                    usage.total_tokens,
                    self._limits.max_total_tokens,
                    turns,
                    tool_call_count,
                )

            if turn.tool_calls:
                if turn.finish_reason != "tool_calls":
                    raise AgentProtocolError(
                        "Model returned tool calls without finish_reason=tool_calls"
                    )
                tool_budget_exceeded = (
                    tool_call_count + len(turn.tool_calls)
                    > self._limits.max_tool_calls
                )
                if tool_budget_exceeded and not terminal_submission_turn:
                    if self._terminal_tool_name:
                        # The over-budget batch was not executed; keep the
                        # message protocol balanced for emergency finalization.
                        history.pop()
                        emergency_reason = "tool_call_limit"
                        break
                    raise AgentLimitExceededError(
                        "Agent exceeded the tool call limit."
                    )

                for tool_call in turn.tool_calls:
                    tool_call_count += 1
                    cache_key = self._cache_key(tool_call)
                    cache_hit = (
                        self._tool_registry.is_cacheable(tool_call.name)
                        and cache_key in result_cache
                    )
                    result = (
                        result_cache[cache_key]
                        if cache_hit
                        else await self._tool_registry.execute(tool_call)
                    )
                    if (
                        not cache_hit
                        and self._tool_registry.is_cacheable(tool_call.name)
                        and not result.is_error
                    ):
                        result_cache[cache_key] = result

                    tool_message = result.to_model_message(tool_call.id)
                    if cache_hit:
                        payload = json.loads(tool_message["content"])
                        payload["already_executed"] = True
                        tool_message["content"] = json.dumps(
                            payload,
                            ensure_ascii=False,
                        )
                    history.append(tool_message)

                    made_progress = not cache_hit and not result.is_error
                    event = {
                        "event": "tool_call",
                        "turn": len(turns),
                        "tool": tool_call.name,
                        "arguments": self._argument_summary(tool_call),
                        "success": not result.is_error,
                        "result_chars": len(tool_message["content"]),
                        "result_hash": hashlib.sha256(
                            tool_message["content"].encode("utf-8")
                        ).hexdigest()[:16],
                        "cache_hit": cache_hit,
                        "made_progress": made_progress,
                    }
                    if result.is_error:
                        event["error"] = self._terminal_error_message(result)
                    trace.append(event)
                    self._notify_progress(
                        progress_callback,
                        ToolCallingAgentProgress(
                            len(turns),
                            tool_call_count,
                            usage,
                            turn.finish_reason,
                            event,
                        ),
                    )

                    if (
                        tool_call.name == self._terminal_tool_name
                        and result.terminal_content is not None
                        and not result.is_error
                    ):
                        model_event["tools"] = self._tool_trace_summaries(
                            trace,
                            len(turns),
                        )
                        self._notify_progress(
                            progress_callback,
                            ToolCallingAgentProgress(
                                len(turns),
                                tool_call_count,
                                usage,
                                turn.finish_reason,
                                model_event,
                            ),
                        )
                        return self._result(
                            result.terminal_content,
                            history,
                            turns,
                            usage,
                            tool_call_count,
                            (
                                "emergency_finalize"
                                if emergency_reason
                                else "terminal_tool"
                            ),
                            trace,
                        )

                    if tool_call.name == self._terminal_tool_name:
                        if cache_key in invalid_terminal_payloads:
                            event["duplicate_submission"] = True
                            history.append(
                                self._meta_message(
                                    "你重复提交了完全相同且已被拒绝的结果。"
                                    "不要再次原样提交；请按工具错误重新读取对应"
                                    "文件范围并修正或删除该 Finding。"
                                )
                            )
                        else:
                            invalid_terminal_payloads.add(cache_key)
                            invalid_terminal_submissions += 1
                            terminal_validation_errors.append(
                                self._terminal_error_message(result)
                            )

                    consecutive_no_progress = (
                        0 if made_progress else consecutive_no_progress + 1
                    )

                model_event["tools"] = self._tool_trace_summaries(
                    trace,
                    len(turns),
                )
                self._notify_progress(
                    progress_callback,
                    ToolCallingAgentProgress(
                        len(turns),
                        tool_call_count,
                        usage,
                        turn.finish_reason,
                        model_event,
                    ),
                )

                if (
                    invalid_terminal_submissions
                    >= self._limits.max_invalid_terminal_submissions
                ):
                    raise AgentProtocolError(
                        self._terminal_submission_failure_message(
                            invalid_terminal_submissions,
                            terminal_validation_errors,
                        )
                    )

                # A terminal submission is allowed to finish the current turn
                # even when that response crosses a hard budget. If validation
                # failed, do not resume exploration beyond the hard limit.
                if token_budget_exceeded:
                    emergency_reason = "cumulative_token_limit"
                    break
                if tool_budget_exceeded:
                    emergency_reason = "tool_call_limit"
                    break

                if (
                    consecutive_no_progress
                    >= self._limits.no_progress_limit
                ):
                    history.append(
                        self._meta_message(
                            "最近三次操作没有产生新证据。请检查当前覆盖情况；"
                            "如果代表性证据已经充分，请调用 "
                            f"{self._terminal_tool_name or '完成工具'}；"
                            "否则只调查一个明确缺失项。"
                        )
                    )
                    consecutive_no_progress = 0
                    trace.append(
                        {
                            "event": "no_progress_nudge",
                            "turn": len(turns),
                        }
                    )

                if (
                    not soft_warning_sent
                    and (
                        len(turns) >= self._limits.soft_model_turns
                        or tool_call_count >= self._limits.soft_tool_calls
                        or usage.total_tokens
                        >= int(self._limits.max_total_tokens * 0.7)
                    )
                ):
                    history.append(
                        self._meta_message(
                            "探索已进入软预算区。请检查已覆盖事项，"
                            "只补充明确缺失的关键证据；如果证据已经充分，"
                            f"现在调用 {self._terminal_tool_name or '完成工具'}。"
                        )
                    )
                    soft_warning_sent = True
                    trace.append(
                        {"event": "soft_budget_nudge", "turn": len(turns)}
                    )
                continue

            if turn.finish_reason == "tool_calls":
                raise AgentProtocolError(
                    "Model returned finish_reason=tool_calls without any tool calls"
                )
            if turn.finish_reason == "stop":
                if not turn.content.strip():
                    raise AgentProtocolError(
                        "Model stopped without a final answer."
                    )
                if self._terminal_tool_name:
                    invalid_terminal_submissions += 1
                    terminal_validation_errors.append(
                        "Model stopped without calling the required "
                        f"{self._terminal_tool_name} tool."
                    )
                    if (
                        invalid_terminal_submissions
                        >= self._limits.max_invalid_terminal_submissions
                    ):
                        raise AgentProtocolError(
                            self._terminal_submission_failure_message(
                                invalid_terminal_submissions,
                                terminal_validation_errors,
                            )
                        )
                    history.append(
                        self._meta_message(
                            "不要用普通文本结束。请调用 "
                            f"{self._terminal_tool_name} 提交结构化结果。"
                        )
                    )
                    continue
                return self._result(
                    turn.content,
                    history,
                    turns,
                    usage,
                    tool_call_count,
                    "model_stop",
                    trace,
                )
            if turn.finish_reason == "length":
                raise AgentProtocolError(
                    "Model output was truncated by the token limit."
                )
            if turn.finish_reason == "content_filter":
                raise AgentProtocolError(
                    "Model output was blocked by the content filter."
                )
            if turn.finish_reason == "insufficient_system_resource":
                raise ToolCallingAgentError(
                    "Model generation stopped because the provider had "
                    "insufficient resources."
                )
            raise AgentProtocolError(
                f"Model returned an unsupported finish_reason: {turn.finish_reason}"
            )
        else:
            emergency_reason = "model_turn_limit"

        if self._terminal_tool_name:
            return await self._emergency_finalize(
                history=history,
                turns=turns,
                usage=usage,
                tool_call_count=tool_call_count,
                trace=trace,
                options=model_options,
                reason=emergency_reason or "safety_limit",
                progress_callback=progress_callback,
            )
        raise AgentLimitExceededError("Agent exceeded the model turn limit.")

    async def _emergency_finalize(
        self,
        history: List[Dict[str, Any]],
        turns: List[ModelTurn],
        usage: TokenUsage,
        tool_call_count: int,
        trace: List[Dict[str, Any]],
        options: ModelOptions,
        reason: str,
        progress_callback: Optional[AgentProgressCallback],
    ) -> ToolCallingAgentResult:
        """关闭探索工具并保留一次强制 terminal 提交机会。"""

        terminal_name = self._terminal_tool_name
        if terminal_name is None:
            raise AgentLimitExceededError("Agent exceeded a safety limit.")
        terminal_schema = self._tool_registry.model_schema(terminal_name)
        compacted_history = self._compact_tool_results(history, keep_recent=2)
        compacted_history.append(
            self._meta_message(
                "探索预算已经结束。不得继续探索。请立即根据现有证据调用 "
                f"{terminal_name}。无法确认的事项标记为 not_found 并写入 warnings。"
            )
        )
        emergency_options = replace(
            options,
            response_format=None,
            tool_choice={
                "type": "function",
                "function": {"name": terminal_name},
            },
        )
        turn = await self._model.complete(
            messages=compacted_history,
            tools=[terminal_schema],
            options=emergency_options,
        )
        turns.append(turn)
        usage = self._add_usage(usage, turn.usage)
        emergency_event = {
            "event": "model_turn",
            "turn": len(turns),
            "request_chars": self._estimate_request_chars(
                compacted_history,
                [terminal_schema],
            ),
            "finish_reason": turn.finish_reason,
            "requested_tools": len(turn.tool_calls),
            "total_tokens": usage.total_tokens,
            "decision": "emergency_finalize",
            "emergency_reason": reason,
        }
        trace.append(emergency_event)
        self._notify_progress(
            progress_callback,
            ToolCallingAgentProgress(
                len(turns),
                tool_call_count,
                usage,
                turn.finish_reason,
                emergency_event,
            ),
        )
        if len(turn.tool_calls) != 1 or turn.tool_calls[0].name != terminal_name:
            raise AgentLimitExceededError(
                "Agent reached the safety limit and failed to submit a result."
            )
        tool_call = turn.tool_calls[0]
        result = await self._tool_registry.execute(tool_call)
        tool_call_count += 1
        compacted_history.append(dict(turn.assistant_message))
        compacted_history.append(result.to_model_message(tool_call.id))
        if result.is_error or result.terminal_content is None:
            raise AgentLimitExceededError(
                "Agent reached the safety limit and submitted an invalid result: "
                + self._terminal_error_message(result)
            )
        emergency_event["tools"] = [
            {
                "name": terminal_name,
                "arguments": self._argument_summary(tool_call),
                "success": True,
                "cache_hit": False,
                "made_progress": True,
            }
        ]
        self._notify_progress(
            progress_callback,
            ToolCallingAgentProgress(
                len(turns),
                tool_call_count,
                usage,
                turn.finish_reason,
                emergency_event,
            ),
        )
        trace.append(
            {
                "event": "emergency_finalize",
                "reason": reason,
                "turn": len(turns),
                "tool": terminal_name,
                "success": True,
            }
        )
        return self._result(
            result.terminal_content,
            compacted_history,
            turns,
            usage,
            tool_call_count,
            "emergency_finalize",
            trace,
        )

    def _compact_history_if_needed(
        self,
        history: List[Dict[str, Any]],
        tool_schemas: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        if (
            self._estimate_request_chars(history, tool_schemas)
            <= self._limits.context_compact_chars
        ):
            return history, False
        compacted = self._compact_tool_results(
            history,
            keep_recent=self._limits.recent_tool_results,
        )
        return compacted, compacted != history

    @staticmethod
    def _compact_tool_results(
        history: List[Dict[str, Any]],
        keep_recent: int,
    ) -> List[Dict[str, Any]]:
        tool_indexes = [
            index
            for index, message in enumerate(history)
            if message.get("role") == "tool"
        ]
        replace_indexes = set(tool_indexes[:-keep_recent] if keep_recent else tool_indexes)
        compacted: List[Dict[str, Any]] = []
        for index, message in enumerate(history):
            cloned = dict(message)
            if index in replace_indexes:
                original = str(cloned.get("content") or "")
                if ToolCallingAgent._is_compacted_tool_result(original):
                    compacted.append(cloned)
                    continue
                cloned["content"] = json.dumps(
                    {
                        "success": True,
                        "data": {
                            "compacted": True,
                            "original_chars": len(original),
                            "content_hash": hashlib.sha256(
                                original.encode("utf-8")
                            ).hexdigest()[:16],
                            "summary": ToolCallingAgent._tool_result_summary(
                                original
                            ),
                            "instruction": (
                                "Re-run a focused read/search if exact source text "
                                "is needed for final evidence."
                            ),
                        },
                    },
                    ensure_ascii=False,
                )
            compacted.append(cloned)
        return compacted

    @staticmethod
    def _is_compacted_tool_result(content: str) -> bool:
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return False
        data = payload.get("data") if isinstance(payload, dict) else None
        return isinstance(data, dict) and data.get("compacted") is True

    @staticmethod
    def _tool_result_summary(content: str) -> Dict[str, Any]:
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return {"preview": content[:500]}
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            return {"preview": content[:500]}
        summary: Dict[str, Any] = {}
        for key in (
            "path",
            "start_line",
            "end_line",
            "total_lines",
            "has_more",
            "next_start_line",
            "query",
            "path_prefix",
            "returned_count",
            "total_safe_files",
            "argv",
            "exit_code",
        ):
            if key in data:
                summary[key] = data[key]
        if isinstance(data.get("paths"), list):
            summary["paths"] = data["paths"][:30]
        if isinstance(data.get("matches"), list):
            summary["matches"] = data["matches"][:10]
        if isinstance(data.get("stdout"), str):
            summary["stdout_preview"] = data["stdout"][:500]
        return summary or {"preview": content[:500]}

    @staticmethod
    def _cache_key(tool_call: ModelToolCall) -> str:
        return json.dumps(
            [tool_call.name, tool_call.arguments],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _is_terminal_submission_turn(self, turn: ModelTurn) -> bool:
        return (
            self._terminal_tool_name is not None
            and len(turn.tool_calls) == 1
            and turn.tool_calls[0].name == self._terminal_tool_name
        )

    def _turn_decision(self, turn: ModelTurn) -> str:
        if self._is_terminal_submission_turn(turn):
            return "validate_terminal_submission"
        if turn.tool_calls:
            return "execute_tools"
        if turn.finish_reason == "stop":
            return (
                "request_terminal_submission"
                if self._terminal_tool_name
                else "finish"
            )
        return "handle_provider_finish_reason"

    @staticmethod
    def _tool_trace_summaries(
        trace: List[Dict[str, Any]],
        turn_number: int,
    ) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        for event in trace:
            if (
                event.get("event") != "tool_call"
                or event.get("turn") != turn_number
            ):
                continue
            summary = {
                "name": event["tool"],
                "arguments": event["arguments"],
                "success": event["success"],
                "result_hash": event["result_hash"],
                "cache_hit": event["cache_hit"],
                "made_progress": event["made_progress"],
            }
            if "error" in event:
                summary["error"] = event["error"]
            if event.get("duplicate_submission"):
                summary["duplicate_submission"] = True
            summaries.append(summary)
        return summaries

    @staticmethod
    def _argument_summary(tool_call: ModelToolCall) -> str:
        return json.dumps(
            tool_call.arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )[:1000]

    @staticmethod
    def _meta_message(content: str) -> Dict[str, Any]:
        return {"role": "user", "content": content}

    @staticmethod
    def _terminal_error_message(result: AgentToolResult) -> str:
        error = result.data.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or "invalid terminal submission")
        return "invalid terminal submission"

    @staticmethod
    def _terminal_submission_failure_message(
        attempts: int,
        errors: List[str],
    ) -> str:
        """生成可直接持久化和展示的 terminal 校验失败说明。"""

        header = (
            "Model failed to produce a valid terminal submission "
            f"after {attempts} attempts."
        )
        if not errors:
            return header
        details = "\n".join(
            f"{index}. {message[:180]}"
            for index, message in enumerate(errors, start=1)
        )
        return f"{header}\nValidation errors:\n{details}"

    @staticmethod
    def _result(
        content: str,
        history: List[Dict[str, Any]],
        turns: List[ModelTurn],
        usage: TokenUsage,
        tool_call_count: int,
        completion_reason: str,
        trace: List[Dict[str, Any]],
    ) -> ToolCallingAgentResult:
        return ToolCallingAgentResult(
            content=content,
            messages=history,
            turns=turns,
            usage=usage,
            tool_call_count=tool_call_count,
            completion_reason=completion_reason,
            trace=trace,
        )

    @staticmethod
    def _limit_error(
        label: str,
        actual: int,
        maximum: int,
        turns: List[ModelTurn],
        tool_call_count: int,
    ) -> AgentLimitExceededError:
        return AgentLimitExceededError(
            f"Agent exceeded the {label}: {actual}/{maximum} "
            + ("cumulative tokens " if label == "cumulative token budget" else "")
            + f"after {len(turns)} model turns and "
            f"{tool_call_count} executed tool calls."
        )

    @staticmethod
    def _add_usage(current: TokenUsage, addition: TokenUsage) -> TokenUsage:
        return TokenUsage(
            current.prompt_tokens + addition.prompt_tokens,
            current.completion_tokens + addition.completion_tokens,
            current.total_tokens + addition.total_tokens,
        )

    @staticmethod
    def _notify_progress(
        callback: Optional[AgentProgressCallback],
        progress: ToolCallingAgentProgress,
    ) -> None:
        if callback is not None:
            callback(progress)

    @staticmethod
    def _estimate_request_chars(
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> int:
        return len(
            json.dumps(
                {"messages": messages, "tools": tools},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
