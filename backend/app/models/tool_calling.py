from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ModelOptions:
    model: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.0
    thinking: Optional[Dict[str, str]] = None
    response_format: Optional[Dict[str, str]] = None
    tool_choice: Optional[Any] = None


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass(frozen=True)
class ModelToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class ModelTurn:
    assistant_message: Dict[str, Any]
    tool_calls: List[ModelToolCall]
    content: str
    reasoning_content: Optional[str]
    finish_reason: str
    usage: TokenUsage
    provider: str
    model: str
