from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# 描述一个 Agent 可以调用的工具及其约束。
class ToolSpec(BaseModel):
    name: str
    description: str = ""
    mutating: bool = False
    effect: Optional[str] = None
    approval: Optional[str] = None
    required_slots: List[str] = Field(default_factory=list)
    optional_slots: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)


# 承载外部服务调用后的结构化结果。
class ToolResult(BaseModel):
    tool_name: str
    success: bool
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)
