from typing import Any, Dict

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)
