from typing import Optional

from pydantic import BaseModel, Field


# 定义接口请求体的数据结构。
class DebugLLMRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    system_prompt: Optional[str] = Field(
        default="You are OfferPilot, a concise job-search preparation assistant.",
        max_length=2000,
    )
    model: Optional[str] = Field(default=None, max_length=128)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)


# 定义接口响应体的数据结构。
class DebugLLMResponse(BaseModel):
    provider: str
    model: str
    content: str
