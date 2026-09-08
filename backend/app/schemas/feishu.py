from typing import Optional

from pydantic import BaseModel

from app.schemas.agent import AgentRunResponse


# 定义接口响应体的数据结构。
class FeishuEventProcessResponse(BaseModel):
    handled: bool
    event_type: Optional[str] = None
    message: Optional[str] = None
    user_id: Optional[str] = None
    agent_response: Optional[AgentRunResponse] = None
    reply_sent: Optional[bool] = None
    reply_error: Optional[str] = None
    reply_message_id: Optional[str] = None
