from pydantic import BaseModel


# 定义接口响应体的数据结构。
class HealthResponse(BaseModel):
    status: str

