from fastapi import APIRouter

from app.schemas.health import HealthResponse

router = APIRouter()


# 返回服务健康检查结果。
@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")

