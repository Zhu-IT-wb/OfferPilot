import pytest
from fastapi.testclient import TestClient

from app.api.routes import debug
from app.main import app
from app.services import llm_service


def test_debug_llm_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLMService:
        async def generate_text(self, **kwargs):
            raise llm_service.LLMConfigurationError(
                "LLM API key is not configured. Set DEEPSEEK_API_KEY or OFFERPILOT_LLM_API_KEY."
            )

    monkeypatch.setattr(debug, "LLMService", FakeLLMService)
    client = TestClient(app)

    response = client.post("/api/debug/llm", json={"prompt": "hello"})

    assert response.status_code == 503
    assert "API key" in response.json()["detail"]


def test_debug_llm_returns_model_content(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_generate_text(
        self,
        prompt,
        system_prompt=None,
        model=None,
        temperature=0.3,
        max_tokens=512,
    ):
        return llm_service.LLMResult(
            provider="deepseek",
            model=model or "deepseek-v4-flash",
            content=f"mock reply: {prompt}",
            raw_response={},
        )

    monkeypatch.setattr(llm_service.LLMService, "generate_text", fake_generate_text)
    client = TestClient(app)

    response = client.post("/api/debug/llm", json={"prompt": "今日任务是什么？"})

    assert response.status_code == 200
    assert response.json() == {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "content": "mock reply: 今日任务是什么？",
    }
