from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import settings


# 表示当前模块抛出的业务异常。
class LLMConfigurationError(RuntimeError):
    """Raised when the LLM provider is not configured correctly."""


# 表示当前模块抛出的业务异常。
class LLMRequestError(RuntimeError):
    """Raised when the LLM provider request fails."""


# 承载外部服务调用后的结构化结果。
@dataclass(frozen=True)
class LLMResult:
    provider: str
    model: str
    content: str
    raw_response: Dict[str, Any]


# 定义 LLMService 相关的数据结构或领域对象。
class LLMService:
    # 初始化当前组件所需的依赖和配置。
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.provider = provider or settings.llm_provider
        self.default_model = default_model or settings.llm_model
        self.timeout_seconds = timeout_seconds or settings.llm_timeout_seconds

    # 处理 generate_text 相关逻辑。
    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> LLMResult:
        if not self.api_key:
            raise LLMConfigurationError(
                "LLM API key is not configured. Set DEEPSEEK_API_KEY or OFFERPILOT_LLM_API_KEY."
            )

        selected_model = model or self.default_model
        messages = self._build_messages(prompt=prompt, system_prompt=system_prompt)
        payload: Dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        try:
            response_data = await self._post_chat_completions(payload)
        except httpx.HTTPStatusError as exc:
            raise LLMRequestError(
                f"LLM provider returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMRequestError(f"LLM provider request failed: {exc}") from exc

        content = self._extract_content(response_data)
        return LLMResult(
            provider=self.provider,
            model=selected_model,
            content=content,
            raw_response=response_data,
        )

    # 发送 POST 请求处理 chat completions。
    async def _post_chat_completions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    # 构造 messages。
    @staticmethod
    def _build_messages(prompt: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    # 从输入数据中提取 content。
    @staticmethod
    def _extract_content(response_data: Dict[str, Any]) -> str:
        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError("LLM provider response does not contain message content.") from exc

        if not isinstance(content, str):
            raise LLMRequestError("LLM provider message content is not a string.")

        return content
