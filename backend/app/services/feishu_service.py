import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from app.core.config import settings


class FeishuConfigurationError(RuntimeError):
    """Raised when Feishu app credentials are not configured."""


class FeishuRequestError(RuntimeError):
    """Raised when Feishu OpenAPI returns or causes an error."""


@dataclass(frozen=True)
class FeishuMessageResult:
    message_id: Optional[str]
    raw_response: Dict[str, Any]


class FeishuMessageService:
    def __init__(
        self,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.app_id = app_id if app_id is not None else settings.feishu_app_id
        self.app_secret = app_secret if app_secret is not None else settings.feishu_app_secret
        self.base_url = (base_url or settings.feishu_api_base_url).rstrip("/")
        self.timeout_seconds = timeout_seconds or settings.feishu_timeout_seconds
        self._tenant_access_token: Optional[str] = None
        self._tenant_access_token_expires_at = 0.0

    def is_configured(self) -> bool:
        return bool(self.app_id and self.app_secret)

    async def send_text_message(
        self,
        receive_id: str,
        text: str,
        receive_id_type: str = "open_id",
    ) -> FeishuMessageResult:
        token = await self.get_tenant_access_token()
        payload = {
            "receive_id": receive_id,
            "msg_type": "text",
            "content": json.dumps({"text": text}, ensure_ascii=False),
        }
        response_data = await self._post_json(
            path="/im/v1/messages",
            payload=payload,
            headers={"Authorization": f"Bearer {token}"},
            params={"receive_id_type": receive_id_type},
        )
        self._raise_for_feishu_code(response_data)

        data = response_data.get("data")
        message_id = data.get("message_id") if isinstance(data, dict) else None
        return FeishuMessageResult(
            message_id=message_id,
            raw_response=response_data,
        )

    async def get_tenant_access_token(self) -> str:
        if not self.is_configured():
            raise FeishuConfigurationError(
                "Feishu app credentials are not configured. Set FEISHU_APP_ID and FEISHU_APP_SECRET."
            )

        now = time.time()
        if self._tenant_access_token and now < self._tenant_access_token_expires_at:
            return self._tenant_access_token

        response_data = await self._post_json(
            path="/auth/v3/tenant_access_token/internal",
            payload={
                "app_id": self.app_id,
                "app_secret": self.app_secret,
            },
        )
        self._raise_for_feishu_code(response_data)

        token = response_data.get("tenant_access_token")
        if not isinstance(token, str) or not token:
            raise FeishuRequestError("Feishu token response does not contain tenant_access_token.")

        expire_seconds = response_data.get("expire", 7200)
        if not isinstance(expire_seconds, int):
            expire_seconds = 7200

        self._tenant_access_token = token
        self._tenant_access_token_expires_at = now + max(expire_seconds - 60, 60)
        return token

    async def _post_json(
        self,
        path: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers=request_headers,
                    params=params,
                )
                response.raise_for_status()
                response_data = response.json()
        except httpx.HTTPStatusError as exc:
            raise FeishuRequestError(
                f"Feishu OpenAPI returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise FeishuRequestError(f"Feishu OpenAPI request failed: {exc}") from exc
        except ValueError as exc:
            raise FeishuRequestError("Feishu OpenAPI response is not valid JSON.") from exc

        if not isinstance(response_data, dict):
            raise FeishuRequestError("Feishu OpenAPI response must be a JSON object.")

        return response_data

    @staticmethod
    def _raise_for_feishu_code(response_data: Dict[str, Any]) -> None:
        code = response_data.get("code", 0)
        if code not in (0, None):
            message = response_data.get("msg") or response_data.get("message") or "Unknown Feishu OpenAPI error."
            raise FeishuRequestError(f"Feishu OpenAPI returned code {code}: {message}")
