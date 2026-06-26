import asyncio

import pytest

from app.services.feishu_service import (
    FeishuConfigurationError,
    FeishuMessageService,
    FeishuRequestError,
)


def test_feishu_service_requires_credentials() -> None:
    service = FeishuMessageService(app_id="", app_secret="")

    with pytest.raises(FeishuConfigurationError):
        asyncio.run(service.get_tenant_access_token())


def test_feishu_service_gets_and_caches_tenant_access_token(monkeypatch) -> None:
    calls = []
    service = FeishuMessageService(app_id="app_id", app_secret="app_secret")

    async def fake_post_json(path, payload, headers=None, params=None):
        calls.append(
            {
                "path": path,
                "payload": payload,
                "headers": headers,
                "params": params,
            }
        )
        return {
            "code": 0,
            "tenant_access_token": "tenant_token",
            "expire": 7200,
        }

    monkeypatch.setattr(service, "_post_json", fake_post_json)

    first_token = asyncio.run(service.get_tenant_access_token())
    second_token = asyncio.run(service.get_tenant_access_token())

    assert first_token == "tenant_token"
    assert second_token == "tenant_token"
    assert calls == [
        {
            "path": "/auth/v3/tenant_access_token/internal",
            "payload": {"app_id": "app_id", "app_secret": "app_secret"},
            "headers": None,
            "params": None,
        }
    ]


def test_feishu_service_sends_text_message(monkeypatch) -> None:
    calls = []
    service = FeishuMessageService(app_id="app_id", app_secret="app_secret")

    async def fake_post_json(path, payload, headers=None, params=None):
        calls.append(
            {
                "path": path,
                "payload": payload,
                "headers": headers,
                "params": params,
            }
        )
        if path == "/auth/v3/tenant_access_token/internal":
            return {
                "code": 0,
                "tenant_access_token": "tenant_token",
                "expire": 7200,
            }
        return {
            "code": 0,
            "data": {"message_id": "om_test"},
        }

    monkeypatch.setattr(service, "_post_json", fake_post_json)

    result = asyncio.run(
        service.send_text_message(
            receive_id="ou_test",
            text="今天的任务：1. LeetCode 206. 反转链表",
        )
    )

    assert result.message_id == "om_test"
    assert calls[1] == {
        "path": "/im/v1/messages",
        "payload": {
            "receive_id": "ou_test",
            "msg_type": "text",
            "content": '{"text": "今天的任务：1. LeetCode 206. 反转链表"}',
        },
        "headers": {"Authorization": "Bearer tenant_token"},
        "params": {"receive_id_type": "open_id"},
    }


def test_feishu_service_raises_for_feishu_error_code() -> None:
    with pytest.raises(FeishuRequestError):
        FeishuMessageService._raise_for_feishu_code(
            {
                "code": 999,
                "msg": "bad request",
            }
        )
