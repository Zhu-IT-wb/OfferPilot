# 第十三阶段编码报告：飞书 Tenant Access Token 与消息回复

日期：2026-06-25  
项目：OfferPilot  
任务：实现飞书 tenant access token 获取，并封装文本消息发送服务  

## 1. 背景

第十一、十二阶段已经让 OfferPilot 可以：

- 接收飞书事件回调；
- 校验 verification token；
- 提取文本消息；
- 调用 `AgentOrchestrator`。

但此前 Agent 的回复只体现在 HTTP 响应里，还不能真正发回飞书用户。本阶段目标是封装飞书消息发送能力：

```text
飞书事件
-> 提取用户文本
-> Agent 处理
-> 获取 tenant_access_token
-> 调用飞书发送消息 API
-> 回复飞书用户
```

## 2. 本次实现内容

新增或修改文件：

| 文件 | 作用 |
| --- | --- |
| `backend/app/services/feishu_service.py` | 新增飞书 tenant token 和文本消息发送服务 |
| `backend/app/core/config.py` | 增加飞书 App ID / Secret / API Base URL 配置 |
| `backend/app/api/routes/feishu.py` | 事件处理后尝试主动回复飞书用户 |
| `backend/app/schemas/feishu.py` | 响应中增加回复发送状态 |
| `backend/.env.example` | 增加飞书应用凭证配置示例 |
| `backend/tests/test_feishu_service.py` | 增加飞书服务测试 |
| `backend/tests/test_feishu_events.py` | 增加事件回调发送回复测试 |
| `backend/README.md` | 更新飞书事件说明 |

## 3. 新增配置

`.env` 中增加：

```env
FEISHU_APP_ID=your_app_id
FEISHU_APP_SECRET=your_app_secret
FEISHU_API_BASE_URL=https://open.feishu.cn/open-apis
FEISHU_TIMEOUT_SECONDS=15
```

说明：

- `FEISHU_APP_ID`：飞书应用的 App ID；
- `FEISHU_APP_SECRET`：飞书应用的 App Secret；
- `FEISHU_API_BASE_URL`：飞书 OpenAPI 基础地址；
- `FEISHU_TIMEOUT_SECONDS`：调用飞书 API 超时时间。

## 4. FeishuMessageService

新增服务：

```python
FeishuMessageService
```

核心方法：

```python
get_tenant_access_token()
send_text_message(receive_id, text, receive_id_type="open_id")
```

### 4.1 获取 tenant_access_token

调用飞书接口：

```text
POST /auth/v3/tenant_access_token/internal
```

请求体：

```json
{
  "app_id": "...",
  "app_secret": "..."
}
```

返回中读取：

```json
{
  "tenant_access_token": "...",
  "expire": 7200
}
```

当前服务会在实例内缓存 token，避免同一个实例重复获取。

### 4.2 发送文本消息

调用飞书接口：

```text
POST /im/v1/messages?receive_id_type=open_id
```

请求头：

```text
Authorization: Bearer <tenant_access_token>
```

请求体：

```json
{
  "receive_id": "ou_xxx",
  "msg_type": "text",
  "content": "{\"text\":\"今天的任务...\"}"
}
```

## 5. 事件回调行为

当 `/api/feishu/events` 收到文本消息时：

1. 校验 verification token；
2. 提取 `open_id`；
3. 提取文本；
4. 调用 `AgentOrchestrator`；
5. 如果配置了 `FEISHU_APP_ID` 和 `FEISHU_APP_SECRET`，调用飞书 API 发回回复；
6. 返回处理状态。

响应中新增字段：

| 字段 | 说明 |
| --- | --- |
| `reply_sent` | 是否成功发送飞书回复 |
| `reply_error` | 未发送或发送失败原因 |
| `reply_message_id` | 飞书返回的消息 ID |

本地未配置飞书应用凭证时，事件仍会正常处理，但返回：

```json
{
  "reply_sent": false,
  "reply_error": "Feishu app credentials are not configured; reply was not sent."
}
```

## 6. 测试方法

### 6.1 全量测试

```bash
cd /Users/will/Developer/OfferPilot/backend
.venv/bin/python -m pytest
```

预期：

```text
49 passed
```

### 6.2 单独测试飞书服务

```bash
.venv/bin/python -m pytest tests/test_feishu_service.py -q
```

覆盖：

- 未配置 App ID / Secret 时抛出配置错误；
- 可以获取并缓存 tenant access token；
- 发送文本消息时 payload 和 Authorization header 正确；
- 飞书 OpenAPI 错误码会转成 `FeishuRequestError`。

### 6.3 单独测试飞书事件

```bash
.venv/bin/python -m pytest tests/test_feishu_events.py -q
```

覆盖：

- 文本消息处理后会尝试发送飞书回复；
- 未配置凭证时不会发送，但会返回原因；
- 发送失败时会返回错误原因；
- token 校验仍然生效。

### 6.4 手动测试

`.env` 中配置：

```env
FEISHU_VERIFICATION_TOKEN=your_verification_token
FEISHU_APP_ID=your_app_id
FEISHU_APP_SECRET=your_app_secret
```

启动服务：

```bash
.venv/bin/python -m uvicorn app.main:app --reload
```

使用飞书事件回调或 curl 模拟文本消息：

```bash
curl -X POST http://127.0.0.1:8000/api/feishu/events \
  -H "Content-Type: application/json" \
  -d '{"schema":"2.0","header":{"event_type":"im.message.receive_v1","token":"your_verification_token"},"event":{"sender":{"sender_id":{"open_id":"ou_test"}},"message":{"message_type":"text","content":"{\"text\":\"今天任务是什么？\"}"}}}'
```

如果 `ou_test` 是真实可接收消息的 open_id，且应用权限正确，飞书会收到 Agent 回复。

## 7. 本次测试结果

运行命令：

```bash
cd /Users/will/Developer/OfferPilot/backend
.venv/bin/python -m pytest
```

测试结果：

```text
collected 49 items
tests/test_agent_message_route.py ...       [  6%]
tests/test_agent_orchestrator.py ......     [ 18%]
tests/test_config.py ...                    [ 24%]
tests/test_debug_llm.py ..                  [ 28%]
tests/test_debug_routes.py ..               [ 32%]
tests/test_feishu_events.py ..........      [ 53%]
tests/test_feishu_service.py ....           [ 61%]
tests/test_health.py .                      [ 63%]
tests/test_intent_classifier.py .....        [ 73%]
tests/test_offerpilot_repository.py ....     [ 81%]
tests/test_offerpilot_tools.py ....          [ 89%]
tests/test_sqlite_offerpilot_repository.py ..... [100%]

49 passed in 1.70s
```

## 8. 后续开发建议

下一阶段建议继续飞书真实接入：

1. 在飞书开放平台配置消息事件订阅；
2. 使用 ngrok 或 Cloudflare Tunnel 暴露本地 HTTPS 回调；
3. 配置必要权限并发布应用版本；
4. 用真实飞书消息触发 `/api/feishu/events`；
5. 根据真实事件 payload 修正字段兼容。

这样 OfferPilot 就能进入“飞书里发消息，Agent 真正回复”的联调阶段。
