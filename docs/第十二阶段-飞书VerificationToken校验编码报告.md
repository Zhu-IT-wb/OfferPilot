# 第十二阶段编码报告：飞书 Verification Token 校验

日期：2026-06-25  
项目：OfferPilot  
任务：为 `/api/feishu/events` 增加 verification token 校验  

## 1. 背景

第十一阶段已经完成飞书事件回调草案：

```text
POST /api/feishu/events
```

支持：

- URL 验证 challenge；
- 文本消息提取；
- 调用 `AgentOrchestrator`。

但当时没有任何校验，请求可以被伪造。本阶段目标是补上飞书事件订阅中的 verification token 校验。

## 2. 本次实现内容

新增或修改文件：

| 文件 | 作用 |
| --- | --- |
| `backend/app/core/config.py` | 增加 `feishu_verification_token` 配置 |
| `backend/app/api/routes/feishu.py` | 回调入口增加 token 校验 |
| `backend/.env.example` | 增加 `FEISHU_VERIFICATION_TOKEN` 示例 |
| `backend/tests/test_feishu_events.py` | 增加 token 校验测试 |
| `backend/README.md` | 更新飞书事件说明 |

## 3. 配置

`.env` 中增加：

```env
FEISHU_VERIFICATION_TOKEN=your_verification_token
```

行为：

- 未配置时：允许本地调试；
- 已配置时：请求必须携带匹配 token；
- token 错误或缺失：返回 `403 Forbidden`。

## 4. 当前支持的 token 位置

为了兼容飞书不同事件格式，当前会从以下位置提取 token：

```text
payload["token"]
payload["header"]["token"]
payload["event"]["token"]
```

只要其中一个位置的 token 与 `FEISHU_VERIFICATION_TOKEN` 匹配，即视为通过。

## 5. 接口行为

### 5.1 challenge 校验通过

请求：

```json
{
  "type": "url_verification",
  "token": "verify_token",
  "challenge": "challenge_token"
}
```

响应：

```json
{
  "challenge": "challenge_token"
}
```

### 5.2 文本消息校验通过

请求：

```json
{
  "schema": "2.0",
  "header": {
    "event_type": "im.message.receive_v1",
    "token": "verify_token"
  },
  "event": {
    "message": {
      "message_type": "text",
      "content": {
        "text": "今天任务是什么？"
      }
    }
  }
}
```

响应中会包含：

```json
{
  "handled": true,
  "message": "今天任务是什么？",
  "agent_response": {
    "action": "list_today_tasks"
  }
}
```

### 5.3 token 错误

响应：

```http
403 Forbidden
```

```json
{
  "detail": "Invalid Feishu verification token."
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
43 passed
```

### 6.2 单独测试飞书事件

```bash
.venv/bin/python -m pytest tests/test_feishu_events.py -q
```

覆盖：

- 未配置 token 时允许本地调试；
- top-level token 可以通过；
- header token 可以通过；
- token 错误返回 `403`；
- token 缺失返回 `403`。

### 6.3 手动测试

配置 `.env`：

```env
FEISHU_VERIFICATION_TOKEN=verify_token
```

启动服务：

```bash
.venv/bin/python -m uvicorn app.main:app --reload
```

正确 token：

```bash
curl -X POST http://127.0.0.1:8000/api/feishu/events \
  -H "Content-Type: application/json" \
  -d '{"type":"url_verification","token":"verify_token","challenge":"challenge_token"}'
```

预期：

```json
{"challenge":"challenge_token"}
```

错误 token：

```bash
curl -X POST http://127.0.0.1:8000/api/feishu/events \
  -H "Content-Type: application/json" \
  -d '{"type":"url_verification","token":"wrong_token","challenge":"challenge_token"}'
```

预期：

```http
403 Forbidden
```

## 7. 本次测试结果

运行命令：

```bash
cd /Users/will/Developer/OfferPilot/backend
.venv/bin/python -m pytest
```

测试结果：

```text
collected 43 items
tests/test_agent_message_route.py ...        [  6%]
tests/test_agent_orchestrator.py ......      [ 20%]
tests/test_config.py ...                     [ 27%]
tests/test_debug_llm.py ..                   [ 32%]
tests/test_debug_routes.py ..                [ 37%]
tests/test_feishu_events.py ........         [ 55%]
tests/test_health.py .                       [ 58%]
tests/test_intent_classifier.py .....         [ 69%]
tests/test_offerpilot_repository.py ....      [ 79%]
tests/test_offerpilot_tools.py ....           [ 88%]
tests/test_sqlite_offerpilot_repository.py ..... [100%]

43 passed in 2.43s
```

## 8. 后续开发建议

下一阶段建议继续飞书主线：

1. 增加飞书应用凭证配置：`FEISHU_APP_ID`、`FEISHU_APP_SECRET`；
2. 实现 tenant access token 获取；
3. 封装 `FeishuMessageService`；
4. 在事件回调中调用飞书 API 主动回复用户。

这样 OfferPilot 就能从“收到飞书消息并处理”推进到“处理后真正回复飞书用户”。
