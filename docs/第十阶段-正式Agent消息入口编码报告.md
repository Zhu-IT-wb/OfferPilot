# 第十阶段编码报告：正式 Agent 消息入口

日期：2026-06-25  
项目：OfferPilot  
任务：新增非 debug 的正式 Agent 消息入口 `/api/agent/message`  

## 1. 背景

前面几个阶段为了调试，提供了：

```text
/api/debug/llm
/api/debug/intent
/api/debug/agent
```

这些接口只用于本地/开发环境，不应该作为未来正式对外入口。根据需求文档中的用户消息处理流程：

```text
飞书事件回调
  -> 验签
  -> 提取用户消息
  -> 读取用户上下文
  -> 意图识别
  -> 选择工具
  -> 执行工具
  -> 模型生成回复
  -> 写入日志
  -> 飞书回复用户
```

当前还不直接接飞书，但需要先有一个正式的、渠道无关的 Agent 消息入口，未来飞书回调可以复用同一条内部链路。

## 2. 本次实现内容

新增或修改文件：

| 文件 | 作用 |
| --- | --- |
| `backend/app/api/routes/agent.py` | 新增正式 Agent 消息路由 |
| `backend/app/main.py` | 注册 Agent 路由 |
| `backend/app/schemas/agent.py` | 增加 `AgentMessageRequest` |
| `backend/tests/test_agent_message_route.py` | 增加正式消息入口测试 |
| `backend/README.md` | 补充接口说明 |

## 3. 接口设计

### 3.1 正式 Agent 消息入口

```http
POST /api/agent/message
Content-Type: application/json
```

请求体：

```json
{
  "message": "新增投递深信服开发实习，明天下午三点一面",
  "confirmed": true,
  "user_id": "local_user",
  "source": "api"
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `message` | 用户自然语言消息 |
| `confirmed` | 是否确认执行写入类动作 |
| `user_id` | 用户标识，当前预留 |
| `source` | 消息来源，当前预留，例如 `api` / `feishu` |

响应体沿用 `AgentResponse`：

```json
{
  "intent": "add_application",
  "confidence": 0.86,
  "action": "create_application",
  "reply": "已创建投递记录：深信服 - 开发实习。",
  "need_confirmation": false,
  "slots": {
    "company": "深信服",
    "role": "开发实习"
  },
  "missing_slots": [],
  "tool_result": {
    "tool_name": "create_application",
    "success": true,
    "message": "已创建投递记录：深信服 - 开发实习。",
    "data": {}
  }
}
```

## 4. 与 debug 接口的区别

| 接口 | 定位 | 生产环境 |
| --- | --- | --- |
| `/api/debug/llm` | 调试 LLM 调用 | 默认关闭 |
| `/api/debug/intent` | 调试意图识别 | 默认关闭 |
| `/api/debug/agent` | 调试 Agent 编排 | 默认关闭 |
| `/api/agent/message` | 正式 Agent 消息入口 | 保留 |

未来飞书回调接口不需要 HTTP 调 `/api/debug/agent`，而是可以直接复用：

```python
AgentOrchestrator.handle_message(...)
```

或者在 API 层复用同样的请求/响应结构。

## 5. 当前链路

```text
POST /api/agent/message
-> AgentOrchestrator
-> IntentClassifier
-> ToolRegistry
-> OfferPilotRepository
-> Memory / SQLite
-> AgentResponse
```

这与 OfferPilot 的核心主线一致：

```text
用户自然语言输入 -> Agent 编排 -> 工具执行 -> 数据沉淀
```

## 6. 测试方法

### 6.1 自动化测试

```bash
cd /Users/will/Developer/OfferPilot/backend
.venv/bin/python -m pytest
```

预期：

```text
35 passed
```

### 6.2 单独测试正式消息入口

```bash
.venv/bin/python -m pytest tests/test_agent_message_route.py -q
```

覆盖：

- `/api/agent/message` 能返回 Agent 编排结果；
- `confirmed=true` 能传给编排层；
- debug 路由关闭时，正式 Agent 消息入口仍然可用。

### 6.3 Swagger 手动测试

启动服务：

```bash
.venv/bin/python -m uvicorn app.main:app --reload
```

打开：

```text
http://127.0.0.1:8000/docs
```

调用：

```text
POST /api/agent/message
```

请求：

```json
{
  "message": "今天任务是什么？"
}
```

再测试写入类动作：

```json
{
  "message": "新增投递深信服开发实习，明天下午三点一面",
  "confirmed": true
}
```

## 7. 本次测试结果

运行命令：

```bash
cd /Users/will/Developer/OfferPilot/backend
.venv/bin/python -m pytest
```

测试结果：

```text
collected 35 items
tests/test_agent_message_route.py ...        [  8%]
tests/test_agent_orchestrator.py ......      [ 25%]
tests/test_config.py ...                     [ 34%]
tests/test_debug_llm.py ..                   [ 40%]
tests/test_debug_routes.py ..                [ 45%]
tests/test_health.py .                       [ 48%]
tests/test_intent_classifier.py .....         [ 62%]
tests/test_offerpilot_repository.py ....      [ 74%]
tests/test_offerpilot_tools.py ....           [ 85%]
tests/test_sqlite_offerpilot_repository.py ..... [100%]

35 passed in 1.80s
```

## 8. 后续开发建议

下一阶段可以开始接飞书前置能力，但仍然保持小步：

1. 新增 `FeishuEventRequest` schema；
2. 新增 `/api/feishu/events` 回调草案；
3. 先支持 challenge 验证；
4. 再支持提取文本消息并调用 `AgentOrchestrator`。

这样就能从“本地 API 调 Agent”进入“飞书消息调 Agent”。
