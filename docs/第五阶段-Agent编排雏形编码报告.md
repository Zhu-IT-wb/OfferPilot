# 第五阶段编码报告：Agent 编排雏形与调试接口

日期：2026-06-25  
项目：OfferPilot  
任务：新增 Agent 编排雏形，并提供 `/api/debug/agent` 调试接口  

## 1. 实验目的

第三阶段已经完成“意图识别雏形”，可以把用户自然语言转成：

```text
intent + confidence + slots
```

但这还只是理解用户输入，并没有决定下一步要做什么。本阶段目标是在不接飞书、不写数据库、不真正执行工具的前提下，先把 Agent 主链路推进到：

```text
用户输入
-> 意图识别
-> AgentOrchestrator 决定下一步动作
-> 返回结构化 Agent 响应
```

## 2. 本次实现内容

新增或修改文件：

| 文件 | 作用 |
| --- | --- |
| `backend/app/schemas/agent.py` | 定义 Agent 动作枚举、调试请求、编排响应 |
| `backend/app/agents/orchestrator.py` | 实现 Agent 编排雏形 |
| `backend/app/api/routes/debug.py` | 新增 `/api/debug/agent` 调试接口 |
| `backend/tests/test_agent_orchestrator.py` | 增加 Agent 编排测试 |
| `backend/README.md` | 补充 Agent 调试接口说明 |

## 3. 接口设计

### 3.1 Agent 编排调试接口

请求：

```http
POST /api/debug/agent
Content-Type: application/json
```

请求体：

```json
{
  "message": "新增投递深信服开发实习，明天下午三点一面"
}
```

响应示例：

```json
{
  "intent": "add_application",
  "confidence": 0.86,
  "action": "create_application",
  "reply": "我识别到你要新增投递记录，公司是 深信服，岗位是 开发实习，时间是 明天下午三点，轮次是 一面。下一步会写入投递表，执行前需要你确认。",
  "need_confirmation": true,
  "slots": {
    "company": "深信服",
    "role": "开发实习",
    "interview_time": "明天下午三点",
    "round": "一面"
  },
  "missing_slots": []
}
```

## 4. 当前支持的动作

本阶段定义了 `AgentActionName`：

| action | 含义 |
| --- | --- |
| `list_today_tasks` | 查询今日任务 |
| `create_application` | 创建投递记录 |
| `update_application` | 更新投递状态 |
| `complete_task` | 完成任务 |
| `postpone_task` | 延期任务 |
| `create_interview_review` | 创建面试复盘 |
| `start_mock_interview` | 开始模拟面试 |
| `record_answer` | 记录并评分用户回答 |
| `answer_help` | 普通咨询回复 |
| `summarize_week` | 周复盘 |
| `ask_clarification` | 信息不足时追问 |
| `no_op` | 暂不执行动作 |

## 5. 实现说明

### 5.1 编排职责

`AgentOrchestrator` 当前只负责决策，不直接操作数据库或飞书。

当前职责：

- 调用 `IntentClassifier`；
- 根据 `intent` 选择 `action`；
- 判断是否需要用户确认；
- 判断关键槽位是否缺失；
- 生成面向用户的下一步回复。

### 5.2 为什么暂不执行工具

本阶段刻意不做真正的写入动作，是为了保持项目节奏小而稳。

当前链路是：

```text
识别 -> 决策
```

下一阶段再加入：

```text
识别 -> 决策 -> 工具执行
```

这样可以避免一边设计 Agent、一边设计数据存储、一边接飞书，导致边界混乱。

### 5.3 确认机制

会改变状态的动作默认需要确认，例如：

- `create_application`
- `update_application`
- `complete_task`
- `postpone_task`
- `create_interview_review`
- `record_answer`

只读或咨询类动作暂不需要确认，例如：

- `list_today_tasks`
- `answer_help`
- `summarize_week`

这为后续接飞书时的“确认后再写入多维表格”预留了接口。

## 6. 本次测试

新增测试覆盖：

- 新增投递意图可以编排为 `create_application`；
- 缺少岗位时返回 `ask_clarification`；
- 查询今日任务意图可以编排为 `list_today_tasks`；
- `/api/debug/agent` 路由可以返回结构化 Agent 响应。

运行命令：

```bash
cd /Users/will/Developer/OfferPilot/backend
.venv/bin/python -m pytest
```

测试结果：

```text
collected 14 items
tests/test_agent_orchestrator.py .... [ 28%]
tests/test_config.py ..              [ 42%]
tests/test_debug_llm.py ..           [ 57%]
tests/test_health.py .               [ 64%]
tests/test_intent_classifier.py ..... [100%]

14 passed in 0.14s
```

## 7. 后续开发建议

下一阶段建议做 Tool Registry 和本地内存版工具：

1. 定义 `ToolResult`；
2. 定义 `ToolRegistry`；
3. 实现 `list_today_tasks`、`create_application`、`complete_task` 的本地内存版；
4. 让 `AgentOrchestrator` 从“只决策”升级为“决策后调用工具”；
5. 等工具接口稳定后，再接飞书多维表格。

这样 OfferPilot 会从“能决定动作”继续推进到“能真实执行动作”。
