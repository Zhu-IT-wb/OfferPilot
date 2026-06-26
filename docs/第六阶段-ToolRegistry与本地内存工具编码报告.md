# 第六阶段编码报告：Tool Registry 与本地内存工具

日期：2026-06-25  
项目：OfferPilot  
任务：新增工具注册与本地内存版工具执行  

## 1. 实验目的

第五阶段已经完成 Agent 编排雏形，可以把用户输入转成：

```text
intent -> action -> reply
```

但当时还没有真正执行动作。本阶段目标是在不接数据库、不接飞书多维表格的前提下，先加入一层工具执行能力，让部分 action 可以真的产生状态变化。

本阶段推进后的链路：

```text
用户输入
-> 意图识别
-> AgentOrchestrator 选择 action
-> ToolRegistry 调用本地工具
-> 返回工具执行结果
```

## 2. 本次实现内容

新增或修改文件：

| 文件 | 作用 |
| --- | --- |
| `backend/app/schemas/tool.py` | 定义统一工具执行结果 `ToolResult` |
| `backend/app/tools/registry.py` | 定义 `ToolRegistry` 和工具查找错误 |
| `backend/app/tools/offerpilot_tools.py` | 实现本地内存版 OfferPilot 工具 |
| `backend/app/schemas/agent.py` | 为 `/debug/agent` 增加 `confirmed` 和 `tool_result` |
| `backend/app/agents/orchestrator.py` | 接入工具注册表和确认执行逻辑 |
| `backend/app/api/routes/debug.py` | `/api/debug/agent` 支持确认执行 |
| `backend/tests/test_agent_orchestrator.py` | 增加编排执行测试 |
| `backend/tests/test_offerpilot_tools.py` | 增加工具层测试 |
| `backend/README.md` | 更新 Agent 调试接口说明 |

## 3. 当前工具能力

当前支持的本地工具：

| action / tool_name | 能力 |
| --- | --- |
| `list_today_tasks` | 返回内存中的今日任务 |
| `create_application` | 创建投递记录 |
| `complete_task` | 完成匹配到的任务 |
| `postpone_task` | 延期匹配到的任务 |
| `create_interview_review` | 创建面试复盘记录 |
| `start_mock_interview` | 返回模拟面试第一问 |

当前内存中预置了 3 个任务：

| id | title | task_type |
| --- | --- | --- |
| `task_1` | `LeetCode 206. 反转链表` | `leetcode` |
| `task_2` | `HashMap 扩容机制` | `interview_question` |
| `task_3` | `云聚图库 Caffeine + Redis 两级缓存设计` | `project_deep_dive` |

## 4. 接口变化

### 4.1 `/api/debug/agent`

请求体新增：

```json
{
  "message": "新增投递深信服开发实习，明天下午三点一面",
  "confirmed": true
}
```

其中：

- `confirmed=false`：只做识别和编排，不执行写入类工具；
- `confirmed=true`：允许执行需要确认的工具。

### 4.2 示例：查看今日任务

请求：

```json
{
  "message": "今天任务是什么？"
}
```

响应会直接包含工具执行结果：

```json
{
  "intent": "get_today_tasks",
  "action": "list_today_tasks",
  "need_confirmation": false,
  "reply": "今天的任务：...",
  "tool_result": {
    "tool_name": "list_today_tasks",
    "success": true,
    "message": "今天的任务：...",
    "data": {
      "tasks": []
    }
  }
}
```

### 4.3 示例：新增投递但未确认

请求：

```json
{
  "message": "新增投递深信服开发实习，明天下午三点一面"
}
```

响应：

```json
{
  "intent": "add_application",
  "action": "create_application",
  "need_confirmation": true,
  "tool_result": null
}
```

此时不会写入内存投递记录。

### 4.4 示例：确认新增投递

请求：

```json
{
  "message": "新增投递深信服开发实习，明天下午三点一面",
  "confirmed": true
}
```

响应会包含工具执行结果：

```json
{
  "intent": "add_application",
  "action": "create_application",
  "need_confirmation": false,
  "tool_result": {
    "tool_name": "create_application",
    "success": true,
    "message": "已创建投递记录：深信服 - 开发实习。",
    "data": {
      "application": {
        "id": "app_1",
        "company": "深信服",
        "role": "开发实习",
        "status": "interview_1",
        "interview_time": "明天下午三点",
        "round": "一面"
      }
    }
  }
}
```

## 5. 设计说明

### 5.1 ToolResult

工具统一返回：

```json
{
  "tool_name": "create_application",
  "success": true,
  "message": "已创建投递记录：深信服 - 开发实习。",
  "data": {}
}
```

这样后续无论工具背后是内存、SQLite、飞书多维表格还是第三方 API，Agent 编排层都可以用同一种方式接收结果。

### 5.2 ToolRegistry

`ToolRegistry` 负责：

- 注册工具名和处理函数；
- 判断某个 action 是否有工具；
- 根据工具名执行工具；
- 未注册时抛出 `ToolNotFoundError`。

### 5.3 本地内存 Store

`InMemoryOfferPilotStore` 当前负责保存：

- `tasks`
- `applications`
- `interview_reviews`

它只适合本地调试，不适合生产持久化。服务重启后数据会清空。

### 5.4 确认机制

本阶段保留第五阶段设计：

- 查询类动作可以直接执行；
- 写入类动作必须带 `confirmed=true` 才会执行；
- 未确认时只返回计划动作和确认提示。

这为后续飞书机器人里的“确认后写入多维表格”打基础。

## 6. 测试结果

新增测试覆盖：

- `ToolRegistry` 可以注册和执行工具；
- 未注册工具会抛出 `ToolNotFoundError`；
- 本地工具可以创建投递记录；
- 本地工具可以完成预置任务；
- 编排器不会在未确认时执行写入类动作；
- 编排器在确认后可以执行 `create_application`；
- 查询今日任务会直接执行 `list_today_tasks`。

运行命令：

```bash
cd /Users/will/Developer/OfferPilot/backend
.venv/bin/python -m pytest
```

测试结果：

```text
collected 20 items
tests/test_agent_orchestrator.py ...... [ 30%]
tests/test_config.py ..                [ 40%]
tests/test_debug_llm.py ..             [ 50%]
tests/test_health.py .                 [ 55%]
tests/test_intent_classifier.py .....   [ 80%]
tests/test_offerpilot_tools.py ....     [100%]

20 passed in 0.14s
```

## 7. 后续开发建议

下一阶段建议做持久化前的“数据模型与仓储边界”：

1. 定义 `Application`、`Task`、`InterviewReview` 的内部数据模型；
2. 抽象 repository 接口；
3. 把当前内存 store 包装成 repository 实现；
4. 再决定是先接 SQLite，还是直接接飞书多维表格。

这样后续替换存储层时，工具和 Agent 编排层不会大面积改动。
