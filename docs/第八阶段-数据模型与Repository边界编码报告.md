# 第八阶段编码报告：数据模型与 Repository 边界

日期：2026-06-25  
项目：OfferPilot  
任务：抽离领域模型和数据访问边界，为后续数据库或飞书多维表格接入做准备  

## 1. 背景

第六阶段已经实现了 Tool Registry 和本地内存工具，但当时工具层直接管理 `dict/list`：

```text
tasks: List[Dict[str, Any]]
applications: List[Dict[str, Any]]
interview_reviews: List[Dict[str, Any]]
```

这在 MVP 早期能跑，但后续会出现几个问题：

- 字段名容易写错；
- 工具层和存储层耦合；
- 后续替换 SQLite、PostgreSQL 或飞书多维表格时改动面太大；
- 不利于表达真实后端项目的分层设计。

本阶段目标是把数据结构从工具层抽出来，形成：

```text
AgentOrchestrator
-> ToolRegistry
-> OfferPilot tools
-> Repository
-> Domain models
```

## 2. 本次实现内容

新增或修改文件：

| 文件 | 作用 |
| --- | --- |
| `backend/app/models/task.py` | 定义任务模型、任务类型、任务状态、优先级 |
| `backend/app/models/application.py` | 定义投递模型、投递状态、面试轮次到状态的转换 |
| `backend/app/models/interview_review.py` | 定义面试复盘模型和状态 |
| `backend/app/repositories/offerpilot_repository.py` | 定义 `OfferPilotRepository` 协议和内存实现 |
| `backend/app/tools/offerpilot_tools.py` | 改为通过 repository 读写数据 |
| `backend/tests/test_offerpilot_repository.py` | 增加 repository 测试 |
| `backend/tests/test_offerpilot_tools.py` | 更新工具测试，验证工具通过 repository 工作 |
| `backend/tests/test_agent_orchestrator.py` | 更新编排测试，使用新的内存 repository |
| `backend/README.md` | 更新当前能力说明 |

## 3. 当前领域模型

### 3.1 Task

字段：

| 字段 | 说明 |
| --- | --- |
| `id` | 任务 ID |
| `title` | 任务标题 |
| `task_type` | `leetcode` / `interview_question` / `project_deep_dive` 等 |
| `status` | `pending` / `in_progress` / `postponed` / `passed` 等 |
| `priority` | `high` / `medium` / `low` |

### 3.2 Application

字段：

| 字段 | 说明 |
| --- | --- |
| `id` | 投递 ID |
| `company` | 公司 |
| `role` | 岗位 |
| `status` | 投递状态 |
| `interview_time` | 面试时间，当前保留自然语言表达 |
| `round` | 面试轮次 |
| `jd_keywords` | JD 关键词 |

### 3.3 InterviewReview

字段：

| 字段 | 说明 |
| --- | --- |
| `id` | 复盘 ID |
| `company` | 公司 |
| `round` | 面试轮次 |
| `topics` | 被问知识点 |
| `raw_message` | 原始复盘输入 |
| `status` | 复盘状态 |

## 4. Repository 边界

当前定义了 `OfferPilotRepository` 协议，核心方法包括：

```python
list_today_tasks()
create_application(...)
complete_task(...)
postpone_task(...)
create_interview_review(...)
```

当前实现：

```python
InMemoryOfferPilotRepository
```

后续可以替换为：

```text
SQLiteOfferPilotRepository
FeishuBitableOfferPilotRepository
PostgresOfferPilotRepository
```

工具层不需要知道底层数据到底来自内存、数据库还是飞书多维表格。

## 5. 工具层变化

重构前：

```text
offerpilot_tools.py
-> 自己维护 tasks/applications/interview_reviews
```

重构后：

```text
offerpilot_tools.py
-> 调用 OfferPilotRepository
-> 将领域模型转换为 ToolResult.data
```

这让工具层职责更清晰：

- 校验工具参数；
- 调用 repository；
- 包装 `ToolResult`；
- 不再直接关心数据存储结构。

## 6. 本阶段测试方法

### 6.1 自动化测试

进入后端目录：

```bash
cd /Users/will/Developer/OfferPilot/backend
```

运行：

```bash
.venv/bin/python -m pytest
```

预期结果：

```text
27 passed
```

### 6.2 单独测试 repository

```bash
.venv/bin/python -m pytest tests/test_offerpilot_repository.py -q
```

用于验证：

- 默认任务列表；
- 创建投递记录；
- 完成任务；
- 创建面试复盘。

### 6.3 单独测试工具层

```bash
.venv/bin/python -m pytest tests/test_offerpilot_tools.py -q
```

用于验证：

- Tool Registry 可以注册和执行工具；
- 工具可以通过 repository 创建投递记录；
- 工具可以完成预置任务。

### 6.4 手动接口测试

启动服务：

```bash
.venv/bin/python -m uvicorn app.main:app --reload
```

打开 Swagger UI：

```text
http://127.0.0.1:8000/docs
```

测试 `POST /api/debug/agent`：

```json
{
  "message": "新增投递深信服开发实习，明天下午三点一面",
  "confirmed": true
}
```

预期响应中包含：

```json
{
  "action": "create_application",
  "tool_result": {
    "success": true,
    "data": {
      "application": {
        "company": "深信服",
        "role": "开发实习",
        "status": "interview_1"
      }
    }
  }
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
collected 27 items
tests/test_agent_orchestrator.py ......     [ 22%]
tests/test_config.py ...                    [ 33%]
tests/test_debug_llm.py ..                  [ 40%]
tests/test_debug_routes.py ..               [ 48%]
tests/test_health.py .                      [ 51%]
tests/test_intent_classifier.py .....        [ 70%]
tests/test_offerpilot_repository.py ....     [ 85%]
tests/test_offerpilot_tools.py ....          [100%]

27 passed in 1.85s
```

## 8. 后续开发建议

下一阶段可以开始做真正的持久化选择：

1. 如果想快速验证闭环，可以先接 SQLite；
2. 如果想尽快贴近飞书工作台，可以直接设计 Feishu Bitable repository；
3. 无论选哪种，都尽量保持 `OfferPilotRepository` 接口不变。

建议下一步优先做 SQLite，本地调试更稳定，也方便写自动化测试。等业务字段稳定后，再把同一套 repository 接到飞书多维表格。
