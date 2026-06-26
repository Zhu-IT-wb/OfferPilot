# 第九阶段编码报告：SQLite 持久化 Repository

日期：2026-06-25  
项目：OfferPilot  
任务：新增 SQLite 版 Repository，为本地持久化和后续正式存储打基础  

## 1. 背景

第八阶段已经定义了领域模型和 Repository 边界：

```text
Tool -> OfferPilotRepository -> Domain Models
```

当时只有 `InMemoryOfferPilotRepository`，数据会随着服务重启而丢失。本阶段新增 SQLite 实现，让同一套工具层可以切换到本地持久化存储。

## 2. 本次实现内容

新增或修改文件：

| 文件 | 作用 |
| --- | --- |
| `backend/app/repositories/sqlite_offerpilot_repository.py` | 新增 SQLite Repository 实现 |
| `backend/app/core/config.py` | 增加存储配置项 |
| `backend/app/tools/offerpilot_tools.py` | 支持按配置选择默认 repository |
| `backend/.env.example` | 增加 SQLite 配置示例 |
| `backend/tests/test_sqlite_offerpilot_repository.py` | 增加 SQLite Repository 测试 |
| `backend/README.md` | 补充存储后端说明 |

## 3. 新增配置

### 3.1 默认内存模式

```env
OFFERPILOT_STORAGE_BACKEND=memory
```

适合快速调试和测试，服务重启后数据清空。

### 3.2 SQLite 模式

```env
OFFERPILOT_STORAGE_BACKEND=sqlite
OFFERPILOT_SQLITE_PATH=./data/offerpilot.db
```

适合本地持久化，服务重启后数据保留。

## 4. SQLite 表设计

### 4.1 tasks

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | TEXT PRIMARY KEY | 任务 ID |
| `title` | TEXT | 任务标题 |
| `task_type` | TEXT | 任务类型 |
| `status` | TEXT | 任务状态 |
| `priority` | TEXT | 优先级 |

### 4.2 applications

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | TEXT PRIMARY KEY | 投递 ID |
| `company` | TEXT | 公司 |
| `role` | TEXT | 岗位 |
| `status` | TEXT | 投递状态 |
| `interview_time` | TEXT | 面试时间 |
| `round` | TEXT | 面试轮次 |
| `jd_keywords` | TEXT | JSON 字符串 |

### 4.3 interview_reviews

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | TEXT PRIMARY KEY | 复盘 ID |
| `company` | TEXT | 公司 |
| `round` | TEXT | 面试轮次 |
| `topics` | TEXT | JSON 字符串 |
| `raw_message` | TEXT | 原始复盘内容 |
| `status` | TEXT | 复盘状态 |

## 5. 当前行为

初始化 SQLite Repository 时会：

1. 自动创建数据库目录；
2. 自动创建三张表；
3. 如果任务表为空，写入 3 条默认任务：
   - `LeetCode 206. 反转链表`
   - `HashMap 扩容机制`
   - `云聚图库 Caffeine + Redis 两级缓存设计`

工具层仍然只依赖 `OfferPilotRepository`，所以可以用同一套工具调用内存或 SQLite。

## 6. 本阶段测试方法

### 6.1 全量测试

```bash
cd /Users/will/Developer/OfferPilot/backend
.venv/bin/python -m pytest
```

预期：

```text
32 passed
```

### 6.2 单独测试 SQLite Repository

```bash
.venv/bin/python -m pytest tests/test_sqlite_offerpilot_repository.py -q
```

覆盖：

- 自动建表；
- 自动种默认任务；
- 创建投递；
- SQLite 文件跨实例保留数据；
- 完成任务；
- 创建面试复盘；
- 工具层可以使用 SQLite repository。

### 6.3 手动切换 SQLite 测试

编辑：

```text
/Users/will/Developer/OfferPilot/backend/.env
```

加入或修改：

```env
OFFERPILOT_STORAGE_BACKEND=sqlite
OFFERPILOT_SQLITE_PATH=./data/offerpilot.db
```

启动服务：

```bash
cd /Users/will/Developer/OfferPilot/backend
.venv/bin/python -m uvicorn app.main:app --reload
```

打开：

```text
http://127.0.0.1:8000/docs
```

调用 `POST /api/debug/agent`：

```json
{
  "message": "新增投递深信服开发实习，明天下午三点一面",
  "confirmed": true
}
```

预期响应里有：

```json
{
  "action": "create_application",
  "tool_result": {
    "success": true,
    "data": {
      "application": {
        "id": "app_1",
        "company": "深信服",
        "role": "开发实习",
        "status": "interview_1"
      }
    }
  }
}
```

重启服务后再次创建一条投递，ID 应该继续递增，例如 `app_2`，说明 SQLite 文件保留了历史数据。

## 7. 本次测试结果

运行命令：

```bash
cd /Users/will/Developer/OfferPilot/backend
.venv/bin/python -m pytest
```

测试结果：

```text
collected 32 items
tests/test_agent_orchestrator.py ......        [ 18%]
tests/test_config.py ...                       [ 28%]
tests/test_debug_llm.py ..                     [ 34%]
tests/test_debug_routes.py ..                  [ 40%]
tests/test_health.py .                         [ 43%]
tests/test_intent_classifier.py .....           [ 59%]
tests/test_offerpilot_repository.py ....        [ 71%]
tests/test_offerpilot_tools.py ....             [ 84%]
tests/test_sqlite_offerpilot_repository.py ..... [100%]

32 passed in 1.54s
```

## 8. 后续开发建议

下一阶段可以开始做正式业务接口或飞书接入前的准备：

1. 增加 repository 查询接口，例如按公司查投递记录；
2. 增加正式非 debug 的 Agent 入口设计；
3. 或者开始接飞书事件回调，把飞书消息交给 `AgentOrchestrator`。

建议先做“正式 Agent 入口草案”，再接飞书。这样可以继续保持外部 API 少而清晰。
