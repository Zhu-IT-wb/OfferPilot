# OfferPilot

OfferPilot 是一个面向计算机求职准备的飞书原生 AI Agent 系统。它把投递管理、面试安排、复盘沉淀、每日任务和多维表格协同整合到同一条执行链路中，让求职准备从零散记录升级为可追踪、可确认、可迭代的工作流。

项目当前聚焦个人秋招与实习准备场景，采用自研后端承载 Agent 编排、工具调用、状态管理和飞书集成；飞书机器人、多维表格与日历作为用户交互和数据可视化入口。

## Product Positioning

OfferPilot 不是一个只负责回答问题的聊天机器人，而是一个围绕“求职执行闭环”设计的智能工作台：

```text
用户消息
  -> 对话路由与上下文恢复
  -> LLM Planner / 规则 Planner
  -> 工具调用与写入确认
  -> Repository 持久化
  -> 飞书消息 / 多维表格 / 日历同步
  -> 下一轮任务与复盘
```

核心目标是把下面这些高频动作变成稳定流程：

- 管理公司、岗位、投递状态、面试轮次和下一步动作；
- 在飞书中通过自然语言新增、查询和更新投递记录；
- 自动同步投递数据到飞书多维表格，并支持表格侧反向同步；
- 对面试时间进行追问、确认、标准化，并写入飞书日历提醒；
- 对写入类操作建立确认机制，避免误操作；
- 通过 Agent Planner 将复杂用户表达拆解为明确、可执行的系统动作。

## Current Capabilities

| Area | Capability |
| --- | --- |
| Agent Orchestration | 基于 LangGraph 的消息处理图，覆盖上下文加载、待确认动作、Planner、路由、意图识别、工具执行和状态同步。 |
| LLM Planner | 支持 DeepSeek-compatible LLM Planner，并提供规则 fallback，保证在模型不可用时仍可完成核心求职流程。 |
| Conversation State | 支持用户级会话上下文、待确认动作、投递草稿、取消与确认语义。 |
| Application Workflow | 支持新增投递、查询公司状态、更新投递状态、记录面试轮次、生成后续动作。 |
| Feishu Bot | 支持飞书事件回调、URL verification、文本消息提取、Agent 回复发送和事件审计。 |
| Feishu Bitable | 支持多维表格自动创建、记录推送同步、记录拉取同步、事件订阅和协作者授权。 |
| Feishu Calendar | 支持 OfferPilot 专属日历、面试日程创建、参与人同步和提醒确认。 |
| Persistence | Repository 边界清晰，支持内存存储与 SQLite 本地持久化。 |
| Quality | 后端测试覆盖 Agent、Planner、Feishu 事件、多维表格同步、工具层和 Repository。 |
| Documentation | `docs/` 下保留阶段性编码报告和产品技术方案，便于追踪架构演进。 |

## Architecture

```text
+-------------------+        +------------------------+
| Feishu Bot        |        | Feishu Bitable         |
| Messages / Events |        | Records / Collaboration|
+---------+---------+        +-----------+------------+
          |                              ^
          v                              |
+---------+------------------------------+------------+
|                  OfferPilot Backend                 |
|                                                     |
|  FastAPI Routes                                     |
|    - /api/agent/message                             |
|    - /api/feishu/events                             |
|    - /api/health                                    |
|    - /api/debug/*                                   |
|                                                     |
|  Agent Layer                                        |
|    - LangGraph Orchestrator                         |
|    - Message Router                                 |
|    - LLM / Rule-based Planner                       |
|    - Application Dialogue Manager                   |
|                                                     |
|  Tool Layer                                         |
|    - Application tools                              |
|    - Task tools                                     |
|    - Interview review tools                         |
|    - Feishu calendar / bitable sync tools           |
|                                                     |
|  Data Layer                                         |
|    - OfferPilotRepository interface                 |
|    - In-memory implementation                       |
|    - SQLite implementation                          |
+---------------------+-------------------------------+
                      |
                      v
              +-------+-------+
              | DeepSeek API  |
              | LLM Planning  |
              +---------------+
```

## Repository Layout

```text
.
├── backend/
│   ├── app/
│   │   ├── agents/          # Agent 编排、Planner、路由与对话状态
│   │   ├── api/             # FastAPI 路由
│   │   ├── core/            # 配置与环境变量加载
│   │   ├── models/          # 领域模型
│   │   ├── repositories/    # Repository 接口与实现
│   │   ├── schemas/         # API 与 Agent schema
│   │   ├── services/        # 飞书、LLM、多维表格同步服务
│   │   └── tools/           # OfferPilot 工具注册与执行
│   ├── tests/               # 后端测试套件
│   ├── pyproject.toml       # Python 项目配置
│   └── README.md            # 后端运行说明
└── docs/                    # 产品方案与阶段性编码报告
```

## Quick Start

### 1. Prepare Backend Environment

```bash
cd backend
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e ".[dev]"
```

### 2. Configure Runtime

```bash
cp .env.example .env
```

本地开发推荐配置：

```env
OFFERPILOT_ENV=local
OFFERPILOT_ENABLE_DEBUG_ROUTES=true
OFFERPILOT_STORAGE_BACKEND=sqlite
OFFERPILOT_SQLITE_PATH=./data/offerpilot.db
```

如果需要启用 LLM Planner：

```env
DEEPSEEK_API_KEY=your_api_key
OFFERPILOT_LLM_PROVIDER=deepseek
OFFERPILOT_LLM_MODEL=deepseek-v4-flash
OFFERPILOT_LLM_PLANNER_ENABLED=true
OFFERPILOT_LLM_PLANNER_FALLBACK_ENABLED=true
```

如果需要接入飞书：

```env
FEISHU_APP_ID=your_app_id
FEISHU_APP_SECRET=your_app_secret
FEISHU_VERIFICATION_TOKEN=your_verification_token
FEISHU_BITABLE_SYNC_ENABLED=true
FEISHU_CALENDAR_SYNC_ENABLED=true
```

### 3. Run Tests

```bash
.venv/bin/python -m pytest
```

### 4. Start API Service

```bash
.venv/bin/python -m uvicorn app.main:app --reload
```

默认服务地址：

```text
http://127.0.0.1:8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/api/health
```

Agent 消息入口：

```bash
curl -X POST http://127.0.0.1:8000/api/agent/message \
  -H "Content-Type: application/json" \
  -d '{"message":"新增投递字节跳动后端开发实习，明天下午三点一面"}'
```

## Configuration Highlights

| Variable | Purpose |
| --- | --- |
| `OFFERPILOT_ENV` | 运行环境，影响 debug routes 默认开关。 |
| `OFFERPILOT_ENABLE_DEBUG_ROUTES` | 是否开放 `/api/debug/*`。生产环境建议关闭。 |
| `OFFERPILOT_STORAGE_BACKEND` | 存储后端，可选 `memory` 或 `sqlite`。 |
| `OFFERPILOT_SQLITE_PATH` | SQLite 数据库路径。 |
| `DEEPSEEK_API_KEY` / `OFFERPILOT_LLM_API_KEY` | LLM 调用凭证。 |
| `OFFERPILOT_LLM_PLANNER_ENABLED` | 是否启用 LLM Planner。 |
| `FEISHU_APP_ID` / `FEISHU_APP_SECRET` | 飞书应用凭证。 |
| `FEISHU_VERIFICATION_TOKEN` | 飞书事件订阅校验 token。 |
| `FEISHU_BITABLE_SYNC_ENABLED` | 是否启用飞书多维表格同步。 |
| `FEISHU_CALENDAR_SYNC_ENABLED` | 是否启用飞书日历同步。 |

完整配置请参考 [backend/.env.example](backend/.env.example)。

## API Surface

| Endpoint | Description |
| --- | --- |
| `GET /api/health` | 服务健康检查。 |
| `POST /api/agent/message` | 正式 Agent 消息入口，供 API 或飞书事件回调复用。 |
| `POST /api/feishu/events` | 飞书事件回调入口，支持 verification、文本消息和多维表格事件。 |
| `POST /api/debug/llm` | LLM 调试接口，仅建议本地或开发环境使用。 |
| `POST /api/debug/intent` | 意图识别调试接口。 |
| `POST /api/debug/agent` | Agent 编排调试接口。 |

## Engineering Principles

- 飞书负责入口、提醒与结构化数据展示；
- 后端负责业务状态、Agent 编排和工具执行；
- LLM 负责理解复杂表达和生成可执行计划；
- 写入类动作必须可解释、可确认、可追踪；
- Repository、Service、Tool、Agent 分层保持清晰边界；
- 所有外部集成能力都应有本地 fallback 或可测试边界。

## Roadmap

- 引入更完整的任务计划与每日推送机制；
- 扩展面试复盘、薄弱点抽取和后续训练任务生成；
- 接入简历、项目文档和面经知识库检索；
- 增强飞书多维表格模板和字段迁移能力；
- 在稳定单用户闭环后，再评估多用户、权限和 Web 控制台。

## Documentation

- [产品需求与技术方案](docs/OfferPilot-需求与技术方案.md)
- [后端运行说明](backend/README.md)
- `docs/` 下的阶段性编码报告记录了从后端骨架、Agent 编排、飞书事件、日历同步、多维表格同步到 LLM Planner 接入的演进过程。

