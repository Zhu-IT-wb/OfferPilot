# OfferPilot Backend

OfferPilot 后端基于 FastAPI，承载 Agent 编排、受控 Tool Calling、求职业务状态、飞书集成、学习中心、GitHub 项目分析与项目面试训练。

## Quick Start

```bash
cd OfferPilot/backend
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m uvicorn app.main:app --reload
```

启动后访问：

```text
http://127.0.0.1:8000/api/health
```

期望返回：

```json
{"status":"ok"}
```

## Current Scope

- FastAPI 应用入口；
- `/api/health` 健康检查接口；
- `POST /api/agent/runs` 启动 run，`GET /api/agent/runs/{thread_id}` 读取状态，`POST /api/agent/runs/{thread_id}/resume` 恢复中断；
- `/api/feishu/events` 飞书事件回调，支持 challenge、文本消息提取、Agent 回复与事件审计；
- `/api/debug/llm` 大模型调用调试接口，仅用于本地/开发环境；
- `/api/debug/agent` Agent 编排调试接口，仅用于本地/开发环境；
- 统一 LangGraph ReAct Runtime、动态 plan、checkpoint/interrupt、上下文预算与失败后继续规划；
- 投递、任务、面试、StudyPlan、日历与多维表格工具执行，写操作按影响级别审批；
- LeetCode Hot 100 本地目录、每日推荐、反馈、间隔复习与飞书定时推送；
- GitHub 项目发现、只读代码分析、项目画像确认与项目面试训练；
- 领域模型与 Repository 边界，支持内存与 SQLite 持久化；
- 基础配置模块；
- 学习中心和项目训练 Dashboard。

## Debug Routes

`/api/debug/*` 是无身份隔离的开发调试入口，不是最终对外 API，因此默认关闭。

仅在服务没有通过 ngrok 等方式暴露时，才可临时开启：

```env
OFFERPILOT_ENV=local
OFFERPILOT_ENABLE_DEBUG_ROUTES=true
OFFERPILOT_STORAGE_BACKEND=memory
```

生产环境应关闭：

```env
OFFERPILOT_ENV=production
OFFERPILOT_ENABLE_DEBUG_ROUTES=false
```

关闭后，`/api/health` 仍然可用，但 `/api/debug/*` 会返回 `404`。

## Main Agent Runtime

主 Agent 的所有自然语言任务都进入同一个循环状态图。应用启动时只编译一次 LangGraph，所有 API 和飞书消息共享同一个 Runtime：

```text
START → prepare_turn → agent
                       ├─ tool_calls → validate → human_gate? → execute
                       │                                  ↑          │
                       │                                  └─ resume  └─ observation → agent
                       └─ final_draft → completion_guard → verifier? → finalize → END
```

- Agent 每次取得工具成功、失败或部分结果后都会重新决策，直到满足完成条件或触发运行预算；
- `update_execution_plan` 维护可选、用户可见且可修改的 plan，它是工作记忆，不是固定执行器；
- `completion_guard` 只检查 pending 调用、中断、AI/Tool 消息配对、实际写入回执和结构化 plan 等运行不变量，不再通过关键词重新解释用户意图；启用 verifier 时，每份最终草稿都会由无工具模型对照原始目标与执行回执检查语义完整性；
- 工具声明 `read / local_write / external_write / destructive` 影响级别以及审批策略；
- 原生 `interrupt` 暂停审批、补充信息和选择，`AsyncSqliteSaver` 用同一 `thread_id` 跨请求恢复；
- checkpoint 保存图控制状态；`agent_tool_operations` 和 `agent_ingress_receipts` 分别保存工具幂等回执和入口请求去重；
- 每轮模型调用注入可信的当前绝对时间、时区和本周边界；个人投递、面试、任务、薄弱点等结论必须存在对应的成功读取回执；
- 非日历写入出现 `unknown` 时使用经人工审批的 `reconcile_write_operation` 解析账本，日历 session 则使用领域专用协调工具，二者都禁止盲目重试；
- 上下文预算包含 system prompt、工具 Schema 和消息：已经闭合的历史工具调用会转成保留状态、错误码、operation key、artifact 和结果摘要的紧凑回执，较老的普通对话轮次会写入结构化 conversation summary；最近对话、unknown 结果、未配对调用、当前任务和 pending interaction 保持原样。压缩后的消息与摘要会写回 checkpoint，完整原始事件仍单独保留用于审计。

## Agent Runs

启动一个 run：

```bash
export OFFERPILOT_AGENT_TOKEN="$(.venv/bin/python scripts/issue_agent_api_token.py --user-id local_user)"
curl -X POST http://127.0.0.1:8000/api/agent/runs \
  -H "Authorization: Bearer $OFFERPILOT_AGENT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"查询本周面试，结合薄弱知识点生成复习计划，并同步到飞书日历","user_id":"local_user","source":"api","conversation_scope":"autumn-recruiting","request_id":"demo-start-1"}'
```

响应包含 `thread_id`、`run_id`、`status`、`reply`、动态 `plan`、待处理 `interaction`、工具执行摘要、artifacts、warnings 和 usage。`status` 为 `completed / waiting_for_input / partial / failed / degraded` 之一。

通用 Agent HTTP 接口要求短期 Bearer Actor Token。先在 `.env` 配置独立随机的
`OFFERPILOT_AGENT_API_SIGNING_SECRET`；服务端只信任验签 Token 的用户身份，旧
`user_id/source` 字段仅做兼容校验，不能切换用户。读取当前状态不会推进图：

```bash
curl -H "Authorization: Bearer $OFFERPILOT_AGENT_TOKEN" \
  "http://127.0.0.1:8000/api/agent/runs/agent_xxx"
```

当响应为 `waiting_for_input` 时，使用同一 `thread_id` 和 `interaction.id` 恢复。首次补充复习偏好的示例：

```bash
curl -X POST http://127.0.0.1:8000/api/agent/runs/agent_xxx/resume \
  -H "Authorization: Bearer $OFFERPILOT_AGENT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"interaction_id":"interaction_xxx","decision":"answer","value":{"timezone":"Asia/Shanghai","weekday_windows":[{"start_time":"19:30","end_time":"22:00"}],"weekend_windows":[{"start_time":"09:00","end_time":"12:00"}],"daily_max_minutes":120,"session_minutes":45},"user_id":"local_user","source":"api","request_id":"demo-resume-1"}'
```

日历同步等外部写会返回新的 approval interaction；使用新的 `interaction.id` 再调用同一接口，并把 `decision` 设为 `approve`。其他恢复动作是 `reject`、`revise` 和 `cancel`。

## User Isolation

- 飞书入口只接受已验签事件中的 `open_id`，HTTP Agent 入口只接受签名 Actor Token 中的 `sub`；请求正文不能切换身份。
- Runtime 以来源和用户生成 owner，checkpoint、请求回执、工具操作账本及会话锁均绑定该 owner/thread。
- 投递、面试、任务、复习计划、训练记录和检索证据的读写都带 owner 条件；关联对象跨 owner 时直接拒绝。
- 每个 owner 使用独立的多维表格和托管日历配置。表格回调通过服务端 `app_token + table_id -> owner` 映射归属，不能通过修改隐藏字段转移数据。
- 旧 `local_user` 数据不会被新用户自动认领；历史资源迁移必须由管理员显式绑定并通过冲突检查。

## StudyPlan Workflow

复习计划是主 Agent 的纵向多工具闭环：并行读取本周面试和薄弱知识点，缺少偏好时 interrupt 收集时区、可用窗口、每日上限和单次时长，再由确定性调度器避开本地面试、已有复习 session 和可用的日历忙闲。计划、sessions 和未排期项先写入本地表，并可通过 `get_study_plan / list_study_plans` 重新读取；`sync_study_plan_to_calendar(plan_id)` 始终在外部写审批后执行，并按 operation key 和日历 event ID 幂等同步。外部写结果为 `unknown` 时不会自动重试，必须先通过 `reconcile_study_calendar_session` 查询远端状态再决定补偿动作。

## Feishu Events

飞书事件回调草案：

```text
POST /api/feishu/events
```

飞书回调默认要求配置 `FEISHU_VERIFICATION_TOKEN`，缺失或不匹配时不会执行 Agent 或任何写操作。仅离线本地测试可显式设置 `OFFERPILOT_FEISHU_ALLOW_UNVERIFIED_EVENTS=true`；该开关在 production 环境无效，也不应在 ngrok、局域网或公网可访问的服务上开启。

文本、卡片和多维表格事件还必须携带稳定的事件 ID，入口回执按事件 ID 去重，避免飞书重试导致重复写入。用户身份只接受已验签事件中的 `open_id`；多维表格回写的 owner 则由服务端保存的表格资源映射确定，事件操作者和表格字段不能转移数据归属。

当前支持 URL 验证 challenge：

```bash
curl -X POST http://127.0.0.1:8000/api/feishu/events \
  -H "Content-Type: application/json" \
  -d '{"type":"url_verification","token":"your_verification_token","challenge":"challenge_token"}'
```

预期返回：

```json
{"challenge":"challenge_token"}
```

当前也支持文本消息事件提取，并调用 Agent 编排：

```bash
curl -X POST http://127.0.0.1:8000/api/feishu/events \
  -H "Content-Type: application/json" \
  -d '{"schema":"2.0","header":{"event_type":"im.message.receive_v1","token":"your_verification_token"},"event":{"sender":{"sender_id":{"open_id":"ou_test"}},"message":{"message_type":"text","content":"{\"text\":\"今天任务是什么？\"}"}}}'
```

如果配置了飞书应用凭证，后端会把 Agent 回复通过飞书消息 API 发回用户：

```env
FEISHU_APP_ID=your_app_id
FEISHU_APP_SECRET=your_app_secret
```

如果未配置应用凭证，接口仍会处理事件，但响应里会标记 `reply_sent=false`。

说明：当前已支持 verification token 校验和文本消息主动回复；还未实现 encrypted event 解密。

## Debug LLM

先复制配置文件：

```bash
cd OfferPilot/backend
cp .env.example .env
```

默认使用 DeepSeek：

```env
DEEPSEEK_API_KEY=your_api_key
```

也可以切换为 OpenAI Chat Completions：

```env
OFFERPILOT_LLM_PROVIDER=openai
OFFERPILOT_LLM_BASE_URL=https://api.openai.com/v1
OFFERPILOT_LLM_MODEL=gpt-4.1-mini
OPENAI_API_KEY=your_openai_api_key
```

`OFFERPILOT_LLM_API_KEY` 的优先级高于 provider 专用 Key。OpenAI 配置需要 OpenAI Platform API Key，不能使用 ChatGPT 网页账号或订阅代替。

服务启动时会自动读取 `backend/.env`。如果你同时在 shell 里 `export` 了同名变量，shell 里的值优先级更高。

启动服务后调用：

```bash
curl -X POST http://127.0.0.1:8000/api/debug/llm \
  -H "Content-Type: application/json" \
  -d '{"prompt":"用一句话介绍 OfferPilot"}'
```

如果未配置 API Key，接口会返回 `503`，提示需要设置 `OFFERPILOT_LLM_API_KEY`、`OPENAI_API_KEY` 或 `DEEPSEEK_API_KEY`。

## Debug Agent

调试 Agent 编排：

```bash
curl -X POST http://127.0.0.1:8000/api/debug/agent \
  -H "Content-Type: application/json" \
  -d '{"message":"新增投递深信服开发实习，明天下午三点一面"}'
```

接口使用同一个 ReAct Runtime，并返回与正式 run API 相同的状态、plan、interaction、工具摘要和 artifacts；需要恢复时仍调用正式的 `/api/agent/runs/{thread_id}/resume`。

## Storage Backend

当前支持两种本地存储：

```env
OFFERPILOT_STORAGE_BACKEND=memory
```

或者：

```env
OFFERPILOT_STORAGE_BACKEND=sqlite
OFFERPILOT_SQLITE_PATH=./data/offerpilot.db
```

`memory` 适合跑测试和快速调试，服务重启后数据会清空。`sqlite` 会把任务、投递和面试复盘写入本地 SQLite 文件，服务重启后仍然保留。

Agent checkpoint 使用独立配置，避免和业务事务混在同一个连接生命周期中：

```env
OFFERPILOT_AGENT_CHECKPOINT_BACKEND=sqlite
OFFERPILOT_AGENT_CHECKPOINT_PATH=./data/offerpilot_checkpoints.db
OFFERPILOT_AGENT_API_SIGNING_SECRET=replace-with-a-separate-random-secret
```

测试可把 checkpoint backend 设为 `memory`。SQLite 模式下，LangGraph checkpoint 写入独立文件；API 请求及飞书文本、卡片和 Bitable 事件的 ingress receipt，与工具 operation ledger 一起写入业务 SQLite。checkpoint 解决“从哪个节点恢复”，ledger 解决“恢复时是否已经执行过外部副作用”，两者不能互相替代。

## Tests

```bash
.venv/bin/python -m pytest
```

测试数量和平台差异会随实现变化，请以当前环境的 `pytest` 输出为准。

## LeetCode Hot 100

在飞书中发送以下命令使用第一版刷题计划：

```text
开启每日刷题
今天刷什么
第1题独立完成
LRU 看题解完成
第2题没做出来
第3题延期
关闭每日刷题
```

开启后默认在 `Asia/Shanghai` 时区 08:00 推送当天计划，12:00 和 18:00 只提醒仍未反馈的题目。每日推荐由确定性工作流生成，不消耗 LLM Token；正常为两道新题和一道到期复习题，没有到期复习时推荐三道新题。

推荐与反馈的主界面是“刷题计划”网页。定时消息只保留进度摘要和工作台入口，题目结果在网页中点击记录；文本回复继续作为兼容入口。

```env
OFFERPILOT_DASHBOARD_PUBLIC_BASE_URL=https://your-domain.example
OFFERPILOT_DASHBOARD_OAUTH_SCOPE=auth:user.id:read
OFFERPILOT_DASHBOARD_SESSION_SECRET=replace-with-a-random-secret
```

后端只负责提供该网页，不会通过代码自动创建飞书客户端标签页。开发测试时，可在
OfferPilot 机器人会话顶部点击 `+`，添加网页链接并命名为“刷题计划”；或者在飞书
开放平台为同一个应用添加“网页应用”能力，将桌面端主页和移动端主页都设为：

```text
https://your-domain.example/leetcode/dashboard
```

应用能力变更后需要创建版本并发布。

临时隧道只适合本地联调。正式演示应使用没有访问确认页的稳定 HTTPS 域名，避免飞书
WebView 被第三方确认页面阻断。

飞书开放平台“安全设置”中的 OAuth 重定向 URL：

```text
https://your-domain.example/leetcode/dashboard/auth/callback
```

还需要把公网域名加入 H5 可信域名。域名变更时必须同步更新网页应用主页或会话标签页、
OAuth 重定向 URL、H5 可信域名和 `OFFERPILOT_DASHBOARD_PUBLIC_BASE_URL`。

运行时只读取 `app/data/leetcode_hot100.json`，不会访问力扣。手动更新公开元数据快照：

```bash
.venv/bin/python scripts/sync_leetcode_hot100.py
```

也可以用已经下载的页面离线验证解析器；只有解析到恰好 100 道唯一题目时才会原子替换旧快照：

```bash
.venv/bin/python scripts/sync_leetcode_hot100.py --html /path/to/top-100-liked.html
```

## 八股学习中心

登录后的页面入口：

```text
https://your-public-domain.example/study/knowledge
```

页面提供左侧知识导航、一题一卡文字回答、AI 结构化评价、简答版和完整解析。首次打开页面会为当前飞书用户启用每日八股订阅；08:00 推送今日计划，12:00 和 18:00 只提醒未完成题目。薄弱题按掌握度自动进入后续复习队列。

本地私有资料默认从仓库根目录的 `data/knowledge` 读取。启动时，`KnowledgeCorpus` 会过滤介绍页和聚合页，将 AI 专题的一题一文件以及传统开发的大篇章 Markdown 统一拆成一题一条，生成稳定 ID、简答、完整解析、来源信息、内容哈希和初版评分点，再增量同步到 SQLite。`app/data/interview_knowledge.json` 只在 Markdown 目录不存在或没有可用题目时作为最小回退题库。

当前本地语料可解析为 814 道题。原始 `data/` 已被 Git 忽略，不会随代码提交。手动同步命令：

私有目录中的 `.offerpilot-corpus.json` 记录最低 Markdown 文件数和最低题目数，用来避免首次部署时因目录未完整挂载而把残缺题库写入数据库。题库有意扩容后可同步更新该清单；有意删题时使用下方的 `--force`。

```bash
.venv/bin/python scripts/sync_knowledge.py
```

同步会先校验题目数量和稳定 ID；默认不允许自动删除或替换任何现有题目，会保留上一次可用题库并拒绝写入。确认是有意迁移后，才使用：

```bash
.venv/bin/python scripts/sync_knowledge.py --force
```

可以通过以下配置覆盖目录或关闭启动同步：

```text
OFFERPILOT_KNOWLEDGE_SOURCE_PATH=../data/knowledge
OFFERPILOT_KNOWLEDGE_MARKDOWN_SYNC_ENABLED=true
```

## MCP + Hybrid RAG

OfferPilot runs a local `Career Knowledge` MCP server over stdio. The main Agent
keeps one persistent MCP client and exposes three read-only evidence tools:

- `search_interview_knowledge`: only interview knowledge;
- `search_project_evidence`: only the current actor's project and source-code evidence;
- `read_evidence`: load full content for selected stable evidence IDs.

These adapters return bounded raw evidence and metadata instead of invoking a second
answer-composition LLM. The ReAct Agent decides whether to search, read more evidence,
or synthesize the final answer. MCP output remains untrusted data and cannot override
system or runtime instructions; the backend injects `owner_id` rather than accepting it
from model arguments.

SQLite remains the source of truth. Qdrant Local stores a rebuildable index under
`backend/data/qdrant`, using `BAAI/bge-small-zh-v1.5` dense embeddings and Qdrant
BM25 sparse embeddings. Dense and sparse candidates are fused with RRF. Every hit
retains a stable `evidence_id`, source path, code line range, project version and
content hash. Access filters are applied to both retrieval branches before fusion.

```env
OFFERPILOT_RAG_ENABLED=true
OFFERPILOT_RAG_QDRANT_PATH=./data/qdrant
OFFERPILOT_RAG_COLLECTION_NAME=career_knowledge
OFFERPILOT_RAG_DENSE_MODEL=BAAI/bge-small-zh-v1.5
OFFERPILOT_RAG_SPARSE_MODEL=Qdrant/bm25
OFFERPILOT_RAG_FASTEMBED_CACHE_PATH=./data/fastembed-cache
OFFERPILOT_RAG_TOP_K=5
```

The first query downloads the local embedding models. Run an end-to-end MCP smoke
test with:

```bash
.venv/bin/python scripts/smoke_mcp_rag.py "Why does InnoDB use a B+ tree?"
```

Retrieval quality can be measured with the reusable evaluator in
`app/rag/evaluation.py`; it reports Recall@K, MRR and p95 latency from gold evidence
IDs. Project evidence is private: the backend injects `owner_id`, and model-produced
arguments cannot override the authenticated actor.

每日推荐、展示和评分仍按稳定题目 ID 读取结构化数据；Qdrant 只承担可重建的语义检索索引。模型负责识别命中的评分点、选择证据和组织回答，最终分数、掌握状态、复习日期与 StudyPlan 排期仍由服务端确定性计算。
