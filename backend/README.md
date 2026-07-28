# OfferPilot Backend

OfferPilot 后端服务第一阶段采用 FastAPI 搭建，当前只包含最小可运行骨架和健康检查接口。

## Quick Start

```bash
cd /Users/will/Developer/OfferPilot/backend
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
- `/api/agent/message` 正式 Agent 消息入口；
- `/api/feishu/events` 飞书事件回调草案，支持 challenge、文本消息提取和文本回复发送；
- `/api/debug/llm` 大模型调用调试接口，仅用于本地/开发环境；
- `/api/debug/intent` 意图识别调试接口，仅用于本地/开发环境；
- `/api/debug/agent` Agent 编排调试接口，仅用于本地/开发环境；
- 本地内存版工具执行，包括今日任务、投递创建、任务完成等；
- LeetCode Hot 100 本地目录、每日推荐、反馈、间隔复习与飞书定时推送；
- 领域模型与 repository 边界，当前以内存实现承载数据；
- 基础配置模块；
- 后续模块目录预留。

## Debug Routes

`/api/debug/*` 是开发调试入口，不是最终对外 API。默认只在 `local`、`dev`、`development`、`test` 环境开启。

本地开发可以在 `.env` 中保留：

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

## Agent Message

正式 Agent 消息入口：

```bash
curl -X POST http://127.0.0.1:8000/api/agent/message \
  -H "Content-Type: application/json" \
  -d '{"message":"今天任务是什么？"}'
```

写入类动作需要确认：

```bash
curl -X POST http://127.0.0.1:8000/api/agent/message \
  -H "Content-Type: application/json" \
  -d '{"message":"新增投递深信服开发实习，明天下午三点一面","confirmed":true}'
```

这个接口不是 debug 接口，未来飞书事件回调会提取用户消息后调用同一个 Agent 编排链路。

## Feishu Events

飞书事件回调草案：

```text
POST /api/feishu/events
```

如果配置了 `FEISHU_VERIFICATION_TOKEN`，请求必须携带匹配的 token。未配置时允许本地调试。

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

配置 DeepSeek API Key：

```bash
cd /Users/will/Developer/OfferPilot/backend
cp .env.example .env
```

然后编辑 `backend/.env`：

```env
DEEPSEEK_API_KEY=your_api_key
```

服务启动时会自动读取 `backend/.env`。如果你同时在 shell 里 `export` 了同名变量，shell 里的值优先级更高。

启动服务后调用：

```bash
curl -X POST http://127.0.0.1:8000/api/debug/llm \
  -H "Content-Type: application/json" \
  -d '{"prompt":"用一句话介绍 OfferPilot"}'
```

如果未配置 API Key，接口会返回 `503`，提示需要设置 `DEEPSEEK_API_KEY` 或 `OFFERPILOT_LLM_API_KEY`。

## Debug Agent

调试 Agent 编排：

```bash
curl -X POST http://127.0.0.1:8000/api/debug/agent \
  -H "Content-Type: application/json" \
  -d '{"message":"新增投递深信服开发实习，明天下午三点一面"}'
```

接口会返回识别到的意图、下一步动作、是否需要确认、槽位信息和给用户的回复文本。

写入类动作默认不会立刻执行，需要确认后再调用：

```bash
curl -X POST http://127.0.0.1:8000/api/debug/agent \
  -H "Content-Type: application/json" \
  -d '{"message":"新增投递深信服开发实习，明天下午三点一面","confirmed":true}'
```

工具执行结果通过 repository 保存；使用 `memory` 时服务重启后会清空，使用 `sqlite` 时会持久化恢复。

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
OFFERPILOT_DASHBOARD_PUBLIC_BASE_URL=https://sculptor-jester-deskwork.ngrok-free.dev
OFFERPILOT_DASHBOARD_OAUTH_SCOPE=auth:user.id:read
OFFERPILOT_DASHBOARD_SESSION_SECRET=replace-with-a-random-secret
```

后端只负责提供该网页，不会通过代码自动创建飞书客户端标签页。开发测试时，可在
OfferPilot 机器人会话顶部点击 `+`，添加网页链接并命名为“刷题计划”；或者在飞书
开放平台为同一个应用添加“网页应用”能力，将桌面端主页和移动端主页都设为：

```text
https://sculptor-jester-deskwork.ngrok-free.dev/leetcode/dashboard
```

应用能力变更后需要创建版本并发布。

`ngrok-free.dev` 只适合临时开发。ngrok 免费版会对首次浏览器访问显示
`ERR_NGROK_6024` 安全确认页；手机端可点击 **Visit Site** 继续访问，但后端无法替
飞书 WebView 添加 `ngrok-skip-browser-warning` 请求头。正式使用时应换成无访问确认
页的稳定 HTTPS 域名（或付费 ngrok 域名）。

飞书开放平台“安全设置”中的 OAuth 重定向 URL：

```text
https://sculptor-jester-deskwork.ngrok-free.dev/leetcode/dashboard/auth/callback
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

当前没有引入向量数据库：每日推荐、展示和评分都按稳定题目 ID 读取结构化数据；后续自由追问或跨题语义检索需要时，可在 `KnowledgeCorpus.search()` 的内部实现中增加混合检索，而不改变业务调用接口。模型只识别命中的评分点和回答证据，最终分数、掌握状态与复习日期仍由服务端计算。
