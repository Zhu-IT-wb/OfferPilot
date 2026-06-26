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

当前工具执行结果通过 repository 写入进程内存，服务重启后会清空。后续可以把 repository 实现替换为 SQLite、PostgreSQL 或飞书多维表格。

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
