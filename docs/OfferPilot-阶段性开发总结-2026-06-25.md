# OfferPilot 阶段性开发总结

日期：2026-06-25  
项目：OfferPilot  
当前定位：面向计算机秋招准备的个人求职 Agent  
当前阶段：后端 Agent 主链路与飞书入口雏形已打通  

## 1. 项目目标回顾

OfferPilot 的目标不是做一个泛聊天机器人，而是做一个围绕计算机秋招准备的个人执行 Agent。

第一版 MVP 聚焦：

- 每日任务生成和打卡；
- LeetCode / 八股 / 项目深挖任务管理；
- 投递记录管理；
- 面试复盘；
- 根据复盘生成薄弱点和后续补强任务；
- 使用飞书作为入口和工作台；
- 后端负责 Agent 编排、模型调用、工具执行和数据沉淀。

当前技术路线：

```text
飞书入口
-> FastAPI 后端
-> Agent 编排
-> DeepSeek
-> Tool Registry
-> Repository
-> SQLite / 后续飞书多维表格
```

## 2. 当前整体架构

当前代码已经形成基础分层：

```text
API 层
  /api/health
  /api/agent/message
  /api/feishu/events
  /api/debug/*

Agent 层
  IntentClassifier
  AgentOrchestrator

Tool 层
  ToolRegistry
  OfferPilot tools

Repository 层
  OfferPilotRepository
  InMemoryOfferPilotRepository
  SQLiteOfferPilotRepository

Model 层
  Task
  Application
  InterviewReview

Service 层
  LLMService
  FeishuMessageService
```

主链路：

```text
用户自然语言
-> IntentClassifier 识别意图
-> AgentOrchestrator 选择 action
-> ToolRegistry 调用工具
-> Repository 读写数据
-> AgentResponse
-> 飞书消息回复
```

## 3. 已完成阶段

### 第一阶段：FastAPI 后端骨架

完成内容：

- 创建 FastAPI 应用入口；
- 新增健康检查接口；
- 建立基础目录结构；
- 增加测试。

接口：

```text
GET /api/health
```

返回：

```json
{"status":"ok"}
```

### 第二阶段：DeepSeek 调用封装

完成内容：

- 新增 `LLMService`；
- 支持 DeepSeek OpenAI-compatible API；
- 支持 `.env` / 环境变量配置；
- 新增 LLM 调试接口；
- 无 API Key 时返回明确错误。

接口：

```text
POST /api/debug/llm
```

### 第三阶段：意图识别雏形

完成内容：

- 新增 `IntentClassifier`；
- 定义 OfferPilot MVP 意图枚举；
- 支持 LLM JSON 解析；
- 无 API Key 时支持本地规则兜底；
- 能识别今日任务、新增投递、完成任务、面试复盘、模拟面试等意图。

示例：

```text
新增投递深信服开发实习，明天下午三点一面
```

识别结果：

```json
{
  "intent": "add_application",
  "slots": {
    "company": "深信服",
    "role": "开发实习",
    "interview_time": "明天下午三点",
    "round": "一面"
  }
}
```

### 第四阶段：本地 `.env` 自动加载

完成内容：

- 后端启动时自动读取 `backend/.env`；
- 支持常见 `.env` 写法；
- shell/export 环境变量优先级高于 `.env`；
- 不需要每次手动 `--env-file`。

### 第五阶段：Agent 编排雏形

完成内容：

- 新增 `AgentOrchestrator`；
- 根据意图选择下一步 action；
- 生成结构化 Agent 响应；
- 引入确认机制。

示例：

```json
{
  "intent": "add_application",
  "action": "create_application",
  "need_confirmation": true
}
```

### 第六阶段：Tool Registry 与本地内存工具

完成内容：

- 新增 `ToolRegistry`；
- 新增统一 `ToolResult`；
- 实现本地工具：
  - `list_today_tasks`
  - `create_application`
  - `complete_task`
  - `postpone_task`
  - `create_interview_review`
  - `start_mock_interview`
- 写入类动作必须 `confirmed=true` 才执行。

### 第七阶段：Debug 路由环境隔离

完成内容：

- 新增 `OFFERPILOT_ENABLE_DEBUG_ROUTES`；
- `/api/debug/*` 默认只在 local/dev/test 开启；
- 生产环境可关闭 debug 接口；
- `/api/health` 和正式业务入口不受影响。

### 第八阶段：数据模型与 Repository 边界

完成内容：

- 新增领域模型：
  - `Task`
  - `Application`
  - `InterviewReview`
- 新增 `OfferPilotRepository` 协议；
- 将工具层从直接操作 dict/list 改为调用 repository；
- 为后续 SQLite / 飞书多维表格接入打基础。

核心边界：

```text
model = 数据长什么样
repository = 数据怎么读写、存在哪里
tool = Agent 可以执行什么动作
```

### 第九阶段：SQLite 持久化 Repository

完成内容：

- 新增 `SQLiteOfferPilotRepository`；
- 支持自动建表；
- 支持默认任务种子数据；
- 支持任务、投递、面试复盘持久化；
- 支持通过配置在 `memory` 和 `sqlite` 间切换。

配置：

```env
OFFERPILOT_STORAGE_BACKEND=sqlite
OFFERPILOT_SQLITE_PATH=./data/offerpilot.db
```

### 第十阶段：正式 Agent 消息入口

完成内容：

- 新增正式接口：

```text
POST /api/agent/message
```

- 与 `/api/debug/agent` 区分；
- 未来飞书回调会复用同一条 Agent 编排链路；
- debug 接口不再是唯一调用 Agent 的方式。

### 第十一阶段：飞书事件回调草案

完成内容：

- 新增飞书事件回调接口：

```text
POST /api/feishu/events
```

- 支持 URL verification challenge；
- 支持提取文本消息；
- 支持调用 `AgentOrchestrator`；
- 非文本消息暂不处理。

### 第十二阶段：飞书 Verification Token 校验

完成内容：

- 新增配置：

```env
FEISHU_VERIFICATION_TOKEN=
```

- 如果配置 token，请求必须携带匹配 token；
- token 错误或缺失返回 `403`；
- 未配置 token 时允许本地调试。

支持 token 位置：

```text
payload["token"]
payload["header"]["token"]
payload["event"]["token"]
```

### 第十三阶段：飞书 Tenant Access Token 与消息回复

完成内容：

- 新增 `FeishuMessageService`；
- 支持使用 `FEISHU_APP_ID` / `FEISHU_APP_SECRET` 获取 `tenant_access_token`；
- 支持调用飞书消息 API 发送文本消息；
- 飞书事件处理后会尝试把 Agent 回复发回用户；
- 未配置飞书应用凭证时，事件仍能处理，但不会发送回复。

配置：

```env
FEISHU_APP_ID=
FEISHU_APP_SECRET=
FEISHU_API_BASE_URL=https://open.feishu.cn/open-apis
FEISHU_TIMEOUT_SECONDS=15
```

## 4. 当前可用接口

### 健康检查

```text
GET /api/health
```

### 正式 Agent 消息入口

```text
POST /api/agent/message
```

请求示例：

```json
{
  "message": "今天任务是什么？"
}
```

写入类动作：

```json
{
  "message": "新增投递深信服开发实习，明天下午三点一面",
  "confirmed": true
}
```

### 飞书事件回调

```text
POST /api/feishu/events
```

支持：

- challenge；
- token 校验；
- 文本消息提取；
- 调用 Agent；
- 尝试发送飞书文本回复。

### Debug 接口

仅用于本地/开发环境：

```text
POST /api/debug/llm
POST /api/debug/intent
POST /api/debug/agent
```

## 5. 当前测试状态

运行命令：

```bash
cd /Users/will/Developer/OfferPilot/backend
.venv/bin/python -m pytest
```

当前结果：

```text
49 passed
```

覆盖范围：

- FastAPI 健康检查；
- `.env` 自动加载；
- Debug 路由隔离；
- DeepSeek LLM 封装；
- 意图识别；
- Agent 编排；
- 工具注册与执行；
- 内存 repository；
- SQLite repository；
- 正式 Agent 消息入口；
- 飞书事件回调；
- 飞书 verification token 校验；
- 飞书 tenant access token 和消息发送服务。

## 6. 当前配置项

核心配置：

```env
OFFERPILOT_APP_NAME=OfferPilot API
OFFERPILOT_APP_VERSION=0.1.0
OFFERPILOT_API_PREFIX=/api
OFFERPILOT_ENV=local
OFFERPILOT_ENABLE_DEBUG_ROUTES=true
```

存储配置：

```env
OFFERPILOT_STORAGE_BACKEND=sqlite
OFFERPILOT_SQLITE_PATH=./data/offerpilot.db
```

DeepSeek 配置：

```env
DEEPSEEK_API_KEY=
OFFERPILOT_LLM_PROVIDER=deepseek
OFFERPILOT_LLM_BASE_URL=https://api.deepseek.com
OFFERPILOT_LLM_MODEL=deepseek-v4-flash
OFFERPILOT_LLM_TIMEOUT_SECONDS=30
```

飞书配置：

```env
FEISHU_VERIFICATION_TOKEN=
FEISHU_APP_ID=
FEISHU_APP_SECRET=
FEISHU_API_BASE_URL=https://open.feishu.cn/open-apis
FEISHU_TIMEOUT_SECONDS=15
```

注意：真实 `.env` 不应提交到 GitHub。

## 7. 当前限制

当前还没有完成：

- 飞书 encrypted event 解密；
- 飞书事件去重；
- 真实公网回调联调；
- 飞书多维表格读写；
- 每日定时任务生成；
- 面试复盘薄弱点提取；
- 真实多轮模拟面试状态管理；
- 生产级日志、监控、鉴权。

## 8. 下一步建议

建议继续沿着飞书真实联调推进：

1. 使用 ngrok 或 Cloudflare Tunnel 暴露本地服务；
2. 在飞书开放平台配置事件订阅；
3. 发布飞书应用版本；
4. 用真实飞书消息触发 `/api/feishu/events`；
5. 根据真实事件 payload 修正兼容逻辑；
6. 接入飞书多维表格，将 SQLite 过渡为工作台数据展示。

## 9. 简历表述草案

可以写成：

```text
OfferPilot：面向计算机秋招准备的个人求职 Agent

基于 FastAPI、DeepSeek OpenAI-compatible API 和飞书开放平台构建个人秋招执行 Agent，围绕每日任务、投递管理、面试复盘和模拟面试构建闭环。设计并实现意图识别、Agent 编排、工具注册、Repository 数据访问层和 SQLite 持久化；接入飞书事件回调、verification token 校验、tenant access token 获取和机器人文本回复能力，为后续多维表格数据沉淀和飞书工作台联动打基础。
```
