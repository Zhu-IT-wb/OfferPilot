# 第二阶段编码报告：DeepSeek 调用封装与调试接口

日期：2026-06-24  
项目：OfferPilot  
阶段：Phase 1 继续推进  
任务：封装 OpenAI-compatible LLM 调用，并提供 `/api/debug/llm` 调试接口  

## 1. 实验目的

第一阶段已经完成 FastAPI 后端骨架和 `/api/health` 健康检查。本阶段目标是在不接入飞书、不引入数据库的前提下，让后端具备调用大模型的最小能力。

本阶段验收目标：

- 后端配置中支持 LLM 相关环境变量；
- 封装统一 `LLMService`；
- 支持 DeepSeek OpenAI-compatible Chat Completions 调用；
- 新增 `/api/debug/llm` 调试接口；
- 未配置 API Key 时返回明确错误；
- 单元测试覆盖正常响应和缺少 API Key 两种情况。

## 2. 本次实现内容

新增或修改文件：

| 文件 | 作用 |
| --- | --- |
| `backend/app/services/llm_service.py` | 封装 LLM 请求、错误处理、响应解析 |
| `backend/app/api/routes/debug.py` | 新增调试路由 |
| `backend/app/schemas/llm.py` | 定义 LLM 调试接口请求和响应模型 |
| `backend/app/core/config.py` | 增加 LLM 配置项 |
| `backend/app/main.py` | 注册 debug 路由 |
| `backend/.env.example` | 增加 DeepSeek 配置示例 |
| `backend/pyproject.toml` | 将 `httpx` 提升为运行时依赖 |
| `backend/tests/test_debug_llm.py` | 增加调试接口测试 |
| `backend/README.md` | 补充 LLM 调试接口使用方式 |

## 3. 接口设计

### 3.1 LLM 调试接口

请求：

```http
POST /api/debug/llm
Content-Type: application/json
```

请求体：

```json
{
  "prompt": "用一句话介绍 OfferPilot",
  "system_prompt": "You are OfferPilot, a concise job-search preparation assistant.",
  "model": "deepseek-v4-flash",
  "temperature": 0.3,
  "max_tokens": 512
}
```

其中只有 `prompt` 必填，其余字段都有默认值。

响应：

```json
{
  "provider": "deepseek",
  "model": "deepseek-v4-flash",
  "content": "..."
}
```

未配置 API Key 时返回：

```http
503 Service Unavailable
```

```json
{
  "detail": "LLM API key is not configured. Set DEEPSEEK_API_KEY or OFFERPILOT_LLM_API_KEY."
}
```

## 4. DeepSeek 配置

根据 DeepSeek 官方 API 文档，DeepSeek API 支持 OpenAI-compatible API 格式，OpenAI-compatible `base_url` 为：

```text
https://api.deepseek.com
```

当前默认模型使用：

```text
deepseek-v4-flash
```

原因是旧模型名 `deepseek-chat` 和 `deepseek-reasoner` 将在 2026-07-24 废弃，第一版直接使用新模型名可以减少后续迁移成本。

环境变量：

```bash
export DEEPSEEK_API_KEY="your_api_key"
export OFFERPILOT_LLM_PROVIDER=deepseek
export OFFERPILOT_LLM_BASE_URL=https://api.deepseek.com
export OFFERPILOT_LLM_MODEL=deepseek-v4-flash
export OFFERPILOT_LLM_TIMEOUT_SECONDS=30
```

系统同时支持：

```bash
export OFFERPILOT_LLM_API_KEY="your_api_key"
```

如果同时设置，优先读取 `OFFERPILOT_LLM_API_KEY`，否则读取 `DEEPSEEK_API_KEY`。

## 5. 代码结构说明

### 5.1 `LLMService`

`LLMService` 负责：

- 读取模型配置；
- 组装 OpenAI-compatible messages；
- 发送 `/chat/completions` 请求；
- 解析 `choices[0].message.content`；
- 屏蔽底层 HTTP 异常；
- 返回统一 `LLMResult`。

当前实现保持简单，后续可以扩展：

- JSON Mode；
- Tool Calling；
- 流式输出；
- 多模型 provider；
- 请求日志；
- token usage 统计；
- 重试和限流。

### 5.2 错误类型

当前定义了两个错误：

| 错误 | 场景 |
| --- | --- |
| `LLMConfigurationError` | API Key 缺失等配置错误 |
| `LLMRequestError` | 模型服务 HTTP 错误、网络错误、响应格式错误 |

路由层将它们转换为 HTTP 响应：

- 配置错误：`503 Service Unavailable`；
- 上游请求错误：`502 Bad Gateway`。

## 6. 运行方式

进入后端目录：

```bash
cd /Users/will/Developer/OfferPilot/backend
```

安装或更新依赖：

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

配置 API Key：

```bash
export DEEPSEEK_API_KEY="your_api_key"
```

启动服务：

```bash
.venv/bin/python -m uvicorn app.main:app --reload
```

调用接口：

```bash
curl -X POST http://127.0.0.1:8000/api/debug/llm \
  -H "Content-Type: application/json" \
  -d '{"prompt":"用一句话介绍 OfferPilot"}'
```

运行测试：

```bash
.venv/bin/python -m pytest
```

### 6.1 本次实际验证记录

本次已刷新项目 editable 安装：

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

单元测试结果：

```text
collected 3 items
tests/test_debug_llm.py .. [ 66%]
tests/test_health.py .     [100%]
3 passed in 0.13s
```

本次未进行真实 DeepSeek API 调用，因为当前项目环境没有配置 `DEEPSEEK_API_KEY`。调试接口已经覆盖了“缺少 API Key 返回 503”和“模型服务返回内容后正常响应”两种路径。

## 7. 后续开发建议

下一步建议做飞书消息回调之前，先做一个很轻的 Agent Orchestrator 雏形：

1. 定义用户意图枚举；
2. 创建 `IntentClassifier`；
3. 创建一个 `/api/debug/intent` 调试接口；
4. 输入自然语言，输出意图和结构化参数；
5. 再把它接入飞书消息回调。

也可以直接进入飞书链路：

```text
飞书消息 -> 后端回调 -> LLMService -> 机器人回复
```

但建议至少保留 `/api/debug/llm`，后续调问题会方便很多。

## 8. 本阶段结论

本阶段完成了 OfferPilot 后端从“能启动”到“能调用模型”的关键一步。系统现在已经具备接入 Agent 编排的基础能力。

接下来所有需要 AI 的模块，例如每日任务生成、八股评分、面试复盘和模拟面试，都可以复用 `LLMService`，而不是在业务代码中直接写 HTTP 请求。
