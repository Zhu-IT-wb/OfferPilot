# 第七阶段编码报告：Debug 路由环境隔离

日期：2026-06-25  
项目：OfferPilot  
任务：限制 `/api/debug/*` 只在本地/开发环境开放  

## 1. 背景

前几个阶段为了快速调试，暴露了多个 debug 接口：

```text
/api/debug/llm
/api/debug/intent
/api/debug/agent
```

这些接口适合开发阶段在 Swagger UI 或 curl 中测试模块能力，但不应该作为生产环境的对外 API。后端内部模块之间也不会通过 HTTP 调用这些接口，而是直接调用类和函数，例如：

```text
AgentOrchestrator
-> IntentClassifier
-> ToolRegistry
-> Repository / Store
```

本阶段目标是把调试入口和未来正式外部入口区分开。

## 2. 本次实现内容

新增或修改文件：

| 文件 | 作用 |
| --- | --- |
| `backend/app/core/config.py` | 增加 `debug_routes_enabled` 配置 |
| `backend/app/main.py` | 根据配置决定是否注册 debug 路由 |
| `backend/.env.example` | 增加 `OFFERPILOT_ENABLE_DEBUG_ROUTES` 示例 |
| `backend/tests/test_debug_routes.py` | 覆盖 debug 路由开关 |
| `backend/tests/test_config.py` | 覆盖布尔环境变量解析 |
| `backend/README.md` | 补充 debug 路由环境说明 |

## 3. 配置规则

新增环境变量：

```env
OFFERPILOT_ENABLE_DEBUG_ROUTES=true
```

默认规则：

| 环境 | 默认是否开启 debug 路由 |
| --- | --- |
| `local` | 开启 |
| `dev` | 开启 |
| `development` | 开启 |
| `test` | 开启 |
| `production` | 关闭 |
| `prod` | 关闭 |

如果显式配置 `OFFERPILOT_ENABLE_DEBUG_ROUTES`，则以该变量为准。

## 4. 当前行为

本地开发：

```env
OFFERPILOT_ENV=local
OFFERPILOT_ENABLE_DEBUG_ROUTES=true
```

可访问：

```text
/api/health
/api/debug/llm
/api/debug/intent
/api/debug/agent
```

生产环境：

```env
OFFERPILOT_ENV=production
OFFERPILOT_ENABLE_DEBUG_ROUTES=false
```

可访问：

```text
/api/health
```

不可访问：

```text
/api/debug/*
```

会返回：

```text
404 Not Found
```

## 5. 设计说明

### 5.1 为什么不是让 Agent 调 debug API

debug API 只是外部调试入口。Agent 内部不应该 HTTP 调自己，而应该直接调用模块：

```text
AgentOrchestrator.handle_message()
IntentClassifier.classify()
ToolRegistry.run()
```

这样更简单、更快，也更容易测试。

### 5.2 未来正式入口

未来接飞书时，对外入口应该是少量正式 API，例如：

```text
POST /api/feishu/events
POST /api/feishu/card/callback
GET  /api/health
```

这些正式 API 内部再调用 Agent 编排层。

## 6. 测试结果

新增测试覆盖：

- `debug_routes_enabled=false` 时 `/api/debug/agent` 返回 `404`；
- `debug_routes_enabled=false` 时 `/api/health` 仍然可用；
- `debug_routes_enabled=true` 时 `/api/debug/agent` 可用；
- 布尔环境变量可以解析 `true`、`0` 等常见写法。

运行命令：

```bash
cd /Users/will/Developer/OfferPilot/backend
.venv/bin/python -m pytest
```

测试结果：

```text
collected 23 items
tests/test_agent_orchestrator.py ...... [ 26%]
tests/test_config.py ...               [ 39%]
tests/test_debug_llm.py ..             [ 47%]
tests/test_debug_routes.py ..          [ 56%]
tests/test_health.py .                 [ 60%]
tests/test_intent_classifier.py .....   [ 82%]
tests/test_offerpilot_tools.py ....     [100%]

23 passed in 1.40s
```

## 7. 后续开发建议

下一阶段可以回到业务主线，做数据模型与仓储边界：

1. 定义 `Task`、`Application`、`InterviewReview` 内部模型；
2. 定义 repository 接口；
3. 把当前内存 store 改造成 repository 实现；
4. 后续再替换为 SQLite 或飞书多维表格实现。
