# 第四阶段编码报告：本地 `.env` 自动加载

日期：2026-06-25  
项目：OfferPilot  
任务：后端启动时自动读取 `backend/.env` 配置  

## 1. 背景

前两阶段已经支持通过环境变量配置 DeepSeek API Key，但本地开发时每次手动 `export DEEPSEEK_API_KEY=...` 或给 `uvicorn` 传 `--env-file` 不够方便。

本阶段目标是让后端启动时自动读取：

```text
/Users/will/Developer/OfferPilot/backend/.env
```

## 2. 实现内容

修改文件：

| 文件 | 作用 |
| --- | --- |
| `backend/app/core/config.py` | 增加轻量 `.env` 文件加载逻辑 |
| `backend/tests/test_config.py` | 增加配置加载测试 |
| `backend/README.md` | 更新本地配置 API Key 的说明 |

## 3. 当前行为

默认行为：

- 服务启动时自动读取 `backend/.env`；
- 支持 `KEY=value`；
- 支持 `export KEY=value`；
- 支持单双引号；
- 支持注释行和行内注释；
- 不覆盖 shell 中已经存在的环境变量。

优先级：

```text
shell/export 环境变量 > backend/.env > 代码默认值
```

也就是说，如果同时存在：

```bash
export DEEPSEEK_API_KEY=from_shell
```

和：

```env
DEEPSEEK_API_KEY=from_env_file
```

最终会使用 `from_shell`。

## 4. 使用方式

进入后端目录：

```bash
cd /Users/will/Developer/OfferPilot/backend
```

复制配置模板：

```bash
cp .env.example .env
```

编辑 `backend/.env`：

```env
DEEPSEEK_API_KEY=your_api_key
OFFERPILOT_LLM_PROVIDER=deepseek
OFFERPILOT_LLM_BASE_URL=https://api.deepseek.com
OFFERPILOT_LLM_MODEL=deepseek-v4-flash
OFFERPILOT_LLM_TIMEOUT_SECONDS=30
```

启动服务：

```bash
.venv/bin/python -m uvicorn app.main:app --reload
```

不再需要额外传 `--env-file .env`。

## 5. 测试结果

运行命令：

```bash
cd /Users/will/Developer/OfferPilot/backend
.venv/bin/python -m pytest
```

结果：

```text
collected 10 items
tests/test_config.py ..              [ 20%]
tests/test_debug_llm.py ..           [ 40%]
tests/test_health.py .               [ 50%]
tests/test_intent_classifier.py ..... [100%]

10 passed in 0.17s
```

## 6. 说明

本阶段没有引入 `python-dotenv` 依赖，而是在 `core/config.py` 中实现了项目当前够用的轻量加载逻辑。这样本地启动不需要额外安装新包，后续如果配置规则变复杂，再替换为成熟依赖也很容易。
