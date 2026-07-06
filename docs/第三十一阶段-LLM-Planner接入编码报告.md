# 第三十一阶段：LLM Planner 接入编码报告

## 阶段目标

本阶段目标是把 OfferPilot 的 Agent 决策层从“规则路由 + 意图识别 + 固定映射”升级为“LLM 先生成结构化计划，后端再校验并执行工具”。

这一步不是替换飞书、数据库或工具层，而是重构 Agent 的大脑层：

1. 用户自然语言先进入 LLM Planner。
2. LLM 只负责规划，不直接执行。
3. 后端 Plan Validator 校验安全性、必填字段和工具合法性。
4. 合法计划交给 ToolRegistry 执行。
5. LLM 不可用或返回非法 JSON 时，回退到原来的规则链路。

## 主要改动

### 1. 拆分规则 Planner

文件：

- `backend/app/agents/planner.py`

原来的 `AgentPlanner` 规则逻辑被拆成 `RuleBasedAgentPlanner`，继续保留作为兜底。

新的 `AgentPlanner` 变为 LLM-first Planner，对外仍然输出统一的 `AgentPlan`，因此后续工具执行链路不用整体推翻。

### 2. 接入 LLM Planner

新增：

- `AgentPlannerContext`
- `AgentPlanner.plan_message(...)`

LLM Planner 会拿到：

- 用户原始消息
- 当前会话 ID
- 用户 ID
- 来源渠道
- 当前 pending action
- 可用工具列表

然后要求模型只返回严格 JSON 格式的 `AgentPlan`。

### 3. ToolRegistry 增加工具描述

文件：

- `backend/app/tools/registry.py`
- `backend/app/schemas/tool.py`
- `backend/app/tools/offerpilot_tools.py`

新增 `ToolSpec`，每个工具现在可以声明：

- 工具名称
- 工具用途
- 是否会修改数据
- 必填槽位
- 可选槽位
- 示例表达

这样 LLM Planner 不是随便“想象工具”，而是在后端给定的工具目录里做选择。

### 4. LangGraph 流程调整

文件：

- `backend/app/agents/orchestrator.py`

新消息处理顺序改为：

1. 加载上下文
2. 处理确认/取消/pending
3. 优先进入 LLM Planner
4. 如果 LLM Planner 产出合法计划，执行计划
5. 如果 LLM 不可用或非法，回退到旧的 MessageRouter + IntentClassifier + RuleBasedPlanner

这样“我有哪些面试”不会再先被旧路由判成普通消息，而是可以直接规划成 `query_application`。

### 5. Plan Validator

LLM 生成的计划会被后端校验：

- 写操作必须强制确认
- 工具名必须存在于 ToolRegistry
- 新增投递必须有公司和岗位
- 安排面试必须有公司、轮次、具体面试时间
- “明天下午”这类不够具体的时间不会执行，会追问具体几点
- 查询面试会自动补成 `query_type=upcoming_interviews`

## 新增配置

文件：

- `backend/.env.example`

新增：

```env
OFFERPILOT_LLM_PLANNER_ENABLED=true
OFFERPILOT_LLM_PLANNER_FALLBACK_ENABLED=true
```

含义：

- `OFFERPILOT_LLM_PLANNER_ENABLED`：是否启用 LLM Planner。
- `OFFERPILOT_LLM_PLANNER_FALLBACK_ENABLED`：LLM 不可用或返回非法 JSON 时，是否回退到旧规则链路。

## 测试覆盖

新增和补充测试：

- “我有哪些面试”能规划成 `query_application`
- LLM 返回非法 JSON 时能回退到规则链路
- LLM 想直接执行写操作时，后端会强制要求确认
- 面试时间只有“明天下午”时，会追问具体时间
- “你好”会走普通帮助回复，不会被硬塞进秋招写操作

测试结果：

```text
112 passed, 77 warnings
```

## 当前边界

本阶段解决的是 Planner 层，而不是一次性补全所有工具能力。

例如“取消面试”现在可以被 LLM Planner 更自然地理解，但如果要真正取消飞书日历事件、更新数据库状态、同步多维表格，还需要后续新增明确的 `cancel_interview` 或扩展 `update_application` 工具语义。

也就是说：

- 这一阶段让 Agent 更像 Agent：先规划、再校验、再执行。
- 下一阶段应该继续补工具能力：取消面试、改期面试、查询候选记录并让用户选择。
