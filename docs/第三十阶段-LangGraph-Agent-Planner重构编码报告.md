# 第三十阶段：LangGraph Agent Planner 重构编码报告

## 阶段目标

本阶段目标不是继续给 `orchestrator.py` 增加规则，而是把 OfferPilot 的消息处理从“线性 if/else 工作流”升级为“LangGraph 状态图 + Agent Planner”的结构，为后续接入 LLM Planner、MCP-ready Tools 和更复杂的多工具编排打基础。

## 本阶段完成内容

### 1. 引入 LangGraph

在 `backend/pyproject.toml` 中新增依赖：

```toml
langgraph>=0.6.0
```

当前本地安装版本为 `langgraph 0.6.11`。

### 2. 重构 AgentOrchestrator 为 LangGraph 图

文件：

- `backend/app/agents/orchestrator.py`

新增 `AgentGraphState`，表示一次用户消息处理过程中在图节点之间流转的状态。

当前图结构：

```text
START
  -> load_context
  -> cancel_pending / confirm_pending / handle_pending / route_message
  -> handle_general / classify_intent
  -> plan_response
  -> execute_tool
  -> sync_pending
  -> END
```

每个节点职责：

- `load_context`：生成 conversation_id，读取 pending action。
- `cancel_pending`：处理用户取消当前待确认动作。
- `confirm_pending`：处理用户确认并执行待确认动作。
- `handle_pending`：尝试把当前消息补到上一轮待确认动作里。
- `route_message`：判断是普通问答还是秋招业务动作。
- `handle_general`：处理你好、你是谁、你能做什么、领域问答等。
- `classify_intent`：调用 IntentClassifier 做业务意图识别。
- `plan_response`：调用 AgentPlanner 生成结构化计划。
- `execute_tool`：在不需要确认或用户已确认时执行工具。
- `sync_pending`：把未执行或缺槽动作保存为 pending。

`handle_message` 现在只负责：

```python
final_state = await self._graph.ainvoke(...)
return final_state["response"]
```

这意味着 orchestrator 的角色从“写死业务流程”变成了“运行 Agent 图”。

### 3. 新增结构化 AgentPlan

文件：

- `backend/app/schemas/agent_plan.py`

新增：

- `AgentPlanStep`
- `AgentPlan`

`AgentPlan` 用来描述 Planner 输出的结构化计划：

```json
{
  "intent": "get_today_tasks",
  "action": "list_today_tasks",
  "need_confirmation": false,
  "steps": [
    {
      "tool_name": "list_today_tasks",
      "arguments": {},
      "reason": "read_today_tasks"
    }
  ]
}
```

目前 `AgentPlan` 可以通过 `to_response()` 转成原有接口兼容的 `AgentResponse`。

### 4. 新增 AgentPlanner

文件：

- `backend/app/agents/planner.py`

`AgentPlanner` 负责把 `IntentClassification` 转成 `AgentPlan`。

当前仍然保留规则式 Planner，但已经把“规划职责”从 `orchestrator.py` 中抽出来。后续可以把 `AgentPlanner.plan()` 替换为：

```text
上下文检索 + LLM 结构化计划生成 + Validator 校验
```

而不用重写飞书回调、工具执行、会话状态这些外层链路。

## 与之前的区别

之前：

```text
handle_message
  -> 一长串 if/else
  -> intent_classifier
  -> _plan_response
  -> _maybe_execute_tool
```

现在：

```text
handle_message
  -> LangGraph
      -> load_context
      -> route / pending / confirm / cancel
      -> classify_intent
      -> AgentPlanner.plan
      -> execute_tool
      -> sync_pending
```

这一步还不是最终形态的“全自主 Agent”，但已经把后续升级点留出来了：

- Planner 可以替换成 LLM Planner。
- Tool 可以逐步变成 MCP-ready schema。
- LangGraph 可以继续加 `retrieve_context`、`validate_plan`、`ask_confirmation`、`replan_on_failure` 等节点。

## 新增测试

文件：

- `backend/tests/test_agent_planner.py`

新增测试点：

- `AgentPlanner` 可以输出结构化 `AgentPlan.steps`。
- `AgentOrchestrator` 持有可异步执行的 LangGraph 编译图。

## 测试结果

执行：

```bash
cd /Users/will/Developer/OfferPilot/backend
.venv/bin/python -m pytest
```

结果：

```text
107 passed
```

## 下一步建议

下一阶段建议继续做：

**第三十一阶段：ContextRetriever + LLM Planner 雏形**

目标：

1. 自动读取最近投递、近期面试、待确认动作；
2. 把上下文喂给 Planner；
3. 让 LLM 输出结构化 `AgentPlan`；
4. 加 `PlanValidator`，防止模型直接执行高风险动作；
5. 支持“取消面试”“改面试时间”这类未写死的新动作规划。
