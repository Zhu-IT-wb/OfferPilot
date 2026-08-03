import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.agents.application_dialogue import ApplicationDialogueManager
from app.agents.conversation import PendingAgentAction, RecentAgentContext
from app.agents.intent_classifier import IntentClassifier
from app.core.config import settings
from app.schemas.agent import AgentActionName, AgentResponse
from app.schemas.agent_plan import AgentPlan, AgentPlanStep
from app.schemas.intent import IntentClassification, IntentName
from app.schemas.tool import ToolSpec
from app.services.feishu_service import parse_chinese_datetime
from app.services.llm_service import LLMConfigurationError, LLMRequestError, LLMService


# 承载 LLM Planner 所需的轻量上下文。
@dataclass(frozen=True)
class AgentPlannerContext:
    conversation_id: str
    user_id: str
    source: str
    pending: Optional[PendingAgentAction] = None
    recent_context: Optional[RecentAgentContext] = None


# 将旧意图识别结果转换为结构化 Agent 执行计划，作为 LLM Planner 的稳定兜底。
class RuleBasedAgentPlanner:
    # 初始化当前组件所需的依赖和配置。
    def __init__(self, application_dialogue: Optional[ApplicationDialogueManager] = None) -> None:
        self.application_dialogue = application_dialogue or ApplicationDialogueManager()

    # 根据意图识别结果生成 AgentPlan。
    def plan(self, classification: IntentClassification) -> AgentPlan:
        application_response = self.application_dialogue.plan(classification)
        if application_response is not None:
            return self._from_response(
                response=application_response,
                reason="application_dialogue",
            )

        intent = classification.intent
        slots = classification.slots

        if intent == IntentName.GET_TODAY_TASKS:
            return self._plan(
                classification=classification,
                action=AgentActionName.LIST_TODAY_TASKS,
                reply="我识别到你想查看今日任务。下一步会调用任务工具读取今天的 LeetCode、八股、项目深挖和投递安排。",
                reason="read_today_tasks",
            )

        if intent == IntentName.ENABLE_LEETCODE_PLAN:
            return self._plan(
                classification=classification,
                action=AgentActionName.ENABLE_LEETCODE_PLAN,
                reply="正在开启每日 LeetCode 推送。",
                reason="enable_leetcode_plan",
            )

        if intent == IntentName.DISABLE_LEETCODE_PLAN:
            return self._plan(
                classification=classification,
                action=AgentActionName.DISABLE_LEETCODE_PLAN,
                reply="正在关闭每日 LeetCode 推送。",
                reason="disable_leetcode_plan",
            )

        if intent == IntentName.GET_TODAY_LEETCODE:
            return self._plan(
                classification=classification,
                action=AgentActionName.GET_TODAY_LEETCODE,
                reply="我来读取今天的 LeetCode 推荐。",
                reason="get_today_leetcode",
            )

        if intent == IntentName.RECORD_LEETCODE_RESULT:
            if not slots.get("result"):
                return self._plan(
                    classification=classification,
                    action=AgentActionName.ASK_CLARIFICATION,
                    reply="请说明结果：独立完成、提示后完成、看题解完成、未完成、延期或跳过。",
                    missing_slots=["result"],
                    reason="missing_leetcode_result",
                )
            return self._plan(
                classification=classification,
                action=AgentActionName.RECORD_LEETCODE_RESULT,
                reply="正在记录 LeetCode 训练结果。",
                reason="record_leetcode_result",
            )

        if intent == IntentName.ADD_APPLICATION:
            return self._plan_add_application(classification)

        if intent == IntentName.QUERY_APPLICATION:
            return self._plan(
                classification=classification,
                action=AgentActionName.QUERY_APPLICATION,
                reply="我识别到你想查询投递进度。下一步会读取投递记录。",
                reason="query_application",
            )

        if intent == IntentName.UPDATE_APPLICATION:
            return self._plan_update_application(classification)

        if intent == IntentName.COMPLETE_TASK:
            return self._plan(
                classification=classification,
                action=AgentActionName.COMPLETE_TASK,
                reply=self._build_task_reply("完成", slots),
                need_confirmation=True,
                reason="complete_task",
            )

        if intent == IntentName.POSTPONE_TASK:
            return self._plan(
                classification=classification,
                action=AgentActionName.POSTPONE_TASK,
                reply=self._build_task_reply("延期", slots),
                need_confirmation=True,
                reason="postpone_task",
            )

        if intent == IntentName.ADD_INTERVIEW_REVIEW:
            return self._plan(
                classification=classification,
                action=AgentActionName.CREATE_INTERVIEW_REVIEW,
                reply=self._build_interview_review_reply(slots),
                need_confirmation=True,
                reason="interview_review",
            )

        if intent == IntentName.START_MOCK_INTERVIEW:
            return self._plan(
                classification=classification,
                action=AgentActionName.START_MOCK_INTERVIEW,
                reply=self._build_mock_interview_reply(slots),
                reason="mock_interview",
            )

        if intent == IntentName.START_PROJECT_TRAINING:
            return self._plan(
                classification=classification,
                action=AgentActionName.START_PROJECT_TRAINING,
                reply="正在创建或恢复项目训练会话。",
                reason="start_project_training",
            )

        if intent == IntentName.RESUME_PROJECT_TRAINING:
            return self._plan(
                classification=classification,
                action=AgentActionName.RESUME_PROJECT_TRAINING,
                reply="正在查找未完成的项目训练。",
                reason="resume_project_training",
            )

        if intent == IntentName.GET_PROJECT_TRAINING_SUMMARY:
            return self._plan(
                classification=classification,
                action=AgentActionName.GET_PROJECT_TRAINING_SUMMARY,
                reply="正在读取最近的项目训练总结。",
                reason="get_project_training_summary",
            )

        if intent == IntentName.ANSWER_QUESTION:
            return self._plan(
                classification=classification,
                action=AgentActionName.RECORD_ANSWER,
                reply="我识别到你正在提交一道题或一个面试问题的回答。下一步会交给评分工具判断核心点、复杂度和表达完整度。",
                need_confirmation=True,
                reason="answer_question",
            )

        if intent == IntentName.ASK_HELP:
            return self._plan(
                classification=classification,
                action=AgentActionName.ANSWER_HELP,
                reply="我识别到你是在咨询问题。下一步会直接生成解释或建议，不写入任务、投递或复盘数据。",
                reason="help",
            )

        if intent == IntentName.SUMMARIZE_WEEK:
            return self._plan(
                classification=classification,
                action=AgentActionName.SUMMARIZE_WEEK,
                reply="我识别到你想做周复盘。下一步会汇总本周任务完成情况、投递进展、面试复盘和薄弱点。",
                reason="weekly_summary",
            )

        return self._plan(
            classification=classification,
            action=AgentActionName.NO_OP,
            reply="我还没能明确判断这句话要执行哪个秋招动作。你可以换成类似“今天任务是什么”“新增投递某公司某岗位”或“复盘今天的面试”。",
            reason="unknown",
        )

    # 兼容旧调用方，直接返回 AgentResponse。
    def plan_response(self, classification: IntentClassification) -> AgentResponse:
        return self.plan(classification).to_response()

    # 规划新增投递记录动作。
    def _plan_add_application(self, classification: IntentClassification) -> AgentPlan:
        missing_slots = self._missing_required_slots(
            classification.slots,
            required_slots=["company", "role"],
        )
        if missing_slots:
            return self._plan(
                classification=classification,
                action=AgentActionName.ASK_CLARIFICATION,
                reply=self._build_missing_application_reply(missing_slots),
                missing_slots=missing_slots,
                reason="missing_application_slots",
            )

        return self._plan(
            classification=classification,
            action=AgentActionName.CREATE_APPLICATION,
            reply=self._build_create_application_reply(classification.slots),
            need_confirmation=True,
            reason="create_application",
        )

    # 规划投递状态或面试安排更新动作。
    def _plan_update_application(self, classification: IntentClassification) -> AgentPlan:
        required_slots = ["company"]
        if classification.slots.get("update_type") == "schedule_interview":
            required_slots.extend(["round", "interview_time"])

        missing_slots = self._missing_required_slots(
            classification.slots,
            required_slots=required_slots,
        )
        if (
            classification.slots.get("update_type") == "schedule_interview"
            and classification.slots.get("interview_time")
            and not self._is_specific_interview_time(classification.slots.get("interview_time"))
            and "interview_time" not in missing_slots
        ):
            missing_slots.append("interview_time")

        if missing_slots:
            return self._plan(
                classification=classification,
                action=AgentActionName.ASK_CLARIFICATION,
                reply=self._build_missing_update_application_reply(missing_slots),
                missing_slots=missing_slots,
                reason="missing_update_application_slots",
            )

        return self._plan(
            classification=classification,
            action=AgentActionName.UPDATE_APPLICATION,
            reply=self._build_update_application_reply(classification.slots),
            need_confirmation=True,
            reason="update_application",
        )

    # 创建一份标准 AgentPlan。
    def _plan(
        self,
        classification: IntentClassification,
        action: AgentActionName,
        reply: str,
        need_confirmation: bool = False,
        missing_slots: Optional[List[str]] = None,
        reason: str = "",
    ) -> AgentPlan:
        steps = []
        if action not in {AgentActionName.ASK_CLARIFICATION, AgentActionName.NO_OP, AgentActionName.ANSWER_HELP}:
            steps.append(
                AgentPlanStep(
                    tool_name=action.value,
                    arguments=classification.slots.copy(),
                    reason=reason,
                )
            )
        return AgentPlan(
            intent=classification.intent,
            confidence=classification.confidence,
            action=action,
            reply=reply,
            need_confirmation=need_confirmation,
            slots=classification.slots,
            missing_slots=missing_slots or [],
            steps=steps,
            reason=reason,
        )

    # 将旧的 AgentResponse 包装成 AgentPlan。
    @staticmethod
    def _from_response(response: AgentResponse, reason: str) -> AgentPlan:
        steps = []
        if response.action not in {
            AgentActionName.ASK_CLARIFICATION,
            AgentActionName.NO_OP,
            AgentActionName.ANSWER_HELP,
        }:
            steps.append(
                AgentPlanStep(
                    tool_name=response.action.value,
                    arguments=response.slots.copy(),
                    reason=reason,
                )
            )
        return AgentPlan(
            intent=response.intent,
            confidence=response.confidence,
            action=response.action,
            reply=response.reply,
            need_confirmation=response.need_confirmation,
            slots=response.slots,
            missing_slots=response.missing_slots,
            steps=steps,
            reason=reason,
        )

    # 计算缺失的必要槽位。
    @staticmethod
    def _missing_required_slots(slots: Dict[str, Any], required_slots: List[str]) -> List[str]:
        return [
            slot
            for slot in required_slots
            if slot not in slots or slots.get(slot) is None or slots.get(slot) == ""
        ]

    # 判断面试时间是否可以解析到具体日期时间。
    @staticmethod
    def _is_specific_interview_time(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        return parse_chinese_datetime(value) is not None

    # 构造新增投递缺槽追问。
    @staticmethod
    def _build_missing_application_reply(missing_slots: List[str]) -> str:
        labels = {
            "company": "公司",
            "role": "岗位",
        }
        missing = "、".join(labels.get(slot, slot) for slot in missing_slots)
        return f"我识别到你想新增投递记录，但还缺少{missing}。请补充后我再生成投递创建动作。"

    # 构造更新投递缺槽追问。
    @staticmethod
    def _build_missing_update_application_reply(missing_slots: List[str]) -> str:
        labels = {
            "company": "公司",
            "round": "轮次",
            "interview_time": "具体面试时间（例如明天下午三点）",
            "calendar_reminder": "是否同步飞书日历提醒（回复需要/不需要）",
        }
        missing = "、".join(labels.get(slot, slot) for slot in missing_slots)
        return f"我识别到你想更新投递进度，但还缺少{missing}。请补充后我再生成投递更新动作。"

    # 构造新增投递确认文案。
    @staticmethod
    def _build_create_application_reply(slots: Dict[str, Any]) -> str:
        company = slots["company"]
        role = slots["role"]
        details = [f"公司是 {company}", f"岗位是 {role}"]
        if slots.get("interview_time"):
            details.append(f"时间是 {slots['interview_time']}")
        if slots.get("round"):
            details.append(f"轮次是 {slots['round']}")

        return f"我识别到你要新增投递记录，{ '，'.join(details) }。下一步会写入投递表，执行前需要你确认。"

    # 构造投递更新确认文案。
    @staticmethod
    def _build_update_application_reply(slots: Dict[str, Any]) -> str:
        company = slots.get("company")
        update_type = slots.get("update_type")
        details = []
        if company:
            details.append(f"公司是 {company}")
        if slots.get("round"):
            details.append(f"轮次是 {slots['round']}")
        if slots.get("interview_time"):
            details.append(f"时间是 {slots['interview_time']}")
        if update_type == "schedule_interview" and "calendar_reminder" in slots:
            if slots.get("calendar_reminder"):
                details.append("需要同步飞书日历提醒，默认提前 30 分钟")
            else:
                details.append("不需要同步飞书日历提醒")

        if update_type == "schedule_interview" and details:
            if "calendar_reminder" not in slots:
                return (
                    f"我识别到你要记录面试安排，{ '，'.join(details) }。"
                    "默认会同步飞书日历并提前 30 分钟提醒；如果不需要，可以先回复“不需要提醒”，否则回复“确认”执行。"
                )
            return f"我识别到你要记录面试安排，{ '，'.join(details) }。下一步会更新投递进度并记录面试安排，执行前需要你确认。"
        if update_type == "pass_round" and details:
            return f"我识别到你要记录面试/笔试通过，{ '，'.join(details) }。下一步会更新投递进度，执行前需要你确认。"
        if update_type == "offer" and company:
            return f"我识别到你拿到了 {company} 的 Offer。下一步会更新投递进度，执行前需要你确认。"
        if update_type == "reject" and company:
            return f"我识别到 {company} 这条投递已结束或未通过。下一步会更新投递进度，执行前需要你确认。"
        if update_type == "withdraw" and company:
            if slots.get("apply_to_all"):
                return f"我识别到你要放弃 {company} 的所有投递岗位。下一步会把匹配投递标记为已放弃，执行前需要你确认。"
            return f"我识别到你要放弃 {company} 这条投递。下一步会把投递状态标记为已放弃，执行前需要你确认。"
        if company:
            return f"我识别到你想更新 {company} 的投递状态。下一步会定位投递记录并更新状态，执行前需要你确认。"
        return "我识别到你想更新投递状态。下一步需要先定位对应公司或岗位，执行前需要你确认。"

    # 构造任务操作确认文案。
    @staticmethod
    def _build_task_reply(operation: str, slots: Dict[str, Any]) -> str:
        task_title = slots.get("task_title")
        task_type = slots.get("task_type")
        if task_title:
            return f"我识别到你要{operation}任务：{task_title}。下一步会更新任务状态，执行前需要你确认。"
        if task_type:
            return f"我识别到你要{operation}一个 {task_type} 任务。下一步会匹配最近任务并更新状态，执行前需要你确认。"
        return f"我识别到你要{operation}任务。下一步需要匹配具体任务，执行前需要你确认。"

    # 构造面试复盘确认文案。
    @staticmethod
    def _build_interview_review_reply(slots: Dict[str, Any]) -> str:
        if slots.get("company"):
            return f"我识别到你要提交 {slots['company']} 的面试复盘。下一步会提取被问知识点、薄弱点和补强任务，执行前需要你确认。"
        return "我识别到你要提交面试复盘。下一步会提取被问知识点、薄弱点和补强任务，执行前需要你确认。"

    # 构造模拟面试启动文案。
    @staticmethod
    def _build_mock_interview_reply(slots: Dict[str, Any]) -> str:
        project = slots.get("project")
        role = slots.get("role")
        if project and role:
            return f"我识别到你想开始模拟面试，岗位是 {role}，项目是 {project}。下一步会进入文本模拟面试流程。"
        if project:
            return f"我识别到你想围绕 {project} 开始模拟面试。下一步会进入项目深挖模拟面试流程。"
        return "我识别到你想开始模拟面试。下一步会先确认岗位、项目或面试类型。"


# LLM-first Planner：让模型生成计划，后端负责校验，规则 Planner 负责兜底。
class AgentPlanner:
    SYSTEM_PROMPT = """
你是 OfferPilot 的 Agent Planner，不是聊天机器人。
你的任务是把用户消息规划成一个严格 JSON 的 AgentPlan，不要执行工具，不要编造工具。

只允许返回一个 JSON object，不要 Markdown，不要解释。

可选 intent:
- get_today_tasks
- enable_leetcode_plan
- disable_leetcode_plan
- get_today_leetcode
- record_leetcode_result
- complete_task
- postpone_task
- add_application
- query_application
- update_application
- add_interview_review
- start_mock_interview
- start_project_training
- resume_project_training
- get_project_training_summary
- answer_question
- ask_help
- summarize_week
- unknown

可选 action:
- list_today_tasks
- enable_leetcode_plan
- disable_leetcode_plan
- get_today_leetcode
- record_leetcode_result
- create_application
- query_application
- update_application
- reschedule_interview
- cancel_interview
- complete_task
- postpone_task
- create_interview_review
- start_mock_interview
- start_project_training
- resume_project_training
- get_project_training_summary
- record_answer
- answer_help
- summarize_week
- ask_clarification
- no_op

规划规则：
1. 只能选择可用工具列表里的工具；没有合适工具时用 answer_help、ask_clarification 或 no_op。
2. 新增投递、更新投递、完成任务、延期任务、创建复盘属于写操作，need_confirmation 必须为 true。
3. 查询今日任务、查询投递、查询面试安排属于读操作，need_confirmation 为 false，可以直接执行。
4. 用户问“我有哪些面试/最近有什么面试/查看面试安排/笔试面试安排”时，使用 query_application，slots.query_type = "upcoming_interviews"。
5. 用户问“我投了哪些/投递记录/投递列表”时，使用 query_application，slots.query_type = "list"。
6. 用户问某家公司进度时，使用 query_application，slots.query_type = "company_status"，并抽取 company。
7. 用户说“我投递了某公司某岗位”时，使用 create_application，必须抽取 company 和 role。
8. 用户说“某公司约我一面/明天三点某公司面试”时，使用 update_application，slots.update_type = "schedule_interview"，必须抽取 company、round、interview_time、status。
8.1 用户要求面试改期时使用 reschedule_interview；取消已有面试时使用 cancel_interview。两者都属于写操作。
9. 具体面试时间必须包含日期/相对日期和小时，例如“明天下午三点”。只有“明天下午”不够，必须 ask_clarification，missing_slots 包含 interview_time。
10. 普通问候、你是谁、你能做什么，使用 answer_help，不要进入写操作。
11. 如果用户追问“先准备哪个/哪个更急/怎么排序”，并且 recent_context.tool_result 里有近期面试列表，使用 answer_help，直接基于最近工具结果给优先级建议，不要追问 tasks_to_prioritize 之类不存在的字段。
12. 如果需要追问，action 使用 ask_clarification，steps 为空，missing_slots 只能使用后端已定义的槽位，例如 company、role、round、interview_time、task_title、task_type。不要创造新槽位。
13. “开启每日刷题/关闭每日刷题/今天刷什么”分别使用对应 LeetCode action。
14. 用户明确回复“第1题独立完成/提示后完成/看题解完成/没做出来/延期/跳过”时，使用 record_leetcode_result，抽取 problem_index 和 result，回复本身就是确认，不再二次确认。

AgentPlan JSON 字段：
{
  "intent": "query_application",
  "confidence": 0.9,
  "action": "query_application",
  "reply": "我来查看你近期的笔试/面试安排。",
  "need_confirmation": false,
  "slots": {"query_type": "upcoming_interviews"},
  "missing_slots": [],
  "steps": [{"tool_name": "query_application", "arguments": {"query_type": "upcoming_interviews"}, "reason": "query upcoming interviews"}],
  "reason": "..."
}
""".strip()

    _TOOL_LESS_ACTIONS = {
        AgentActionName.ASK_CLARIFICATION,
        AgentActionName.NO_OP,
        AgentActionName.ANSWER_HELP,
    }

    # 初始化 LLM Planner 及规则兜底 Planner。
    def __init__(
        self,
        application_dialogue: Optional[ApplicationDialogueManager] = None,
        llm_service: Optional[LLMService] = None,
        llm_planner_enabled: Optional[bool] = None,
        fallback_enabled: Optional[bool] = None,
    ) -> None:
        self.rule_planner = RuleBasedAgentPlanner(application_dialogue=application_dialogue)
        self.llm_service = llm_service or LLMService()
        self.llm_planner_enabled = (
            settings.llm_planner_enabled if llm_planner_enabled is None else llm_planner_enabled
        )
        self.fallback_enabled = (
            settings.llm_planner_fallback_enabled if fallback_enabled is None else fallback_enabled
        )

    # 根据旧意图识别结果生成 AgentPlan，供兜底路径和旧测试继续使用。
    def plan(self, classification: IntentClassification) -> AgentPlan:
        return self.rule_planner.plan(classification)

    # 兼容旧调用方，直接返回 AgentResponse。
    def plan_response(self, classification: IntentClassification) -> AgentResponse:
        return self.rule_planner.plan_response(classification)

    # 用 LLM 直接从用户原话规划 AgentPlan；失败时返回 None，让 LangGraph 进入旧兜底链路。
    async def plan_message(
        self,
        message: str,
        context: AgentPlannerContext,
        tool_specs: List[ToolSpec],
    ) -> Optional[AgentPlan]:
        rule_classification = IntentClassifier().classify_by_rules(message)
        if rule_classification.intent in {
            IntentName.ENABLE_LEETCODE_PLAN,
            IntentName.DISABLE_LEETCODE_PLAN,
            IntentName.GET_TODAY_LEETCODE,
            IntentName.RECORD_LEETCODE_RESULT,
        }:
            return self.rule_planner.plan(rule_classification)
        if not self.llm_planner_enabled:
            return None
        if isinstance(self.llm_service, LLMService) and not self.llm_service.api_key:
            return None

        prompt = self._build_prompt(
            message=message,
            context=context,
            tool_specs=tool_specs,
        )
        try:
            result = await self.llm_service.generate_text(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=1200,
            )
            raw_plan = self._parse_agent_plan(result.content)
            return self._validate_plan(
                raw_plan,
                message=message,
                context=context,
                tool_specs=tool_specs,
            )
        except (LLMConfigurationError, LLMRequestError, ValueError):
            if self.fallback_enabled:
                return None
            return self._unknown_plan(
                reply="Planner 暂时不可用，我还不能稳定地规划这个动作。请稍后重试。",
                reason="llm_planner_failed",
            )

    # 构造给 LLM 的用户提示。
    def _build_prompt(
        self,
        message: str,
        context: AgentPlannerContext,
        tool_specs: List[ToolSpec],
    ) -> str:
        payload = {
            "user_message": message,
            "conversation": {
                "conversation_id": context.conversation_id,
                "user_id": context.user_id,
                "source": context.source,
                "pending_action": self._serialize_pending_action(context.pending),
                "recent_context": self._serialize_recent_context(context.recent_context),
            },
            "available_tools": [self._dump_model(spec) for spec in tool_specs],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    # 从 LLM 文本里解析 AgentPlan JSON。
    def _parse_agent_plan(self, content: str) -> AgentPlan:
        text = content.strip()
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fenced_match:
            text = fenced_match.group(1)
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end < start:
                raise ValueError("LLM planner did not return JSON object.")
            text = text[start : end + 1]

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM planner JSON is invalid.") from exc

        if hasattr(AgentPlan, "model_validate"):
            return AgentPlan.model_validate(data)
        return AgentPlan.parse_obj(data)

    # 校验和修正 LLM 提出的计划，确保不会越权执行或带着缺槽执行。
    def _validate_plan(
        self,
        plan: AgentPlan,
        message: str,
        context: AgentPlannerContext,
        tool_specs: List[ToolSpec],
    ) -> AgentPlan:
        tool_map = {spec.name: spec for spec in tool_specs}
        slots = plan.slots.copy()
        action = plan.action

        if self._is_interview_priority_follow_up(message):
            priority_plan = self._prioritize_recent_interviews_plan(
                context=context,
                confidence=plan.confidence,
            )
            if priority_plan is not None and (
                action in self._TOOL_LESS_ACTIONS or plan.missing_slots
            ):
                return priority_plan

        if action == AgentActionName.QUERY_APPLICATION:
            slots = self._normalize_query_slots(message=message, slots=slots)

        if action == AgentActionName.CREATE_APPLICATION:
            slots = self._normalize_create_application_slots(message=message, slots=slots)

        if action in {
            AgentActionName.UPDATE_APPLICATION,
            AgentActionName.RESCHEDULE_INTERVIEW,
            AgentActionName.CANCEL_INTERVIEW,
        }:
            slots = self._normalize_update_slots(message=message, slots=slots)

        slots = self._inherit_recent_application_focus(
            action=action,
            slots=slots,
            context=context,
        )
        missing_slots = self._sanitize_missing_slots(
            plan=plan,
            slots=slots,
            tool_map=tool_map,
        )
        if any(slots.get(key) for key in ("company", "application_id", "schedule_id")):
            missing_slots = [
                slot
                for slot in missing_slots
                if slot not in {"company", "application_id", "schedule_id"}
            ]

        if action == AgentActionName.ASK_CLARIFICATION and plan.missing_slots and not missing_slots:
            return self._unknown_missing_slot_plan(plan=plan, message=message)

        required_slots = self._required_slots_for_action(action=action, slots=slots, tool_map=tool_map)
        missing_slots.extend(
            slot
            for slot in self.rule_planner._missing_required_slots(slots, required_slots)
            if slot not in missing_slots
        )

        if self._requires_specific_interview_time(action=action, slots=slots):
            interview_time = slots.get("interview_time")
            if (
                interview_time
                and not self.rule_planner._is_specific_interview_time(interview_time)
                and "interview_time" not in missing_slots
            ):
                missing_slots.append("interview_time")

        if missing_slots:
            return self._clarification_plan(plan=plan, slots=slots, missing_slots=missing_slots)

        need_confirmation = plan.need_confirmation
        tool_spec = tool_map.get(action.value)
        if tool_spec and tool_spec.mutating:
            need_confirmation = True

        steps = self._validated_steps(plan=plan, slots=slots, tool_map=tool_map)
        reply = self._confirmation_reply_for_action(action=action, slots=slots) if need_confirmation else ""
        if not reply:
            reply = plan.reply.strip() or self._default_reply_for_action(action, slots)
        return AgentPlan(
            intent=plan.intent,
            confidence=plan.confidence,
            action=action,
            reply=reply,
            need_confirmation=need_confirmation,
            slots=slots,
            missing_slots=[],
            steps=steps,
            reason=plan.reason,
        )

    # 写操作进入待确认状态时，使用后端固定文案，避免 LLM 说成“已经执行”。
    def _confirmation_reply_for_action(
        self,
        action: AgentActionName,
        slots: Dict[str, Any],
    ) -> str:
        if action == AgentActionName.CREATE_APPLICATION:
            return self.rule_planner._build_create_application_reply(slots)
        if action == AgentActionName.UPDATE_APPLICATION:
            return self.rule_planner._build_update_application_reply(slots)
        if action in {AgentActionName.RESCHEDULE_INTERVIEW, AgentActionName.CANCEL_INTERVIEW}:
            return self.rule_planner._build_update_application_reply(slots)
        if action == AgentActionName.COMPLETE_TASK:
            return self.rule_planner._build_task_reply("完成", slots)
        if action == AgentActionName.POSTPONE_TASK:
            return self.rule_planner._build_task_reply("延期", slots)
        if action == AgentActionName.CREATE_INTERVIEW_REVIEW:
            return self.rule_planner._build_interview_review_reply(slots)
        return ""

    # 根据用户原话补齐查询类默认参数。
    @staticmethod
    def _normalize_query_slots(message: str, slots: Dict[str, Any]) -> Dict[str, Any]:
        normalized = slots.copy()
        compact = re.sub(r"\s+", "", message).lower()
        if "query_type" not in normalized:
            if any(word in compact for word in ("面试", "笔试", "约面", "安排")):
                normalized["query_type"] = "upcoming_interviews"
            elif any(word in compact for word in ("投了哪些", "投过哪些", "投递记录", "投递列表")):
                normalized["query_type"] = "list"
            elif normalized.get("company"):
                normalized["query_type"] = "company_status"
            else:
                normalized["query_type"] = "list"
        return normalized

    # 用户省略公司时，沿用当前会话最近一次明确定位的投递记录。
    @staticmethod
    def _inherit_recent_application_focus(
        action: AgentActionName,
        slots: Dict[str, Any],
        context: AgentPlannerContext,
    ) -> Dict[str, Any]:
        normalized = slots.copy()
        if action not in {
            AgentActionName.UPDATE_APPLICATION,
            AgentActionName.RESCHEDULE_INTERVIEW,
            AgentActionName.CANCEL_INTERVIEW,
        }:
            return normalized
        if any(normalized.get(key) for key in ("company", "application_id", "schedule_id")):
            return normalized
        if context.recent_context is None:
            return normalized

        focus = context.recent_context.application_focus
        for key in ("application_id", "company", "role"):
            value = focus.get(key)
            if value:
                normalized[key] = value
        return normalized

    # 标准化新增投递类参数，修正 LLM 偶发的公司/岗位错槽位。
    @staticmethod
    def _normalize_create_application_slots(message: str, slots: Dict[str, Any]) -> Dict[str, Any]:
        normalized = slots.copy()
        rule_slots = IntentClassifier()._extract_application_slots(message)
        rule_company = rule_slots.get("company")
        rule_role = rule_slots.get("role")

        company = str(normalized.get("company") or "").strip()
        role = str(normalized.get("role") or "").strip()
        if rule_company and (
            not company
            or AgentPlanner._looks_like_role_value(company)
            or (role and company.replace(" ", "").startswith(role.replace(" ", "")))
        ):
            normalized["company"] = rule_company

        if rule_role and not role:
            normalized["role"] = rule_role

        if "interview_time" in normalized and not AgentPlanner._has_interview_timing_context(message):
            normalized.pop("interview_time", None)
            normalized.pop("round", None)

        return normalized

    # 标准化更新投递类参数。
    @staticmethod
    def _normalize_update_slots(message: str, slots: Dict[str, Any]) -> Dict[str, Any]:
        normalized = slots.copy()
        rule_slots = IntentClassifier()._extract_application_update_slots(message)
        for key in ("company", "role", "round", "interview_time", "update_type", "status", "apply_to_all"):
            if key not in normalized and key in rule_slots:
                normalized[key] = rule_slots[key]

        update_type = normalized.get("update_type")
        if update_type is None and (normalized.get("interview_time") or normalized.get("round")):
            normalized["update_type"] = "schedule_interview"
        if normalized.get("update_type") == "withdraw" and "status" not in normalized:
            normalized["status"] = "withdrawn"
        if normalized.get("update_type") == "schedule_interview" and "status" not in normalized:
            round_name = str(normalized.get("round", ""))
            if "二面" in round_name:
                normalized["status"] = "interview_2"
            elif "三面" in round_name:
                normalized["status"] = "interview_3"
            else:
                normalized["status"] = "interview_1"
        return normalized

    # 计算某个动作真正需要的槽位。
    @staticmethod
    def _required_slots_for_action(
        action: AgentActionName,
        slots: Dict[str, Any],
        tool_map: Dict[str, ToolSpec],
    ) -> List[str]:
        if action == AgentActionName.CREATE_APPLICATION:
            return ["company", "role"]
        if action == AgentActionName.RECORD_LEETCODE_RESULT:
            return ["result"]
        if action == AgentActionName.UPDATE_APPLICATION:
            required = []
            if not any(slots.get(slot) for slot in ("company", "application_id")):
                required.append("company")
            if slots.get("update_type") == "schedule_interview":
                required.extend(["round", "interview_time"])
            return required
        if action == AgentActionName.RESCHEDULE_INTERVIEW:
            required = ["interview_time"]
            if not any(slots.get(slot) for slot in ("company", "application_id", "schedule_id")):
                required.append("company")
            return required
        if action == AgentActionName.CANCEL_INTERVIEW:
            return [] if any(slots.get(slot) for slot in ("company", "application_id", "schedule_id")) else ["company"]
        spec = tool_map.get(action.value)
        return spec.required_slots if spec else []

    # 过滤 LLM 编造的 missing slot，避免把内部字段暴露给用户。
    def _sanitize_missing_slots(
        self,
        plan: AgentPlan,
        slots: Dict[str, Any],
        tool_map: Dict[str, ToolSpec],
    ) -> List[str]:
        allowed_slots = self._allowed_missing_slots(plan=plan, slots=slots, tool_map=tool_map)
        return [slot for slot in plan.missing_slots if slot in allowed_slots]

    # 根据动作和意图确定允许追问的槽位。
    def _allowed_missing_slots(
        self,
        plan: AgentPlan,
        slots: Dict[str, Any],
        tool_map: Dict[str, ToolSpec],
    ) -> set[str]:
        if plan.intent == IntentName.ADD_APPLICATION or plan.action == AgentActionName.CREATE_APPLICATION:
            return {"company", "role", "round", "interview_time"}
        if plan.intent == IntentName.UPDATE_APPLICATION or plan.action in {
            AgentActionName.UPDATE_APPLICATION,
            AgentActionName.RESCHEDULE_INTERVIEW,
            AgentActionName.CANCEL_INTERVIEW,
        }:
            return {
                "company", "role", "round", "interview_time", "calendar_reminder", "application_id", "schedule_id"
            }
        if plan.action in {AgentActionName.COMPLETE_TASK, AgentActionName.POSTPONE_TASK}:
            return {"task_title", "task_type"}
        if plan.action == AgentActionName.RECORD_LEETCODE_RESULT:
            return {"assignment_id", "problem_index", "problem_title", "result"}
        if plan.action == AgentActionName.CREATE_INTERVIEW_REVIEW:
            return {"company", "round", "topics"}
        return set(self._required_slots_for_action(plan.action, slots, tool_map))

    # 判断面试更新时间是否必须可被解析成具体时间。
    @staticmethod
    def _requires_specific_interview_time(action: AgentActionName, slots: Dict[str, Any]) -> bool:
        return action == AgentActionName.RESCHEDULE_INTERVIEW or (
            action == AgentActionName.UPDATE_APPLICATION and slots.get("update_type") == "schedule_interview"
        )

    # 判断槽位值是否更像岗位而不是公司。
    @staticmethod
    def _looks_like_role_value(value: str) -> bool:
        compact = re.sub(r"\s+", "", value).lower()
        return any(
            word in compact
            for word in (
                "岗位",
                "职位",
                "方向",
                "java后端",
                "java开发",
                "python后端",
                "ai应用开发",
                "agent开发",
                "后端开发",
                "开发实习",
                "算法工程师",
                "实习生",
            )
        )

    # 判断新增投递语句里的时间是否真的在描述笔试/面试安排。
    @staticmethod
    def _has_interview_timing_context(message: str) -> bool:
        compact = re.sub(r"\s+", "", message).lower()
        return any(word in compact for word in ("面试", "笔试", "一面", "二面", "三面", "hr面", "约我", "安排"))

    # 判断用户是否在追问上一轮面试列表的优先级。
    @staticmethod
    def _is_interview_priority_follow_up(message: str) -> bool:
        compact = re.sub(r"\s+", "", message).lower()
        return any(
            phrase in compact
            for phrase in (
                "先准备哪一个",
                "先准备哪个",
                "优先准备",
                "哪个先准备",
                "哪一个先准备",
                "哪个更急",
                "哪个最急",
                "怎么排序",
                "准备顺序",
                "先看哪个",
            )
        )

    # 基于上一轮查询到的面试列表生成优先级建议。
    def _prioritize_recent_interviews_plan(
        self,
        context: AgentPlannerContext,
        confidence: float,
    ) -> Optional[AgentPlan]:
        items = self._extract_recent_interview_items(context.recent_context)
        if not items:
            return None

        ranked_items = sorted(
            enumerate(items),
            key=lambda item: (self._interview_sort_timestamp(item[1]), item[0]),
        )
        lines = ["我建议按“时间最近优先，其次看目标公司/岗位匹配度”来排："]
        for index, (_, item) in enumerate(ranked_items[:5], start=1):
            company = item.get("company") or "公司待补充"
            role = item.get("role") or "岗位待补充"
            round_name = item.get("round") or "面试"
            time_text = item.get("time_text") or "时间待补充"
            if index == 1:
                reason = "时间最靠前，先准备它"
            elif self._interview_sort_timestamp(item) == self._interview_sort_timestamp(ranked_items[0][1]):
                reason = "时间也很近，排在同一优先级"
            else:
                reason = "排在后面继续准备"
            lines.append(f"{index}. {company} - {role}（{round_name}，{time_text}）：{reason}")

        if len(ranked_items) > 5:
            lines.append("剩下的可以等最近一两场准备完后再排。")
        lines.append("如果这些时间里有你最想去的公司，可以把它上调一档。")

        return AgentPlan(
            intent=IntentName.ASK_HELP,
            confidence=max(confidence, 0.78),
            action=AgentActionName.ANSWER_HELP,
            reply="\n".join(lines),
            need_confirmation=False,
            slots={"based_on": "recent_upcoming_interviews"},
            missing_slots=[],
            steps=[],
            reason="prioritize_recent_interviews",
        )

    # 提取最近工具结果中的面试数据。
    @staticmethod
    def _extract_recent_interview_items(
        recent_context: Optional[RecentAgentContext],
    ) -> List[Dict[str, Any]]:
        if recent_context is None or not recent_context.tool_result:
            return []

        tool_result = recent_context.tool_result
        if tool_result.get("tool_name") != AgentActionName.QUERY_APPLICATION.value:
            return []

        data = tool_result.get("data") or {}
        if data.get("query_type") != "upcoming_interviews":
            return []

        items: List[Dict[str, Any]] = []
        for schedule in data.get("interview_schedules") or []:
            items.append(
                {
                    "company": schedule.get("company"),
                    "role": schedule.get("role"),
                    "round": schedule.get("round"),
                    "time_text": schedule.get("start_time"),
                    "start_at": schedule.get("start_at"),
                }
            )

        if items:
            return items

        for application in data.get("applications") or []:
            if not application.get("interview_time") and not application.get("round"):
                continue
            items.append(
                {
                    "company": application.get("company"),
                    "role": application.get("role"),
                    "round": application.get("round"),
                    "time_text": application.get("interview_time"),
                    "start_at": None,
                }
            )
        return items

    # 将面试时间转换成可排序时间戳，无法解析的排到最后。
    @staticmethod
    def _interview_sort_timestamp(item: Dict[str, Any]) -> float:
        start_at = item.get("start_at")
        if isinstance(start_at, str) and start_at:
            try:
                return datetime.fromisoformat(start_at.replace("Z", "+00:00")).timestamp()
            except ValueError:
                pass

        time_text = item.get("time_text")
        if isinstance(time_text, str) and time_text:
            parsed = parse_chinese_datetime(time_text)
            if parsed is not None:
                return parsed.timestamp()
        return float("inf")

    # 模型追问了不存在的字段时，改成用户可理解的回复。
    @staticmethod
    def _unknown_missing_slot_plan(plan: AgentPlan, message: str) -> AgentPlan:
        return AgentPlan(
            intent=IntentName.ASK_HELP,
            confidence=min(plan.confidence, 0.65),
            action=AgentActionName.ANSWER_HELP,
            reply=(
                "我理解你是在继续讨论上一个结果，但这句话不需要补内部字段。"
                "如果你想让我排序，请先让我查看一次面试安排，或者直接把几个面试的公司和时间发给我。"
            ),
            need_confirmation=False,
            slots={"raw_message": message},
            missing_slots=[],
            steps=[],
            reason="unknown_missing_slot_sanitized",
        )

    # 从计划中筛出可信步骤，并自动补齐单步工具调用。
    def _validated_steps(
        self,
        plan: AgentPlan,
        slots: Dict[str, Any],
        tool_map: Dict[str, ToolSpec],
    ) -> List[AgentPlanStep]:
        if plan.action in self._TOOL_LESS_ACTIONS:
            return []
        if plan.action.value not in tool_map:
            return []

        valid_steps = [
            AgentPlanStep(
                tool_name=step.tool_name,
                arguments=step.arguments.copy(),
                reason=step.reason,
            )
            for step in plan.steps
            if step.tool_name == plan.action.value and step.tool_name in tool_map
        ]
        if valid_steps:
            valid_steps[0].arguments.update(slots)
            return valid_steps

        return [
            AgentPlanStep(
                tool_name=plan.action.value,
                arguments=slots.copy(),
                reason=plan.reason,
            )
        ]

    # 将缺槽计划改写成追问计划，避免带着不完整信息调用工具。
    def _clarification_plan(
        self,
        plan: AgentPlan,
        slots: Dict[str, Any],
        missing_slots: List[str],
    ) -> AgentPlan:
        if plan.intent == IntentName.ADD_APPLICATION:
            reply = self.rule_planner._build_missing_application_reply(missing_slots)
        elif plan.intent == IntentName.UPDATE_APPLICATION:
            reply = self.rule_planner._build_missing_update_application_reply(missing_slots)
        else:
            labels = {
                "company": "公司",
                "role": "岗位",
                "round": "轮次",
                "interview_time": "具体面试时间（例如明天下午三点）",
                "task_title": "任务名称",
                "task_type": "任务类型",
                "problem_index": "第几题",
                "problem_title": "题目名称",
                "result": "做题结果",
            }
            missing = "、".join(labels.get(slot, slot) for slot in missing_slots)
            reply = f"我识别到你想执行这个秋招动作，但还缺少{missing}。请补充后我再继续。"

        return AgentPlan(
            intent=plan.intent,
            confidence=plan.confidence,
            action=AgentActionName.ASK_CLARIFICATION,
            reply=reply,
            need_confirmation=False,
            slots=slots,
            missing_slots=missing_slots,
            steps=[],
            reason="missing_required_slots",
        )

    # 构造无可执行动作时的稳定计划。
    @staticmethod
    def _unknown_plan(reply: str, reason: str) -> AgentPlan:
        return AgentPlan(
            intent=IntentName.UNKNOWN,
            confidence=0.2,
            action=AgentActionName.NO_OP,
            reply=reply,
            need_confirmation=False,
            slots={},
            missing_slots=[],
            steps=[],
            reason=reason,
        )

    # 为 LLM 计划提供默认回复兜底。
    @staticmethod
    def _default_reply_for_action(action: AgentActionName, slots: Dict[str, Any]) -> str:
        if action == AgentActionName.QUERY_APPLICATION:
            if slots.get("query_type") == "upcoming_interviews":
                return "我来查看你近期的笔试/面试安排。"
            return "我来查看你的投递记录。"
        if action == AgentActionName.LIST_TODAY_TASKS:
            return "我来查看今天的秋招任务。"
        return "我已经理解你的请求，下一步会按计划处理。"

    # 序列化 PendingAction，避免把 Python 对象原样塞给 LLM。
    @staticmethod
    def _serialize_pending_action(pending: Optional[PendingAgentAction]) -> Optional[Dict[str, Any]]:
        if pending is None:
            return None
        return {
            "intent": pending.intent.value,
            "action": pending.action.value,
            "reply": pending.reply,
            "need_confirmation": pending.need_confirmation,
            "slots": pending.slots,
            "missing_slots": pending.missing_slots,
            "original_message": pending.original_message,
        }

    # 序列化最近一轮上下文，并限制列表长度，避免 Planner prompt 过大。
    @staticmethod
    def _serialize_recent_context(
        recent_context: Optional[RecentAgentContext],
    ) -> Optional[Dict[str, Any]]:
        if recent_context is None:
            return None

        payload = recent_context.to_prompt_context()
        tool_result = payload.get("tool_result")
        if isinstance(tool_result, dict):
            data = tool_result.get("data")
            if isinstance(data, dict):
                for list_key in ("applications", "interview_schedules", "tasks"):
                    value = data.get(list_key)
                    if isinstance(value, list) and len(value) > 10:
                        data[list_key] = value[:10]
        return payload

    # 兼容 Pydantic v1/v2 的模型转 dict。
    @staticmethod
    def _dump_model(model: Any) -> Dict[str, Any]:
        if hasattr(model, "model_dump"):
            return model.model_dump()
        return model.dict()
