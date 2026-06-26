from typing import Any, Dict, List, Optional

from app.agents.intent_classifier import IntentClassifier
from app.schemas.agent import AgentActionName, AgentResponse
from app.schemas.intent import IntentClassification, IntentName
from app.schemas.tool import ToolResult
from app.tools.offerpilot_tools import get_default_tool_registry
from app.tools.registry import ToolRegistry


class AgentOrchestrator:
    def __init__(
        self,
        intent_classifier: Optional[IntentClassifier] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.intent_classifier = intent_classifier or IntentClassifier()
        self.tool_registry = tool_registry or get_default_tool_registry()

    async def handle_message(self, message: str, confirmed: bool = False) -> AgentResponse:
        classification = await self.intent_classifier.classify(message)
        planned_response = self._plan_response(classification)
        return self._maybe_execute_tool(planned_response, message=message, confirmed=confirmed)

    def _plan_response(self, classification: IntentClassification) -> AgentResponse:
        intent = classification.intent
        slots = classification.slots

        if intent == IntentName.GET_TODAY_TASKS:
            return self._response(
                classification=classification,
                action=AgentActionName.LIST_TODAY_TASKS,
                reply="我识别到你想查看今日任务。下一步会调用任务工具读取今天的 LeetCode、八股、项目深挖和投递安排。",
            )

        if intent == IntentName.ADD_APPLICATION:
            return self._handle_add_application(classification)

        if intent == IntentName.UPDATE_APPLICATION:
            return self._response(
                classification=classification,
                action=AgentActionName.UPDATE_APPLICATION,
                reply=self._build_update_application_reply(slots),
                need_confirmation=True,
            )

        if intent == IntentName.COMPLETE_TASK:
            return self._response(
                classification=classification,
                action=AgentActionName.COMPLETE_TASK,
                reply=self._build_task_reply("完成", slots),
                need_confirmation=True,
            )

        if intent == IntentName.POSTPONE_TASK:
            return self._response(
                classification=classification,
                action=AgentActionName.POSTPONE_TASK,
                reply=self._build_task_reply("延期", slots),
                need_confirmation=True,
            )

        if intent == IntentName.ADD_INTERVIEW_REVIEW:
            return self._response(
                classification=classification,
                action=AgentActionName.CREATE_INTERVIEW_REVIEW,
                reply=self._build_interview_review_reply(slots),
                need_confirmation=True,
            )

        if intent == IntentName.START_MOCK_INTERVIEW:
            return self._response(
                classification=classification,
                action=AgentActionName.START_MOCK_INTERVIEW,
                reply=self._build_mock_interview_reply(slots),
            )

        if intent == IntentName.ANSWER_QUESTION:
            return self._response(
                classification=classification,
                action=AgentActionName.RECORD_ANSWER,
                reply="我识别到你正在提交一道题或一个面试问题的回答。下一步会交给评分工具判断核心点、复杂度和表达完整度。",
                need_confirmation=True,
            )

        if intent == IntentName.ASK_HELP:
            return self._response(
                classification=classification,
                action=AgentActionName.ANSWER_HELP,
                reply="我识别到你是在咨询问题。下一步会直接生成解释或建议，不写入任务、投递或复盘数据。",
            )

        if intent == IntentName.SUMMARIZE_WEEK:
            return self._response(
                classification=classification,
                action=AgentActionName.SUMMARIZE_WEEK,
                reply="我识别到你想做周复盘。下一步会汇总本周任务完成情况、投递进展、面试复盘和薄弱点。",
            )

        return self._response(
            classification=classification,
            action=AgentActionName.NO_OP,
            reply="我还没能明确判断这句话要执行哪个秋招动作。你可以换成类似“今天任务是什么”“新增投递某公司某岗位”或“复盘今天的面试”。",
        )

    def _handle_add_application(self, classification: IntentClassification) -> AgentResponse:
        missing_slots = self._missing_required_slots(
            classification.slots,
            required_slots=["company", "role"],
        )
        if missing_slots:
            return self._response(
                classification=classification,
                action=AgentActionName.ASK_CLARIFICATION,
                reply=self._build_missing_application_reply(missing_slots),
                missing_slots=missing_slots,
            )

        return self._response(
            classification=classification,
            action=AgentActionName.CREATE_APPLICATION,
            reply=self._build_create_application_reply(classification.slots),
            need_confirmation=True,
        )

    def _maybe_execute_tool(
        self,
        response: AgentResponse,
        message: str,
        confirmed: bool,
    ) -> AgentResponse:
        tool_name = response.action.value
        if not self.tool_registry.has_tool(tool_name):
            return response

        if response.need_confirmation and not confirmed:
            return response

        arguments = response.slots.copy()
        arguments["raw_message"] = message
        tool_result = self.tool_registry.run(tool_name, arguments)
        return self._with_tool_result(response, tool_result)

    @staticmethod
    def _with_tool_result(response: AgentResponse, tool_result: ToolResult) -> AgentResponse:
        return AgentResponse(
            intent=response.intent,
            confidence=response.confidence,
            action=response.action,
            reply=tool_result.message,
            need_confirmation=False,
            slots=response.slots,
            missing_slots=response.missing_slots,
            tool_result=tool_result,
        )

    @staticmethod
    def _response(
        classification: IntentClassification,
        action: AgentActionName,
        reply: str,
        need_confirmation: bool = False,
        missing_slots: Optional[List[str]] = None,
    ) -> AgentResponse:
        return AgentResponse(
            intent=classification.intent,
            confidence=classification.confidence,
            action=action,
            reply=reply,
            need_confirmation=need_confirmation,
            slots=classification.slots,
            missing_slots=missing_slots or [],
        )

    @staticmethod
    def _missing_required_slots(slots: Dict[str, Any], required_slots: List[str]) -> List[str]:
        return [slot for slot in required_slots if not slots.get(slot)]

    @staticmethod
    def _build_missing_application_reply(missing_slots: List[str]) -> str:
        labels = {
            "company": "公司",
            "role": "岗位",
        }
        missing = "、".join(labels.get(slot, slot) for slot in missing_slots)
        return f"我识别到你想新增投递记录，但还缺少{missing}。请补充后我再生成投递创建动作。"

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

    @staticmethod
    def _build_update_application_reply(slots: Dict[str, Any]) -> str:
        if slots.get("company"):
            return f"我识别到你想更新 {slots['company']} 的投递状态。下一步会定位投递记录并更新状态，执行前需要你确认。"
        return "我识别到你想更新投递状态。下一步需要先定位对应公司或岗位，执行前需要你确认。"

    @staticmethod
    def _build_task_reply(operation: str, slots: Dict[str, Any]) -> str:
        task_title = slots.get("task_title")
        task_type = slots.get("task_type")
        if task_title:
            return f"我识别到你要{operation}任务：{task_title}。下一步会更新任务状态，执行前需要你确认。"
        if task_type:
            return f"我识别到你要{operation}一个 {task_type} 任务。下一步会匹配最近任务并更新状态，执行前需要你确认。"
        return f"我识别到你要{operation}任务。下一步需要匹配具体任务，执行前需要你确认。"

    @staticmethod
    def _build_interview_review_reply(slots: Dict[str, Any]) -> str:
        if slots.get("company"):
            return f"我识别到你要提交 {slots['company']} 的面试复盘。下一步会提取被问知识点、薄弱点和补强任务，执行前需要你确认。"
        return "我识别到你要提交面试复盘。下一步会提取被问知识点、薄弱点和补强任务，执行前需要你确认。"

    @staticmethod
    def _build_mock_interview_reply(slots: Dict[str, Any]) -> str:
        project = slots.get("project")
        role = slots.get("role")
        if project and role:
            return f"我识别到你想开始模拟面试，岗位是 {role}，项目是 {project}。下一步会进入文本模拟面试流程。"
        if project:
            return f"我识别到你想围绕 {project} 开始模拟面试。下一步会进入项目深挖模拟面试流程。"
        return "我识别到你想开始模拟面试。下一步会先确认岗位、项目或面试类型。"
