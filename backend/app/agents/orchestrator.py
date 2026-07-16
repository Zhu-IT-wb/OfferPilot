import re
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.agents.application_dialogue import ApplicationDialogueManager
from app.agents.conversation import (
    InMemoryConversationStore,
    PendingAgentAction,
    RecentAgentContext,
    get_default_conversation_store,
)
from app.agents.general_responder import GeneralResponder
from app.agents.intent_classifier import IntentClassifier
from app.agents.message_router import MessageRouter, MessageRoute, MessageRouteName
from app.agents.planner import AgentPlanner, AgentPlannerContext
from app.schemas.agent import AgentActionName, AgentResponse
from app.schemas.agent_plan import AgentPlan, AgentPlanStep
from app.schemas.intent import IntentClassification, IntentName
from app.schemas.tool import ToolResult
from app.services.feishu_service import parse_chinese_datetime
from app.tools.offerpilot_tools import get_default_tool_registry
from app.tools.registry import ToolRegistry


# 定义 LangGraph 在一次消息处理中流转的共享状态。
class AgentGraphState(TypedDict, total=False):
    message: str
    confirmed: bool
    user_id: str
    source: str
    conversation_id: str
    pending: Optional[PendingAgentAction]
    recent_context: Optional[RecentAgentContext]
    route: MessageRoute
    classification: IntentClassification
    agent_plan: AgentPlan
    planned_response: AgentResponse
    response: AgentResponse
    original_message: str
    pending_handled: bool
    planner_handled: bool


# 编排意图识别、会话状态、工具执行和最终回复。
class AgentOrchestrator:
    # 初始化当前组件所需的依赖和配置。
    def __init__(
        self,
        intent_classifier: Optional[IntentClassifier] = None,
        message_router: Optional[MessageRouter] = None,
        general_responder: Optional[GeneralResponder] = None,
        tool_registry: Optional[ToolRegistry] = None,
        conversation_store: Optional[InMemoryConversationStore] = None,
        application_dialogue: Optional[ApplicationDialogueManager] = None,
        planner: Optional[AgentPlanner] = None,
    ) -> None:
        self.intent_classifier = intent_classifier or IntentClassifier()
        self.message_router = message_router or MessageRouter()
        self.general_responder = general_responder or GeneralResponder()
        self.tool_registry = tool_registry or get_default_tool_registry()
        self.conversation_store = conversation_store or get_default_conversation_store()
        self.application_dialogue = application_dialogue or ApplicationDialogueManager()
        llm_planner_enabled = None
        if planner is None and intent_classifier is not None:
            llm_planner_enabled = False
        self.planner = planner or AgentPlanner(
            application_dialogue=self.application_dialogue,
            llm_planner_enabled=llm_planner_enabled,
        )
        self._graph = self._build_graph()

    # 处理一条用户消息的完整 Agent 流程。
    async def handle_message(
        self,
        message: str,
        confirmed: bool = False,
        user_id: str = "local_user",
        source: str = "api",
    ) -> AgentResponse:
        final_state = await self._graph.ainvoke(
            {
                "message": message,
                "confirmed": confirmed,
                "user_id": user_id,
                "source": source,
                "original_message": message,
            }
        )
        response = final_state.get("response")
        if response is not None:
            return response

        return self._manual_response(
            intent=IntentName.UNKNOWN,
            confidence=0.2,
            action=AgentActionName.NO_OP,
            reply="Agent 图执行结束，但没有生成可返回的响应。请稍后重试。",
        )

    # 构建 OfferPilot 消息处理图。
    def _build_graph(self):
        graph = StateGraph(AgentGraphState)
        graph.add_node("load_context", self._graph_load_context)
        graph.add_node("cancel_pending", self._graph_cancel_pending)
        graph.add_node("confirm_pending", self._graph_confirm_pending)
        graph.add_node("handle_pending", self._graph_handle_pending)
        graph.add_node("plan_message", self._graph_plan_message)
        graph.add_node("route_message", self._graph_route_message)
        graph.add_node("handle_general", self._graph_handle_general)
        graph.add_node("classify_intent", self._graph_classify_intent)
        graph.add_node("plan_response", self._graph_plan_response)
        graph.add_node("execute_tool", self._graph_execute_tool)
        graph.add_node("sync_pending", self._graph_sync_pending)

        graph.add_edge(START, "load_context")
        graph.add_conditional_edges(
            "load_context",
            self._graph_route_after_context,
            {
                "cancel": "cancel_pending",
                "confirm": "confirm_pending",
                "pending": "handle_pending",
                "plan": "plan_message",
            },
        )
        graph.add_edge("cancel_pending", END)
        graph.add_edge("confirm_pending", END)
        graph.add_conditional_edges(
            "handle_pending",
            self._graph_route_after_pending,
            {
                "handled": "sync_pending",
                "plan": "plan_message",
            },
        )
        graph.add_conditional_edges(
            "plan_message",
            self._graph_route_after_plan_message,
            {
                "planned": "execute_tool",
                "fallback": "route_message",
            },
        )
        graph.add_conditional_edges(
            "route_message",
            self._graph_route_after_message_route,
            {
                "general": "handle_general",
                "business": "classify_intent",
            },
        )
        graph.add_edge("handle_general", END)
        graph.add_edge("classify_intent", "plan_response")
        graph.add_edge("plan_response", "execute_tool")
        graph.add_edge("execute_tool", "sync_pending")
        graph.add_edge("sync_pending", END)
        return graph.compile()

    # 从会话存储中加载当前用户上下文。
    def _graph_load_context(self, state: AgentGraphState) -> Dict[str, Any]:
        conversation_id = self._conversation_id(
            source=state.get("source", "api"),
            user_id=state.get("user_id", "local_user"),
        )
        return {
            "conversation_id": conversation_id,
            "pending": self.conversation_store.get_pending_action(conversation_id),
            "recent_context": self.conversation_store.get_recent_context(conversation_id),
        }

    # 根据上下文判断下一步进入确认、取消、待确认动作还是新消息路由。
    def _graph_route_after_context(self, state: AgentGraphState) -> str:
        message = state.get("message", "")
        if self._is_cancel_message(message):
            return "cancel"
        if self._is_confirm_message(message):
            return "confirm"
        if state.get("pending") and not state.get("confirmed", False):
            return "pending"
        return "plan"

    # 取消当前会话中的待确认动作。
    def _graph_cancel_pending(self, state: AgentGraphState) -> Dict[str, Any]:
        self.conversation_store.clear_pending_action(state["conversation_id"])
        return {
            "response": self._manual_response(
                intent=IntentName.UNKNOWN,
                confidence=1.0,
                action=AgentActionName.NO_OP,
                reply="已取消当前待确认的秋招动作。",
            )
        }

    # 执行用户对待确认动作的确认。
    def _graph_confirm_pending(self, state: AgentGraphState) -> Dict[str, Any]:
        response = self._handle_confirmation(
            pending=state.get("pending"),
            conversation_id=state["conversation_id"],
        )
        self._store_recent_context(
            conversation_id=state["conversation_id"],
            response=response,
            original_message=state.get("message", ""),
        )
        return {
            "response": response
        }

    # 尝试把当前消息接到上一轮待确认动作上。
    def _graph_handle_pending(self, state: AgentGraphState) -> Dict[str, Any]:
        pending = state.get("pending")
        if pending is None:
            return {"pending_handled": False}

        message = state.get("message", "")
        if self._is_pending_status_question(message):
            response = self._manual_response(
                intent=pending.intent,
                confidence=pending.confidence,
                action=pending.action,
                reply=self._build_pending_status_reply(pending),
                need_confirmation=pending.need_confirmation,
                slots=pending.slots,
                missing_slots=pending.missing_slots,
            )
            return {
                "response": response,
                "original_message": pending.original_message,
                "pending_handled": True,
            }

        application_plan = self.application_dialogue.try_handle_pending(
            pending=pending,
            message=message,
        )
        if application_plan is not None:
            response = self._maybe_execute_tool(
                application_plan.response,
                message=application_plan.original_message,
                confirmed=False,
            )
            return {
                "response": response,
                "original_message": application_plan.original_message,
                "pending_handled": True,
            }

        if pending.missing_slots:
            slot_updates = self._extract_slot_updates(message, pending.missing_slots)
            if slot_updates:
                merged_slots = pending.slots.copy()
                merged_slots.update(slot_updates)
                classification = IntentClassification(
                    intent=pending.intent,
                    confidence=max(pending.confidence, 0.8),
                    slots=merged_slots,
                )
                planned_response = self._plan_response(classification)
                original_message = f"{pending.original_message}\n{message}"
                response = self._maybe_execute_tool(
                    planned_response,
                    message=original_message,
                    confirmed=False,
                )
                return {
                    "response": response,
                    "original_message": original_message,
                    "pending_handled": True,
                }

        return {"pending_handled": False}

    # 根据待确认动作处理结果决定继续补状态还是当作新消息。
    @staticmethod
    def _graph_route_after_pending(state: AgentGraphState) -> str:
        return "handled" if state.get("pending_handled") else "plan"

    # 优先让 LLM Planner 从原始消息生成结构化计划。
    async def _graph_plan_message(self, state: AgentGraphState) -> Dict[str, Any]:
        agent_plan = await self.planner.plan_message(
            message=state.get("message", ""),
            context=AgentPlannerContext(
                conversation_id=state["conversation_id"],
                user_id=state.get("user_id", "local_user"),
                source=state.get("source", "api"),
                pending=state.get("pending"),
                recent_context=state.get("recent_context"),
            ),
            tool_specs=self.tool_registry.describe_tools(),
        )
        if agent_plan is None:
            return {"planner_handled": False}

        agent_plan = self._with_actor_context_for_plan(
            plan=agent_plan,
            user_id=state.get("user_id", "local_user"),
            source=state.get("source", "api"),
        )
        return {
            "agent_plan": agent_plan,
            "planned_response": agent_plan.to_response(),
            "planner_handled": True,
        }

    # 根据 LLM Planner 是否产出计划决定执行或进入旧兜底链路。
    @staticmethod
    def _graph_route_after_plan_message(state: AgentGraphState) -> str:
        return "planned" if state.get("planner_handled") else "fallback"

    # 对新消息做轻量路由，区分普通问答和业务动作。
    def _graph_route_message(self, state: AgentGraphState) -> Dict[str, Any]:
        return {"route": self.message_router.route(state.get("message", ""))}

    # 根据消息路由选择普通回复或业务意图识别。
    @staticmethod
    def _graph_route_after_message_route(state: AgentGraphState) -> str:
        route = state["route"]
        return "business" if route.route == MessageRouteName.JOB_ACTION else "general"

    # 生成闲聊、帮助说明和领域问答响应。
    async def _graph_handle_general(self, state: AgentGraphState) -> Dict[str, Any]:
        response = await self._handle_general_message(
            message=state.get("message", ""),
            route=state["route"],
        )
        self._store_recent_context(
            conversation_id=state["conversation_id"],
            response=response,
            original_message=state.get("message", ""),
        )
        return {
            "response": response
        }

    # 对业务消息执行意图识别，并补充入口上下文。
    async def _graph_classify_intent(self, state: AgentGraphState) -> Dict[str, Any]:
        classification = await self.intent_classifier.classify(state.get("message", ""))
        classification = self._with_actor_context(
            classification=classification,
            user_id=state.get("user_id", "local_user"),
            source=state.get("source", "api"),
        )
        return {"classification": classification}

    # 根据意图识别结果生成 Agent 响应计划。
    def _graph_plan_response(self, state: AgentGraphState) -> Dict[str, Any]:
        agent_plan = self.planner.plan(state["classification"])
        return {
            "agent_plan": agent_plan,
            "planned_response": agent_plan.to_response(),
        }

    # 在计划允许时执行工具。
    def _graph_execute_tool(self, state: AgentGraphState) -> Dict[str, Any]:
        response = self._maybe_execute_tool(
            state["planned_response"],
            message=state.get("message", ""),
            confirmed=state.get("confirmed", False),
        )
        return {"response": response}

    # 将待确认动作写回会话状态。
    def _graph_sync_pending(self, state: AgentGraphState) -> Dict[str, Any]:
        self._sync_pending_action(
            conversation_id=state["conversation_id"],
            response=state["response"],
            original_message=state.get("original_message") or state.get("message", ""),
        )
        self._store_recent_context(
            conversation_id=state["conversation_id"],
            response=state["response"],
            original_message=state.get("original_message") or state.get("message", ""),
        )
        return {}

    # 处理闲聊、帮助说明和领域问答。
    async def _handle_general_message(self, message: str, route: MessageRoute) -> AgentResponse:
        reply = await self.general_responder.respond(message=message, route=route)
        if route.route == MessageRouteName.UNKNOWN:
            return self._manual_response(
                intent=IntentName.UNKNOWN,
                confidence=route.confidence,
                action=AgentActionName.NO_OP,
                reply=reply,
                slots={"message_route": route.route.value, "route_reason": route.reason},
            )

        return self._manual_response(
            intent=IntentName.ASK_HELP,
            confidence=route.confidence,
            action=AgentActionName.ANSWER_HELP,
            reply=reply,
            slots={"message_route": route.route.value, "route_reason": route.reason},
        )

    # 根据意图识别结果规划下一步 Agent 响应。
    def _plan_response(self, classification: IntentClassification) -> AgentResponse:
        return self.planner.plan_response(classification)

    # 生成新增投递记录的待确认动作。
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

    # 生成更新投递进度或面试安排的待确认动作。
    def _handle_update_application(self, classification: IntentClassification) -> AgentResponse:
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
            return self._response(
                classification=classification,
                action=AgentActionName.ASK_CLARIFICATION,
                reply=self._build_missing_update_application_reply(missing_slots),
                missing_slots=missing_slots,
            )

        return self._response(
            classification=classification,
            action=AgentActionName.UPDATE_APPLICATION,
            reply=self._build_update_application_reply(classification.slots),
            need_confirmation=True,
        )

    # 在用户确认后执行对应工具。
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

    # 处理用户对待确认动作的确认消息。
    def _handle_confirmation(
        self,
        pending: Optional[PendingAgentAction],
        conversation_id: str,
    ) -> AgentResponse:
        if pending is None:
            return self._manual_response(
                intent=IntentName.UNKNOWN,
                confidence=1.0,
                action=AgentActionName.NO_OP,
                reply="现在没有待确认的秋招动作。你可以先说“新增投递某公司某岗位”或“今天任务是什么”。",
            )

        if pending.missing_slots:
            if pending.intent in {IntentName.ADD_APPLICATION, IntentName.UPDATE_APPLICATION}:
                reply = self.application_dialogue.build_missing_reply(
                    intent=pending.intent,
                    missing_slots=pending.missing_slots,
                )
            else:
                reply = self._build_missing_application_reply(pending.missing_slots)
            return self._manual_response(
                intent=pending.intent,
                confidence=pending.confidence,
                action=AgentActionName.ASK_CLARIFICATION,
                reply=reply,
                slots=pending.slots,
                missing_slots=pending.missing_slots,
            )

        response = self._manual_response(
            intent=pending.intent,
            confidence=pending.confidence,
            action=pending.action,
            reply=pending.reply,
            slots=pending.slots,
            need_confirmation=pending.need_confirmation,
        )
        executed_response = self._maybe_execute_tool(
            response,
            message=pending.original_message,
            confirmed=True,
        )
        if executed_response.missing_slots:
            self.conversation_store.set_pending_action(
                conversation_id,
                PendingAgentAction.from_response(executed_response, pending.original_message),
            )
        else:
            self.conversation_store.clear_pending_action(conversation_id)
        return executed_response

    # 构造待确认动作的真实状态回复。
    def _build_pending_status_reply(self, pending: PendingAgentAction) -> str:
        if pending.missing_slots:
            if pending.intent in {IntentName.ADD_APPLICATION, IntentName.UPDATE_APPLICATION}:
                missing_reply = self.application_dialogue.build_missing_reply(
                    intent=pending.intent,
                    missing_slots=pending.missing_slots,
                )
            else:
                missing_reply = self._build_missing_application_reply(pending.missing_slots)
            return f"还没有写入。{missing_reply}"

        action_label = "这个动作"
        if pending.action == AgentActionName.CREATE_APPLICATION:
            company = pending.slots.get("company") or "这家公司"
            role = pending.slots.get("role") or "这个岗位"
            action_label = f"{company} 的 {role} 投递记录"
        elif pending.action == AgentActionName.UPDATE_APPLICATION:
            company = pending.slots.get("company") or "这家公司"
            action_label = f"{company} 的投递进度"

        return f"还没有写入。{action_label}还在待确认状态；回复“确认”或“对的”后，我才会真正写入数据库并同步飞书多维表格。"

    # 尝试用用户补充消息填充待确认动作缺失槽位。
    def _try_fill_pending_slots(
        self,
        pending: PendingAgentAction,
        message: str,
        conversation_id: str,
    ) -> Optional[AgentResponse]:
        slot_updates = self._extract_slot_updates(message, pending.missing_slots)
        if not slot_updates:
            return None

        merged_slots = pending.slots.copy()
        merged_slots.update(slot_updates)
        classification = IntentClassification(
            intent=pending.intent,
            confidence=max(pending.confidence, 0.8),
            slots=merged_slots,
        )
        planned_response = self._plan_response(classification)
        original_message = f"{pending.original_message}\n{message}"
        response = self._maybe_execute_tool(planned_response, message=original_message, confirmed=False)
        self._sync_pending_action(
            conversation_id=conversation_id,
            response=response,
            original_message=original_message,
        )
        return response

    # 尝试更新待确认面试安排的日历提醒偏好。
    def _try_update_pending_calendar_reminder(
        self,
        pending: PendingAgentAction,
        message: str,
        conversation_id: str,
    ) -> Optional[AgentResponse]:
        if pending.action != AgentActionName.UPDATE_APPLICATION:
            return None
        if pending.slots.get("update_type") != "schedule_interview":
            return None

        calendar_reminder = self._extract_calendar_reminder_preference(message)
        if calendar_reminder is None:
            return None

        slots = pending.slots.copy()
        slots["calendar_reminder"] = calendar_reminder
        classification = IntentClassification(
            intent=pending.intent,
            confidence=max(pending.confidence, 0.8),
            slots=slots,
        )
        response = self._plan_response(classification)
        self._sync_pending_action(
            conversation_id=conversation_id,
            response=response,
            original_message=pending.original_message,
        )
        return response

    # 把新的待确认动作同步到会话状态中。
    def _sync_pending_action(
        self,
        conversation_id: str,
        response: AgentResponse,
        original_message: str,
    ) -> None:
        if response.need_confirmation or response.missing_slots:
            self.conversation_store.set_pending_action(
                conversation_id,
                PendingAgentAction.from_response(response, original_message),
            )
            return

        self.conversation_store.clear_pending_action(conversation_id)

    # 保存最近一轮回复，支持下一轮基于上文追问。
    def _store_recent_context(
        self,
        conversation_id: str,
        response: AgentResponse,
        original_message: str,
    ) -> None:
        self.conversation_store.set_recent_context(
            conversation_id,
            RecentAgentContext.from_response(response, original_message),
        )

    # 把工具执行结果合并到 Agent 响应里。
    @staticmethod
    def _with_tool_result(response: AgentResponse, tool_result: ToolResult) -> AgentResponse:
        slots = response.slots.copy()
        raw_missing_slots = tool_result.data.get("missing_slots", [])
        missing_slots = [
            slot
            for slot in raw_missing_slots
            if isinstance(slot, str) and slot
        ] if isinstance(raw_missing_slots, list) else []
        if tool_result.data.get("requires_selection"):
            selection_slot = tool_result.data.get("selection_slot")
            if isinstance(selection_slot, str) and selection_slot:
                missing_slots = [selection_slot]
            slots["candidates"] = tool_result.data.get("candidates", [])
        return AgentResponse(
            intent=response.intent,
            confidence=response.confidence,
            action=response.action,
            reply=tool_result.message,
            need_confirmation=False,
            slots=slots,
            missing_slots=missing_slots,
            tool_result=tool_result,
        )

    # 处理 response 相关逻辑。
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

    # 构造不依赖意图识别结果的 Agent 响应。
    @staticmethod
    def _manual_response(
        intent: IntentName,
        confidence: float,
        action: AgentActionName,
        reply: str,
        need_confirmation: bool = False,
        slots: Optional[Dict[str, Any]] = None,
        missing_slots: Optional[List[str]] = None,
    ) -> AgentResponse:
        return AgentResponse(
            intent=intent,
            confidence=confidence,
            action=action,
            reply=reply,
            need_confirmation=need_confirmation,
            slots=slots or {},
            missing_slots=missing_slots or [],
        )

    # 根据入口和用户 ID 生成会话 ID。
    @staticmethod
    def _conversation_id(source: str, user_id: str) -> str:
        normalized_source = source.strip() or "api"
        normalized_user_id = user_id.strip() or "local_user"
        return f"{normalized_source}:{normalized_user_id}"

    # 根据入口和用户 ID 生成业务数据归属 ID。
    @staticmethod
    def _owner_id(source: str, user_id: str) -> str:
        normalized_source = (source or "api").strip().lower() or "api"
        normalized_user_id = (user_id or "local_user").strip() or "local_user"
        if normalized_user_id == "unknown_feishu_user":
            return "local_user"
        if normalized_source == "api" and normalized_user_id == "local_user":
            return "local_user"
        return f"{normalized_source}:{normalized_user_id}"

    # 判断用户消息是否表达确认。
    @staticmethod
    def _is_confirm_message(message: str) -> bool:
        compact = re.sub(r"\s+", "", message).lower()
        return compact in {
            "确认",
            "确定",
            "对",
            "对的",
            "是",
            "是的",
            "没错",
            "好",
            "好的",
            "可以",
            "可以的",
            "执行",
            "没问题",
            "嗯",
            "嗯嗯",
            "yes",
            "y",
            "ok",
        }

    # 判断用户消息是否表达取消当前待确认动作。
    @staticmethod
    def _is_cancel_message(message: str) -> bool:
        compact = re.sub(r"\s+", "", message).lower()
        return compact in {
            "取消",
            "不用",
            "不用了",
            "算了",
            "先不做",
            "撤销",
            "cancel",
        }

    # 判断用户是否在询问待确认动作有没有真正执行。
    @staticmethod
    def _is_pending_status_question(message: str) -> bool:
        compact = re.sub(r"\s+", "", message).lower().rstrip("？?。！!")
        return any(
            phrase in compact
            for phrase in (
                "记录了吗",
                "写入了吗",
                "保存了吗",
                "同步了吗",
                "执行了吗",
                "记了吗",
                "录了吗",
                "有没有记录",
                "是否记录",
                "记录了没",
                "写入了没",
                "保存了没",
                "同步了没",
            )
        )

    # 从输入数据中提取 slot updates。
    @staticmethod
    def _extract_slot_updates(message: str, missing_slots: List[str]) -> Dict[str, Any]:
        updates: Dict[str, Any] = {}
        text = message.strip()

        if "company" in missing_slots:
            company_match = re.search(r"(?:公司|企业|厂)[是为:：]?([^，,。；;\s]+)", text)
            if company_match:
                updates["company"] = company_match.group(1).strip()
            elif (
                len(text) <= 20
                and not AgentOrchestrator._looks_like_time_expression(text)
                and not any(word in text for word in ("岗位", "职位", "方向", "时间"))
            ):
                updates["company"] = text

        if "role" in missing_slots:
            role_match = re.search(r"(?:岗位|职位|方向)[是为:：]?([^，,。；;]+)", text)
            if role_match:
                updates["role"] = role_match.group(1).strip()
            elif len(text) <= 30 and any(
                word in text.lower()
                for word in ("java", "ai", "agent", "后端", "开发", "算法", "实习")
            ):
                updates["role"] = text

        if "interview_time" in missing_slots:
            time_match = re.search(r"(?:时间|面试时间)[是为:：]?([^，,。；;]+)", text)
            if time_match:
                updates["interview_time"] = time_match.group(1).strip()
            elif any(word in text for word in ("今天", "今晚", "明天", "后天", "上午", "下午", "晚上")):
                updates["interview_time"] = text

        if "calendar_reminder" in missing_slots:
            calendar_reminder = AgentOrchestrator._extract_calendar_reminder_preference(text)
            if calendar_reminder is not None:
                updates["calendar_reminder"] = calendar_reminder

        if "round" in missing_slots:
            round_match = re.search(r"(笔试|一面|二面|三面|hr\s*面|HR\s*面)", text, flags=re.IGNORECASE)
            if round_match:
                updates["round"] = round_match.group(1)

        return {key: value for key, value in updates.items() if value}

    # 计算缺失的 required slots。
    @staticmethod
    def _missing_required_slots(slots: Dict[str, Any], required_slots: List[str]) -> List[str]:
        return [
            slot
            for slot in required_slots
            if slot not in slots or slots.get(slot) is None or slots.get(slot) == ""
        ]

    # 判断 specific interview time 是否成立。
    @staticmethod
    def _is_specific_interview_time(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        return parse_chinese_datetime(value) is not None

    # 判断输入是否像 time expression。
    @staticmethod
    def _looks_like_time_expression(value: str) -> bool:
        compact = re.sub(r"\s+", "", value)
        return bool(
            parse_chinese_datetime(compact)
            or re.search(r"(今天|明天|后天|大后天|上午|下午|晚上|今晚|早上|周[一二三四五六日天]|星期[一二三四五六日天]|礼拜[一二三四五六日天])", compact)
            or re.search(r"[零一二三四五六七八九十两\d]{1,3}点", compact)
        )

    # 从输入数据中提取 calendar reminder preference。
    @staticmethod
    def _extract_calendar_reminder_preference(message: str) -> Optional[bool]:
        compact = re.sub(r"\s+", "", message).lower()
        if any(word in compact for word in ("不需要提醒", "不用提醒", "不要提醒", "不加日历", "不同步日历", "不用日历")):
            return False
        if any(word in compact for word in ("需要提醒", "要提醒", "加日历", "同步日历", "日历提醒", "需要日历")):
            return True
        if compact in {"需要", "要", "加", "同步", "提醒"}:
            return True
        if compact in {"不需要", "不用", "不要", "否", "不加"}:
            return False
        return None

    # 把飞书用户上下文补充进意图槽位。
    @staticmethod
    def _with_actor_context(
        classification: IntentClassification,
        user_id: str,
        source: str,
    ) -> IntentClassification:
        normalized_source = source.strip().lower()
        normalized_user_id = user_id.strip()
        slots = classification.slots.copy()
        slots.setdefault(
            "owner_id",
            AgentOrchestrator._owner_id(source=source, user_id=user_id),
        )
        if normalized_source == "feishu" and normalized_user_id and normalized_user_id != "unknown_feishu_user":
            slots.setdefault("attendee_user_id", normalized_user_id)
            slots.setdefault("attendee_user_id_type", "open_id")
            slots.setdefault("bitable_collaborator_user_id", normalized_user_id)
            slots.setdefault("bitable_collaborator_user_id_type", "open_id")
        return IntentClassification(
            intent=classification.intent,
            confidence=classification.confidence,
            slots=slots,
        )

    # 把飞书用户上下文补充进 LLM Planner 生成的计划槽位。
    @staticmethod
    def _with_actor_context_for_plan(
        plan: AgentPlan,
        user_id: str,
        source: str,
    ) -> AgentPlan:
        classification = IntentClassification(
            intent=plan.intent,
            confidence=plan.confidence,
            slots=plan.slots,
        )
        enriched = AgentOrchestrator._with_actor_context(
            classification=classification,
            user_id=user_id,
            source=source,
        )
        if enriched.slots == plan.slots:
            return plan

        steps = []
        for step in plan.steps:
            arguments = step.arguments.copy()
            arguments.update(enriched.slots)
            steps.append(
                AgentPlanStep(
                    tool_name=step.tool_name,
                    arguments=arguments,
                    reason=step.reason,
                )
            )
        return AgentPlan(
            intent=plan.intent,
            confidence=plan.confidence,
            action=plan.action,
            reply=plan.reply,
            need_confirmation=plan.need_confirmation,
            slots=enriched.slots,
            missing_slots=plan.missing_slots,
            steps=steps,
            reason=plan.reason,
        )

    # 构造 missing application reply。
    @staticmethod
    def _build_missing_application_reply(missing_slots: List[str]) -> str:
        labels = {
            "company": "公司",
            "role": "岗位",
        }
        missing = "、".join(labels.get(slot, slot) for slot in missing_slots)
        return f"我识别到你想新增投递记录，但还缺少{missing}。请补充后我再生成投递创建动作。"

    # 构造 missing update application reply。
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

    # 构造 create application reply。
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

    # 构造 update application reply。
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
        if company:
            return f"我识别到你想更新 {company} 的投递状态。下一步会定位投递记录并更新状态，执行前需要你确认。"
        return "我识别到你想更新投递状态。下一步需要先定位对应公司或岗位，执行前需要你确认。"

    # 构造 task reply。
    @staticmethod
    def _build_task_reply(operation: str, slots: Dict[str, Any]) -> str:
        task_title = slots.get("task_title")
        task_type = slots.get("task_type")
        if task_title:
            return f"我识别到你要{operation}任务：{task_title}。下一步会更新任务状态，执行前需要你确认。"
        if task_type:
            return f"我识别到你要{operation}一个 {task_type} 任务。下一步会匹配最近任务并更新状态，执行前需要你确认。"
        return f"我识别到你要{operation}任务。下一步需要匹配具体任务，执行前需要你确认。"

    # 构造 interview review reply。
    @staticmethod
    def _build_interview_review_reply(slots: Dict[str, Any]) -> str:
        if slots.get("company"):
            return f"我识别到你要提交 {slots['company']} 的面试复盘。下一步会提取被问知识点、薄弱点和补强任务，执行前需要你确认。"
        return "我识别到你要提交面试复盘。下一步会提取被问知识点、薄弱点和补强任务，执行前需要你确认。"

    # 构造 mock interview reply。
    @staticmethod
    def _build_mock_interview_reply(slots: Dict[str, Any]) -> str:
        project = slots.get("project")
        role = slots.get("role")
        if project and role:
            return f"我识别到你想开始模拟面试，岗位是 {role}，项目是 {project}。下一步会进入文本模拟面试流程。"
        if project:
            return f"我识别到你想围绕 {project} 开始模拟面试。下一步会进入项目深挖模拟面试流程。"
        return "我识别到你想开始模拟面试。下一步会先确认岗位、项目或面试类型。"
