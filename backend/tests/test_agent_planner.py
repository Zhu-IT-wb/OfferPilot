import asyncio
import json

from app.agents.orchestrator import AgentOrchestrator
from app.agents.conversation import InMemoryConversationStore
from app.agents.planner import AgentPlanner
from app.schemas.agent_plan import AgentPlan
from app.schemas.agent import AgentActionName
from app.schemas.intent import IntentClassification, IntentName
from app.services.llm_service import LLMResult
from app.tools.offerpilot_tools import build_offerpilot_tool_registry


def test_agent_planner_returns_structured_plan_for_today_tasks() -> None:
    planner = AgentPlanner()

    plan = planner.plan(
        IntentClassification(
            intent=IntentName.GET_TODAY_TASKS,
            confidence=0.72,
            slots={},
        )
    )

    assert plan.action == AgentActionName.LIST_TODAY_TASKS
    assert plan.steps[0].tool_name == "list_today_tasks"
    assert plan.to_response().action == AgentActionName.LIST_TODAY_TASKS


def test_agent_planner_routes_interview_reschedule_to_domain_tool() -> None:
    planner = AgentPlanner()

    plan = planner.plan(
        IntentClassification(
            intent=IntentName.UPDATE_APPLICATION,
            confidence=0.9,
            slots={
                "company": "美团",
                "round": "一面",
                "interview_time": "后天下午四点",
                "update_type": "reschedule_interview",
            },
        )
    )

    assert plan.action == AgentActionName.RESCHEDULE_INTERVIEW
    assert plan.need_confirmation is True
    assert plan.steps[0].tool_name == "reschedule_interview"


def test_agent_planner_routes_interview_cancellation_to_domain_tool() -> None:
    planner = AgentPlanner()

    plan = planner.plan(
        IntentClassification(
            intent=IntentName.UPDATE_APPLICATION,
            confidence=0.9,
            slots={"schedule_id": "schedule_1", "update_type": "cancel_interview"},
        )
    )

    assert plan.action == AgentActionName.CANCEL_INTERVIEW
    assert plan.need_confirmation is True
    assert plan.steps[0].tool_name == "cancel_interview"


def test_orchestrator_uses_compiled_langgraph() -> None:
    orchestrator = AgentOrchestrator()

    assert hasattr(orchestrator._graph, "ainvoke")


def test_llm_planner_queries_upcoming_interviews_from_free_form_message() -> None:
    class FakeLLMService:
        async def generate_text(self, **kwargs):
            return LLMResult(
                provider="fake",
                model="fake",
                content=json.dumps(
                    {
                        "intent": "query_application",
                        "confidence": 0.91,
                        "action": "query_application",
                        "reply": "我来查看你近期的笔试/面试安排。",
                        "need_confirmation": False,
                        "slots": {},
                        "missing_slots": [],
                        "steps": [],
                        "reason": "query interviews",
                    },
                    ensure_ascii=False,
                ),
                raw_response={},
            )

    planner = AgentPlanner(llm_service=FakeLLMService(), llm_planner_enabled=True)
    orchestrator = AgentOrchestrator(
        planner=planner,
        tool_registry=build_offerpilot_tool_registry(calendar_service=None, bitable_service=None),
        conversation_store=InMemoryConversationStore(),
    )

    result = asyncio.run(orchestrator.handle_message("我有哪些面试"))

    assert result.action == AgentActionName.QUERY_APPLICATION
    assert result.need_confirmation is False
    assert result.slots["query_type"] == "upcoming_interviews"
    assert result.tool_result is not None


def test_llm_planner_invalid_json_falls_back_to_rule_path() -> None:
    class FakeLLMService:
        async def generate_text(self, **kwargs):
            return LLMResult(
                provider="fake",
                model="fake",
                content="not json",
                raw_response={},
            )

    class FakeIntentClassifier:
        async def classify(self, message):
            return IntentClassification(
                intent=IntentName.GET_TODAY_TASKS,
                confidence=0.7,
                slots={},
            )

    planner = AgentPlanner(llm_service=FakeLLMService(), llm_planner_enabled=True)
    orchestrator = AgentOrchestrator(
        intent_classifier=FakeIntentClassifier(),
        planner=planner,
        conversation_store=InMemoryConversationStore(),
    )

    result = asyncio.run(orchestrator.handle_message("今天任务是什么"))

    assert result.action == AgentActionName.LIST_TODAY_TASKS
    assert result.tool_result is not None


def test_llm_planner_forces_confirmation_for_mutating_action() -> None:
    class FakeLLMService:
        async def generate_text(self, **kwargs):
            return LLMResult(
                provider="fake",
                model="fake",
                content=json.dumps(
                    {
                        "intent": "add_application",
                        "confidence": 0.93,
                        "action": "create_application",
                        "reply": "我会记录这条投递。",
                        "need_confirmation": False,
                        "slots": {"company": "美团", "role": "Java 后端实习"},
                        "missing_slots": [],
                        "steps": [],
                        "reason": "create application",
                    },
                    ensure_ascii=False,
                ),
                raw_response={},
            )

    planner = AgentPlanner(llm_service=FakeLLMService(), llm_planner_enabled=True)
    orchestrator = AgentOrchestrator(
        planner=planner,
        tool_registry=build_offerpilot_tool_registry(calendar_service=None, bitable_service=None),
        conversation_store=InMemoryConversationStore(),
    )

    result = asyncio.run(orchestrator.handle_message("我投递了美团 Java 后端实习"))

    assert result.action == AgentActionName.CREATE_APPLICATION
    assert result.need_confirmation is True
    assert result.tool_result is None


def test_llm_planner_repairs_misparsed_natural_application_message() -> None:
    class FakeLLMService:
        async def generate_text(self, **kwargs):
            return LLMResult(
                provider="fake",
                model="fake",
                content=json.dumps(
                    {
                        "intent": "add_application",
                        "confidence": 0.9,
                        "action": "create_application",
                        "reply": "我会记录这条投递。",
                        "need_confirmation": False,
                        "slots": {
                            "company": "ai应用开发岗位",
                            "role": "ai应用开发",
                            "interview_time": "今天",
                        },
                        "missing_slots": [],
                        "steps": [],
                        "reason": "create application",
                    },
                    ensure_ascii=False,
                ),
                raw_response={},
            )

    planner = AgentPlanner(llm_service=FakeLLMService(), llm_planner_enabled=True)
    orchestrator = AgentOrchestrator(
        planner=planner,
        tool_registry=build_offerpilot_tool_registry(calendar_service=None, bitable_service=None),
        conversation_store=InMemoryConversationStore(),
    )

    result = asyncio.run(
        orchestrator.handle_message("小猪，我今天投递了 农夫山泉公司的 ai应用开发岗位")
    )

    assert result.action == AgentActionName.CREATE_APPLICATION
    assert result.need_confirmation is True
    assert result.slots["company"] == "农夫山泉"
    assert result.slots["role"] == "ai应用开发"
    assert "interview_time" not in result.slots


def test_llm_planner_asks_for_specific_interview_time() -> None:
    class FakeLLMService:
        async def generate_text(self, **kwargs):
            return LLMResult(
                provider="fake",
                model="fake",
                content=json.dumps(
                    {
                        "intent": "update_application",
                        "confidence": 0.9,
                        "action": "update_application",
                        "reply": "我会记录面试安排。",
                        "need_confirmation": False,
                        "slots": {
                            "company": "美团",
                            "role": "Java 后端实习",
                            "round": "一面",
                            "interview_time": "明天下午",
                            "update_type": "schedule_interview",
                            "status": "interview_1",
                        },
                        "missing_slots": [],
                        "steps": [],
                        "reason": "schedule interview",
                    },
                    ensure_ascii=False,
                ),
                raw_response={},
            )

    planner = AgentPlanner(llm_service=FakeLLMService(), llm_planner_enabled=True)
    orchestrator = AgentOrchestrator(
        planner=planner,
        tool_registry=build_offerpilot_tool_registry(calendar_service=None, bitable_service=None),
        conversation_store=InMemoryConversationStore(),
    )

    result = asyncio.run(orchestrator.handle_message("我之前投递的美团 Java 后端实习，明天下午邀请我一面"))

    assert result.action == AgentActionName.ASK_CLARIFICATION
    assert result.need_confirmation is False
    assert result.missing_slots == ["interview_time"]
    assert result.tool_result is None


def test_llm_planner_keeps_greeting_as_help_response() -> None:
    class FakeLLMService:
        async def generate_text(self, **kwargs):
            return LLMResult(
                provider="fake",
                model="fake",
                content=json.dumps(
                    {
                        "intent": "ask_help",
                        "confidence": 0.96,
                        "action": "answer_help",
                        "reply": "你好，我是 OfferPilot，可以帮你记录投递、安排面试和查看任务。",
                        "need_confirmation": False,
                        "slots": {},
                        "missing_slots": [],
                        "steps": [],
                        "reason": "smalltalk",
                    },
                    ensure_ascii=False,
                ),
                raw_response={},
            )

    planner = AgentPlanner(llm_service=FakeLLMService(), llm_planner_enabled=True)
    orchestrator = AgentOrchestrator(
        planner=planner,
        conversation_store=InMemoryConversationStore(),
    )

    result = asyncio.run(orchestrator.handle_message("你好"))

    assert result.intent == IntentName.ASK_HELP
    assert result.action == AgentActionName.ANSWER_HELP
    assert result.need_confirmation is False
    assert result.tool_result is None
    assert "OfferPilot" in result.reply


def test_llm_planner_prioritizes_recent_interviews_from_previous_tool_result() -> None:
    class FakeLLMService:
        def __init__(self):
            self.calls = 0

        async def generate_text(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                content = {
                    "intent": "query_application",
                    "confidence": 0.91,
                    "action": "query_application",
                    "reply": "我来查看你近期的笔试/面试安排。",
                    "need_confirmation": False,
                    "slots": {"query_type": "upcoming_interviews"},
                    "missing_slots": [],
                    "steps": [],
                    "reason": "query interviews",
                }
            else:
                content = {
                    "intent": "ask_help",
                    "confidence": 0.72,
                    "action": "ask_clarification",
                    "reply": "我识别到你想执行这个秋招动作，但还缺少tasks_to_prioritize。",
                    "need_confirmation": False,
                    "slots": {},
                    "missing_slots": ["tasks_to_prioritize"],
                    "steps": [],
                    "reason": "bad missing slot",
                }

            return LLMResult(
                provider="fake",
                model="fake",
                content=json.dumps(content, ensure_ascii=False),
                raw_response={},
            )

    from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository

    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(
        company="美团",
        role="Java 后端实习",
        round_name="一面",
        interview_time="明天下午三点",
    )
    repository.create_interview_schedule(
        company="美团",
        round_name="一面",
        application_id=application.id,
        role=application.role,
        start_time="明天下午三点",
        start_at=None,
    )
    planner = AgentPlanner(llm_service=FakeLLMService(), llm_planner_enabled=True)
    orchestrator = AgentOrchestrator(
        planner=planner,
        tool_registry=build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None),
        conversation_store=InMemoryConversationStore(),
    )

    first = asyncio.run(orchestrator.handle_message("我现在有哪些面试", user_id="u1"))
    second = asyncio.run(orchestrator.handle_message("你觉得我要先准备哪一个", user_id="u1"))

    assert first.action == AgentActionName.QUERY_APPLICATION
    assert second.action == AgentActionName.ANSWER_HELP
    assert second.missing_slots == []
    assert "tasks_to_prioritize" not in second.reply
    assert "美团" in second.reply
    assert "先准备" in second.reply


def test_llm_planner_reuses_application_after_round_passed_follow_up() -> None:
    class FakeLLMService:
        def __init__(self):
            self.calls = 0

        async def generate_text(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                content = {
                    "intent": "update_application",
                    "confidence": 0.94,
                    "action": "update_application",
                    "reply": "准备记录一面通过。",
                    "need_confirmation": True,
                    "slots": {
                        "company": "A公司",
                        "role": "java开发",
                        "round": "一面",
                        "update_type": "pass_round",
                        "status": "interview_1_passed",
                    },
                    "missing_slots": [],
                    "steps": [],
                    "reason": "passed first interview",
                }
            elif self.calls == 2:
                content = {
                    "intent": "ask_help",
                    "confidence": 0.9,
                    "action": "answer_help",
                    "reply": "已经确认你一面通过了！",
                    "need_confirmation": False,
                    "slots": {},
                    "missing_slots": [],
                    "steps": [],
                    "reason": "acknowledge previous update",
                }
            else:
                content = {
                    "intent": "update_application",
                    "confidence": 0.93,
                    "action": "update_application",
                    "reply": "准备记录二面安排。",
                    "need_confirmation": True,
                    "slots": {
                        "round": "二面",
                        "interview_time": "明天下午三点",
                        "update_type": "schedule_interview",
                        "status": "interview_2",
                    },
                    "missing_slots": ["company"],
                    "steps": [],
                    "reason": "schedule second interview",
                }

            return LLMResult(
                provider="fake",
                model="fake",
                content=json.dumps(content, ensure_ascii=False),
                raw_response={},
            )

    from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository

    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="A公司", role="java开发", round_name="一面")
    planner = AgentPlanner(llm_service=FakeLLMService(), llm_planner_enabled=True)
    conversation_store = InMemoryConversationStore()
    orchestrator = AgentOrchestrator(
        planner=planner,
        tool_registry=build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None),
        conversation_store=conversation_store,
    )

    passed_plan = asyncio.run(orchestrator.handle_message("A公司的java开发一面通过了", user_id="u1"))
    passed = asyncio.run(orchestrator.handle_message("确认", user_id="u1"))
    acknowledged = asyncio.run(orchestrator.handle_message("我已经一面通过了", user_id="u1"))
    scheduled_plan = asyncio.run(
        orchestrator.handle_message("我明天下午三点要进行二面", user_id="u1")
    )

    assert passed_plan.need_confirmation is True
    assert passed.tool_result is not None and passed.tool_result.success is True
    assert acknowledged.action == AgentActionName.ANSWER_HELP
    recent_context = conversation_store.get_recent_context("api:u1")
    assert recent_context is not None
    assert recent_context.application_focus["application_id"] == application.id
    assert scheduled_plan.action == AgentActionName.UPDATE_APPLICATION
    assert scheduled_plan.need_confirmation is True
    assert scheduled_plan.missing_slots == []
    assert scheduled_plan.slots["application_id"] == application.id
    assert scheduled_plan.slots["company"] == "A公司"
