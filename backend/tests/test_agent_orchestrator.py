import asyncio
import json

from fastapi.testclient import TestClient

from app.agents.conversation import InMemoryConversationStore
from app.agents.intent_classifier import IntentClassifier
from app.agents.orchestrator import AgentOrchestrator
from app.agents.planner import AgentPlanner
from app.api.routes import debug
from app.main import app
from app.repositories.offerpilot_repository import InMemoryOfferPilotRepository
from app.schemas.agent import AgentActionName, AgentResponse
from app.schemas.intent import IntentClassification, IntentName
from app.services.feishu_service import (
    FeishuBitableRecordResult,
    FeishuCalendarAttendeeResult,
    FeishuCalendarEventResult,
    FeishuCalendarResult,
)
from app.services.llm_service import LLMResult
from app.tools.offerpilot_tools import build_offerpilot_tool_registry


def test_orchestrator_plans_create_application_with_confirmation() -> None:
    class FakeIntentClassifier:
        async def classify(self, message):
            assert message == "新增投递深信服开发实习，明天下午三点一面"
            return IntentClassification(
                intent=IntentName.ADD_APPLICATION,
                confidence=0.86,
                slots={
                    "company": "深信服",
                    "role": "开发实习",
                    "interview_time": "明天下午三点",
                    "round": "一面",
                },
            )

    orchestrator = AgentOrchestrator(intent_classifier=FakeIntentClassifier())

    result = asyncio.run(orchestrator.handle_message("新增投递深信服开发实习，明天下午三点一面"))

    assert result.intent == IntentName.ADD_APPLICATION
    assert result.action == AgentActionName.CREATE_APPLICATION
    assert result.need_confirmation is True
    assert result.missing_slots == []
    assert result.slots["company"] == "深信服"
    assert "写入投递表" in result.reply


def test_orchestrator_asks_clarification_when_application_slots_missing() -> None:
    class FakeIntentClassifier:
        async def classify(self, message):
            assert message == "新增投递深信服"
            return IntentClassification(
                intent=IntentName.ADD_APPLICATION,
                confidence=0.62,
                slots={"company": "深信服"},
            )

    orchestrator = AgentOrchestrator(intent_classifier=FakeIntentClassifier())

    result = asyncio.run(orchestrator.handle_message("新增投递深信服"))

    assert result.action == AgentActionName.ASK_CLARIFICATION
    assert result.need_confirmation is False
    assert result.missing_slots == ["role"]
    assert "岗位" in result.reply


def test_orchestrator_plans_today_tasks_read_action() -> None:
    class FakeIntentClassifier:
        async def classify(self, message):
            return IntentClassification(
                intent=IntentName.GET_TODAY_TASKS,
                confidence=0.72,
                slots={},
            )

    orchestrator = AgentOrchestrator(intent_classifier=FakeIntentClassifier())

    result = asyncio.run(orchestrator.handle_message("今天任务是什么？"))

    assert result.action == AgentActionName.LIST_TODAY_TASKS
    assert result.need_confirmation is False
    assert "今天的任务" in result.reply
    assert result.tool_result is not None
    assert result.tool_result.success is True
    assert len(result.tool_result.data["tasks"]) == 3


def test_orchestrator_executes_application_query_without_confirmation() -> None:
    class FakeIntentClassifier:
        async def classify(self, message):
            return IntentClassification(
                intent=IntentName.QUERY_APPLICATION,
                confidence=0.74,
                slots={"query_type": "company_status", "company": "深信服"},
            )

    repository = InMemoryOfferPilotRepository()
    repository.create_application(company="深信服", role="AI 应用开发", round_name="二面")
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)
    orchestrator = AgentOrchestrator(
        intent_classifier=FakeIntentClassifier(),
        tool_registry=registry,
    )

    result = asyncio.run(orchestrator.handle_message("深信服现在什么状态？"))

    assert result.intent == IntentName.QUERY_APPLICATION
    assert result.action == AgentActionName.QUERY_APPLICATION
    assert result.need_confirmation is False
    assert result.tool_result is not None
    assert "深信服 当前进度" in result.reply


def test_orchestrator_does_not_execute_mutating_action_without_confirmation() -> None:
    class FakeIntentClassifier:
        async def classify(self, message):
            return IntentClassification(
                intent=IntentName.ADD_APPLICATION,
                confidence=0.86,
                slots={"company": "深信服", "role": "开发实习"},
            )

    repository = InMemoryOfferPilotRepository()
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)
    orchestrator = AgentOrchestrator(
        intent_classifier=FakeIntentClassifier(),
        tool_registry=registry,
    )

    result = asyncio.run(orchestrator.handle_message("新增投递深信服开发实习"))

    assert result.action == AgentActionName.CREATE_APPLICATION
    assert result.need_confirmation is True
    assert result.tool_result is None
    assert repository.applications == []


def test_orchestrator_executes_create_application_when_confirmed() -> None:
    class FakeIntentClassifier:
        async def classify(self, message):
            return IntentClassification(
                intent=IntentName.ADD_APPLICATION,
                confidence=0.86,
                slots={
                    "company": "深信服",
                    "role": "开发实习",
                    "interview_time": "明天下午三点",
                    "round": "一面",
                },
            )

    repository = InMemoryOfferPilotRepository()
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)
    orchestrator = AgentOrchestrator(
        intent_classifier=FakeIntentClassifier(),
        tool_registry=registry,
    )

    result = asyncio.run(
        orchestrator.handle_message(
            "新增投递深信服开发实习，明天下午三点一面",
            confirmed=True,
        )
    )

    assert result.action == AgentActionName.CREATE_APPLICATION
    assert result.need_confirmation is False
    assert result.tool_result is not None
    assert result.tool_result.success is True
    assert result.tool_result.data["application"]["id"] == "app_1"
    assert result.tool_result.data["application"]["status"] == "interview_1"
    assert len(repository.applications) == 1


def test_orchestrator_confirms_application_update_and_records_schedule() -> None:
    class FakeIntentClassifier:
        async def classify(self, message):
            assert message == "明天早上八点，深信服约我一面"
            return IntentClassification(
                intent=IntentName.UPDATE_APPLICATION,
                confidence=0.82,
                slots={
                    "company": "深信服",
                    "round": "一面",
                    "interview_time": "明天早上八点",
                    "update_type": "schedule_interview",
                    "status": "interview_1",
                },
            )

    repository = InMemoryOfferPilotRepository()
    repository.create_application(company="深信服", role="AI 应用开发")
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)
    orchestrator = AgentOrchestrator(
        intent_classifier=FakeIntentClassifier(),
        tool_registry=registry,
        conversation_store=InMemoryConversationStore(),
    )

    first = asyncio.run(
        orchestrator.handle_message(
            "明天早上八点，深信服约我一面",
            user_id="ou_test",
            source="feishu",
        )
    )
    assert first.action == AgentActionName.UPDATE_APPLICATION
    assert first.need_confirmation is True
    assert repository.applications[0].status.value == "planned"

    second = asyncio.run(
        orchestrator.handle_message(
            "确认",
            user_id="ou_test",
            source="feishu",
        )
    )

    assert second.tool_result is not None
    assert second.tool_result.success is True
    assert repository.applications[0].status.value == "interview_1"
    assert repository.interview_schedules[0].company == "深信服"
    assert repository.interview_schedules[0].reminder_minutes == 30


def test_orchestrator_asks_for_specific_interview_time_before_update() -> None:
    class FakeIntentClassifier:
        async def classify(self, message):
            return IntentClassification(
                intent=IntentName.UPDATE_APPLICATION,
                confidence=0.82,
                slots={
                    "company": "美团",
                    "round": "一面",
                    "interview_time": "明天下午",
                    "update_type": "schedule_interview",
                    "status": "interview_1",
                },
            )

    repository = InMemoryOfferPilotRepository()
    repository.create_application(company="美团", role="Java 后端实习")
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)
    orchestrator = AgentOrchestrator(
        intent_classifier=FakeIntentClassifier(),
        tool_registry=registry,
        conversation_store=InMemoryConversationStore(),
    )

    result = asyncio.run(
        orchestrator.handle_message(
            "我之前投递的美团 Java 后端实习，明天下午邀请我一面",
            user_id="ou_test",
            source="feishu",
        )
    )

    assert result.action == AgentActionName.ASK_CLARIFICATION
    assert result.need_confirmation is False
    assert result.missing_slots == ["interview_time"]
    assert "具体面试时间" in result.reply
    assert repository.applications[0].status.value == "planned"
    assert repository.interview_schedules == []


def test_orchestrator_allows_skipping_calendar_reminder_before_confirmation() -> None:
    class FakeIntentClassifier:
        async def classify(self, message):
            return IntentClassification(
                intent=IntentName.UPDATE_APPLICATION,
                confidence=0.82,
                slots={
                    "company": "美团",
                    "round": "一面",
                    "interview_time": "明天下午",
                    "update_type": "schedule_interview",
                    "status": "interview_1",
                },
            )

    repository = InMemoryOfferPilotRepository()
    repository.create_application(company="美团", role="Java 后端实习")
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)
    orchestrator = AgentOrchestrator(
        intent_classifier=FakeIntentClassifier(),
        tool_registry=registry,
        conversation_store=InMemoryConversationStore(),
    )

    first = asyncio.run(
        orchestrator.handle_message(
            "我之前投递的美团 Java 后端实习，明天下午邀请我一面",
            user_id="ou_test",
            source="feishu",
        )
    )
    assert first.missing_slots == ["interview_time"]

    second = asyncio.run(
        orchestrator.handle_message(
            "明天下午三点",
            user_id="ou_test",
            source="feishu",
        )
    )
    assert second.action == AgentActionName.UPDATE_APPLICATION
    assert second.need_confirmation is True
    assert second.slots["interview_time"] == "明天下午三点"
    assert "如果不需要" in second.reply

    third = asyncio.run(
        orchestrator.handle_message(
            "不需要提醒",
            user_id="ou_test",
            source="feishu",
        )
    )
    assert third.need_confirmation is True
    assert third.slots["calendar_reminder"] is False
    assert "不需要同步飞书日历提醒" in third.reply

    fourth = asyncio.run(
        orchestrator.handle_message(
            "确认",
            user_id="ou_test",
            source="feishu",
        )
    )

    assert fourth.tool_result is not None
    assert fourth.tool_result.success is True
    assert fourth.tool_result.data["calendar_sync"]["status"] == "skipped_by_user"
    assert repository.applications[0].status.value == "interview_1"
    assert repository.interview_schedules[0].start_at is not None
    assert "未添加飞书日历提醒" in fourth.reply


def test_orchestrator_fills_multiple_application_slots_from_labeled_reply() -> None:
    class FakeIntentClassifier:
        async def classify(self, message):
            return IntentClassification(
                intent=IntentName.UPDATE_APPLICATION,
                confidence=0.82,
                slots={
                    "round": "一面",
                    "role": "Java 后端",
                    "interview_time": "明天",
                    "update_type": "schedule_interview",
                    "status": "interview_1",
                },
            )

    repository = InMemoryOfferPilotRepository()
    repository.create_application(company="测试修改推库", role="Java 后端实习")
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)
    orchestrator = AgentOrchestrator(
        intent_classifier=FakeIntentClassifier(),
        tool_registry=registry,
        conversation_store=InMemoryConversationStore(),
    )

    first = asyncio.run(
        orchestrator.handle_message(
            "我之前投 Java 后端实习，明天下午邀请我一面",
            user_id="ou_test",
            source="feishu",
        )
    )
    assert first.action == AgentActionName.ASK_CLARIFICATION
    assert first.missing_slots == ["company", "interview_time"]

    second = asyncio.run(
        orchestrator.handle_message(
            "公司是 测试修改推库  时间是明天下午5点。",
            user_id="ou_test",
            source="feishu",
        )
    )
    assert second.action == AgentActionName.UPDATE_APPLICATION
    assert second.need_confirmation is True
    assert second.slots["company"] == "测试修改推库"
    assert second.slots["interview_time"] == "明天下午5点"
    assert "公司是 测试修改推库" in second.reply

    third = asyncio.run(
        orchestrator.handle_message(
            "确认",
            user_id="ou_test",
            source="feishu",
        )
    )
    assert third.tool_result is not None
    assert third.tool_result.success is True
    assert third.tool_result.data["application"]["company"] == "测试修改推库"
    assert third.tool_result.data["interview_schedule"]["start_at"] is not None


def test_orchestrator_answers_greeting_without_business_intent() -> None:
    class FailingIntentClassifier:
        async def classify(self, message):
            raise AssertionError("smalltalk should not enter business intent classification")

    orchestrator = AgentOrchestrator(
        intent_classifier=FailingIntentClassifier(),
        conversation_store=InMemoryConversationStore(),
    )

    result = asyncio.run(orchestrator.handle_message("你好"))

    assert result.intent == IntentName.ASK_HELP
    assert result.action == AgentActionName.ANSWER_HELP
    assert result.need_confirmation is False
    assert result.tool_result is None
    assert "OfferPilot" in result.reply
    assert result.slots["message_route"] == "smalltalk"


def test_orchestrator_answers_identity_question_without_business_intent() -> None:
    class FailingIntentClassifier:
        async def classify(self, message):
            raise AssertionError("identity question should not enter business intent classification")

    orchestrator = AgentOrchestrator(
        intent_classifier=FailingIntentClassifier(),
        conversation_store=InMemoryConversationStore(),
    )

    result = asyncio.run(orchestrator.handle_message("你是谁？"))

    assert result.action == AgentActionName.ANSWER_HELP
    assert "个人助手" in result.reply
    assert "记录投递" in result.reply
    assert result.slots["message_route"] == "smalltalk"


def test_orchestrator_answers_capability_help_without_business_intent() -> None:
    class FailingIntentClassifier:
        async def classify(self, message):
            raise AssertionError("capability help should not enter business intent classification")

    orchestrator = AgentOrchestrator(
        intent_classifier=FailingIntentClassifier(),
        conversation_store=InMemoryConversationStore(),
    )

    result = asyncio.run(orchestrator.handle_message("你能做什么？"))

    assert result.action == AgentActionName.ANSWER_HELP
    assert "飞书多维表格" in result.reply
    assert "飞书日历" in result.reply
    assert result.slots["message_route"] == "capability_help"


def test_orchestrator_routes_domain_question_to_general_responder() -> None:
    class FailingIntentClassifier:
        async def classify(self, message):
            raise AssertionError("domain question should not enter business intent classification")

    class FakeGeneralResponder:
        async def respond(self, message, route):
            assert message == "Java 后端秋招怎么准备？"
            assert route.route.value == "domain_question"
            return "先抓 Java 基础、数据库、缓存和项目深挖。"

    orchestrator = AgentOrchestrator(
        intent_classifier=FailingIntentClassifier(),
        general_responder=FakeGeneralResponder(),
        conversation_store=InMemoryConversationStore(),
    )

    result = asyncio.run(orchestrator.handle_message("Java 后端秋招怎么准备？"))

    assert result.action == AgentActionName.ANSWER_HELP
    assert result.reply == "先抓 Java 基础、数据库、缓存和项目深挖。"
    assert result.slots["message_route"] == "domain_question"


def test_orchestrator_passes_feishu_user_to_calendar_attendee_sync() -> None:
    class FakeIntentClassifier:
        async def classify(self, message):
            return IntentClassification(
                intent=IntentName.UPDATE_APPLICATION,
                confidence=0.82,
                slots={
                    "company": "深信服",
                    "round": "一面",
                    "interview_time": "明天早上八点",
                    "update_type": "schedule_interview",
                    "status": "interview_1",
                },
            )

    class FakeCalendarService:
        calendar_id = "primary"

        def __init__(self):
            self.added_user_ids = []

        def is_calendar_sync_enabled(self):
            return True

        def should_manage_offerpilot_calendar(self):
            return True

        def create_shared_calendar(self):
            return FeishuCalendarResult(
                calendar_id="feishu.cn_offerpilot@group.calendar.feishu.cn",
                raw_response={"code": 0},
            )

        def create_interview_event(self, **kwargs):
            return FeishuCalendarEventResult(
                event_id="evt_test_1",
                raw_response={"code": 0},
            )

        def add_event_attendee(self, **kwargs):
            self.added_user_ids.append(kwargs["user_id"])
            return FeishuCalendarAttendeeResult(
                attendee_ids=["user_attendee_1"],
                raw_response={"code": 0},
            )

    repository = InMemoryOfferPilotRepository()
    repository.create_application(company="深信服", role="AI 应用开发")
    calendar_service = FakeCalendarService()
    registry = build_offerpilot_tool_registry(
        repository,
        calendar_service=calendar_service,
        bitable_service=None,
    )
    orchestrator = AgentOrchestrator(
        intent_classifier=FakeIntentClassifier(),
        tool_registry=registry,
        conversation_store=InMemoryConversationStore(),
    )

    first = asyncio.run(
        orchestrator.handle_message(
            "明天早上八点，深信服约我一面",
            user_id="ou_test",
            source="feishu",
        )
    )
    second = asyncio.run(
        orchestrator.handle_message(
            "确认",
            user_id="ou_test",
            source="feishu",
        )
    )

    assert first.slots["attendee_user_id"] == "ou_test"
    assert first.slots["bitable_collaborator_user_id"] == "ou_test"
    assert second.tool_result is not None
    assert second.tool_result.data["calendar_sync"]["attendee_sync"]["synced"] is True
    assert calendar_service.added_user_ids == ["ou_test"]


def test_orchestrator_fills_missing_slot_then_confirms_pending_action() -> None:
    class FakeIntentClassifier:
        async def classify(self, message):
            assert message == "我今晚有个面试，深信服二面"
            return IntentClassification(
                intent=IntentName.ADD_APPLICATION,
                confidence=0.76,
                slots={
                    "company": "深信服",
                    "interview_time": "今晚",
                    "round": "二面",
                },
            )

    repository = InMemoryOfferPilotRepository()
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)
    orchestrator = AgentOrchestrator(
        intent_classifier=FakeIntentClassifier(),
        tool_registry=registry,
        conversation_store=InMemoryConversationStore(),
    )

    first = asyncio.run(
        orchestrator.handle_message(
            "我今晚有个面试，深信服二面",
            user_id="ou_test",
            source="feishu",
        )
    )
    second = asyncio.run(
        orchestrator.handle_message(
            "岗位是 AI 应用开发",
            user_id="ou_test",
            source="feishu",
        )
    )
    third = asyncio.run(
        orchestrator.handle_message(
            "确认",
            user_id="ou_test",
            source="feishu",
        )
    )

    assert first.action == AgentActionName.ASK_CLARIFICATION
    assert first.missing_slots == ["role"]
    assert second.action == AgentActionName.CREATE_APPLICATION
    assert second.need_confirmation is True
    assert second.slots["role"] == "AI 应用开发"
    assert third.action == AgentActionName.CREATE_APPLICATION
    assert third.tool_result is not None
    assert third.tool_result.success is True
    assert repository.applications[0].company == "深信服"
    assert repository.applications[0].role == "AI 应用开发"
    assert repository.applications[0].round == "二面"


def test_orchestrator_accepts_natural_confirmation_synonym_for_pending_action() -> None:
    class FakeIntentClassifier:
        async def classify(self, message):
            assert message == "我今天投递了农夫山泉公司的 AI 应用开发岗位"
            return IntentClassification(
                intent=IntentName.ADD_APPLICATION,
                confidence=0.82,
                slots={
                    "company": "农夫山泉",
                    "role": "AI 应用开发",
                },
            )

    repository = InMemoryOfferPilotRepository()
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)
    orchestrator = AgentOrchestrator(
        intent_classifier=FakeIntentClassifier(),
        tool_registry=registry,
        conversation_store=InMemoryConversationStore(),
    )

    first = asyncio.run(
        orchestrator.handle_message(
            "我今天投递了农夫山泉公司的 AI 应用开发岗位",
            user_id="ou_test",
            source="feishu",
        )
    )
    second = asyncio.run(
        orchestrator.handle_message(
            "对的",
            user_id="ou_test",
            source="feishu",
        )
    )

    assert first.need_confirmation is True
    assert second.action == AgentActionName.CREATE_APPLICATION
    assert second.tool_result is not None
    assert second.tool_result.success is True
    assert repository.applications[0].company == "农夫山泉"
    assert repository.applications[0].role == "AI 应用开发"


def test_orchestrator_updates_pending_application_company_from_correction() -> None:
    class FakeIntentClassifier:
        async def classify(self, message):
            assert message == "我今天投递了农夫山泉公司的 AI 应用开发岗位"
            return IntentClassification(
                intent=IntentName.ADD_APPLICATION,
                confidence=0.82,
                slots={
                    "company": "农夫山泉公司",
                    "role": "AI 应用开发",
                },
            )

    repository = InMemoryOfferPilotRepository()
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)
    orchestrator = AgentOrchestrator(
        intent_classifier=FakeIntentClassifier(),
        tool_registry=registry,
        conversation_store=InMemoryConversationStore(),
    )

    first = asyncio.run(
        orchestrator.handle_message(
            "我今天投递了农夫山泉公司的 AI 应用开发岗位",
            user_id="ou_test",
            source="feishu",
        )
    )
    second = asyncio.run(
        orchestrator.handle_message(
            "公司错了，公司是 农夫山泉",
            user_id="ou_test",
            source="feishu",
        )
    )
    third = asyncio.run(
        orchestrator.handle_message(
            "确认",
            user_id="ou_test",
            source="feishu",
        )
    )

    assert first.need_confirmation is True
    assert second.action == AgentActionName.CREATE_APPLICATION
    assert second.need_confirmation is True
    assert second.slots["company"] == "农夫山泉"
    assert third.tool_result is not None
    assert third.tool_result.success is True
    assert repository.applications[0].company == "农夫山泉"
    assert repository.applications[0].role == "AI 应用开发"


def test_orchestrator_does_not_claim_pending_application_was_recorded() -> None:
    class FakeLLMService:
        def __init__(self):
            self.calls = 0

        async def generate_text(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                content = {
                    "intent": "add_application",
                    "confidence": 0.9,
                    "action": "create_application",
                    "reply": "好的，我来帮你记录投递农夫山泉公司的 AI 应用开发岗位。",
                    "need_confirmation": False,
                    "slots": {"company": "农夫山泉", "role": "AI 应用开发"},
                    "missing_slots": [],
                    "steps": [],
                    "reason": "create application",
                }
            else:
                content = {
                    "intent": "ask_help",
                    "confidence": 0.9,
                    "action": "answer_help",
                    "reply": "好的，已经帮你记录投递农夫山泉公司的 AI 应用开发岗位。",
                    "need_confirmation": False,
                    "slots": {},
                    "missing_slots": [],
                    "steps": [],
                    "reason": "answer status",
                }
            return LLMResult(
                provider="fake",
                model="fake",
                content=json.dumps(content, ensure_ascii=False),
                raw_response={},
            )

    repository = InMemoryOfferPilotRepository()
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)
    planner = AgentPlanner(llm_service=FakeLLMService(), llm_planner_enabled=True)
    orchestrator = AgentOrchestrator(
        planner=planner,
        tool_registry=registry,
        conversation_store=InMemoryConversationStore(),
    )

    first = asyncio.run(
        orchestrator.handle_message(
            "小猪，我今天投递了 农夫山泉公司的 ai应用开发岗位",
            user_id="ou_test",
            source="feishu",
        )
    )
    assert first.action == AgentActionName.CREATE_APPLICATION
    assert first.need_confirmation is True
    assert "确认" in first.reply
    assert first.tool_result is None
    assert repository.applications == []

    second = asyncio.run(
        orchestrator.handle_message(
            "你记录了吗",
            user_id="ou_test",
            source="feishu",
        )
    )
    assert second.action == AgentActionName.CREATE_APPLICATION
    assert second.need_confirmation is True
    assert second.tool_result is None
    assert "还没有" in second.reply
    assert "确认" in second.reply
    assert repository.applications == []

    third = asyncio.run(
        orchestrator.handle_message(
            "确认",
            user_id="ou_test",
            source="feishu",
        )
    )

    assert third.tool_result is not None
    assert third.tool_result.success is True
    assert repository.applications[0].company == "农夫山泉"
    assert repository.applications[0].role == "AI 应用开发"


def test_orchestrator_repairs_withdraw_all_company_update_plan() -> None:
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
                        "reply": "我识别到你想更新 vivo 的投递状态。下一步会定位投递记录并更新状态，执行前需要你确认。",
                        "need_confirmation": True,
                        "slots": {"company": "vivo"},
                        "missing_slots": [],
                        "steps": [],
                        "reason": "update application",
                    },
                    ensure_ascii=False,
                ),
                raw_response={},
            )

    repository = InMemoryOfferPilotRepository()
    repository.create_application(company="vivo", role="AI 应用开发")
    repository.create_application(company="vivo", role="Java 后端")
    registry = build_offerpilot_tool_registry(repository, calendar_service=None, bitable_service=None)
    planner = AgentPlanner(llm_service=FakeLLMService(), llm_planner_enabled=True)
    orchestrator = AgentOrchestrator(
        planner=planner,
        tool_registry=registry,
        conversation_store=InMemoryConversationStore(),
    )

    first = asyncio.run(
        orchestrator.handle_message(
            "我取消投递了 vivo 的所有岗位，我不想去 vivo 了",
            user_id="local_user",
            source="api",
        )
    )
    second = asyncio.run(
        orchestrator.handle_message(
            "确认",
            user_id="local_user",
            source="api",
        )
    )

    assert first.action == AgentActionName.UPDATE_APPLICATION
    assert first.need_confirmation is True
    assert first.slots["status"] == "withdrawn"
    assert first.slots["apply_to_all"] is True
    assert "放弃" in first.reply
    assert second.tool_result is not None
    assert second.tool_result.success is True
    assert [application.status.value for application in repository.list_applications(company="vivo")] == [
        "withdrawn",
        "withdrawn",
    ]


def test_orchestrator_confirm_without_pending_action_returns_guidance() -> None:
    orchestrator = AgentOrchestrator(conversation_store=InMemoryConversationStore())

    result = asyncio.run(
        orchestrator.handle_message(
            "确认",
            user_id="ou_test",
            source="feishu",
        )
    )

    assert result.action == AgentActionName.NO_OP
    assert "没有待确认" in result.reply


def test_orchestrator_requires_candidate_selection_before_updating_application() -> None:
    class FakeIntentClassifier:
        async def classify(self, message):
            return IntentClassification(
                intent=IntentName.UPDATE_APPLICATION,
                confidence=0.9,
                slots={"company": "美团", "update_type": "reject", "status": "rejected"},
            )

    repository = InMemoryOfferPilotRepository()
    first_application = repository.create_application(company="美团", role="Java 后端")
    second_application = repository.create_application(company="美团", role="AI 应用开发")
    conversation_store = InMemoryConversationStore()
    orchestrator = AgentOrchestrator(
        intent_classifier=FakeIntentClassifier(),
        tool_registry=build_offerpilot_tool_registry(
            repository,
            calendar_service=None,
            bitable_service=None,
        ),
        conversation_store=conversation_store,
    )

    planned = asyncio.run(orchestrator.handle_message("美团一面挂了"))
    candidates = asyncio.run(orchestrator.handle_message("确认"))
    selected = asyncio.run(orchestrator.handle_message(f"application_id: {first_application.id}"))
    executed = asyncio.run(orchestrator.handle_message("确认"))

    assert planned.need_confirmation is True
    assert candidates.missing_slots == ["application_id"]
    assert candidates.tool_result is not None
    assert candidates.tool_result.data["requires_selection"] is True
    assert selected.slots["application_id"] == first_application.id
    assert selected.need_confirmation is True
    assert executed.tool_result is not None
    assert executed.tool_result.success is True
    assert repository.list_applications()[0].status.value == "rejected"
    assert repository.list_applications()[1].id == second_application.id
    assert repository.list_applications()[1].status.value == "planned"


def test_orchestrator_reschedules_and_cancels_interview_with_rule_planner() -> None:
    repository = InMemoryOfferPilotRepository()
    application = repository.create_application(company="美团", role="Java 后端", round_name="一面")
    schedule = repository.create_interview_schedule(
        application_id=application.id,
        company=application.company,
        role=application.role,
        round_name="一面",
        start_time="明天下午三点",
    )
    orchestrator = AgentOrchestrator(
        tool_registry=build_offerpilot_tool_registry(
            repository,
            calendar_service=None,
            bitable_service=None,
        ),
        conversation_store=InMemoryConversationStore(),
    )

    reschedule_plan = asyncio.run(orchestrator.handle_message("美团一面改到后天下午四点"))
    rescheduled = asyncio.run(orchestrator.handle_message("确认"))
    cancel_plan = asyncio.run(orchestrator.handle_message("取消美团一面"))
    cancelled = asyncio.run(orchestrator.handle_message("确认"))

    assert reschedule_plan.action == AgentActionName.RESCHEDULE_INTERVIEW
    assert reschedule_plan.need_confirmation is True
    assert rescheduled.tool_result is not None
    assert rescheduled.tool_result.success is True
    assert rescheduled.tool_result.data["interview_schedule"]["id"] == schedule.id
    assert rescheduled.tool_result.data["interview_schedule"]["start_time"] == "后天下午四点"
    assert cancel_plan.action == AgentActionName.CANCEL_INTERVIEW
    assert cancel_plan.need_confirmation is True
    assert cancelled.tool_result is not None
    assert cancelled.tool_result.data["interview_schedule"]["status"] == "cancelled"


def test_scenario_one_application_schedule_reschedule_cancel_end_to_end() -> None:
    class RuleIntentClassifier:
        async def classify(self, message):
            return IntentClassifier()._classify_by_rules(message)

    class FakeCalendarService:
        calendar_id = "primary"

        def __init__(self):
            self.created = []
            self.updated = []
            self.deleted = []

        def is_calendar_sync_enabled(self):
            return True

        def should_manage_offerpilot_calendar(self):
            return False

        def create_interview_event(self, **kwargs):
            self.created.append(kwargs)
            return FeishuCalendarEventResult(event_id="evt_scenario_1", raw_response={"code": 0})

        def update_interview_event(self, **kwargs):
            self.updated.append(kwargs)
            return FeishuCalendarEventResult(event_id=kwargs["event_id"], raw_response={"code": 0})

        def delete_interview_event(self, **kwargs):
            self.deleted.append(kwargs)
            return FeishuCalendarEventResult(event_id=kwargs["event_id"], raw_response={"code": 0})

    class FakeBitableService:
        app_token = "bascn_offerpilot"
        table_id = "tbl_applications"

        def __init__(self):
            self.updates = []

        def is_bitable_sync_enabled(self):
            return True

        def should_manage_offerpilot_bitable(self):
            return False

        def update_record(self, **kwargs):
            self.updates.append(kwargs["fields"])
            return FeishuBitableRecordResult(record_id=kwargs["record_id"], raw_response={"code": 0})

    repository = InMemoryOfferPilotRepository()
    repository.set_runtime_setting(
        "feishu.offerpilot_bitable_record_id.tbl_applications.app_1",
        "rec_app_1",
    )
    calendar_service = FakeCalendarService()
    bitable_service = FakeBitableService()
    orchestrator = AgentOrchestrator(
        intent_classifier=RuleIntentClassifier(),
        tool_registry=build_offerpilot_tool_registry(repository, calendar_service, bitable_service),
        conversation_store=InMemoryConversationStore(),
    )

    asyncio.run(orchestrator.handle_message("我投递了美团 Java 后端实习"))
    created = asyncio.run(orchestrator.handle_message("确认"))
    asyncio.run(orchestrator.handle_message("美团约我明天下午三点一面"))
    scheduled = asyncio.run(orchestrator.handle_message("确认"))
    asyncio.run(orchestrator.handle_message("美团一面改到后天下午四点"))
    rescheduled = asyncio.run(orchestrator.handle_message("确认"))
    asyncio.run(orchestrator.handle_message("取消美团一面"))
    cancelled = asyncio.run(orchestrator.handle_message("确认"))

    assert created.tool_result is not None and created.tool_result.success is True
    assert scheduled.tool_result is not None and scheduled.tool_result.success is True
    assert rescheduled.tool_result is not None
    assert "operation_status" in rescheduled.tool_result.data, rescheduled.tool_result.data
    assert rescheduled.tool_result.data["operation_status"] == "completed"
    assert cancelled.tool_result is not None
    assert cancelled.tool_result.data["operation_status"] == "completed"
    assert len(calendar_service.created) == 1
    assert len(calendar_service.updated) == 1
    assert len(calendar_service.deleted) == 1
    assert bitable_service.updates[-1]["面试时间文本"] is None
    assert repository.interview_schedules[0].status.value == "cancelled"


def test_debug_agent_route_returns_orchestrated_response(monkeypatch) -> None:
    class FakeAgentOrchestrator:
        async def handle_message(self, message, confirmed=False, user_id="local_user", source="api"):
            assert message == "今天任务是什么？"
            assert confirmed is False
            assert user_id == "local_user"
            assert source == "api"
            return AgentResponse(
                intent=IntentName.GET_TODAY_TASKS,
                confidence=0.9,
                action=AgentActionName.LIST_TODAY_TASKS,
                reply="我识别到你想查看今日任务。",
                slots={},
            )

    monkeypatch.setattr(debug, "AgentOrchestrator", FakeAgentOrchestrator)
    client = TestClient(app)

    response = client.post(
        "/api/debug/agent",
        json={"message": "今天任务是什么？"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "intent": "get_today_tasks",
        "confidence": 0.9,
        "action": "list_today_tasks",
        "reply": "我识别到你想查看今日任务。",
        "need_confirmation": False,
        "slots": {},
        "missing_slots": [],
    }
