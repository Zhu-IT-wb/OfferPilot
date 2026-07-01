import asyncio

from fastapi.testclient import TestClient

from app.agents.intent_classifier import IntentClassifier
from app.api.routes import debug
from app.main import app
from app.schemas.intent import IntentClassification, IntentName
from app.services.llm_service import LLMConfigurationError, LLMRequestError, LLMResult


def test_intent_classifier_parses_llm_json() -> None:
    class FakeLLMService:
        async def generate_text(self, **kwargs):
            assert kwargs["temperature"] == 0.0
            assert "User message:" in kwargs["prompt"]
            return LLMResult(
                provider="deepseek",
                model="deepseek-v4-flash",
                content=(
                    "```json\n"
                    '{"intent":"add_application","confidence":0.86,'
                    '"slots":{"company":"深信服","role":"开发实习生",'
                    '"interview_time":"明天下午三点"}}\n'
                    "```"
                ),
                raw_response={},
            )

    classifier = IntentClassifier(llm_service=FakeLLMService())

    result = asyncio.run(classifier.classify("新增投递深信服开发实习，明天下午三点一面"))

    assert result.intent == IntentName.ADD_APPLICATION
    assert result.confidence == 0.86
    assert result.slots == {
        "company": "深信服",
        "role": "开发实习生",
        "interview_time": "明天下午三点",
    }


def test_intent_classifier_falls_back_without_api_key() -> None:
    class FakeLLMService:
        async def generate_text(self, **kwargs):
            raise LLMConfigurationError("missing api key")

    classifier = IntentClassifier(llm_service=FakeLLMService())

    result = asyncio.run(classifier.classify("今天任务是什么？"))

    assert result.intent == IntentName.GET_TODAY_TASKS
    assert 0.0 <= result.confidence <= 1.0
    assert result.slots == {}


def test_intent_classifier_falls_back_when_llm_request_fails() -> None:
    class FakeLLMService:
        async def generate_text(self, **kwargs):
            raise LLMRequestError("network failed")

    classifier = IntentClassifier(llm_service=FakeLLMService())

    result = asyncio.run(classifier.classify("今天任务是什么？"))

    assert result.intent == IntentName.GET_TODAY_TASKS
    assert result.slots == {}


def test_intent_classifier_rule_extracts_application_slots() -> None:
    class FakeLLMService:
        async def generate_text(self, **kwargs):
            raise LLMConfigurationError("missing api key")

    classifier = IntentClassifier(llm_service=FakeLLMService())

    result = asyncio.run(classifier.classify("新增投递深信服开发实习，明天下午三点一面"))

    assert result.intent == IntentName.ADD_APPLICATION
    assert result.slots["company"] == "深信服"
    assert result.slots["role"] == "开发实习"
    assert result.slots["interview_time"] == "明天下午三点"
    assert result.slots["round"] == "一面"


def test_intent_classifier_rule_extracts_interview_arrangement_without_application_keyword() -> None:
    class FakeLLMService:
        async def generate_text(self, **kwargs):
            raise LLMConfigurationError("missing api key")

    classifier = IntentClassifier(llm_service=FakeLLMService())

    result = asyncio.run(classifier.classify("我今晚有个 AI 应用开发面试，深信服二面"))

    assert result.intent == IntentName.ADD_APPLICATION
    assert result.slots["company"] == "深信服"
    assert result.slots["role"] == "AI 应用开发"
    assert result.slots["interview_time"] == "今晚"
    assert result.slots["round"] == "二面"


def test_intent_classifier_rule_extracts_application_list_query() -> None:
    class FakeLLMService:
        async def generate_text(self, **kwargs):
            raise LLMConfigurationError("missing api key")

    classifier = IntentClassifier(llm_service=FakeLLMService())

    result = asyncio.run(classifier.classify("我投了哪些公司？"))

    assert result.intent == IntentName.QUERY_APPLICATION
    assert result.slots["query_type"] == "list"


def test_intent_classifier_rule_extracts_company_status_query() -> None:
    class FakeLLMService:
        async def generate_text(self, **kwargs):
            raise LLMConfigurationError("missing api key")

    classifier = IntentClassifier(llm_service=FakeLLMService())

    result = asyncio.run(classifier.classify("深信服现在什么状态？"))

    assert result.intent == IntentName.QUERY_APPLICATION
    assert result.slots["query_type"] == "company_status"
    assert result.slots["company"] == "深信服"


def test_intent_classifier_rule_extracts_upcoming_interview_query() -> None:
    class FakeLLMService:
        async def generate_text(self, **kwargs):
            raise LLMConfigurationError("missing api key")

    classifier = IntentClassifier(llm_service=FakeLLMService())

    result = asyncio.run(classifier.classify("最近有哪些面试？"))

    assert result.intent == IntentName.QUERY_APPLICATION
    assert result.slots["query_type"] == "upcoming_interviews"


def test_intent_classifier_rule_extracts_application_update_schedule() -> None:
    class FakeLLMService:
        async def generate_text(self, **kwargs):
            raise LLMConfigurationError("missing api key")

    classifier = IntentClassifier(llm_service=FakeLLMService())

    result = asyncio.run(classifier.classify("明天早上八点，深信服约我一面"))

    assert result.intent == IntentName.UPDATE_APPLICATION
    assert result.slots["company"] == "深信服"
    assert result.slots["round"] == "一面"
    assert result.slots["interview_time"] == "明天早上八点"
    assert result.slots["update_type"] == "schedule_interview"
    assert result.slots["status"] == "interview_1"


def test_intent_classifier_rule_extracts_application_round_passed() -> None:
    class FakeLLMService:
        async def generate_text(self, **kwargs):
            raise LLMConfigurationError("missing api key")

    classifier = IntentClassifier(llm_service=FakeLLMService())

    result = asyncio.run(classifier.classify("深信服一面过了"))

    assert result.intent == IntentName.UPDATE_APPLICATION
    assert result.slots["company"] == "深信服"
    assert result.slots["round"] == "一面"
    assert result.slots["update_type"] == "pass_round"
    assert result.slots["status"] == "interview_1_passed"


def test_intent_classifier_rule_extracts_application_offer() -> None:
    class FakeLLMService:
        async def generate_text(self, **kwargs):
            raise LLMConfigurationError("missing api key")

    classifier = IntentClassifier(llm_service=FakeLLMService())

    result = asyncio.run(classifier.classify("深信服给 offer 了"))

    assert result.intent == IntentName.UPDATE_APPLICATION
    assert result.slots["company"] == "深信服"
    assert result.slots["update_type"] == "offer"
    assert result.slots["status"] == "offer"


def test_intent_classifier_rule_handles_generic_interview_review() -> None:
    class FakeLLMService:
        async def generate_text(self, **kwargs):
            raise LLMConfigurationError("missing api key")

    classifier = IntentClassifier(llm_service=FakeLLMService())

    result = asyncio.run(classifier.classify("复盘一下今天的面试"))

    assert result.intent == IntentName.ADD_INTERVIEW_REVIEW
    assert "company" not in result.slots


def test_debug_intent_route_returns_structured_intent(monkeypatch) -> None:
    class FakeIntentClassifier:
        async def classify(self, message):
            assert message == "开始模拟面试，项目问云聚图库"
            return IntentClassification(
                intent=IntentName.START_MOCK_INTERVIEW,
                confidence=0.91,
                slots={"project": "云聚图库"},
            )

    monkeypatch.setattr(debug, "IntentClassifier", FakeIntentClassifier)
    client = TestClient(app)

    response = client.post(
        "/api/debug/intent",
        json={"message": "开始模拟面试，项目问云聚图库"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "intent": "start_mock_interview",
        "confidence": 0.91,
        "slots": {"project": "云聚图库"},
    }
