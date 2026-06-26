import json
import re
from typing import Any, Dict, Optional

from pydantic import ValidationError

from app.schemas.intent import IntentClassification, IntentName
from app.services.llm_service import LLMConfigurationError, LLMService


class IntentClassifier:
    SYSTEM_PROMPT = """
You are the intent classifier for OfferPilot, a personal job-search execution agent
for computer science campus recruiting. Classify the user's Chinese or English
message into one of the allowed intents and extract only useful slots.

Allowed intents:
- get_today_tasks: user wants today's tasks or schedule.
- complete_task: user reports finishing a task, LeetCode problem, review item, or drill.
- postpone_task: user wants to postpone, skip, or delay a task.
- add_application: user wants to add a new job application or interview arrangement.
- update_application: user wants to update an existing application status.
- add_interview_review: user submits or starts an interview review.
- start_mock_interview: user wants to start a mock interview.
- answer_question: user is answering an interview question or task prompt.
- ask_help: user asks for general help or explanation.
- summarize_week: user wants a weekly review.
- unknown: none of the above is clear.

Return strict JSON only, without markdown fences:
{
  "intent": "add_application",
  "confidence": 0.86,
  "slots": {
    "company": "深信服",
    "role": "开发实习生",
    "interview_time": "明天下午三点"
  }
}

Slot guidance:
- Keep relative time expressions as written, such as "明天下午三点".
- Use short scalar values or arrays only.
- Do not invent missing user data.
""".strip()

    def __init__(self, llm_service: Optional[LLMService] = None) -> None:
        self.llm_service = llm_service or LLMService()

    async def classify(self, message: str) -> IntentClassification:
        try:
            result = await self.llm_service.generate_text(
                prompt=self._build_prompt(message),
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=512,
            )
        except LLMConfigurationError:
            return self._classify_by_rules(message)

        try:
            payload = self._parse_json_object(result.content)
            return IntentClassification(**payload)
        except (json.JSONDecodeError, TypeError, ValidationError):
            return self._classify_by_rules(message)

    @staticmethod
    def _build_prompt(message: str) -> str:
        return f"User message:\n{message.strip()}"

    @staticmethod
    def _parse_json_object(content: str) -> Dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise json.JSONDecodeError("No JSON object found", text, 0)

        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise TypeError("Intent classification payload must be a JSON object.")

        return parsed

    def _classify_by_rules(self, message: str) -> IntentClassification:
        text = message.strip()
        compact = re.sub(r"\s+", "", text).lower()

        if self._is_today_task_query(compact):
            return self._result(IntentName.GET_TODAY_TASKS, 0.72)

        if self._is_add_application(compact):
            return self._result(
                IntentName.ADD_APPLICATION,
                0.76,
                self._extract_application_slots(text),
            )

        if self._is_interview_review(compact):
            return self._result(
                IntentName.ADD_INTERVIEW_REVIEW,
                0.72,
                self._extract_interview_review_slots(text),
            )

        if self._is_mock_interview(compact):
            return self._result(
                IntentName.START_MOCK_INTERVIEW,
                0.74,
                self._extract_mock_interview_slots(text),
            )

        if self._is_weekly_summary(compact):
            return self._result(IntentName.SUMMARIZE_WEEK, 0.7)

        if self._is_complete_task(compact):
            return self._result(
                IntentName.COMPLETE_TASK,
                0.7,
                self._extract_task_slots(text),
            )

        if self._is_postpone_task(compact):
            return self._result(
                IntentName.POSTPONE_TASK,
                0.68,
                self._extract_task_slots(text),
            )

        if self._is_update_application(compact):
            return self._result(
                IntentName.UPDATE_APPLICATION,
                0.65,
                self._extract_application_slots(text),
            )

        if self._looks_like_answer(text, compact):
            return self._result(IntentName.ANSWER_QUESTION, 0.6)

        if self._is_help_request(compact):
            return self._result(IntentName.ASK_HELP, 0.58)

        return self._result(IntentName.UNKNOWN, 0.35)

    @staticmethod
    def _result(
        intent: IntentName,
        confidence: float,
        slots: Optional[Dict[str, Any]] = None,
    ) -> IntentClassification:
        return IntentClassification(
            intent=intent,
            confidence=confidence,
            slots=slots or {},
        )

    @staticmethod
    def _is_today_task_query(compact: str) -> bool:
        return compact in {"/today", "today"} or (
            ("今天" in compact or "今日" in compact) and any(word in compact for word in ("任务", "安排", "计划"))
        )

    @staticmethod
    def _is_add_application(compact: str) -> bool:
        return "新增投递" in compact or "添加投递" in compact or (
            "投递" in compact and any(word in compact for word in ("岗位", "实习", "一面", "二面", "笔试", "面试"))
        )

    @staticmethod
    def _is_update_application(compact: str) -> bool:
        status_words = ("offer", "挂了", "拒了", "通过", "约了", "收到面试", "进入二面", "hr面", "已投递")
        return any(word in compact for word in status_words) and any(
            word in compact for word in ("投递", "公司", "面试", "一面", "二面", "三面")
        )

    @staticmethod
    def _is_interview_review(compact: str) -> bool:
        return "复盘" in compact and any(word in compact for word in ("面试", "一面", "二面", "三面", "hr", "笔试"))

    @staticmethod
    def _is_mock_interview(compact: str) -> bool:
        return "模拟面试" in compact or "mockinterview" in compact

    @staticmethod
    def _is_weekly_summary(compact: str) -> bool:
        return any(word in compact for word in ("周复盘", "本周复盘", "这周复盘", "周总结", "本周总结"))

    @staticmethod
    def _is_complete_task(compact: str) -> bool:
        return any(word in compact for word in ("完成", "做完", "刷完", "打卡", "/done"))

    @staticmethod
    def _is_postpone_task(compact: str) -> bool:
        return any(word in compact for word in ("延期", "推迟", "改到明天", "明天再做", "跳过"))

    @staticmethod
    def _looks_like_answer(text: str, compact: str) -> bool:
        technical_words = ("hashmap", "redis", "mysql", "spring", "jvm", "链表", "复杂度", "缓存", "线程池")
        answer_words = ("思路", "我觉得", "我的回答", "可以用", "时间复杂度", "空间复杂度")
        return len(text) >= 20 and (
            any(word in compact for word in answer_words) or any(word in compact for word in technical_words)
        )

    @staticmethod
    def _is_help_request(compact: str) -> bool:
        return any(word in compact for word in ("怎么", "如何", "帮我", "解释", "为什么", "建议"))

    def _extract_application_slots(self, text: str) -> Dict[str, Any]:
        slots: Dict[str, Any] = {}
        normalized = re.sub(r"^(新增|添加|记录)?投递[:：]?", "", text.strip())
        first_part = re.split(r"[，,。；;]", normalized, maxsplit=1)[0].strip()

        role_match = re.search(
            r"(java\s*后端|java\s*开发|python\s*后端|ai\s*应用开发|agent\s*开发|后端开发|开发实习生?|开发实习|算法工程师?|实习生?)",
            first_part,
            flags=re.IGNORECASE,
        )
        if role_match:
            company = first_part[: role_match.start()].strip(" ：:,，")
            role = role_match.group(0).strip()
            if company:
                slots["company"] = company
            slots["role"] = role
        elif first_part:
            slots["company"] = first_part

        if "role" not in slots:
            separated_role = re.search(r"(岗位|职位|方向)[是:：]?([^，,。；;]+)", text)
            if separated_role:
                slots["role"] = separated_role.group(2).strip()

        interview_time = self._extract_time_expression(text)
        if interview_time:
            slots["interview_time"] = interview_time

        round_match = re.search(r"(笔试|一面|二面|三面|hr\s*面|HR\s*面)", text, flags=re.IGNORECASE)
        if round_match:
            slots["round"] = round_match.group(1)

        keywords = self._extract_keywords(text)
        if keywords:
            slots["jd_keywords"] = keywords

        return slots

    def _extract_interview_review_slots(self, text: str) -> Dict[str, Any]:
        slots: Dict[str, Any] = {}
        company_match = re.search(
            r"复盘(?:一下)?([^：:，,。；;\s]+?)(一面|二面|三面|hr\s*面|HR\s*面|面试)",
            text,
            flags=re.IGNORECASE,
        )
        if company_match and company_match.group(1):
            company = company_match.group(1).strip()
            if company and not any(word in company for word in ("一下", "今天", "这次", "面试", "的")):
                slots["company"] = company
        if company_match and company_match.group(2):
            round_name = company_match.group(2)
            if round_name != "面试":
                slots["round"] = round_name

        keywords = self._extract_keywords(text)
        if keywords:
            slots["topics"] = keywords

        return slots

    def _extract_mock_interview_slots(self, text: str) -> Dict[str, Any]:
        slots: Dict[str, Any] = {}
        role_match = re.search(r"(岗位|职位)[是:：]?([^，,。；;]+)", text)
        if role_match:
            slots["role"] = role_match.group(2).strip()

        project_match = re.search(r"项目(?:选|问|是|：|:)?([^，,。；;\s]+)", text)
        if project_match:
            slots["project"] = project_match.group(1).strip()

        return slots

    def _extract_task_slots(self, text: str) -> Dict[str, Any]:
        slots: Dict[str, Any] = {}
        compact = re.sub(r"\s+", "", text).lower()
        if any(word in compact for word in ("leetcode", "力扣", "反转链表")):
            slots["task_type"] = "leetcode"
        elif any(word in compact for word in ("八股", "hashmap", "redis", "mysql", "jvm")):
            slots["task_type"] = "interview_question"
        elif "项目" in compact:
            slots["task_type"] = "project_deep_dive"

        title_match = re.search(r"(?:完成|做完|刷完|打卡|延期|推迟)([^，,。；;]+)", text)
        if title_match:
            title = title_match.group(1).strip(" 了")
            if title:
                slots["task_title"] = title

        return slots

    @staticmethod
    def _extract_time_expression(text: str) -> Optional[str]:
        match = re.search(
            r"((?:今天|明天|后天|大后天|下周[一二三四五六日天]?|周[一二三四五六日天])?"
            r"(?:早上|上午|中午|下午|晚上)?"
            r"(?:[零一二三四五六七八九十两\d]{1,3})点(?:半|[零一二三四五六七八九十\d]{1,2}分)?)",
            text,
        )
        if not match:
            return None
        return match.group(1)

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        keyword_patterns = {
            "Java": r"\bjava\b|Java",
            "Spring Boot": r"spring\s*boot",
            "MySQL": r"\bmysql\b|MySQL",
            "Redis": r"\bredis\b|Redis",
            "RAG": r"\brag\b|RAG",
            "Agent": r"\bagent\b|Agent",
            "LLM": r"\bllm\b|LLM|大模型",
            "HashMap": r"hashmap|HashMap",
            "JVM": r"\bjvm\b|JVM",
        }
        return [
            label
            for label, pattern in keyword_patterns.items()
            if re.search(pattern, text, flags=re.IGNORECASE)
        ]
