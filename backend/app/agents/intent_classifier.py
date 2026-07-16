import json
import re
from typing import Any, Dict, Optional

from pydantic import ValidationError

from app.schemas.intent import IntentClassification, IntentName
from app.services.llm_service import LLMConfigurationError, LLMRequestError, LLMService


# 将用户自然语言识别为秋招业务意图。
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
- query_application: user wants to query job applications, company status, or upcoming interviews.
- update_application: user wants to update an existing application status, such as
  pass/fail, offer, or a newly scheduled interview for an existing application.
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
- For update_application, use update_type when clear:
  schedule_interview, reschedule_interview, cancel_interview, pass_round, reject,
  offer, submitted, or status_update.
- For schedule_interview, use calendar_reminder when the user clearly says whether
  Feishu calendar reminder is needed. Use true for "需要提醒/同步日历", false for
  "不用提醒/不同步日历". Omit it when unclear.
- Use short scalar values or arrays only.
- Do not invent missing user data.
""".strip()

    # 初始化当前组件所需的依赖和配置。
    def __init__(self, llm_service: Optional[LLMService] = None) -> None:
        self.llm_service = llm_service or LLMService()

    # 执行自然语言意图识别。
    async def classify(self, message: str) -> IntentClassification:
        try:
            result = await self.llm_service.generate_text(
                prompt=self._build_prompt(message),
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=512,
            )
        except (LLMConfigurationError, LLMRequestError):
            return self._classify_by_rules(message)

        try:
            payload = self._parse_json_object(result.content)
            return IntentClassification(**payload)
        except (json.JSONDecodeError, TypeError, ValidationError):
            return self._classify_by_rules(message)

    # 构造 prompt。
    @staticmethod
    def _build_prompt(message: str) -> str:
        return f"User message:\n{message.strip()}"

    # 解析 json object。
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

    # 处理 classify_by_rules 相关逻辑。
    def _classify_by_rules(self, message: str) -> IntentClassification:
        text = message.strip()
        compact = re.sub(r"\s+", "", text).lower()

        if self._is_today_task_query(compact):
            return self._result(IntentName.GET_TODAY_TASKS, 0.72)

        if self._is_interview_review(compact):
            return self._result(
                IntentName.ADD_INTERVIEW_REVIEW,
                0.72,
                self._extract_interview_review_slots(text),
            )

        if self._is_query_application(compact):
            return self._result(
                IntentName.QUERY_APPLICATION,
                0.74,
                self._extract_application_query_slots(text),
            )

        if self._is_update_application(compact):
            return self._result(
                IntentName.UPDATE_APPLICATION,
                0.72,
                self._extract_application_update_slots(text),
            )

        if self._is_add_application(compact):
            return self._result(
                IntentName.ADD_APPLICATION,
                0.76,
                self._extract_application_slots(text),
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

        if self._looks_like_answer(text, compact):
            return self._result(IntentName.ANSWER_QUESTION, 0.6)

        if self._is_help_request(compact):
            return self._result(IntentName.ASK_HELP, 0.58)

        return self._result(IntentName.UNKNOWN, 0.35)

    # 处理 result 相关逻辑。
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

    # 判断 today task query 是否成立。
    @staticmethod
    def _is_today_task_query(compact: str) -> bool:
        return compact in {"/today", "today"} or (
            ("今天" in compact or "今日" in compact) and any(word in compact for word in ("任务", "安排", "计划"))
        )

    # 判断 add application 是否成立。
    @staticmethod
    def _is_add_application(compact: str) -> bool:
        return "新增投递" in compact or "添加投递" in compact or (
            "投递" in compact and any(word in compact for word in ("岗位", "实习", "一面", "二面", "笔试", "面试"))
        ) or (
            "面试" in compact and any(word in compact for word in ("今晚", "今天", "明天", "一面", "二面", "三面", "hr面"))
        ) or (
            "约我" in compact and any(word in compact for word in ("笔试", "一面", "二面", "三面", "hr面"))
        )

    # 判断 query application 是否成立。
    @staticmethod
    def _is_query_application(compact: str) -> bool:
        if any(word in compact for word in ("投递列表", "投递记录", "投递进度", "投了哪些", "投过哪些")):
            return True
        if any(word in compact for word in ("哪些公司", "哪些岗位")) and any(
            word in compact for word in ("投", "面试", "笔试")
        ):
            return True
        has_time_word = any(word in compact for word in ("最近", "这周", "本周", "明天", "今天"))
        has_interview_word = any(
            word in compact for word in ("面试", "笔试")
        )
        looks_like_new_arrangement = any(word in compact for word in ("约我", "有个", "安排了"))
        if has_time_word and has_interview_word and not looks_like_new_arrangement:
            return True
        if any(word in compact for word in ("什么状态", "什么进度", "到哪", "到哪一步", "进展")):
            return True
        return False

    # 判断 update application 是否成立。
    @staticmethod
    def _is_update_application(compact: str) -> bool:
        round_words = ("笔试", "一面", "二面", "三面", "hr面", "hr")
        has_round = any(word in compact for word in round_words)
        has_schedule_word = any(
            word in compact
            for word in ("约我", "约了", "安排", "通知", "邀我", "邀请", "收到面试", "发来面试")
        )
        has_pass_word = any(word in compact for word in ("过了", "通过", "进了", "进入"))
        has_reschedule_word = any(word in compact for word in ("改期", "改到", "调整到", "推迟到", "提前到"))
        has_cancel_interview_word = "取消" in compact and (has_round or "面试" in compact or "笔试" in compact)
        has_terminal_word = any(
            word in compact
            for word in ("offer", "挂了", "拒了", "拒绝", "没过", "凉了", "取消投递", "撤回投递", "不想去", "放弃")
        )

        if has_schedule_word and (has_round or "面试" in compact or "笔试" in compact):
            return True
        if has_reschedule_word and (has_round or "面试" in compact or "笔试" in compact):
            return True
        if has_cancel_interview_word and "取消投递" not in compact:
            return True
        if has_round and has_pass_word:
            return True
        return has_terminal_word

    # 判断 interview review 是否成立。
    @staticmethod
    def _is_interview_review(compact: str) -> bool:
        return "复盘" in compact and any(word in compact for word in ("面试", "一面", "二面", "三面", "hr", "笔试"))

    # 判断 mock interview 是否成立。
    @staticmethod
    def _is_mock_interview(compact: str) -> bool:
        return "模拟面试" in compact or "mockinterview" in compact

    # 判断 weekly summary 是否成立。
    @staticmethod
    def _is_weekly_summary(compact: str) -> bool:
        return any(word in compact for word in ("周复盘", "本周复盘", "这周复盘", "周总结", "本周总结"))

    # 判断 complete task 是否成立。
    @staticmethod
    def _is_complete_task(compact: str) -> bool:
        return any(word in compact for word in ("完成", "做完", "刷完", "打卡", "/done"))

    # 判断 postpone task 是否成立。
    @staticmethod
    def _is_postpone_task(compact: str) -> bool:
        return any(word in compact for word in ("延期", "推迟", "改到明天", "明天再做", "跳过"))

    # 判断输入是否像 answer。
    @staticmethod
    def _looks_like_answer(text: str, compact: str) -> bool:
        technical_words = ("hashmap", "redis", "mysql", "spring", "jvm", "链表", "复杂度", "缓存", "线程池")
        answer_words = ("思路", "我觉得", "我的回答", "可以用", "时间复杂度", "空间复杂度")
        return len(text) >= 20 and (
            any(word in compact for word in answer_words) or any(word in compact for word in technical_words)
        )

    # 判断 help request 是否成立。
    @staticmethod
    def _is_help_request(compact: str) -> bool:
        return any(word in compact for word in ("怎么", "如何", "帮我", "解释", "为什么", "建议"))

    # 从输入数据中提取 application slots。
    def _extract_application_slots(self, text: str) -> Dict[str, Any]:
        slots: Dict[str, Any] = {}
        normalized = self._normalize_application_message(text)
        first_part = re.split(r"[，,。；;]", normalized, maxsplit=1)[0].strip()

        role_match = re.search(
            r"(java\s*后端|java\s*开发|python\s*后端|ai\s*应用开发|agent\s*开发|后端开发|开发实习生?|开发实习|算法工程师?|实习生?)",
            first_part,
            flags=re.IGNORECASE,
        )
        if role_match:
            company = self._clean_application_company(first_part[: role_match.start()])
            role = role_match.group(0).strip()
            if company:
                slots["company"] = company
            slots["role"] = self._clean_application_role(role)
        elif first_part:
            slots["company"] = self._clean_application_company(first_part)

        if "role" not in slots:
            separated_role = re.search(r"(岗位|职位|方向)[是:：]?([^，,。；;]+)", text)
            if separated_role:
                slots["role"] = self._clean_application_role(separated_role.group(2))

        company_round_match = re.search(
            r"(?P<company>[\u4e00-\u9fa5A-Za-z0-9][\u4e00-\u9fa5A-Za-z0-9_-]{1,20}?)(?P<round>笔试|一面|二面|三面|hr\s*面|HR\s*面)",
            text,
            flags=re.IGNORECASE,
        )
        if company_round_match:
            company = company_round_match.group("company").strip()
            if company and (not slots.get("company") or slots["company"].startswith(("我", "今晚", "今天", "明天"))):
                slots["company"] = company
            slots["round"] = company_round_match.group("round")

        if self._has_interview_timing_context(text):
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

    # 标准化自然投递语句，去掉称呼和“我今天投递了”等叙述前缀。
    @staticmethod
    def _normalize_application_message(text: str) -> str:
        normalized = text.strip()
        greeting_split = re.match(r"^[^，,。；;]{1,12}[，,]\s*(?P<body>.+)$", normalized)
        if greeting_split and "投递" in greeting_split.group("body"):
            normalized = greeting_split.group("body")

        normalized = re.sub(r"^(新增|添加|记录)?投递[:：]?", "", normalized.strip())
        normalized = re.sub(
            r"^(?:我)?(?:今天|今日|昨天|刚刚|已经|现在)?\s*(?:新增|添加|记录)?\s*投递(?:了|过)?\s*",
            "",
            normalized,
        )
        return normalized.strip()

    # 清洗投递公司名。
    @staticmethod
    def _clean_application_company(value: str) -> str:
        cleaned = re.sub(r"\s+", "", value).strip(" ：:,，。；;的")
        cleaned = re.sub(r"(?:公司|企业|厂)?的$", "", cleaned).strip(" ：:,，。；;的")
        if len(cleaned) > 2 and cleaned.endswith("公司"):
            cleaned = cleaned[:-2]
        return cleaned

    # 清洗投递岗位名。
    @staticmethod
    def _clean_application_role(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip(" ：:,，。；;的岗位职位方向")

    # 判断新增投递语句里的时间是否真的在描述笔试/面试安排。
    @staticmethod
    def _has_interview_timing_context(text: str) -> bool:
        compact = re.sub(r"\s+", "", text).lower()
        return any(word in compact for word in ("面试", "笔试", "一面", "二面", "三面", "hr面", "约我", "安排"))

    # 从输入数据中提取 application query slots。
    def _extract_application_query_slots(self, text: str) -> Dict[str, Any]:
        slots: Dict[str, Any] = {}
        compact = re.sub(r"\s+", "", text).lower()

        if any(word in compact for word in ("最近", "这周", "本周", "明天", "今天")) and any(
            word in compact for word in ("面试", "笔试")
        ):
            slots["query_type"] = "upcoming_interviews"
        elif any(word in compact for word in ("什么状态", "什么进度", "到哪", "到哪一步", "进展")):
            slots["query_type"] = "company_status"
        else:
            slots["query_type"] = "list"

        company = self._extract_company_for_query(text)
        if company:
            slots["company"] = company
            if slots["query_type"] == "list":
                slots["query_type"] = "company_status"

        return slots

    # 从输入数据中提取 application update slots。
    def _extract_application_update_slots(self, text: str) -> Dict[str, Any]:
        slots: Dict[str, Any] = {}
        compact = re.sub(r"\s+", "", text).lower()

        round_name = self._extract_round_name(text)
        if round_name:
            slots["round"] = round_name

        interview_time = self._extract_time_expression(text)
        if interview_time:
            slots["interview_time"] = interview_time

        calendar_reminder = self._extract_calendar_reminder_preference(compact)
        if calendar_reminder is not None:
            slots["calendar_reminder"] = calendar_reminder

        role = self._extract_role(text)
        if role:
            slots["role"] = role

        update_type = self._extract_update_type(compact)
        slots["update_type"] = update_type
        if update_type == "withdraw" and any(word in compact for word in ("所有岗位", "全部岗位", "所有投递", "全部投递")):
            slots["apply_to_all"] = True

        status = self._status_value_for_update(update_type, round_name)
        if status:
            slots["status"] = status

        company = self._extract_company_for_update(text)
        if company:
            slots["company"] = company

        return slots

    # 从输入数据中提取 update type。
    @staticmethod
    def _extract_update_type(compact: str) -> str:
        if "offer" in compact:
            return "offer"
        if "取消" in compact and any(word in compact for word in ("面试", "笔试", "一面", "二面", "三面", "hr面")):
            return "cancel_interview"
        if any(word in compact for word in ("改期", "改到", "调整到", "推迟到", "提前到")):
            return "reschedule_interview"
        if any(word in compact for word in ("取消投递", "撤回投递", "不想去", "放弃")):
            return "withdraw"
        if any(word in compact for word in ("挂了", "拒了", "拒绝", "没过", "凉了")):
            return "reject"
        if any(word in compact for word in ("约我", "约了", "安排", "通知", "邀我", "邀请", "收到面试", "发来面试")):
            return "schedule_interview"
        if any(word in compact for word in ("过了", "通过", "进了", "进入")):
            return "pass_round"
        if "已投递" in compact:
            return "submitted"
        return "status_update"

    # 处理 status_value_for_update 相关逻辑。
    @staticmethod
    def _status_value_for_update(update_type: str, round_name: Optional[str]) -> Optional[str]:
        round_key = round_name.replace(" ", "").lower() if round_name else None
        scheduled_by_round = {
            "笔试": "written_test",
            "一面": "interview_1",
            "二面": "interview_2",
            "三面": "interview_3",
            "hr": "hr",
            "hr面": "hr",
        }
        passed_by_round = {
            "笔试": "written_test_passed",
            "一面": "interview_1_passed",
            "二面": "interview_2_passed",
            "三面": "interview_3_passed",
            "hr": "offer",
            "hr面": "offer",
        }
        if update_type == "offer":
            return "offer"
        if update_type == "reject":
            return "rejected"
        if update_type == "withdraw":
            return "withdrawn"
        if update_type == "submitted":
            return "submitted"
        if update_type == "pass_round" and round_key:
            return passed_by_round.get(round_key)
        if update_type == "schedule_interview" and round_key:
            return scheduled_by_round.get(round_key, "interview_scheduled")
        return None

    # 从输入数据中提取 calendar reminder preference。
    @staticmethod
    def _extract_calendar_reminder_preference(compact: str) -> Optional[bool]:
        if any(word in compact for word in ("不需要提醒", "不用提醒", "不要提醒", "不加日历", "不同步日历", "不用日历")):
            return False
        if any(word in compact for word in ("需要提醒", "要提醒", "加日历", "同步日历", "日历提醒", "需要日历")):
            return True
        return None

    # 从输入数据中提取 company for update。
    def _extract_company_for_update(self, text: str) -> Optional[str]:
        text_without_time = self._remove_time_expression(text)
        round_pattern = r"(?:笔试|一面|二面|三面|hr\s*面|HR\s*面)"
        patterns = [
            rf"(?:取消)(?P<company>[\u4e00-\u9fa5A-Za-z0-9_-]{{2,20}}?)(?:的)?{round_pattern}",
            rf"(?P<company>.+?){round_pattern}(?:改期|改到|调整到|推迟到|提前到)",
            r"(?:取消|撤回|放弃)(?:投递)?(?:了)?\s*(?P<company>[\u4e00-\u9fa5A-Za-z0-9_-]{2,20})(?:的)?(?:所有岗位|全部岗位|所有投递|全部投递|岗位|职位)?",
            r"不想去\s*(?P<company>[\u4e00-\u9fa5A-Za-z0-9_-]{2,20})(?:了)?",
            r"(?:之前)?投递[了的]?(?P<company>[\u4e00-\u9fa5A-Za-z0-9_-]{2,20}?)(?:java|ai|agent|后端|前端|算法|开发|实习|岗位|职位|，|,).*?(?:约我|约了|安排(?:了)?|通知|邀我|邀请|收到面试|发来面试)",
            rf"(?P<company>.+?)(?:约我|约了|安排(?:了)?|通知|邀我|邀请|收到面试|发来面试).*?(?:{round_pattern}|面试|笔试)",
            rf"(?P<company>.+?){round_pattern}(?:过了|通过|没过|挂了|拒了|拒绝|凉了)",
            r"(?P<company>.+?)(?:给我|给|发了|发|拿到|收到|收到了)?\s*offer",
            r"(?P<company>.+?)(?:挂了|拒了|拒绝|没过|凉了)",
            rf"(?:进入|进了)(?P<company>.+?){round_pattern}",
        ]
        for pattern in patterns:
            match = re.search(pattern, text_without_time, flags=re.IGNORECASE)
            if match:
                company = self._clean_company_name(match.group("company"))
                if company:
                    return company

        return None

    # 移除 time expression。
    def _remove_time_expression(self, text: str) -> str:
        time_expression = self._extract_time_expression(text)
        if not time_expression:
            return text
        return text.replace(time_expression, "", 1)

    # 清洗 company name。
    @staticmethod
    def _clean_company_name(value: str) -> str:
        company = value.strip(" ，,。；;：:的")
        company = re.split(r"[，,。；;]", company)[-1]
        company = re.sub(r"^(我|我这边|刚刚|刚才|今天|明天|后天|今晚|上午|下午|晚上|早上|有个|有场)+", "", company)
        company = company.strip(" ，,。；;：:的")
        return company

    # 从输入数据中提取 company for query。
    @staticmethod
    def _extract_company_for_query(text: str) -> Optional[str]:
        patterns = [
            r"(.+?)(?:现在|目前)?(?:什么状态|什么进度|到哪(?:一步)?|进展)",
            r"(.+?)(?:的)?(?:投递记录|投递进度|状态)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                company = match.group(1).strip(" ，,。？?的")
                if company and company not in {"我", "最近", "今天", "明天", "本周", "这周"}:
                    return company

        return None

    # 从输入数据中提取 interview review slots。
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

    # 从输入数据中提取 mock interview slots。
    def _extract_mock_interview_slots(self, text: str) -> Dict[str, Any]:
        slots: Dict[str, Any] = {}
        role_match = re.search(r"(岗位|职位)[是:：]?([^，,。；;]+)", text)
        if role_match:
            slots["role"] = role_match.group(2).strip()

        project_match = re.search(r"项目(?:选|问|是|：|:)?([^，,。；;\s]+)", text)
        if project_match:
            slots["project"] = project_match.group(1).strip()

        return slots

    # 从输入数据中提取 role。
    @staticmethod
    def _extract_role(text: str) -> Optional[str]:
        role_match = re.search(
            r"(java\s*后端|java\s*开发|python\s*后端|ai\s*应用开发|agent\s*开发|后端开发|开发实习生?|开发实习|算法工程师?|实习生?)",
            text,
            flags=re.IGNORECASE,
        )
        if role_match:
            return role_match.group(0).strip()

        separated_role = re.search(r"(岗位|职位|方向)[是:：]?([^，,。；;]+)", text)
        if separated_role:
            return separated_role.group(2).strip()

        return None

    # 从输入数据中提取 round name。
    @staticmethod
    def _extract_round_name(text: str) -> Optional[str]:
        round_match = re.search(r"(笔试|一面|二面|三面|hr\s*面|HR\s*面)", text, flags=re.IGNORECASE)
        if round_match:
            return round_match.group(1)
        return None

    # 从输入数据中提取 task slots。
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

    # 从输入数据中提取 time expression。
    @staticmethod
    def _extract_time_expression(text: str) -> Optional[str]:
        match = re.search(
            r"((?:今天|明天|后天|大后天|下周[一二三四五六日天]?|周[一二三四五六日天])?"
            r"(?:早上|上午|中午|下午|晚上)?"
            r"(?:[零一二三四五六七八九十两\d]{1,3})点(?:半|[零一二三四五六七八九十\d]{1,2}分)?)",
            text,
        )
        if not match:
            for word in ("今晚", "今天", "明天", "后天"):
                if word in text:
                    return word
            return None
        return match.group(1)

    # 从输入数据中提取 keywords。
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
