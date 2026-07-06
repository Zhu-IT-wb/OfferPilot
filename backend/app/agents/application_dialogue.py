import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.agents.conversation import PendingAgentAction
from app.schemas.agent import AgentActionName, AgentResponse
from app.schemas.intent import IntentClassification, IntentName
from app.services.feishu_service import parse_chinese_datetime


# 承载投递对话规划后的响应和原始消息。
@dataclass(frozen=True)
class ApplicationDialoguePlan:
    response: AgentResponse
    original_message: str


# 集中管理投递相关多轮对话和槽位补全。
class ApplicationDialogueManager:
    """Plans application-related actions from a structured draft.

    The orchestrator should not need to know how application slots are completed
    or validated. This class keeps that domain workflow in one place.
    """

    _APPLICATION_INTENTS = {
        IntentName.ADD_APPLICATION,
        IntentName.UPDATE_APPLICATION,
    }

    _SLOT_LABELS = {
        "company": "公司",
        "role": "岗位",
        "round": "轮次",
        "interview_time": "具体面试时间（例如明天下午三点）",
    }

    # 根据投递类意图生成多轮对话计划。
    def plan(self, classification: IntentClassification) -> Optional[AgentResponse]:
        if classification.intent not in self._APPLICATION_INTENTS:
            return None

        slots = self._normalize_slots(classification.slots)
        normalized_classification = IntentClassification(
            intent=classification.intent,
            confidence=classification.confidence,
            slots=slots,
        )

        if classification.intent == IntentName.ADD_APPLICATION:
            return self._plan_create_application(normalized_classification)
        if classification.intent == IntentName.UPDATE_APPLICATION:
            return self._plan_update_application(normalized_classification)
        return None

    # 尝试处理投递类待确认动作的补充消息。
    def try_handle_pending(
        self,
        pending: PendingAgentAction,
        message: str,
    ) -> Optional[ApplicationDialoguePlan]:
        if pending.intent not in self._APPLICATION_INTENTS:
            return None

        expected_slots = set(pending.missing_slots)
        if pending.action == AgentActionName.UPDATE_APPLICATION:
            expected_slots.update({"company", "role", "round", "interview_time", "calendar_reminder"})
        elif pending.action == AgentActionName.CREATE_APPLICATION:
            expected_slots.update({"company", "role", "round", "interview_time"})

        slot_updates = self.extract_slot_updates(message, expected_slots=expected_slots)
        if not slot_updates:
            return None

        merged_slots = pending.slots.copy()
        merged_slots.update(slot_updates)
        classification = IntentClassification(
            intent=pending.intent,
            confidence=max(pending.confidence, 0.8),
            slots=merged_slots,
        )
        response = self.plan(classification)
        if response is None:
            return None

        original_message = f"{pending.original_message}\n{message}"
        return ApplicationDialoguePlan(response=response, original_message=original_message)

    # 根据缺失槽位生成追问文案。
    def build_missing_reply(
        self,
        intent: IntentName,
        missing_slots: List[str],
    ) -> str:
        missing = "、".join(self._SLOT_LABELS.get(slot, slot) for slot in missing_slots)
        if intent == IntentName.ADD_APPLICATION:
            return f"我识别到你想新增投递记录，但还缺少{missing}。请补充后我再生成投递创建动作。"
        return f"我识别到你想更新投递进度，但还缺少{missing}。请补充后我再生成投递更新动作。"

    # 处理 plan_create_application 相关逻辑。
    def _plan_create_application(self, classification: IntentClassification) -> AgentResponse:
        missing_slots = self._missing_required_slots(
            classification.slots,
            required_slots=["company", "role"],
        )
        if missing_slots:
            return self._response(
                classification=classification,
                action=AgentActionName.ASK_CLARIFICATION,
                reply=self.build_missing_reply(classification.intent, missing_slots),
                missing_slots=missing_slots,
            )

        return self._response(
            classification=classification,
            action=AgentActionName.CREATE_APPLICATION,
            reply=self._build_create_application_reply(classification.slots),
            need_confirmation=True,
        )

    # 处理 plan_update_application 相关逻辑。
    def _plan_update_application(self, classification: IntentClassification) -> AgentResponse:
        update_type = classification.slots.get("update_type")
        required_slots = ["company"]
        if update_type == "schedule_interview":
            required_slots.extend(["round", "interview_time"])

        missing_slots = self._missing_required_slots(classification.slots, required_slots=required_slots)
        if (
            update_type == "schedule_interview"
            and classification.slots.get("interview_time")
            and not self._is_specific_interview_time(classification.slots.get("interview_time"))
            and "interview_time" not in missing_slots
        ):
            missing_slots.append("interview_time")

        if missing_slots:
            return self._response(
                classification=classification,
                action=AgentActionName.ASK_CLARIFICATION,
                reply=self.build_missing_reply(classification.intent, missing_slots),
                missing_slots=missing_slots,
            )

        return self._response(
            classification=classification,
            action=AgentActionName.UPDATE_APPLICATION,
            reply=self._build_update_application_reply(classification.slots),
            need_confirmation=True,
        )

    # 从用户补充消息里提取可更新槽位。
    @classmethod
    def extract_slot_updates(
        cls,
        message: str,
        expected_slots: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        text = message.strip()
        expected = expected_slots or {"company", "role", "round", "interview_time", "calendar_reminder"}
        updates: Dict[str, Any] = {}

        if "company" in expected:
            company = cls._extract_labeled_value(
                text,
                labels=("公司名称", "公司", "企业", "厂"),
                stop_labels=("岗位", "职位", "方向", "面试时间", "时间", "轮次"),
            )
            if company and not cls._looks_like_time_expression(company):
                updates["company"] = cls._clean_scalar(company)

        if "role" in expected:
            role = cls._extract_labeled_value(
                text,
                labels=("岗位", "职位", "方向"),
                stop_labels=("公司名称", "公司", "企业", "面试时间", "时间", "轮次"),
            )
            if role:
                updates["role"] = cls._clean_scalar(role)
            else:
                role_match = re.search(
                    r"(java\s*后端|java\s*开发|python\s*后端|ai\s*应用开发|agent\s*开发|后端开发|开发实习生?|开发实习|算法工程师?|实习生?)",
                    text,
                    flags=re.IGNORECASE,
                )
                if role_match:
                    updates["role"] = role_match.group(0).strip()

        if "interview_time" in expected:
            interview_time = cls._extract_labeled_value(
                text,
                labels=("具体面试时间", "面试时间", "时间"),
                stop_labels=("公司名称", "公司", "企业", "岗位", "职位", "方向", "轮次"),
            )
            if not interview_time:
                interview_time = cls._extract_time_expression(text)
            if interview_time:
                updates["interview_time"] = cls._clean_scalar(interview_time)

        if "round" in expected:
            round_match = re.search(r"(笔试|一面|二面|三面|hr\s*面|HR\s*面)", text, flags=re.IGNORECASE)
            if round_match:
                updates["round"] = round_match.group(1)

        if "calendar_reminder" in expected:
            calendar_reminder = cls._extract_calendar_reminder_preference(text)
            if calendar_reminder is not None:
                updates["calendar_reminder"] = calendar_reminder

        return {key: value for key, value in updates.items() if value is not None and value != ""}

    # 标准化 slots。
    @classmethod
    def _normalize_slots(cls, slots: Dict[str, Any]) -> Dict[str, Any]:
        normalized = slots.copy()
        for key in ("company", "role", "round", "interview_time"):
            value = normalized.get(key)
            if isinstance(value, str):
                cleaned = cls._clean_scalar(value)
                if key == "company" and cls._is_invalid_company(cleaned):
                    normalized.pop(key, None)
                elif cleaned:
                    normalized[key] = cleaned
                else:
                    normalized.pop(key, None)
        return normalized

    # 从输入数据中提取 labeled value。
    @classmethod
    def _extract_labeled_value(
        cls,
        text: str,
        labels: tuple[str, ...],
        stop_labels: tuple[str, ...],
    ) -> Optional[str]:
        label_pattern = "|".join(re.escape(label) for label in labels)
        stop_pattern = "|".join(re.escape(label) for label in stop_labels)
        pattern = (
            rf"(?:{label_pattern})\s*(?:是|为|叫|名称)?\s*[:：]?\s*"
            rf"(?P<value>.+?)(?=\s*(?:{stop_pattern})\s*(?:是|为|叫|[:：])|[，,。；;]|\n|$)"
        )
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            return None
        return cls._clean_scalar(match.group("value"))

    # 从输入数据中提取 time expression。
    @staticmethod
    def _extract_time_expression(text: str) -> Optional[str]:
        match = re.search(
            r"((?:今天|明天|后天|大后天|下周[一二三四五六日天]?|周[一二三四五六日天]|星期[一二三四五六日天]|礼拜[一二三四五六日天])?"
            r"(?:早上|上午|中午|下午|晚上|今晚)?"
            r"(?:[零一二三四五六七八九十两\d]{1,3})点(?:半|[零一二三四五六七八九十\d]{1,2}分?)?)",
            text,
        )
        if match:
            return match.group(1)

        incomplete_match = re.search(
            r"((?:今天|明天|后天|大后天|下周[一二三四五六日天]?|周[一二三四五六日天]|星期[一二三四五六日天]|礼拜[一二三四五六日天])"
            r"(?:早上|上午|中午|下午|晚上|今晚)?)",
            text,
        )
        if incomplete_match:
            return incomplete_match.group(1)
        return None

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

    # 清洗 scalar。
    @staticmethod
    def _clean_scalar(value: str) -> str:
        cleaned = re.sub(r"\s+", " ", value).strip(" ，,。；;：:.!?！？的 ")
        cleaned = re.sub(r"^(是|为|叫)\s*", "", cleaned).strip(" ，,。；;：:.!?！？的 ")
        return cleaned

    # 判断 invalid company 是否成立。
    @staticmethod
    def _is_invalid_company(value: str) -> bool:
        return not value or value in {"是", "为", "叫", "公司", "公司名称", "企业", "厂"}

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

    # 计算缺失的 required slots。
    @staticmethod
    def _missing_required_slots(slots: Dict[str, Any], required_slots: List[str]) -> List[str]:
        return [
            slot
            for slot in required_slots
            if slot not in slots or slots.get(slot) is None or slots.get(slot) == ""
        ]

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

    # 构造 create application reply。
    @staticmethod
    def _build_create_application_reply(slots: Dict[str, Any]) -> str:
        details = [f"公司是 {slots['company']}", f"岗位是 {slots['role']}"]
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
