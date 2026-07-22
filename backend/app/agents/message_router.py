import re
from dataclasses import dataclass
from enum import Enum


# 枚举消息路由的可选类型。
class MessageRouteName(str, Enum):
    JOB_ACTION = "job_action"
    SMALLTALK = "smalltalk"
    CAPABILITY_HELP = "capability_help"
    DOMAIN_QUESTION = "domain_question"
    UNKNOWN = "unknown"


# 描述一条消息被路由后的类型、置信度和原因。
@dataclass(frozen=True)
class MessageRoute:
    route: MessageRouteName
    confidence: float
    reason: str


# 在业务动作、闲聊和领域问答之间进行轻量路由。
class MessageRouter:
    # 判断消息应该进入哪类处理流程。
    def route(self, message: str) -> MessageRoute:
        text = message.strip()
        compact = re.sub(r"\s+", "", text).lower()
        if not compact:
            return MessageRoute(MessageRouteName.UNKNOWN, 0.4, "empty")

        if self._is_smalltalk(compact):
            return MessageRoute(MessageRouteName.SMALLTALK, 0.95, "smalltalk")
        if self._is_capability_help(compact):
            return MessageRoute(MessageRouteName.CAPABILITY_HELP, 0.92, "capability_help")
        if self._looks_like_job_action(compact):
            return MessageRoute(MessageRouteName.JOB_ACTION, 0.86, "job_action")
        if self._looks_like_domain_question(compact):
            return MessageRoute(MessageRouteName.DOMAIN_QUESTION, 0.74, "domain_question")

        return MessageRoute(MessageRouteName.UNKNOWN, 0.45, "unknown")

    # 判断 smalltalk 是否成立。
    @staticmethod
    def _is_smalltalk(compact: str) -> bool:
        normalized = compact.rstrip("？?。！!")
        exact_phrases = {
            "你好",
            "您好",
            "哈喽",
            "hello",
            "hi",
            "在吗",
            "在不在",
            "早上好",
            "下午好",
            "晚上好",
            "你是谁",
            "你叫什么",
        }
        return normalized in exact_phrases

    # 判断 capability help 是否成立。
    @staticmethod
    def _is_capability_help(compact: str) -> bool:
        normalized = compact.rstrip("？?。！!")
        exact_phrases = {
            "帮助",
            "help",
            "/help",
            "怎么用",
            "如何使用",
            "使用说明",
            "你能做什么",
            "你能干什么",
            "你有什么功能",
        }
        if normalized in exact_phrases:
            return True
        return any(
            phrase in compact
            for phrase in (
                "怎么新增投递",
                "怎么记录投递",
                "怎么安排面试",
                "怎么复盘",
                "怎么查看任务",
            )
        )

    # 判断输入是否像 job action。
    @staticmethod
    def _looks_like_job_action(compact: str) -> bool:
        if compact in {"/today", "today"}:
            return True
        if any(
            phrase in compact
            for phrase in (
                "开启每日刷题",
                "关闭每日刷题",
                "停止每日刷题",
                "今天刷什么",
                "今日刷什么",
                "独立完成",
                "提示后完成",
                "看题解",
                "没做出来",
            )
        ):
            return True
        if ("今天" in compact or "今日" in compact) and any(
            word in compact for word in ("任务", "安排", "计划")
        ):
            return True
        if any(word in compact for word in ("新增投递", "添加投递", "记录投递", "我投递了", "投递了", "投了")):
            return True
        if any(word in compact for word in ("投递列表", "投递记录", "投递进度", "投了哪些", "投过哪些")):
            return True
        if any(word in compact for word in ("什么状态", "什么进度", "到哪一步", "进展")):
            return True
        if any(word in compact for word in ("面试", "笔试", "一面", "二面", "三面", "hr面")) and any(
            word in compact
            for word in (
                "今晚",
                "今天",
                "明天",
                "后天",
                "上午",
                "下午",
                "晚上",
                "一面",
                "二面",
                "三面",
                "hr面",
                "约我",
                "约了",
                "安排",
                "复盘",
            )
        ):
            return True
        if any(word in compact for word in ("完成", "做完", "刷完", "打卡", "/done")):
            return True
        if any(word in compact for word in ("延期", "推迟", "改到明天", "明天再做", "跳过")):
            return True
        if "模拟面试" in compact or "mockinterview" in compact:
            return True
        if any(word in compact for word in ("周复盘", "本周复盘", "周总结", "本周总结")):
            return True
        return False

    # 判断输入是否像 domain question。
    @staticmethod
    def _looks_like_domain_question(compact: str) -> bool:
        question_words = ("?", "？", "怎么", "如何", "为什么", "区别", "准备", "学习", "复习", "讲讲", "解释", "建议")
        domain_words = (
            "秋招",
            "求职",
            "面试",
            "简历",
            "八股",
            "项目",
            "leetcode",
            "java",
            "后端",
            "redis",
            "mysql",
            "spring",
            "jvm",
            "算法",
            "系统设计",
            "agent",
            "ai应用",
        )
        return any(word in compact for word in question_words) and any(
            word in compact for word in domain_words
        )
