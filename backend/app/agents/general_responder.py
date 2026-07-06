from typing import Optional

from app.agents.message_router import MessageRoute, MessageRouteName
from app.services.llm_service import LLMConfigurationError, LLMRequestError, LLMService


# 处理闲聊、能力说明和领域问答等非工具类消息。
class GeneralResponder:
    SYSTEM_PROMPT = """
You are OfferPilot, a practical personal campus recruiting assistant for a CS graduate student.
Answer non-mutating questions briefly in Chinese. Do not claim that you have written data,
created calendars, or updated Feishu unless the user explicitly asked for a supported action
and a tool has run. Keep answers action-oriented and useful for Java backend, AI application,
Agent development, interview preparation, job applications, and review workflows.
""".strip()

    # 初始化当前组件所需的依赖和配置。
    def __init__(self, llm_service: Optional[LLMService] = None) -> None:
        self.llm_service = llm_service or LLMService()

    # 生成非工具类消息的回复。
    async def respond(self, message: str, route: MessageRoute) -> str:
        if route.route == MessageRouteName.SMALLTALK:
            return self._smalltalk_reply(message)
        if route.route == MessageRouteName.CAPABILITY_HELP:
            return self._capability_reply()
        if route.route == MessageRouteName.DOMAIN_QUESTION:
            return await self._domain_question_reply(message)
        return self._unknown_reply()

    # 处理 smalltalk_reply 相关逻辑。
    @staticmethod
    def _smalltalk_reply(message: str) -> str:
        compact = "".join(message.split()).lower()
        if any(phrase in compact for phrase in ("你是谁", "你叫什么")):
            return (
                "我是 OfferPilot，一个帮你把秋招准备落到执行闭环里的个人助手。"
                "你可以让我记录投递、安排面试、查看今日任务，或者复盘面试。"
            )
        return (
            "你好，我是 OfferPilot。你可以直接说“今天任务是什么”、"
            "“我投递了某公司某岗位”或“复盘今天的面试”。"
        )

    # 处理 capability_reply 相关逻辑。
    @staticmethod
    def _capability_reply() -> str:
        return (
            "我现在主要帮你做五件事：\n"
            "1. 记录投递，并同步到飞书多维表格\n"
            "2. 记录面试安排，并同步到飞书日历\n"
            "3. 查看今日 LeetCode、八股和项目深挖任务\n"
            "4. 查询投递进度和后续面试\n"
            "5. 记录面试复盘，后续用于生成薄弱点和补强任务"
        )

    # 处理 domain_question_reply 相关逻辑。
    async def _domain_question_reply(self, message: str) -> str:
        try:
            result = await self.llm_service.generate_text(
                prompt=message.strip(),
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.2,
                max_tokens=500,
            )
        except (LLMConfigurationError, LLMRequestError):
            return self._fallback_domain_reply(message)

        answer = result.content.strip()
        return answer or self._fallback_domain_reply(message)

    # 处理 fallback_domain_reply 相关逻辑。
    @staticmethod
    def _fallback_domain_reply(message: str) -> str:
        compact = "".join(message.split()).lower()
        if "项目" in compact:
            return (
                "项目深挖建议按这条线准备：业务背景、核心链路、技术选型、"
                "高并发/缓存/一致性问题、你亲自做的部分、线上指标和可改进点。"
                "你也可以把项目名发给我，我帮你拆成面试追问清单。"
            )
        if any(word in compact for word in ("java", "后端", "八股", "jvm", "spring", "mysql", "redis")):
            return (
                "Java 后端秋招可以按四块推进：Java 基础与 JVM、并发与线程池、"
                "MySQL/Redis/Spring、项目场景题。建议每天固定一小段八股复述，"
                "再配合项目里的真实场景去解释，不要只背结论。"
            )
        if "leetcode" in compact or "算法" in compact:
            return (
                "算法准备先保证高频题型闭环：数组/哈希、链表、二叉树、栈队列、"
                "动态规划和回溯。每题记录模板、易错点和复盘结论，比单纯刷数量更有用。"
            )
        return (
            "这个问题可以继续展开。你可以告诉我目标岗位、当前进度和卡点，"
            "我会按秋招执行视角给你拆成可落地的下一步。"
        )

    # 处理 unknown_reply 相关逻辑。
    @staticmethod
    def _unknown_reply() -> str:
        return (
            "我现在主要围绕秋招执行来帮你。你可以说“今天任务是什么”、"
            "“我投递了某公司某岗位”、 “明天下午三点某公司一面”，"
            "也可以直接问 Java 后端、项目深挖或面试准备问题。"
        )
