from typing import Any, Dict, List, Optional

from app.models.leetcode import (
    LeetCodeDeliveryType,
    LeetCodeRecommendation,
)


_TYPE_LABELS = {"new": "新题", "review": "复习", "carryover": "顺延"}
_DIFFICULTY_LABELS = {"easy": "简单", "medium": "中等", "hard": "困难"}
_DELIVERY_TEMPLATES = {
    LeetCodeDeliveryType.MORNING: "blue",
    LeetCodeDeliveryType.NOON: "orange",
    LeetCodeDeliveryType.EVENING: "red",
}


def format_leetcode_recommendations(recommendations: List[LeetCodeRecommendation]) -> str:
    if not recommendations:
        return "Hot 100 当前没有需要推荐的题目。"
    lines = [f"今日 LeetCode（共 {len(recommendations)} 题）："]
    for index, item in enumerate(recommendations, start=1):
        problem = item.problem
        lines.extend(
            [
                "",
                f"{index}. {problem.frontend_id}. {problem.title_zh}（{_TYPE_LABELS[item.assignment.assignment_type.value]}）",
                f"难度：{_DIFFICULTY_LABELS[problem.difficulty.value]}｜专题：{problem.category}",
                f"推荐原因：{item.assignment.recommendation_reason}",
                "完成标准：写出可通过的解法，并能说明核心思路和复杂度。",
                f"链接：{problem.url}",
            ]
        )
    lines.extend(
        [
            "",
            "在飞书题目卡片中可直接点击结果；文本入口仍支持：第1题独立完成 / 提示后完成 / 看题解完成 / 没做出来。",
        ]
    )
    return "\n".join(lines)


def build_leetcode_card(
    recommendations: List[LeetCodeRecommendation],
    delivery_type: LeetCodeDeliveryType = LeetCodeDeliveryType.MORNING,
) -> Dict[str, Any]:
    title = _delivery_title(delivery_type, len(recommendations))
    template = _DELIVERY_TEMPLATES[delivery_type]
    elements: List[Dict[str, Any]] = [
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": _card_intro(recommendations, delivery_type),
            },
        }
    ]
    for index, item in enumerate(recommendations, start=1):
        problem = item.problem
        elements.extend(
            [
                {"tag": "hr"},
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": (
                            f"**{index}. {problem.frontend_id}. {problem.title_zh}** "
                            f"（{_TYPE_LABELS[item.assignment.assignment_type.value]}）\n"
                            f"{_DIFFICULTY_LABELS[problem.difficulty.value]} · {problem.category}\n"
                            f"{item.assignment.recommendation_reason}\n"
                            f"[去力扣做题]({problem.url})"
                        ),
                    },
                },
                {
                    "tag": "action",
                    "actions": [
                        _result_button(item.assignment.id, "独立完成", "independent", "primary"),
                        _result_button(item.assignment.id, "提示完成", "with_hint"),
                        _result_button(item.assignment.id, "看题解", "with_solution"),
                    ],
                },
                {
                    "tag": "action",
                    "actions": [
                        _result_button(item.assignment.id, "没做出来", "failed", "danger"),
                        _result_button(item.assignment.id, "明天继续", "postponed"),
                    ],
                },
            ]
        )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": template,
            "title": {"tag": "plain_text", "content": title},
        },
        "elements": elements,
    }


def build_leetcode_reminder_card(
    recommendations: List[LeetCodeRecommendation],
    delivery_type: LeetCodeDeliveryType,
    dashboard_url: Optional[str] = None,
) -> Dict[str, Any]:
    title = _delivery_title(delivery_type, len(recommendations))
    template = _DELIVERY_TEMPLATES[delivery_type]
    problem_lines = [
        (
            f"{index}. {item.problem.frontend_id}. {item.problem.title_zh}"
            f"（{_TYPE_LABELS[item.assignment.assignment_type.value]}）"
        )
        for index, item in enumerate(recommendations, start=1)
    ]
    elements: List[Dict[str, Any]] = [
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": "\n".join(
                    [_card_intro(recommendations, delivery_type), "", *problem_lines]
                ),
            },
        }
    ]
    if dashboard_url:
        elements.append(
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "打开刷题计划"},
                        "type": "primary",
                        "url": dashboard_url,
                    }
                ],
            }
        )
    else:
        elements.append(
            {
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": "请从 OfferPilot 顶部的「刷题计划」标签页打开工作台。",
                    }
                ],
            }
        )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": template,
            "title": {"tag": "plain_text", "content": title},
        },
        "elements": elements,
    }


def _delivery_title(
    delivery_type: LeetCodeDeliveryType,
    recommendation_count: int,
) -> str:
    return {
        LeetCodeDeliveryType.MORNING: (
            f"🎯 今日 LeetCode · {recommendation_count} 题"
        ),
        LeetCodeDeliveryType.NOON: "⏰ 12:00 刷题进度提醒",
        LeetCodeDeliveryType.EVENING: "🔥 18:00 今日最后提醒",
    }[delivery_type]


def _card_intro(
    recommendations: List[LeetCodeRecommendation],
    delivery_type: LeetCodeDeliveryType,
) -> str:
    if delivery_type == LeetCodeDeliveryType.MORNING:
        if any(item.assignment.assignment_type.value == "carryover" for item in recommendations):
            return "昨天没完成不代表失败，但今天别再把欠账交给明天。先拿下一题，行动会把压力变小。"
        return "今天只盯住这三题。做完后直接点击结果按钮，OfferPilot 会自动安排复习。"
    if delivery_type == LeetCodeDeliveryType.NOON:
        return f"还有 {len(recommendations)} 题未完成。午间先解决一题，晚上会轻松很多。"
    return (
        f"今天还有 {len(recommendations)} 题没有反馈。别等状态来了再行动，先做 20 分钟。"
        "未完成的题明天会自动顺延。"
    )


def _result_button(
    assignment_id: str,
    label: str,
    result: str,
    button_type: str = "default",
) -> Dict[str, Any]:
    return {
        "tag": "button",
        "text": {"tag": "plain_text", "content": label},
        "type": button_type,
        "value": {
            "action": "leetcode_result",
            "assignment_id": assignment_id,
            "result": result,
        },
    }
