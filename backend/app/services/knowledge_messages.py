from typing import Any, Dict, List, Optional

from app.models.interview_knowledge import KnowledgeDeliveryType
from app.services.knowledge_recommendation import KnowledgeRecommendation


def build_knowledge_reminder_card(
    recommendations: List[KnowledgeRecommendation],
    delivery_type: KnowledgeDeliveryType,
    dashboard_url: Optional[str],
) -> Dict[str, Any]:
    titles = {
        KnowledgeDeliveryType.MORNING: "📚 今日八股",
        KnowledgeDeliveryType.NOON: "⏰ 午间八股提醒",
        KnowledgeDeliveryType.EVENING: "🌙 今日八股收尾",
    }
    due = sum(
        1 for item in recommendations
        if item.assignment.assignment_type.value == "due_review"
    )
    weak = sum(
        1 for item in recommendations
        if item.assignment.assignment_type.value == "weakness"
    )
    new = sum(
        1 for item in recommendations
        if item.assignment.assignment_type.value == "new"
    )
    summary = f"今天共 {len(recommendations)} 题，预计需要 15 分钟。"
    if delivery_type != KnowledgeDeliveryType.MORNING:
        summary = f"还有 {len(recommendations)} 题没有完成。"
    details = f"到期复习 {due} 题 · 薄弱强化 {weak} 题 · 新题 {new} 题"
    elements: List[Dict[str, Any]] = [
        {"tag": "div", "text": {"tag": "lark_md", "content": summary}},
        {"tag": "div", "text": {"tag": "lark_md", "content": details}},
    ]
    if dashboard_url:
        elements.append(
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "开始今日八股"},
                        "type": "primary",
                        "url": dashboard_url,
                    }
                ],
            }
        )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "blue",
            "title": {"tag": "plain_text", "content": titles[delivery_type]},
        },
        "elements": elements,
    }
