from typing import List

from app.models.leetcode import LeetCodeAssignmentStatus, LeetCodeRecommendation


def format_leetcode_recommendations(recommendations: List[LeetCodeRecommendation]) -> str:
    if not recommendations:
        return "Hot 100 当前没有需要推荐的题目。"
    type_labels = {"new": "新题", "review": "复习", "carryover": "继续完成"}
    difficulty_labels = {"easy": "简单", "medium": "中等", "hard": "困难"}
    lines = [f"今日 LeetCode（共 {len(recommendations)} 题）："]
    for index, item in enumerate(recommendations, start=1):
        problem = item.problem
        lines.extend(
            [
                "",
                f"{index}. {problem.frontend_id}. {problem.title_zh}（{type_labels[item.assignment.assignment_type.value]}）",
                f"难度：{difficulty_labels[problem.difficulty.value]}｜专题：{problem.category}",
                f"推荐原因：{item.assignment.recommendation_reason}",
                "完成标准：写出可通过的解法，并能说明核心思路和复杂度。",
                f"链接：{problem.url}",
            ]
        )
    lines.extend(["", "完成后可回复：第1题独立完成 / 提示后完成 / 看题解完成 / 没做出来。"])
    return "\n".join(lines)


def format_leetcode_feedback_reminder(recommendations: List[LeetCodeRecommendation]) -> str:
    lines = ["今晚还有这些 LeetCode 题没有反馈："]
    for index, item in enumerate(recommendations, start=1):
        if item.assignment.status == LeetCodeAssignmentStatus.PENDING:
            lines.append(f"{index}. {item.problem.frontend_id}. {item.problem.title_zh}")
    lines.extend(
        [
            "",
            "请按今日题目原编号回复，例如：第2题独立完成 / 看题解完成 / 没做出来 / 延期。",
        ]
    )
    return "\n".join(lines)
