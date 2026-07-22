import re
from functools import lru_cache
from typing import Any, Dict, Optional


ENABLE_PHRASES = (
    "开启每日刷题",
    "开启刷题计划",
    "开启leetcode计划",
    "开启力扣计划",
)
DISABLE_PHRASES = (
    "关闭每日刷题",
    "停止每日刷题",
    "关闭leetcode推送",
    "停止力扣推送",
)
FEEDBACK_RESULT_PATTERNS = (
    (("看题解", "题解后"), "with_solution"),
    (("提示后", "提示完成"), "with_hint"),
    (("没做出来", "未完成", "失败"), "failed"),
    (("延期", "明天再做"), "postponed"),
    (("跳过",), "skipped"),
    (("独立完成",), "independent"),
)
AMBIGUOUS_COMPLETION_PHRASES = ("做完", "完成了")


def is_enable_command(compact: str) -> bool:
    return any(phrase in compact for phrase in ENABLE_PHRASES)


def is_disable_command(compact: str) -> bool:
    return any(phrase in compact for phrase in DISABLE_PHRASES)


def is_today_query(compact: str) -> bool:
    return compact in {"今天刷什么", "今日刷什么"} or (
        any(word in compact for word in ("今天", "今日"))
        and any(word in compact for word in ("leetcode", "力扣", "算法题", "刷题"))
    )


def is_feedback(compact: str) -> bool:
    normalized = _normalize_title(compact)
    has_locator = extract_problem_index(compact) is not None or any(
        word in compact for word in ("leetcode", "力扣", "lru")
    ) or any(title in normalized for title in hot100_problem_titles())
    has_result = extract_result(compact) is not None or any(
        word in compact for word in AMBIGUOUS_COMPLETION_PHRASES
    )
    return has_locator and has_result


def looks_like_action(compact: str) -> bool:
    if is_enable_command(compact) or is_disable_command(compact) or is_today_query(compact):
        return True
    return any(
        phrase in compact for phrases, _ in FEEDBACK_RESULT_PATTERNS for phrase in phrases
    )


def extract_feedback_slots(text: str) -> Dict[str, Any]:
    compact = re.sub(r"\s+", "", text).lower()
    slots: Dict[str, Any] = {}
    problem_index = extract_problem_index(compact)
    if problem_index is not None:
        slots["problem_index"] = problem_index
    result = extract_result(compact)
    if result is not None:
        slots["result"] = result
    if problem_index is None:
        title = extract_problem_title(text)
        if title:
            slots["problem_title"] = title
    return slots


def extract_problem_title(text: str) -> str:
    title = re.split(
        r"独立完成|提示后完成|提示完成|看题解(?:后)?完成?|没做出来|未完成|失败|延期|明天再做|跳过",
        text,
        maxsplit=1,
    )[0].strip(" ，,。；;")
    title = re.sub(r"^(?:leetcode|力扣)", "", title, flags=re.IGNORECASE).strip()
    title = re.sub(r"^(?:题目(?:名称)?|名称)[是为:：]?", "", title).strip()
    return title


def extract_problem_index(text: str) -> Optional[int]:
    match = re.search(r"第([一二三123])题", text)
    if match is None:
        return None
    raw_index = match.group(1)
    return {"一": 1, "二": 2, "三": 3}.get(
        raw_index,
        int(raw_index) if raw_index.isdigit() else None,
    )


def extract_result(text: str) -> Optional[str]:
    compact = re.sub(r"\s+", "", text).lower()
    for words, result in FEEDBACK_RESULT_PATTERNS:
        if any(word in compact for word in words):
            return result
    return None


@lru_cache(maxsize=1)
def hot100_problem_titles() -> tuple[str, ...]:
    from app.services.leetcode_catalog import load_hot100_snapshot

    titles = {
        normalized
        for problem in load_hot100_snapshot().problems
        for title in (problem.title_zh, problem.title_en)
        if (normalized := _normalize_title(title))
    }
    return tuple(sorted(titles, key=len, reverse=True))


def _normalize_title(value: str) -> str:
    return re.sub(r"[\s\-—_·:：，,。()（）]+", "", value).lower()
