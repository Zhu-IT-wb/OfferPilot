from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# 枚举 InterviewReviewStatus 的可选值。
class InterviewReviewStatus(str, Enum):
    CREATED = "created"
    ANALYZED = "analyzed"


# 定义 InterviewReview 相关的数据结构或领域对象。
@dataclass
class InterviewReview:
    id: str
    company: Optional[str] = None
    round: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    raw_message: str = ""
    status: InterviewReviewStatus = InterviewReviewStatus.CREATED

    # 将当前领域对象转换为可序列化字典。
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "company": self.company,
            "round": self.round,
            "topics": list(self.topics),
            "raw_message": self.raw_message,
            "status": self.status.value,
        }
