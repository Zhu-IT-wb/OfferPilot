from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class InterviewReviewStatus(str, Enum):
    CREATED = "created"
    ANALYZED = "analyzed"


@dataclass
class InterviewReview:
    id: str
    company: Optional[str] = None
    round: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    raw_message: str = ""
    status: InterviewReviewStatus = InterviewReviewStatus.CREATED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "company": self.company,
            "round": self.round,
            "topics": list(self.topics),
            "raw_message": self.raw_message,
            "status": self.status.value,
        }
