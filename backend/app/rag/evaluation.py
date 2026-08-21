from dataclasses import dataclass
from statistics import mean
from time import perf_counter
from typing import Callable, Dict, List, Sequence

from app.rag.models import RetrievalHit


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    query: str
    expected_evidence_ids: Sequence[str]
    owner_id: str = ""


def evaluate_retrieval(
    cases: Sequence[RetrievalEvaluationCase],
    search: Callable[[RetrievalEvaluationCase], Sequence[RetrievalHit]],
) -> Dict[str, float]:
    reciprocal_ranks: List[float] = []
    recalls: List[float] = []
    latencies_ms: List[float] = []
    for case in cases:
        started = perf_counter()
        hits = list(search(case))
        latencies_ms.append((perf_counter() - started) * 1000)
        expected = set(case.expected_evidence_ids)
        ranked = [hit.evidence_id for hit in hits]
        matched = expected.intersection(ranked)
        recalls.append(len(matched) / len(expected) if expected else 1.0)
        first_rank = next(
            (index for index, evidence_id in enumerate(ranked, start=1) if evidence_id in expected),
            None,
        )
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
    if not cases:
        return {"cases": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "p95_latency_ms": 0.0}
    ordered_latency = sorted(latencies_ms)
    p95_index = min(len(ordered_latency) - 1, int(len(ordered_latency) * 0.95))
    return {
        "cases": float(len(cases)),
        "recall_at_k": round(mean(recalls), 6),
        "mrr": round(mean(reciprocal_ranks), 6),
        "p95_latency_ms": round(ordered_latency[p95_index], 3),
    }
