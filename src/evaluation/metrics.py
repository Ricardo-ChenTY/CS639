from __future__ import annotations

from statistics import median

from src.parsers.answers import answers_match
from src.types import Example, ExperimentRecord


def build_record(
    example: Example,
    method: str,
    prediction: str,
    total_tokens: int,
    latency_sec: float,
) -> ExperimentRecord:
    return ExperimentRecord(
        example=example,
        method=method,
        prediction=prediction,
        is_correct=answers_match(prediction, example.answer),
        total_tokens=total_tokens,
        latency_sec=latency_sec,
    )


def summarize(records: list[ExperimentRecord]) -> dict:
    if not records:
        return {"count": 0, "accuracy": 0.0, "avg_tokens": 0.0, "median_latency": 0.0}

    accuracy = sum(record.is_correct for record in records) / len(records)
    avg_tokens = sum(record.total_tokens for record in records) / len(records)
    median_latency = median(record.latency_sec for record in records)
    return {
        "count": len(records),
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "median_latency": median_latency,
    }
