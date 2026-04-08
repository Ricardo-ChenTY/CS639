from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Example:
    example_id: str
    task_name: str
    question: str
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    text: str
    final_answer: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class SearchNode:
    trace: str
    final_answer: str
    depth: int
    prompt_tokens: int
    cumulative_tokens: int
    latency_sec: float
    correctness_score: float
    utility: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentRecord:
    example: Example
    method: str
    prediction: str
    is_correct: bool
    total_tokens: int
    latency_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)
