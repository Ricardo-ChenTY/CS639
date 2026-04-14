from __future__ import annotations

import math
from typing import Any

from src.parsers.answers import answers_match, extract_final_answer, normalize_answer
from src.types import GenerationResult


def format_constraint_score(final_answer: str, task_name: str) -> float:
    return 1.0 if normalize_answer(final_answer, task_name) else 0.0


def verifier_consistency_score(result: GenerationResult, task_name: str) -> float:
    extracted = extract_final_answer(result.text)
    return 1.0 if answers_match(extracted, result.final_answer, task_name) else 0.0


def logprob_confidence_score(result: GenerationResult) -> float:
    """Convert mean token log-probability to a [0,1] confidence score.

    Uses exp(mean_logprob) — the geometric mean token probability — which
    naturally lives in (0, 1] and is a stronger signal than binary probe
    agreement, at zero extra inference cost.
    """
    logprob = result.metadata.get("answer_logprob")
    if logprob is None:
        return 0.5  # neutral fallback when logprobs are unavailable
    return float(math.exp(max(float(logprob), -10.0)))


def compute_belief_score(
    *,
    result: GenerationResult,
    prompt: str,
    task_name: str,
    llm: Any,
    config: dict,
) -> tuple[float, dict[str, Any]]:
    verifier = verifier_consistency_score(result, task_name)
    confidence = logprob_confidence_score(result)
    format_score = format_constraint_score(result.final_answer, task_name)
    belief = (verifier + confidence + format_score) / 3.0
    return belief, {
        "V": verifier,
        "A": confidence,
        "G": format_score,
        # Zero cost: logprob comes free from the original generation call.
        "probe_prompt_tokens": 0,
        "probe_completion_tokens": 0,
        "probe_latency_sec": 0.0,
    }
