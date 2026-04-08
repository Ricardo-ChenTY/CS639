from __future__ import annotations

from typing import Any

from src.parsers.answers import answers_match, extract_final_answer, normalize_answer
from src.types import GenerationResult


def format_constraint_score(final_answer: str, task_name: str) -> float:
    return 1.0 if normalize_answer(final_answer, task_name) else 0.0


def verifier_consistency_score(result: GenerationResult, task_name: str) -> float:
    extracted = extract_final_answer(result.text)
    return 1.0 if answers_match(extracted, result.final_answer, task_name) else 0.0


def answer_agreement_score(
    *,
    prompt: str,
    candidate_answer: str,
    task_name: str,
    llm: Any,
    config: dict,
) -> tuple[float, list[str], int, int, float]:
    probe_outputs = llm.generate(
        prompt,
        max_new_tokens=config["reasoning"].get("probe_max_new_tokens", 64),
        temperature=max(config["reasoning"].get("probe_temperature", 0.7), 1e-5),
        n=2,
    )
    probe_answers = [extract_final_answer(item["text"]) for item in probe_outputs]
    normalized = [normalize_answer(answer, task_name) for answer in probe_answers if answer]
    candidate = normalize_answer(candidate_answer, task_name)

    agreed = (
        len(normalized) == 2
        and normalized[0] == normalized[1]
        and normalized[0] == candidate
    )
    return (
        1.0 if agreed else 0.0,
        probe_answers,
        sum(item.get("prompt_tokens", 0) for item in probe_outputs),
        sum(item.get("completion_tokens", 0) for item in probe_outputs),
        sum(item.get("latency_sec", 0.0) for item in probe_outputs),
    )


def compute_belief_score(
    *,
    result: GenerationResult,
    prompt: str,
    task_name: str,
    llm: Any,
    config: dict,
) -> tuple[float, dict[str, Any]]:
    verifier = verifier_consistency_score(result, task_name)
    agreement, probe_answers, probe_prompt_tokens, probe_completion_tokens, probe_latency = (
        answer_agreement_score(
            prompt=prompt,
            candidate_answer=result.final_answer,
            task_name=task_name,
            llm=llm,
            config=config,
        )
    )
    format_score = format_constraint_score(result.final_answer, task_name)
    belief = (verifier + agreement + format_score) / 3.0
    return belief, {
        "V": verifier,
        "A": agreement,
        "G": format_score,
        "probe_answers": probe_answers,
        "probe_prompt_tokens": probe_prompt_tokens,
        "probe_completion_tokens": probe_completion_tokens,
        "probe_latency_sec": probe_latency,
    }
