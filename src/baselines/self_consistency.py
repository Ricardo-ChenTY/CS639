from __future__ import annotations

from collections import Counter

from src.parsers.answers import extract_final_answer
from src.prompts.templates import build_cot_prompt
from src.types import Example, GenerationResult


def run_self_consistency(example: Example, llm, config: dict) -> GenerationResult:
    prompt = build_cot_prompt(example)
    outputs = llm.generate(
        prompt,
        max_new_tokens=config["model"]["max_new_tokens"],
        temperature=max(config["model"]["temperature"], 0.7),
        n=config["reasoning"]["self_consistency_samples"],
    )

    answers = [extract_final_answer(item["text"]) for item in outputs]
    vote = Counter(answers).most_common(1)[0][0] if answers else ""
    total_prompt = sum(item.get("prompt_tokens", 0) for item in outputs)
    total_completion = sum(item.get("completion_tokens", 0) for item in outputs)
    total_latency = sum(item.get("latency_sec", 0.0) for item in outputs)

    return GenerationResult(
        text="\n\n".join(item["text"] for item in outputs),
        final_answer=vote,
        prompt_tokens=total_prompt,
        completion_tokens=total_completion,
        latency_sec=total_latency,
        metadata={"prompt": prompt, "answers": answers},
    )
