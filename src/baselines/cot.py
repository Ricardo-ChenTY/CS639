from __future__ import annotations

from src.parsers.answers import extract_final_answer
from src.prompts.templates import build_cot_prompt
from src.types import Example, GenerationResult


def run_cot(example: Example, llm, config: dict) -> GenerationResult:
    prompt = build_cot_prompt(example)
    outputs = llm.generate(
        prompt,
        max_new_tokens=config["model"]["max_new_tokens"],
        temperature=config["model"]["temperature"],
        return_logprobs=True,
    )
    raw = outputs[0]
    metadata: dict = {"prompt": prompt}
    if "answer_logprob" in raw:
        metadata["answer_logprob"] = raw["answer_logprob"]
    return GenerationResult(
        text=raw["text"],
        final_answer=extract_final_answer(raw["text"]),
        prompt_tokens=raw.get("prompt_tokens", 0),
        completion_tokens=raw.get("completion_tokens", 0),
        latency_sec=raw.get("latency_sec", 0.0),
        metadata=metadata,
    )
