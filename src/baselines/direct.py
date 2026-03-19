from __future__ import annotations

from src.parsers.answers import extract_final_answer
from src.prompts.templates import build_direct_prompt
from src.types import Example, GenerationResult


def run_direct(example: Example, llm, config: dict) -> GenerationResult:
    prompt = build_direct_prompt(example)
    outputs = llm.generate(
        prompt,
        max_new_tokens=config["model"]["max_new_tokens"],
        temperature=config["model"]["temperature"],
    )
    raw = outputs[0]
    text = raw["text"]
    return GenerationResult(
        text=text,
        final_answer=extract_final_answer(text),
        prompt_tokens=raw.get("prompt_tokens", 0),
        completion_tokens=raw.get("completion_tokens", 0),
        latency_sec=raw.get("latency_sec", 0.0),
        metadata={"prompt": prompt},
    )
