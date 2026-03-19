from __future__ import annotations

from src.types import Example


def build_direct_prompt(example: Example) -> str:
    return (
        "Answer the following question directly.\n\n"
        f"Question: {example.question}\n"
        "Answer:"
    )


def build_cot_prompt(example: Example) -> str:
    return (
        "Solve the following question step by step, then provide a final answer.\n\n"
        f"Question: {example.question}\n"
        "Reasoning:"
    )


def build_search_prompt(example: Example, partial_trace: str) -> str:
    return (
        "Continue the reasoning below with one strong next step.\n\n"
        f"Question: {example.question}\n"
        f"Current reasoning:\n{partial_trace}\n\n"
        "Next step:"
    )
