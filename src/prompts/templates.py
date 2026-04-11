from __future__ import annotations

from src.types import Example


def _format_choices(example: Example) -> str:
    """Return a formatted choices block if the example has multiple-choice options."""
    choices = example.metadata.get("choices")
    if not choices:
        return ""
    labels = "ABCDEFGHIJ"
    lines = "\n".join(f"({labels[i]}) {choice}" for i, choice in enumerate(choices))
    return f"\n{lines}\n"


def _choice_instruction(example: Example) -> str:
    """Return the answer instruction line appropriate for the task type."""
    if example.metadata.get("choices"):
        return "Answer with the letter of the correct option only (e.g. A, B, C, or D).\n"
    return ""


def build_direct_prompt(example: Example) -> str:
    choices_block = _format_choices(example)
    choice_instr = _choice_instruction(example)
    return (
        "Answer the following question directly.\n\n"
        f"Question: {example.question}"
        f"{choices_block}\n"
        f"{choice_instr}"
        "Answer:"
    )


def build_cot_prompt(example: Example) -> str:
    choices_block = _format_choices(example)
    choice_instr = _choice_instruction(example)
    return (
        "Solve the following question step by step, then provide a final answer.\n\n"
        f"Question: {example.question}"
        f"{choices_block}\n"
        f"{choice_instr}"
        "Reasoning:"
    )


def build_search_prompt(example: Example, partial_trace: str) -> str:
    choices_block = _format_choices(example)
    choice_instr = _choice_instruction(example)
    return (
        "Continue the reasoning below with one strong next step.\n\n"
        f"Question: {example.question}"
        f"{choices_block}\n"
        f"{choice_instr}"
        f"Current reasoning:\n{partial_trace}\n\n"
        "Next step:"
    )
