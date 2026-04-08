from __future__ import annotations

import re


NUMERIC_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
OPTION_PATTERN = re.compile(r"\b(?:option\s*)?\(?([A-D])\)?\b", re.IGNORECASE)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def _extract_numeric_canonical(text: str) -> str | None:
    cleaned = text.replace(",", "")
    matches = NUMERIC_PATTERN.findall(cleaned)
    if not matches:
        return None

    value = matches[-1]
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


def _extract_option_label(text: str) -> str | None:
    matches = OPTION_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].upper()


def normalize_answer(text: str, task_name: str | None = None) -> str:
    collapsed = _normalize_whitespace(text)
    lowered = collapsed.lower()

    numeric = _extract_numeric_canonical(collapsed)
    if numeric is not None and any(
        token in (task_name or "").lower() for token in ("gsm8k", "math", "arithmetic")
    ):
        return numeric

    option = _extract_option_label(collapsed)
    if option is not None and any(
        token in (task_name or "").lower() for token in ("mmlu", "bbh", "choice", "multiple")
    ):
        return option

    return lowered


def extract_final_answer(text: str) -> str:
    patterns = [
        r"final answer\s*[:\-]\s*(.+)",
        r"the answer is\s*[:\-]?\s*(.+)",
        r"answer\s*[:\-]\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def answers_match(prediction: str, target: str, task_name: str | None = None) -> bool:
    return normalize_answer(prediction, task_name) == normalize_answer(target, task_name)
