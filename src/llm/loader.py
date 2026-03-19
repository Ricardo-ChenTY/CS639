from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LLMClient:
    model_name_or_path: str

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        n: int = 1,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "Implement model loading and generation here. "
            "This scaffold intentionally leaves the backend open."
        )


def build_llm(model_config: dict[str, Any]) -> LLMClient:
    return LLMClient(model_name_or_path=model_config["model_name_or_path"])
