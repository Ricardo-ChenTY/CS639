from __future__ import annotations

from src.baselines.cot import run_cot
from src.baselines.direct import run_direct
from src.search.cost_aware import run_cost_aware_search
from src.types import Example, GenerationResult


def compute_stopping_score(result: GenerationResult) -> float:
    """Placeholder for the V/A/G correctness posterior in the proposal."""
    has_final_answer = 1.0 if result.final_answer else 0.0
    has_text = 1.0 if result.text.strip() else 0.0
    reasonable_length = 1.0 if result.total_tokens > 0 else 0.0
    return (has_final_answer + has_text + reasonable_length) / 3.0


def run_adaptive_reasoning(example: Example, llm, config: dict) -> GenerationResult:
    direct = run_direct(example, llm, config)
    score_0 = compute_stopping_score(direct)
    if score_0 >= config["reasoning"]["tau_0"]:
        direct.metadata["route"] = "direct"
        direct.metadata["score"] = score_0
        return direct

    cot = run_cot(example, llm, config)
    score_1 = compute_stopping_score(cot)
    if score_1 >= config["reasoning"]["tau_1"]:
        cot.metadata["route"] = "cot"
        cot.metadata["score"] = score_1
        return cot

    searched = run_cost_aware_search(example, llm, config)
    searched.metadata["route"] = "search"
    searched.metadata["pre_scores"] = {"direct": score_0, "cot": score_1}
    return searched
