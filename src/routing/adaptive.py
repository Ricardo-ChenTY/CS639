from __future__ import annotations

from src.baselines.cot import run_cot
from src.baselines.direct import run_direct
from src.search.cost_aware import run_cost_aware_search
from src.types import Example, GenerationResult
from src.verifiers import compute_belief_score


def _score(result: GenerationResult, example: Example, llm, config: dict) -> tuple[float, dict]:
    return compute_belief_score(
        result=result,
        prompt=result.metadata.get("prompt", ""),
        task_name=example.task_name,
        llm=llm,
        config=config,
    )


def run_adaptive_reasoning(example: Example, llm, config: dict) -> GenerationResult:
    """Full hierarchical method: direct → CoT → search."""
    direct = run_direct(example, llm, config)
    score_0, evidence_0 = _score(direct, example, llm, config)
    direct.metadata.update({"belief": evidence_0, "score": score_0})
    if score_0 >= config["reasoning"]["tau_0"]:
        direct.metadata["route"] = "direct"
        return direct

    cot = run_cot(example, llm, config)
    score_1, evidence_1 = _score(cot, example, llm, config)
    cot.metadata.update({"belief": evidence_1, "score": score_1})
    if score_1 >= config["reasoning"]["tau_1"]:
        cot.metadata["route"] = "cot"
        return cot

    searched = run_cost_aware_search(example, llm, config, seed_result=cot)
    searched.metadata.update({
        "route": "search",
        "pre_scores": {"direct": score_0, "cot": score_1},
    })
    return searched


def run_meta_control_only(example: Example, llm, config: dict) -> GenerationResult:
    """Ablation: direct → CoT routing only, no hard-case search."""
    direct = run_direct(example, llm, config)
    score_0, evidence_0 = _score(direct, example, llm, config)
    direct.metadata.update({"belief": evidence_0, "score": score_0, "route": "direct"})
    if score_0 >= config["reasoning"]["tau_0"]:
        return direct

    cot = run_cot(example, llm, config)
    score_1, evidence_1 = _score(cot, example, llm, config)
    cot.metadata.update({"belief": evidence_1, "score": score_1, "route": "cot"})
    return cot


def run_deliberation_only(example: Example, llm, config: dict) -> GenerationResult:
    """Ablation: always run search after CoT, no selective meta-control."""
    cot = run_cot(example, llm, config)
    _, evidence_1 = _score(cot, example, llm, config)
    cot.metadata["belief"] = evidence_1
    searched = run_cost_aware_search(example, llm, config, seed_result=cot)
    searched.metadata["route"] = "search"
    return searched
