from __future__ import annotations

from src.baselines.cot import run_cot
from src.baselines.direct import run_direct
from src.search.cost_aware import run_cost_aware_search
from src.types import Example, GenerationResult
from src.verifiers import compute_belief_score, compute_belief_score_sc


def _score(result: GenerationResult, example: Example, llm, config: dict) -> tuple[float, dict]:
    return compute_belief_score(
        result=result,
        prompt=result.metadata.get("prompt", ""),
        task_name=example.task_name,
        llm=llm,
        config=config,
    )


def _score_sc(result: GenerationResult, example: Example, llm, config: dict) -> tuple[float, dict]:
    return compute_belief_score_sc(
        result=result,
        prompt=result.metadata.get("prompt", ""),
        task_name=example.task_name,
        llm=llm,
        config=config,
    )


# Tasks where deliberation_lambda0.1 beats direct empirically.
# Source: full-run per-task results (deliberation_lambda0.1 vs direct).
_SEARCH_BENEFICIAL_TASKS: frozenset[str] = frozenset({
    "gsm8k",                           # 0.83 vs 0.76 (+7pp)
    "date_understanding",              # 0.68 vs 0.62 (+6pp)
    "logical_deduction_three_objects", # 0.94 vs 0.88 (+6pp)
    "mmlu_high_school_mathematics",    # 0.40 vs 0.16 (+24pp)
    "mmlu_formal_logic",               # 0.36 vs 0.36 (tie, search preferred)
})


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


def run_task_aware(example: Example, llm, config: dict) -> GenerationResult:
    """Task-type static routing: search for reasoning tasks, direct for knowledge tasks.

    Routing rule derived from empirical per-task results (deliberation_lambda0.1 vs direct).
    Serves as an oracle-guided upper bound showing how much a perfect task classifier
    could recover over belief-score-based adaptive routing.
    """
    if example.task_name in _SEARCH_BENEFICIAL_TASKS:
        cot = run_cot(example, llm, config)
        searched = run_cost_aware_search(example, llm, config, seed_result=cot)
        searched.metadata["route"] = "search"
        return searched
    else:
        result = run_direct(example, llm, config)
        result.metadata["route"] = "direct"
        return result


def run_adaptive_sc_verifier(example: Example, llm, config: dict) -> GenerationResult:
    """Adaptive routing with sampling-consistency verifier replacing logprob confidence.

    The A component of belief score is replaced by the fraction of k=3 samples at
    T=0.7 that agree with the greedy answer.  This is better calibrated than logprob
    for RLHF-tuned models because wrong answers have higher token-level entropy.
    """
    direct = run_direct(example, llm, config)
    score_0, evidence_0 = _score_sc(direct, example, llm, config)
    direct.metadata.update({"belief": evidence_0, "score": score_0})
    if score_0 >= config["reasoning"]["tau_0"]:
        direct.metadata["route"] = "direct"
        return direct

    cot = run_cot(example, llm, config)
    score_1, evidence_1 = _score_sc(cot, example, llm, config)
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
