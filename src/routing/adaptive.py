from __future__ import annotations

from src.baselines.cot import run_cot
from src.baselines.direct import run_direct
from src.search.cost_aware import run_cost_aware_search
from src.types import Example, GenerationResult
from src.verifiers import compute_belief_score


def run_adaptive_reasoning(example: Example, llm, config: dict) -> GenerationResult:
    direct = run_direct(example, llm, config)
    score_0, evidence_0 = compute_belief_score(
        result=direct,
        prompt=direct.metadata["prompt"],
        task_name=example.task_name,
        llm=llm,
        config=config,
    )
    direct.prompt_tokens += evidence_0["probe_prompt_tokens"]
    direct.completion_tokens += evidence_0["probe_completion_tokens"]
    direct.latency_sec += evidence_0["probe_latency_sec"]
    direct.metadata["belief"] = evidence_0
    if score_0 >= config["reasoning"]["tau_0"]:
        direct.metadata["route"] = "direct"
        direct.metadata["score"] = score_0
        return direct

    cot = run_cot(example, llm, config)
    score_1, evidence_1 = compute_belief_score(
        result=cot,
        prompt=cot.metadata["prompt"],
        task_name=example.task_name,
        llm=llm,
        config=config,
    )
    cot.prompt_tokens += evidence_1["probe_prompt_tokens"]
    cot.completion_tokens += evidence_1["probe_completion_tokens"]
    cot.latency_sec += evidence_1["probe_latency_sec"]
    cot.metadata["belief"] = evidence_1
    if score_1 >= config["reasoning"]["tau_1"]:
        cot.metadata["route"] = "cot"
        cot.metadata["score"] = score_1
        return cot

    searched = run_cost_aware_search(example, llm, config, seed_result=cot)
    searched.metadata["route"] = "search"
    searched.metadata["pre_scores"] = {"direct": score_0, "cot": score_1}
    searched.metadata["pre_belief"] = {"direct": evidence_0, "cot": evidence_1}
    return searched
