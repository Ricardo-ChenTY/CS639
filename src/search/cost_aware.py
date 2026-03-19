from __future__ import annotations

from src.parsers.answers import extract_final_answer
from src.prompts.templates import build_search_prompt
from src.types import Example, GenerationResult, SearchNode


def compute_correctness_proxy(answer: str, trace: str) -> float:
    components = [
        1.0 if answer else 0.0,
        1.0 if "final answer" in trace.lower() else 0.0,
        1.0 if trace.strip() else 0.0,
    ]
    return sum(components) / len(components)


def compute_utility(node: SearchNode, config: dict) -> float:
    max_depth = max(config["reasoning"]["max_depth"], 1)
    normalized_cost = (node.cumulative_tokens / max(config["model"]["max_new_tokens"], 1)) + (
        0.1 * node.depth / max_depth
    )
    return node.correctness_score - (config["reasoning"]["lambda_cost"] * normalized_cost)


def run_cost_aware_search(example: Example, llm, config: dict) -> GenerationResult:
    frontier: list[SearchNode] = [
        SearchNode(
            trace="",
            final_answer="",
            depth=0,
            cumulative_tokens=0,
            correctness_score=0.0,
            utility=0.0,
        )
    ]
    best = frontier[0]

    for depth in range(config["reasoning"]["max_depth"]):
        candidates: list[SearchNode] = []
        for node in frontier:
            prompt = build_search_prompt(example, node.trace)
            outputs = llm.generate(
                prompt,
                max_new_tokens=config["model"]["max_new_tokens"],
                temperature=max(config["model"]["temperature"], 0.7),
                n=config["reasoning"]["branching_factor"],
            )
            for item in outputs:
                trace = (node.trace + "\n" + item["text"]).strip()
                answer = extract_final_answer(item["text"])
                candidate = SearchNode(
                    trace=trace,
                    final_answer=answer,
                    depth=depth + 1,
                    cumulative_tokens=node.cumulative_tokens + item.get("completion_tokens", 0),
                    correctness_score=compute_correctness_proxy(answer, trace),
                    utility=0.0,
                    metadata={"prompt": prompt},
                )
                candidate.utility = compute_utility(candidate, config)
                if candidate.utility - node.utility > config["reasoning"]["tau_expand"]:
                    candidates.append(candidate)
                if candidate.utility > best.utility:
                    best = candidate

        if not candidates:
            break

        candidates.sort(key=lambda item: item.utility, reverse=True)
        frontier = candidates[: config["reasoning"]["beam_size"]]

    return GenerationResult(
        text=best.trace,
        final_answer=best.final_answer,
        completion_tokens=best.cumulative_tokens,
        metadata={"best_utility": best.utility, "search_depth": best.depth},
    )
