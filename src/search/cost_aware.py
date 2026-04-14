from __future__ import annotations

from collections import defaultdict

from src.parsers.answers import extract_final_answer
from src.parsers.answers import normalize_answer
from src.prompts.templates import build_search_prompt
from src.types import Example, GenerationResult, SearchNode
from src.verifiers import compute_belief_score


def compute_utility(node: SearchNode, config: dict) -> float:
    max_depth = max(config["reasoning"]["max_depth"], 1)
    normalized_cost = (node.cumulative_tokens / max(config["model"]["max_new_tokens"], 1)) + (
        0.1 * node.depth / max_depth
    )
    return node.correctness_score - (config["reasoning"]["lambda_cost"] * normalized_cost)


def _build_seed_node(seed_result: GenerationResult | None, config: dict) -> SearchNode:
    if seed_result is None:
        return SearchNode(
            trace="",
            final_answer="",
            depth=0,
            prompt_tokens=0,
            cumulative_tokens=0,
            latency_sec=0.0,
            correctness_score=0.0,
            utility=0.0,
        )

    seed_belief = seed_result.metadata.get("belief", {})
    node = SearchNode(
        trace=seed_result.text.strip(),
        final_answer=seed_result.final_answer,
        depth=0,
        prompt_tokens=seed_result.prompt_tokens,
        cumulative_tokens=seed_result.total_tokens,
        latency_sec=seed_result.latency_sec,
        correctness_score=seed_belief.get("V", 0.0) + seed_belief.get("A", 0.0) + seed_belief.get("G", 0.0),
        utility=0.0,
        metadata={"belief": seed_belief, "prompt": seed_result.metadata.get("prompt", "")},
    )
    node.correctness_score /= 3.0
    node.utility = compute_utility(node, config)
    return node


def _aggregate_terminal_answers(nodes: list[SearchNode], task_name: str) -> tuple[SearchNode, dict[str, dict]]:
    answer_support: dict[str, dict] = defaultdict(
        lambda: {"support": 0.0, "belief": 0.0, "count": 0, "best_node": None}
    )
    for node in nodes:
        key = normalize_answer(node.final_answer, task_name)
        if not key:
            continue
        bucket = answer_support[key]
        bucket["support"] += max(node.utility, 0.0)
        bucket["belief"] += node.correctness_score
        bucket["count"] += 1
        if bucket["best_node"] is None or node.utility > bucket["best_node"].utility:
            bucket["best_node"] = node

    if not answer_support:
        fallback = max(nodes, key=lambda item: item.utility)
        return fallback, {}

    best_key, best_bucket = max(
        answer_support.items(),
        key=lambda item: (item[1]["support"], item[1]["belief"], item[1]["count"]),
    )
    return best_bucket["best_node"], {
        key: {
            "support": value["support"],
            "belief": value["belief"],
            "count": value["count"],
            "final_answer": value["best_node"].final_answer if value["best_node"] else "",
        }
        for key, value in answer_support.items()
    }


def run_cost_aware_search(
    example: Example,
    llm,
    config: dict,
    *,
    seed_result: GenerationResult | None = None,
) -> GenerationResult:
    frontier: list[SearchNode] = [_build_seed_node(seed_result, config)]
    terminal_nodes: list[SearchNode] = frontier.copy()

    for depth in range(config["reasoning"]["max_depth"]):
        candidates: list[SearchNode] = []
        for node in frontier:
            prompt = build_search_prompt(example, node.trace)
            outputs = llm.generate(
                prompt,
                max_new_tokens=config["model"]["max_new_tokens"],
                temperature=max(config["model"]["temperature"], 0.7),
                n=config["reasoning"]["branching_factor"],
                return_logprobs=True,
            )
            for item in outputs:
                trace = (node.trace + "\n" + item["text"]).strip()
                answer = extract_final_answer(item["text"])
                node_meta: dict = {"prompt": prompt}
                if "answer_logprob" in item:
                    node_meta["answer_logprob"] = item["answer_logprob"]
                candidate = SearchNode(
                    trace=trace,
                    final_answer=answer,
                    depth=depth + 1,
                    prompt_tokens=node.prompt_tokens + item.get("prompt_tokens", 0),
                    cumulative_tokens=node.cumulative_tokens
                    + item.get("prompt_tokens", 0)
                    + item.get("completion_tokens", 0),
                    latency_sec=node.latency_sec + item.get("latency_sec", 0.0),
                    correctness_score=0.0,
                    utility=0.0,
                    metadata=node_meta,
                )
                belief_score, belief_metadata = compute_belief_score(
                    result=GenerationResult(
                        text=trace,
                        final_answer=answer,
                        prompt_tokens=candidate.prompt_tokens,
                        completion_tokens=max(
                            candidate.cumulative_tokens - candidate.prompt_tokens,
                            0,
                        ),
                        latency_sec=candidate.latency_sec,
                        metadata={"prompt": prompt},
                    ),
                    prompt=prompt,
                    task_name=example.task_name,
                    llm=llm,
                    config=config,
                )
                candidate.prompt_tokens += belief_metadata["probe_prompt_tokens"]
                candidate.cumulative_tokens += (
                    belief_metadata["probe_prompt_tokens"] + belief_metadata["probe_completion_tokens"]
                )
                candidate.latency_sec += belief_metadata["probe_latency_sec"]
                candidate.correctness_score = belief_score
                candidate.metadata["belief"] = belief_metadata
                candidate.utility = compute_utility(candidate, config)
                if candidate.utility - node.utility > config["reasoning"]["tau_expand"]:
                    candidates.append(candidate)
                terminal_nodes.append(candidate)

        if not candidates:
            break

        candidates.sort(key=lambda item: item.utility, reverse=True)
        frontier = candidates[: config["reasoning"]["beam_size"]]
        terminal_nodes.extend(frontier)

    best, answer_support = _aggregate_terminal_answers(terminal_nodes, example.task_name)
    return GenerationResult(
        text=best.trace,
        final_answer=best.final_answer,
        prompt_tokens=best.prompt_tokens,
        completion_tokens=max(best.cumulative_tokens - best.prompt_tokens, 0),
        latency_sec=best.latency_sec,
        metadata={
            "best_utility": best.utility,
            "search_depth": best.depth,
            "answer_support": answer_support,
            "terminal_nodes": len(terminal_nodes),
        },
    )
