from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from src.baselines.cot import run_cot
from src.baselines.direct import run_direct
from src.baselines.self_consistency import run_self_consistency
from src.evaluation.metrics import build_record, summarize
from src.routing.adaptive import (
    run_adaptive_reasoning,
    run_adaptive_sc_verifier,
    run_deliberation_only,
    run_meta_control_only,
    run_task_aware,
)


METHODS = {
    "direct": run_direct,
    "cot": run_cot,
    "self_consistency": run_self_consistency,
    "adaptive": run_adaptive_reasoning,
    "meta_control_only": run_meta_control_only,
    "deliberation_only": run_deliberation_only,
    "task_aware": run_task_aware,
    "adaptive_sc_verifier": run_adaptive_sc_verifier,
}


def run_experiment(config: dict, examples: list, llm, method: str) -> dict:
    runner = METHODS[method]
    records = []

    for example in tqdm(examples, desc=f"Running {method}", leave=False):
        result = runner(example, llm, config)
        records.append(
            build_record(
                example=example,
                method=method,
                prediction=result.final_answer,
                total_tokens=result.total_tokens,
                latency_sec=result.latency_sec,
                metadata=result.metadata,
            )
        )

    summary = summarize(records)
    if config["runtime"]["save_predictions"]:
        save_results(config, method, records, summary)
    return summary


def save_results(config: dict, method: str, records: list, summary: dict) -> None:
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    records_path = output_dir / f"{method}_records.json"
    summary_path = output_dir / f"{method}_summary.json"

    with records_path.open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "example_id": record.example.example_id,
                    "task_name": record.example.task_name,
                    "prediction": record.prediction,
                    "gold": record.example.answer,
                    "is_correct": record.is_correct,
                    "total_tokens": record.total_tokens,
                    "latency_sec": record.latency_sec,
                    "route": record.metadata.get("route"),
                    "score": record.metadata.get("score"),
                    "pre_scores": record.metadata.get("pre_scores"),
                }
                for record in records
            ],
            handle,
            indent=2,
        )

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
