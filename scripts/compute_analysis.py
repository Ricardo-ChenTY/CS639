"""
Offline analysis: bootstrap confidence intervals + matched-budget comparison.
Run after all experiments are complete.
Usage:
    python scripts/compute_analysis.py --output_dir outputs/full/
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(correct: list[bool], n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, lower, upper) accuracy with bootstrap CI."""
    n = len(correct)
    rng = random.Random(42)
    means = []
    for _ in range(n_boot):
        sample = [correct[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1 - ci) / 2
    lo = means[int(alpha * n_boot)]
    hi = means[int((1 - alpha) * n_boot)]
    return sum(correct) / n, lo, hi


def compute_bootstrap(records_dir: Path, methods: list[str]) -> dict:
    results = {}
    for method in methods:
        path = records_dir / f"{method}_records.json"
        if not path.exists():
            continue
        with path.open() as f:
            records = json.load(f)

        # Overall
        correct = [r["is_correct"] for r in records]
        mean, lo, hi = bootstrap_ci(correct)
        results[method] = {
            "overall": {"accuracy": mean, "ci_lo": lo, "ci_hi": hi, "n": len(correct)},
        }

        # Per task
        task_correct = defaultdict(list)
        for r in records:
            task_correct[r["task_name"]].append(r["is_correct"])
        results[method]["per_task"] = {}
        for task, tc in sorted(task_correct.items()):
            m, l, h = bootstrap_ci(tc)
            results[method]["per_task"][task] = {"accuracy": m, "ci_lo": l, "ci_hi": h, "n": len(tc)}

    return results


# ---------------------------------------------------------------------------
# Matched-budget comparison
# ---------------------------------------------------------------------------

def compute_matched_budget(records_dir: Path, methods: list[str]) -> dict:
    """
    For each method, report accuracy at its natural token budget.
    Then find pairs of methods within 20% token budget of each other
    and compare accuracy directly.
    """
    method_stats = {}
    for method in methods:
        path = records_dir / f"{method}_records.json"
        if not path.exists():
            continue
        with path.open() as f:
            records = json.load(f)
        avg_tok = sum(r["total_tokens"] for r in records) / len(records)
        acc = sum(r["is_correct"] for r in records) / len(records)
        method_stats[method] = {"accuracy": acc, "avg_tokens": avg_tok, "n": len(records)}

    # Find matched-budget pairs (within 20% of each other)
    matched_pairs = []
    method_list = list(method_stats.items())
    for i, (m1, s1) in enumerate(method_list):
        for m2, s2 in method_list[i + 1:]:
            ratio = max(s1["avg_tokens"], s2["avg_tokens"]) / max(min(s1["avg_tokens"], s2["avg_tokens"]), 1)
            if ratio <= 1.2:
                matched_pairs.append({
                    "methods": [m1, m2],
                    "tokens": [round(s1["avg_tokens"]), round(s2["avg_tokens"])],
                    "accuracies": [round(s1["accuracy"], 4), round(s2["accuracy"], 4)],
                    "token_ratio": round(ratio, 3),
                })

    return {"method_stats": method_stats, "matched_pairs": matched_pairs}


# ---------------------------------------------------------------------------
# J-score summary  J = acc - mu * normalized_cost
# ---------------------------------------------------------------------------

def compute_j_scores(records_dir: Path, methods: list[str], mu: float = 0.1) -> dict:
    """
    J = mean(acc_i - mu * normalized_cost_i)
    normalized_cost = total_tokens / max_tokens_across_methods
    """
    # First pass: find max avg tokens for normalization
    avg_tokens = {}
    for method in methods:
        path = records_dir / f"{method}_records.json"
        if not path.exists():
            continue
        with path.open() as f:
            records = json.load(f)
        avg_tokens[method] = sum(r["total_tokens"] for r in records) / len(records)

    if not avg_tokens:
        return {}
    max_tok = max(avg_tokens.values())

    j_scores = {}
    for method in methods:
        path = records_dir / f"{method}_records.json"
        if not path.exists():
            continue
        with path.open() as f:
            records = json.load(f)
        j = sum(
            r["is_correct"] - mu * (r["total_tokens"] / max_tok)
            for r in records
        ) / len(records)
        j_scores[method] = {
            "J": round(j, 4),
            "accuracy": round(sum(r["is_correct"] for r in records) / len(records), 4),
            "avg_tokens": round(avg_tokens[method], 1),
            "normalized_cost": round(avg_tokens[method] / max_tok, 4),
        }

    return j_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METHODS = [
    "direct", "cot", "self_consistency", "sc5",
    "adaptive", "meta_control_only", "deliberation_only",
    "deliberation_lambda0.1", "deliberation_lambda0.2", "deliberation_lambda0.3",
    "task_aware", "task_aware_lambda0.1", "adaptive_sc_verifier",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/full/", help="Directory with *_records.json files.")
    parser.add_argument("--mu", type=float, default=0.1, help="Cost penalty for J-score.")
    args = parser.parse_args()

    records_dir = Path(args.output_dir)
    available = [m for m in METHODS if (records_dir / f"{m}_records.json").exists()]
    print(f"Found records for: {available}\n")

    # Bootstrap CIs
    print("=== Bootstrap Confidence Intervals (95%) ===")
    ci_results = compute_bootstrap(records_dir, available)
    for method, res in ci_results.items():
        o = res["overall"]
        print(f"  {method:<30} {o['accuracy']:.3f}  [{o['ci_lo']:.3f}, {o['ci_hi']:.3f}]")

    print()
    print("  Per-task CIs:")
    tasks = sorted({t for m in ci_results.values() for t in m.get("per_task", {})})
    header = "  %-45s" % "Task" + "".join(f"  {m[:8]:<8}" for m in available)
    print(header)
    for task in tasks:
        row = f"  {task:<45}"
        for method in available:
            pt = ci_results.get(method, {}).get("per_task", {}).get(task)
            if pt:
                row += f"  {pt['accuracy']:.2f}±{(pt['ci_hi']-pt['ci_lo'])/2:.2f}"
            else:
                row += "  —       "
        print(row)

    # Matched-budget
    print("\n=== Matched-Budget Comparison (within 20% tokens) ===")
    mb = compute_matched_budget(records_dir, available)
    print("  Method stats:")
    for m, s in sorted(mb["method_stats"].items(), key=lambda x: x[1]["avg_tokens"]):
        print(f"    {m:<30} acc={s['accuracy']:.3f}  avg_tok={s['avg_tokens']:.0f}")
    if mb["matched_pairs"]:
        print("  Matched pairs:")
        for p in mb["matched_pairs"]:
            m1, m2 = p["methods"]
            a1, a2 = p["accuracies"]
            t1, t2 = p["tokens"]
            winner = m1 if a1 >= a2 else m2
            print(f"    {m1}({t1} tok, {a1:.3f}) vs {m2}({t2} tok, {a2:.3f})  → {winner} wins")
    else:
        print("  No pairs within 20% token budget.")

    # J-scores
    print(f"\n=== J-scores (mu={args.mu}) ===")
    j = compute_j_scores(records_dir, available, mu=args.mu)
    for method, s in sorted(j.items(), key=lambda x: -x[1]["J"]):
        print(f"  {method:<30} J={s['J']:.4f}  acc={s['accuracy']:.3f}  norm_cost={s['normalized_cost']:.3f}")

    # Save
    out = {
        "bootstrap_ci": ci_results,
        "matched_budget": mb,
        "j_scores": j,
    }
    out_path = records_dir / "analysis_results.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved full analysis to {out_path}")


if __name__ == "__main__":
    main()
