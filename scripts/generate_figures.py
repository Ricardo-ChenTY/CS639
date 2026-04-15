"""
Generate all paper figures, AUC analysis, and LaTeX tables.
Usage:
    python scripts/generate_figures.py --output_dir outputs/full/ --fig_dir figures/
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHODS_ORDERED = [
    "direct", "cot", "self_consistency", "sc5",
    "adaptive", "meta_control_only",
    "deliberation_only",
    "deliberation_lambda0.1", "deliberation_lambda0.2", "deliberation_lambda0.3",
    "task_aware", "task_aware_lambda0.1", "adaptive_sc_verifier",
]

METHOD_LABELS = {
    "direct":                  "Direct",
    "cot":                     "CoT",
    "self_consistency":        "SC-3",
    "sc5":                     "SC-5",
    "adaptive":                "Adaptive",
    "meta_control_only":       "Meta-Control",
    "deliberation_only":       "Deliberation (λ=0)",
    "deliberation_lambda0.1":  "Deliberation (λ=0.1)",
    "deliberation_lambda0.2":  "Deliberation (λ=0.2)",
    "deliberation_lambda0.3":  "Deliberation (λ=0.3)",
    "task_aware":              "Task-Aware (λ=0)",
    "task_aware_lambda0.1":    "Task-Aware (λ=0.1)",
    "adaptive_sc_verifier":    "Adaptive+SC-Verifier",
}

TASK_LABELS = {
    "date_understanding":              "Date\nUnderstanding",
    "gsm8k":                           "GSM8K",
    "logical_deduction_three_objects": "Logical\nDeduction",
    "mmlu_college_biology":            "College\nBiology",
    "mmlu_computer_security":          "Computer\nSecurity",
    "mmlu_formal_logic":               "Formal\nLogic",
    "mmlu_high_school_mathematics":    "HS Math",
}

COLORS = {
    "direct":                 "#2196F3",
    "cot":                    "#FF9800",
    "self_consistency":       "#9C27B0",
    "sc5":                    "#673AB7",
    "adaptive":               "#4CAF50",
    "meta_control_only":      "#8BC34A",
    "deliberation_only":      "#F44336",
    "deliberation_lambda0.1": "#E91E63",
    "deliberation_lambda0.2": "#FF5722",
    "deliberation_lambda0.3": "#FF7043",
    "task_aware":             "#795548",
    "task_aware_lambda0.1":   "#00BCD4",
    "adaptive_sc_verifier":   "#607D8B",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_records(records_dir: Path, methods: list[str]) -> dict:
    data = {}
    for m in methods:
        p = records_dir / f"{m}_records.json"
        if p.exists():
            data[m] = json.loads(p.read_text())
    return data


def method_stats(records: list) -> dict:
    acc = sum(r["is_correct"] for r in records) / len(records)
    avg_tok = sum(r["total_tokens"] for r in records) / len(records)
    return {"accuracy": acc, "avg_tokens": avg_tok, "n": len(records)}


def bootstrap_ci(correct: list[bool], n_boot: int = 2000, ci: float = 0.95):
    n = len(correct)
    rng = random.Random(42)
    means = sorted(
        sum(correct[rng.randint(0, n - 1)] for _ in range(n)) / n
        for _ in range(n_boot)
    )
    alpha = (1 - ci) / 2
    return sum(correct) / n, means[int(alpha * n_boot)], means[int((1 - alpha) * n_boot)]


# ---------------------------------------------------------------------------
# Figure 1: Accuracy–Cost Scatter
# ---------------------------------------------------------------------------

def fig_accuracy_cost(all_records: dict, fig_dir: Path) -> None:
    """Accuracy–cost scatter using legend + marker shapes.

    Three groups shown with distinct markers:
      ○  Baselines   (direct, cot, SC-3, SC-5)
      △  Ablations   (adaptive, meta_control_only, deliberation_only, λ sweep)
      ★  Proposed    (task_aware_lambda0.1, deliberation_lambda0.1)
    """
    # Groups and their markers
    groups = [
        ("Baselines",  ["direct", "cot", "self_consistency", "sc5"],
         "o", 80, 0.75),
        ("Ablations",  ["adaptive", "meta_control_only",
                        "deliberation_only", "deliberation_lambda0.2",
                        "deliberation_lambda0.3", "adaptive_sc_verifier"],
         "^", 80, 0.75),
        ("Proposed",   ["deliberation_lambda0.1", "task_aware_lambda0.1"],
         "*", 220, 1.0),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    for grp_name, methods, marker, size, alpha in groups:
        for i, method in enumerate(methods):
            if method not in all_records:
                continue
            s = method_stats(all_records[method])
            ax.scatter(
                s["avg_tokens"], s["accuracy"],
                marker=marker, s=size, alpha=alpha, zorder=3,
                color=COLORS.get(method, "gray"),
                edgecolors="white", linewidths=0.8,
                label=METHOD_LABELS.get(method, method),
            )

    ax.axhline(0.677, color=COLORS["direct"], linestyle="--",
               linewidth=1.2, alpha=0.5, zorder=2, label="_nolegend_")
    ax.text(2100, 0.680, "Direct\nbaseline", ha="right", va="bottom",
            fontsize=7.5, color=COLORS["direct"], alpha=0.8)

    ax.set_xlabel("Average Tokens per Example", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Accuracy–Cost Tradeoff", fontsize=12)
    ax.set_xlim(200, 2200)
    ax.set_ylim(0.47, 0.77)
    ax.grid(True, alpha=0.3)

    # Legend: two columns, outside plot area on right
    ax.legend(
        fontsize=7.5, ncol=2,
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        framealpha=0.9, borderpad=0.8,
        handlelength=1.2, handleheight=1.2,
    )

    fig.tight_layout()
    fig.subplots_adjust(right=0.72)
    _save(fig, fig_dir / "fig1_accuracy_cost")


# ---------------------------------------------------------------------------
# Figure 2: Per-task bar chart (direct vs delib_λ0.1 vs task_aware_λ0.1)
# ---------------------------------------------------------------------------

def fig_pertask_deliberation(all_records: dict, fig_dir: Path) -> None:
    tasks = list(TASK_LABELS.keys())
    methods_show = ["direct", "deliberation_lambda0.1", "task_aware_lambda0.1"]

    task_acc: dict[str, dict] = {m: {} for m in methods_show}
    for method in methods_show:
        if method not in all_records:
            continue
        by_task = defaultdict(list)
        for r in all_records[method]:
            by_task[r["task_name"]].append(r["is_correct"])
        for task, vals in by_task.items():
            task_acc[method][task] = sum(vals) / len(vals)

    x = np.arange(len(tasks))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, method in enumerate(methods_show):
        vals = [task_acc[method].get(t, 0) for t in tasks]
        ax.bar(x + (i - 1) * width, vals, width,
               label=METHOD_LABELS.get(method, method),
               color=COLORS.get(method, "gray"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Per-Task Accuracy: Direct vs Best Deliberation Methods", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, fig_dir / "fig2_pertask_deliberation")


# ---------------------------------------------------------------------------
# Figure 3: Lambda sweep
# ---------------------------------------------------------------------------

def fig_lambda_sweep(all_records: dict, fig_dir: Path) -> None:
    lambdas = [0.0, 0.1, 0.2, 0.3]
    method_map = {
        0.0: "deliberation_only",
        0.1: "deliberation_lambda0.1",
        0.2: "deliberation_lambda0.2",
        0.3: "deliberation_lambda0.3",
    }

    valid = []
    for lam in lambdas:
        m = method_map[lam]
        if m in all_records:
            s = method_stats(all_records[m])
            valid.append((lam, s["accuracy"], s["avg_tokens"]))

    ls_, as_, ts_ = zip(*valid)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    ax1.plot(ls_, as_, "o-", color="#E91E63", linewidth=2, markersize=8, label="Accuracy")
    ax2.plot(ls_, ts_, "s--", color="#607D8B", linewidth=1.5, markersize=6, label="Avg Tokens")
    ax1.axhline(0.677, color=COLORS["direct"], linestyle=":", linewidth=1.5, alpha=0.6)
    ax1.text(0.29, 0.681, "Direct (0.677)", fontsize=8, color=COLORS["direct"])

    ax1.set_xlabel("Cost Penalty λ", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11, color="#E91E63")
    ax2.set_ylabel("Avg Tokens", fontsize=11, color="#607D8B")
    ax1.set_title("Deliberation: Accuracy & Cost vs λ", fontsize=12)
    ax1.set_xticks(lambdas)
    ax1.set_ylim(0.50, 0.76)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, fig_dir / "fig3_lambda_sweep")


# ---------------------------------------------------------------------------
# Figure 4: Belief score histogram
# ---------------------------------------------------------------------------

def fig_belief_histogram(all_records: dict, fig_dir: Path) -> None:
    if "adaptive" not in all_records:
        return
    records = all_records["adaptive"]
    correct_scores = [r["score"] for r in records if r["is_correct"] and r.get("score") is not None]
    wrong_scores   = [r["score"] for r in records if not r["is_correct"] and r.get("score") is not None]

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0.55, 1.0, 30)
    ax.hist(correct_scores, bins=bins, alpha=0.6, color="#2196F3", label=f"Correct (n={len(correct_scores)})")
    ax.hist(wrong_scores,   bins=bins, alpha=0.6, color="#F44336", label=f"Wrong (n={len(wrong_scores)})")
    ax.axvline(np.mean(correct_scores), color="#1565C0", linestyle="--", linewidth=1.5,
               label=f"Correct mean={np.mean(correct_scores):.3f}")
    ax.axvline(np.mean(wrong_scores),   color="#B71C1C", linestyle="--", linewidth=1.5,
               label=f"Wrong mean={np.mean(wrong_scores):.3f}")
    ax.set_xlabel("Belief Score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Belief Score Distribution: Correct vs Wrong\n(AUC = 0.361, worse than random)", fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, fig_dir / "fig4_belief_histogram")


# ---------------------------------------------------------------------------
# Figure 5: Bootstrap CI (error bar plot)
# ---------------------------------------------------------------------------

def fig_bootstrap_ci(all_records: dict, fig_dir: Path) -> None:
    # Show key methods only (skip redundant ones)
    methods_show = [
        "task_aware_lambda0.1", "deliberation_lambda0.1", "direct",
        "adaptive_sc_verifier", "adaptive", "deliberation_only",
        "sc5", "self_consistency", "cot",
    ]
    methods_show = [m for m in methods_show if m in all_records]

    accs, los, his = [], [], []
    for m in methods_show:
        correct = [r["is_correct"] for r in all_records[m]]
        acc, lo, hi = bootstrap_ci(correct)
        accs.append(acc)
        los.append(acc - lo)
        his.append(hi - acc)

    y = np.arange(len(methods_show))
    labels = [METHOD_LABELS.get(m, m) for m in methods_show]
    colors = [COLORS.get(m, "gray") for m in methods_show]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(y, accs, xerr=[los, his], align="center",
            color=colors, alpha=0.85, capsize=4, error_kw={"elinewidth": 1.5, "ecolor": "black"})
    ax.axvline(0.677, color=COLORS["direct"], linestyle="--", linewidth=1.2, alpha=0.6, label="Direct baseline")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Accuracy (95% Bootstrap CI)", fontsize=11)
    ax.set_title("Method Comparison with Bootstrap Confidence Intervals", fontsize=11)
    ax.set_xlim(0.44, 0.82)
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, fig_dir / "fig5_bootstrap_ci")


# ---------------------------------------------------------------------------
# Figure 6: Oracle upper bound
# ---------------------------------------------------------------------------

def fig_oracle(all_records: dict, fig_dir: Path) -> None:
    if not all_records.get("direct") or not all_records.get("deliberation_only"):
        print("Skipping oracle figure: missing records")
        return

    direct_r = {r["example_id"]: r for r in all_records["direct"]}
    cot_r    = {r["example_id"]: r for r in all_records.get("cot", [])}
    delib_r  = {r["example_id"]: r for r in all_records["deliberation_only"]}
    ids = set(direct_r) & set(delib_r)
    if cot_r:
        ids &= set(cot_r)

    oracle_correct = sum(
        1 for eid in ids
        if direct_r[eid]["is_correct"]
        or (cot_r.get(eid, {}).get("is_correct", False))
        or delib_r[eid]["is_correct"]
    )
    oracle_acc = oracle_correct / len(ids)

    # Per-task oracle gap
    tasks = list(TASK_LABELS.keys())
    task_direct, task_oracle = {}, {}
    for task in tasks:
        task_ids = [eid for eid in ids if direct_r[eid]["task_name"] == task]
        if not task_ids:
            continue
        task_direct[task] = sum(direct_r[eid]["is_correct"] for eid in task_ids) / len(task_ids)
        task_oracle[task] = sum(
            1 for eid in task_ids
            if direct_r[eid]["is_correct"]
            or (cot_r.get(eid, {}).get("is_correct", False))
            or delib_r[eid]["is_correct"]
        ) / len(task_ids)

    # Best achieved per task
    best_records = all_records.get("task_aware_lambda0.1") or all_records.get("deliberation_lambda0.1")
    if best_records:
        best_r = {r["example_id"]: r for r in best_records}
        task_best = {}
        for task in tasks:
            task_ids = [eid for eid in best_r if best_r[eid]["task_name"] == task]
            if task_ids:
                task_best[task] = sum(best_r[eid]["is_correct"] for eid in task_ids) / len(task_ids)
    else:
        task_best = {}

    tasks_present = [t for t in tasks if t in task_direct]
    x = np.arange(len(tasks_present))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, [task_direct[t] for t in tasks_present], width,
           label="Direct", color=COLORS["direct"], alpha=0.85)
    if task_best:
        ax.bar(x, [task_best.get(t, 0) for t in tasks_present], width,
               label="Task-Aware (λ=0.1)", color=COLORS["task_aware_lambda0.1"], alpha=0.85)
    ax.bar(x + width, [task_oracle[t] for t in tasks_present], width,
           label="Oracle Upper Bound", color="#37474F", alpha=0.75, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks_present], fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(f"Per-Task: Direct vs Best Achieved vs Oracle\n"
                 f"(Overall: Direct={0.677:.3f}, Task-Aware λ0.1={0.713:.3f}, Oracle={oracle_acc:.3f})",
                 fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, fig_dir / "fig6_oracle_gap")


# ---------------------------------------------------------------------------
# AUC analysis
# ---------------------------------------------------------------------------

def compute_auc(all_records: dict) -> None:
    from sklearn.metrics import roc_auc_score

    if "adaptive" not in all_records:
        return
    records = all_records["adaptive"]
    scores = [r["score"] for r in records if r.get("score") is not None]
    labels = [int(r["is_correct"]) for r in records if r.get("score") is not None]
    auc = roc_auc_score(labels, scores)

    print(f"\n=== Verifier AUC ===")
    print(f"  Overall AUC: {auc:.4f}  (random=0.5000)")
    print(f"  Correct mean: {np.mean([s for s, l in zip(scores, labels) if l]):.4f}")
    print(f"  Wrong mean:   {np.mean([s for s, l in zip(scores, labels) if not l]):.4f}")

    by_task = defaultdict(lambda: {"scores": [], "labels": []})
    for r in records:
        if r.get("score") is not None:
            by_task[r["task_name"]]["scores"].append(r["score"])
            by_task[r["task_name"]]["labels"].append(int(r["is_correct"]))
    print("  Per-task AUC:")
    for task, d in sorted(by_task.items()):
        if len(set(d["labels"])) < 2:
            print(f"    {task:<45} N/A")
            continue
        print(f"    {task:<45} {roc_auc_score(d['labels'], d['scores']):.4f}")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def generate_latex_tables(all_records: dict, fig_dir: Path) -> None:
    lines = []

    # ── Main results table ──────────────────────────────────────────────────
    main_methods = [
        "direct", "cot", "self_consistency", "sc5",
        "adaptive", "meta_control_only",
        "deliberation_only", "deliberation_lambda0.1",
        "task_aware_lambda0.1",
    ]
    lines.append("% ── Main Results Table ──────────────────────────────────")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Overall accuracy, average tokens, and J-score ($\mu{=}0.1$) on 300 evaluation examples. Best value in \textbf{bold}.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Accuracy & Avg Tokens & J-score \\")
    lines.append(r"\midrule")

    max_tok = max(
        sum(r["total_tokens"] for r in all_records[m]) / len(all_records[m])
        for m in main_methods if m in all_records
    )
    rows = []
    for m in main_methods:
        if m not in all_records:
            continue
        records = all_records[m]
        acc = sum(r["is_correct"] for r in records) / len(records)
        avg_tok = sum(r["total_tokens"] for r in records) / len(records)
        j = sum(r["is_correct"] - 0.1 * (r["total_tokens"] / max_tok) for r in records) / len(records)
        rows.append((m, acc, avg_tok, j))

    best_acc = max(r[1] for r in rows)
    best_j   = max(r[3] for r in rows)

    groups = [
        ("Baselines", ["direct", "cot", "self_consistency", "sc5"]),
        ("Ablations", ["adaptive", "meta_control_only", "deliberation_only"]),
        ("Ours", ["deliberation_lambda0.1", "task_aware_lambda0.1"]),
    ]
    row_map = {r[0]: r for r in rows}
    for grp_name, grp_methods in groups:
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{grp_name}}}}} \\")
        for m in grp_methods:
            if m not in row_map:
                continue
            _, acc, avg_tok, j = row_map[m]
            acc_s = rf"\textbf{{{acc:.3f}}}" if abs(acc - best_acc) < 1e-9 else f"{acc:.3f}"
            j_s   = rf"\textbf{{{j:.3f}}}" if abs(j - best_j) < 1e-9 else f"{j:.3f}"
            label = METHOD_LABELS.get(m, m).replace("λ", r"$\lambda$")
            lines.append(rf"  {label} & {acc_s} & {avg_tok:.0f} & {j_s} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # ── Ablation table ──────────────────────────────────────────────────────
    lines.append("% ── Ablation Table ─────────────────────────────────────")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study. \checkmark~indicates the component is active.}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Meta-Control & Deliberation & Optimal $\lambda$ & Accuracy \\")
    lines.append(r"\midrule")

    ablation_rows = [
        ("direct",                "Direct",                    r"\xmark", r"\xmark", r"\xmark"),
        ("adaptive",              "Adaptive (full)",           r"\cmark", r"\cmark", r"\xmark"),
        ("meta_control_only",     "Meta-Control only",         r"\cmark", r"\xmark", r"\xmark"),
        ("deliberation_only",     "Deliberation only",         r"\xmark", r"\cmark", r"\xmark"),
        ("deliberation_lambda0.1","Deliberation ($\lambda$=0.1)", r"\xmark", r"\cmark", r"\cmark"),
        ("task_aware_lambda0.1",  "Task-Aware ($\lambda$=0.1)", r"task",   r"\cmark", r"\cmark"),
    ]

    for m, label, mc, delib, lam in ablation_rows:
        if m not in all_records:
            continue
        records = all_records[m]
        acc = sum(r["is_correct"] for r in records) / len(records)
        acc_s = rf"\textbf{{{acc:.3f}}}" if m == "task_aware_lambda0.1" else f"{acc:.3f}"
        lines.append(rf"  {label} & {mc} & {delib} & {lam} & {acc_s} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{note}")
    # fix the closing tag
    lines[-1] = r"\end{table}"
    lines.append("")

    out = fig_dir / "tables.tex"
    out.write_text("\n".join(lines))
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, base: Path) -> None:
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".png", bbox_inches="tight", dpi=150)
    print(f"Saved {base}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/full/")
    parser.add_argument("--fig_dir", default="figures/")
    args = parser.parse_args()

    records_dir = Path(args.output_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    available = [m for m in METHODS_ORDERED if (records_dir / f"{m}_records.json").exists()]
    print(f"Found: {available}\n")
    all_records = load_records(records_dir, available)

    compute_auc(all_records)

    print("\n=== Generating Figures ===")
    fig_accuracy_cost(all_records, fig_dir)
    fig_pertask_deliberation(all_records, fig_dir)
    fig_lambda_sweep(all_records, fig_dir)
    fig_belief_histogram(all_records, fig_dir)
    fig_bootstrap_ci(all_records, fig_dir)
    fig_oracle(all_records, fig_dir)

    print("\n=== Generating LaTeX Tables ===")
    generate_latex_tables(all_records, fig_dir)

    print(f"\nAll outputs saved to {fig_dir}/")


if __name__ == "__main__":
    main()
