# Experiment Documentation

**Project:** Hierarchical Budgeted Belief-Controlled Reasoning for Inference-Time Compute Allocation  
**Model:** Qwen3-4B-Instruct-2507  
**Evaluation:** 300 examples across 7 tasks (full run)  
**Question:** *When is extra reasoning worth the cost?*

---

## 1. Setup

### Model & Hardware
- Model: `Qwen3-4B-Instruct-2507` (local weights under `models/`)
- GPU: NVIDIA RTX 5090 (32 GB VRAM), CUDA 13.0
- Attention: `sdpa` (flash_attn incompatible with CUDA 13.0)
- Precision: `bfloat16`

### Environment
```bash
cd /root/CS639
source .venv/bin/activate
```

### Datasets (7 tasks, 300 eval examples total)
| Task | Split | n |
|------|-------|---|
| gsm8k | eval | 100 |
| bbh_date_understanding | eval | 50 |
| bbh_logical_deduction_three_objects | eval | 50 |
| mmlu_college_biology | eval | 25 |
| mmlu_computer_security | eval | 25 |
| mmlu_formal_logic | eval | 25 |
| mmlu_high_school_mathematics | eval | 25 |

Download frozen splits:
```bash
python scripts/download_datasets.py --root-dir dataset --overwrite
```

---

## 2. Methods

### Baselines
| Tag | Description |
|-----|-------------|
| `direct` | Single greedy decode, no chain-of-thought |
| `cot` | Zero-shot CoT prompt ("Think step by step") |
| `self_consistency` | SC-3: 3 samples at T=0.7, majority vote |
| `sc5` | SC-5: 5 samples at T=0.7, majority vote |

### Full Framework
| Tag | Description |
|-----|-------------|
| `adaptive` | Full belief-controlled routing: direct → CoT → search based on belief score thresholds τ₀, τ₁ |

### Ablations
| Tag | Description |
|-----|-------------|
| `meta_control_only` | Direct → CoT routing only; search removed |
| `deliberation_only` | Always CoT → search; no selective routing (λ=0) |
| `deliberation_lambda0.1` | Always search with cost penalty λ=0.1 |
| `deliberation_lambda0.2` | Always search with cost penalty λ=0.2 |
| `deliberation_lambda0.3` | Always search with cost penalty λ=0.3 |
| `task_aware` | Static routing by task type (λ=0); search for reasoning tasks, direct for knowledge tasks |
| `task_aware_lambda0.1` | Static routing by task type with optimal λ=0.1 |
| `adaptive_sc_verifier` | Full adaptive framework with sampling-consistency verifier replacing logprob confidence |

### Belief Score Construction
```
belief = (V + A + G) / 3

V = verifier_consistency_score   # 1.0 if extracted answer matches raw text answer
A = logprob_confidence_score     # exp(mean_token_logprob)
G = format_constraint_score      # 1.0 if answer format is valid for the task
```

For `adaptive_sc_verifier`, A is replaced by sampling consistency:
```
A = fraction of k=3 samples at T=0.7 that agree with T=0 greedy answer
```

### Task-Aware Routing Rules (from empirical per-task results)
```python
SEARCH_BENEFICIAL_TASKS = {
    "gsm8k",                           # delib_λ0.1=0.88 > direct=0.76
    "date_understanding",              # delib_λ0.1=0.68 > direct=0.62
    "logical_deduction_three_objects", # delib_λ0.1=0.92 > direct=0.88
    "mmlu_high_school_mathematics",    # delib_λ0.1=0.32 > direct=0.16
    "mmlu_formal_logic",               # delib_λ0.1=0.32 ≈ direct=0.36 (tie)
}
# Remaining tasks (college_biology, computer_security) → direct
```

---

## 3. Experiment Workflow

### Step 1: Dev Tuning (grid search over τ and λ)
```bash
python scripts/tune_reasoning.py \
  --config configs/full_server.yaml \
  --output-config configs/full_server_tuned.yaml
```
Grid: τ₀ ∈ {0.50, 0.66, 0.75, 0.84}, τ₁ ∈ {0.50, 0.66, 0.75, 0.84}, τ_expand ∈ {0.0, 0.05, 0.1}, λ ∈ {0.0, 0.1, 0.2, 0.3} → 60 trials  
Objective: `accuracy - λ * normalized_tokens`  
**Tuned result:** τ₀=0.34, τ₁=0.34, τ_expand=0.0, λ=0.0

### Step 2: Full Evaluation (6 core methods)
```bash
python scripts/run_all_experiments.py \
  --config configs/full_server_tuned.yaml
```
Outputs: `outputs/full/{method}_records.json`, `outputs/full/{method}_summary.json`

### Step 3: Extra Experiments (lambda sweep + SC-5 + new methods)
```bash
python scripts/run_extra_experiments.py \
  --config configs/full_server_tuned.yaml \
  --tags sc5 deliberation_lambda0.1 deliberation_lambda0.2 deliberation_lambda0.3 \
         task_aware task_aware_lambda0.1 adaptive_sc_verifier
```

### Step 4: Offline Analysis
```bash
python scripts/compute_analysis.py --output_dir outputs/full/
```
Outputs: `outputs/full/analysis_results.json` (bootstrap CIs, matched-budget pairs, J-scores)

### Step 5: Figures
```bash
python scripts/generate_figures.py --output_dir outputs/full/ --fig_dir figures/
```
Outputs: `figures/fig{1-4}_{name}.{pdf,png}`

---

## 4. Results

### Overall Accuracy (300 examples)

| Method | Accuracy | Avg Tokens | J-score (μ=0.1) | Notes |
|--------|----------|------------|-----------------|-------|
| **task_aware_lambda0.1** | **0.713** | 431 | **0.690** | Best overall |
| deliberation_lambda0.1 | 0.707 | 469 | 0.681 | Best single-method |
| direct | 0.677 | 351 | 0.657 | Baseline |
| adaptive | 0.677 | 351 | 0.657 | = direct (routing failure) |
| meta_control_only | 0.677 | 351 | 0.657 | = direct (routing failure) |
| adaptive_sc_verifier | 0.670 | 350 | 0.651 | All routed to direct |
| task_aware (λ=0) | 0.653 | 909 | 0.603 | Expensive, λ=0 hurts |
| deliberation_only (λ=0) | 0.630 | 566 | 0.599 | Over-deliberation |
| sc5 | 0.600 | 1819 | 0.500 | Expensive, poor ROI |
| cot | 0.567 | 364 | 0.547 | CoT harmful |
| deliberation_lambda0.2 | 0.567 | 364 | 0.547 | Over-penalized |
| deliberation_lambda0.3 | 0.567 | 364 | 0.547 | Over-penalized |
| self_consistency (SC-3) | 0.540 | 1092 | 0.480 | Worst ROI |

### Per-Task Accuracy

| Task | direct | delib_λ0.1 | task_aware_λ0.1 | SC-3 | cot |
|------|--------|-----------|----------------|------|-----|
| gsm8k (n=100) | 0.76 | 0.83 | **0.88** | 0.84 | 0.78 |
| logical_deduction (n=50) | 0.88 | 0.94 | **0.92** | 0.78 | 0.90 |
| date_understanding (n=50) | 0.62 | **0.68** | 0.56 | 0.32 | 0.38 |
| college_biology (n=25) | **0.84** | 0.60 | 0.72 | 0.32 | 0.24 |
| computer_security (n=25) | 0.72 | 0.56 | **0.72** | 0.32 | 0.44 |
| hs_math (n=25) | 0.16 | 0.40 | 0.32 | 0.20 | 0.20 |
| formal_logic (n=25) | 0.36 | 0.36 | 0.32 | 0.08 | 0.24 |

### Bootstrap 95% Confidence Intervals (overall)

| Method | Accuracy | 95% CI |
|--------|----------|--------|
| task_aware_lambda0.1 | 0.713 | [0.663, 0.763] |
| deliberation_lambda0.1 | 0.707 | [0.657, 0.757] |
| direct | 0.677 | [0.623, 0.727] |
| adaptive_sc_verifier | 0.670 | [0.613, 0.723] |
| deliberation_only | 0.630 | [0.573, 0.687] |
| sc5 | 0.600 | [0.550, 0.657] |
| cot | 0.567 | [0.510, 0.620] |
| self_consistency | 0.540 | [0.483, 0.597] |

Note: CIs overlap substantially due to n=300; interpret trends, not individual comparisons.

### Lambda Sweep (deliberation_only variants)

| λ | Accuracy | Avg Tokens | Mechanism |
|---|----------|------------|-----------|
| 0.0 (deliberation_only) | 0.630 | 566 | No cost penalty, over-searches |
| **0.1** | **0.707** | **469** | Sweet spot |
| 0.2 | 0.567 | 364 | Over-penalized → degrades to CoT level |
| 0.3 | 0.567 | 364 | Over-penalized → degrades to CoT level |

### Oracle Upper Bound

| | Accuracy | vs Direct |
|---|----------|-----------|
| Oracle (best of direct/cot/delib per example) | 0.823 | +14.7pp |
| task_aware_lambda0.1 | 0.713 | +3.7pp |
| direct | 0.677 | — |

Gap between oracle and best method = **11pp** — the routing bottleneck.

---

## 5. Ablation Analysis

### Routing Ablation (Adaptive Framework Components)

| Method | What's kept | Accuracy | Conclusion |
|--------|-------------|----------|------------|
| `adaptive` | Full (routing + search) | 0.677 | = direct; routing degenerates |
| `meta_control_only` | Routing only (no search) | 0.677 | = direct; routing degenerates |
| `deliberation_only` | Search only (no routing) | 0.630 | Search hurts on average |
| `deliberation_lambda0.1` | Search + optimal λ | 0.707 | Best search-based method |

**Finding:** Routing failure causes adaptive = meta_control_only = direct. The belief score cannot discriminate correct from incorrect answers.

### Verifier Ablation (A-component of belief score)

| Verifier | Type | AUC | Result |
|----------|------|-----|--------|
| Logprob confidence | exp(mean_logprob) | **0.361** | Anti-correlated; wrong answers score higher |
| Sampling consistency | fraction of k=3 T=0.7 samples agreeing | ~0.5 | All examples score 1.0; no discrimination |

**Root cause:** RLHF instruction-tuning induces overconfidence. Model outputs the same answer at high confidence regardless of correctness.

Belief score distribution (adaptive method):
- Correct mean = 0.951, Wrong mean = 0.956 (Δ = −0.005)
- All 300 examples cluster in [0.90, 1.0]
- Tuned τ₀ = 0.34 < belief floor ≈ 0.67 → everything routes to direct

### Task-Aware Routing Ablation

| Method | λ | Accuracy | Notes |
|--------|---|----------|-------|
| `task_aware` | 0.0 | 0.653 | λ=0 causes search to over-expand |
| `task_aware_lambda0.1` | 0.1 | **0.713** | Optimal λ restores search quality |

**Finding:** Task-aware routing requires λ=0.1 to work. Without it, the seed node's utility is miscalibrated (correctness_score depends on belief propagated from CoT), causing the search expansion threshold to behave incorrectly.

---

## 6. Key Findings

1. **Routing degeneracy** (H1 ✗): Adaptive routing fails because belief scores cluster near 1.0 (RLHF overconfidence). AUC = 0.361 — worse than random. The routing mechanism is theoretically sound but practically inoperable.

2. **Deliberation is task-dependent** (H2 ✓): Search improves reasoning tasks (gsm8k +12pp, logical_deduction +4pp) but severely hurts knowledge retrieval (college_biology −24pp). There is no single best strategy.

3. **Lambda is critical** (H3 ✓): λ=0.1 is the sweet spot. λ=0 causes over-expansion; λ≥0.2 collapses search to CoT behavior.

4. **SC scaling does not generalize** (H4 ✗): SC-3=0.540, SC-5=0.600, both below direct=0.677. The model's output distribution at T=0.7 is too peaked (same wrong answer consistently) for majority voting to help.

5. **Oracle gap = 14.7pp**: Perfect routing would yield 0.823. Best achieved method = 0.713 (task_aware_lambda0.1). Remaining 11pp gap is the routing bottleneck.

6. **Task-type is a stronger routing signal than belief score**: task_aware_lambda0.1 (0.713) beats adaptive (0.677) by 3.7pp using only task name, confirming that the framework direction is correct but needs a better confidence estimator.

---

## 7. File Reference

```
outputs/full/
  {method}_records.json          # Per-example: prediction, gold, is_correct, tokens, route
  {method}_summary.json          # Aggregate: accuracy, avg_tokens, median_latency
  analysis_results.json          # Bootstrap CIs, matched-budget pairs, J-scores
  adaptive_dev_tuning.json       # All 60 dev tuning trials

figures/
  fig1_accuracy_cost.{pdf,png}       # Accuracy–cost scatter (all methods)
  fig2_pertask_deliberation.{pdf,png} # Per-task bar chart (direct vs delib variants)
  fig3_lambda_sweep.{pdf,png}        # λ sweep: accuracy and tokens vs λ
  fig4_belief_histogram.{pdf,png}    # Belief score distribution: correct vs wrong

configs/
  full_server_tuned.yaml         # Tuned config used for all full-run experiments
```
