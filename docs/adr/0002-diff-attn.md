# 0002 — Differential attention as the v1 research bet

**Status:** accepted · **Date:** 2026-09-18

## Question

Does differential attention (Ye et al., 2024) improve code-generation quality
at 485M when trained on our curriculum, versus the same model with standard GQA?

Not yet published in any released open model at ≤ 1B parameters.

## Setup

- Primary run: full model, `attn_kind: diff-gqa`, 100B-token curriculum, Stage-3
  distillation from Qwen2.5-Coder-7B.
- Ablation: identical model, `attn_kind: gqa` (λ frozen at 0), trained for 10B
  tokens only. Compared against the primary at its 10B checkpoint.

## Success / failure

- **Win:** diff run beats GQA baseline by ≥ 1.5 HumanEval points **and** ≥ 3%
  perplexity on the held-out shard, both at 10B tokens.
- **Loss:** diff run underperforms baseline on either metric.
- **Inconclusive:** metrics disagree or move within noise.

## On loss

Enact rollback (ADR-0001), retrain from the 10B checkpoint with `attn_kind:
gqa`, ship as `void-v1-standard`. Publish the negative result.

## Rollback cost

One config line. Optimizer state and data-iterator position are preserved.
